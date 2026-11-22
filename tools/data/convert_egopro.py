"""Convert EgoPro MCAPs (Hugging Face storage bucket LightwheelAI/EgoPro, gated; check the dataset license) into the
Ego2Wrist episode format. MCAPs are fetched into --tmp by tools/data/prefetch_egopro.py, which can run at the same time
(each worker waits up to 3 h for its MCAP to appear).

Per episode -> <out>/<task_slug>/<uuid>/:
  head.mp4 left.mp4 right.mp4   512x384, 30 fps, H.264 CRF 23.
     head  = head_left camera, 1920x1456 -> scaled to 512x388, centre-cropped 2 px top/bottom
     wrist = 1920x1536 (5:4)  -> scaled to 480x384, 16 px black bars left/right
  hands.npz  joints_head (T,2,21,3) hand joints in the head-left camera frame (OpenPose order), root_head (T,2,4,4)
             wrist-joint pose (with orientation) in that frame (for mount calibration), bad (T,2) bad-frame flags
  calib.json intrinsics (source and canvas) for head_left / left_wrist / right_wrist, crop/letterbox parameters
  meta.json  uuid, task, description, environment/scene ids, paradigm, semantic segments, n_frames, usable
Only paradigm == natural_human_hand is converted (others get a meta.json with usable=false, reason).
The MCAP is deleted after processing.

  python tools/data/convert_egopro.py --list $E2W_DATA_ROOT/raw/EgoPro/selection.jsonl \
      --out $E2W_DATA_ROOT/proc/EgoPro --tmp $E2W_DATA_ROOT/raw/EgoPro_tmp
"""
import argparse
import json
import os
import re
import subprocess
import time
from multiprocessing import Pool

import numpy as np

VID = {"head": "/sensor/camera/head_left/video", "left": "/sensor/camera/left_wrist/video",
       "right": "/sensor/camera/right_wrist/video"}
INTR = {"head": "/sensor/camera/head_left/intrinsic", "left": "/sensor/camera/left_wrist/intrinsic",
        "right": "/sensor/camera/right_wrist/intrinsic"}
VF = {"head": "scale=512:388:flags=area,crop=512:384:0:2",
      "left": "scale=480:384:flags=area,pad=512:384:16:0:black",
      "right": "scale=480:384:flags=area,pad=512:384:16:0:black"}
CANVAS = {"head": {"sx": 512 / 1920, "sy": 388 / 1456, "dx": 0, "dy": -2},
          "left": {"sx": 0.25, "sy": 0.25, "dx": 16, "dy": 0}, "right": {"sx": 0.25, "sy": 0.25, "dx": 16, "dy": 0}}
OUT = TMP = None
WAIT_S = 3 * 3600


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80]


def T_of(pos, quat):
    x, y, z, w = quat.x, quat.y, quat.z, quat.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [pos.x, pos.y, pos.z]
    return T


def ts_ns(h):
    return int(h.ts)


def nearest(ts_src, vals, ts_q, tol_ns=20_000_000):
    if not ts_src:
        return [None] * len(ts_q)
    ts = np.asarray(ts_src); order = np.argsort(ts); ts = ts[order]
    out = []
    for t in ts_q:
        i = np.searchsorted(ts, t)
        c = [j for j in (i - 1, i) if 0 <= j < len(ts)]
        j = min(c, key=lambda j: abs(ts[j] - t))
        out.append(vals[order[j]] if abs(ts[j] - t) <= tol_ns else None)
    return out


def one(row):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory as PB
    uid = row["uuid"]
    d = os.path.join(OUT, slug(row["task"]), uid)
    if os.path.exists(os.path.join(d, "meta.json")):
        return json.load(open(os.path.join(d, "meta.json")))
    mc = os.path.join(TMP, uid + ".mcap")
    t0 = time.time()
    while not os.path.exists(mc) and time.time() - t0 < WAIT_S:
        time.sleep(5)
    if not os.path.exists(mc):
        return {"uuid": uid, "usable": False, "error": "mcap never arrived"}
    os.makedirs(d, exist_ok=True)
    try:
        pk = {k: [] for k in VID}; vts = {k: [] for k in VID}; fmt = {}; intr = {}
        head_T = ([], []); hands = {0: ([], []), 1: ([], [])}; bad = {0: [], 1: []}; meta_msg = None; segs = []
        topics = list(VID.values()) + list(INTR.values()) + ["/sensor/camera/head_left/extrinsic", "/pose/left_hand",
                  "/pose/right_hand", "/annotation/bad_frame/pose/hand", "/session/metadata", "/annotation/semantic_segments"]
        with open(mc, "rb") as f:
            for sch, ch, msg, m in make_reader(f, decoder_factories=[PB()]).iter_decoded_messages(topics=topics):
                t = ch.topic
                for k, top in VID.items():
                    if t == top:
                        pk[k].append(m.data); vts[k].append(msg.log_time); fmt[k] = m.format
                for k, top in INTR.items():
                    if t == top and k not in intr:
                        intr[k] = {"K": list(m.K), "D": list(m.D), "model": m.distortion_model,
                                   "width": m.width, "height": m.height}
                if t == "/sensor/camera/head_left/extrinsic":
                    tr = m.transforms[0]
                    head_T[0].append(msg.log_time); head_T[1].append(T_of(tr.translation, tr.rotation))
                elif t in ("/pose/left_hand", "/pose/right_hand"):
                    h = 0 if "left" in t else 1
                    if len(m.transforms) == 21:
                        hands[h][0].append(msg.log_time)
                        hands[h][1].append([T_of(x.pos, x.quat) for x in m.transforms])
                elif t == "/annotation/bad_frame/pose/hand":
                    s = str(m)
                    bad[0].append((msg.log_time, "left" in s.lower() and len(s) > 80))
                    bad[1].append((msg.log_time, "right" in s.lower() and len(s) > 80))
                elif t == "/session/metadata":
                    meta_msg = m
                elif t == "/annotation/semantic_segments":      # one segment per message
                    sg = m.segment
                    for s_ in (sg if hasattr(sg, "__len__") and not hasattr(sg, "subtask_description") else [sg]):
                        segs.append({"subtask": s_.subtask_description, "start": getattr(s_, "start_time", 0.0),
                                     "end": s_.end_time, "task": getattr(m, "task_description", "")})
        paradigm = meta_msg.operator.collection_method.paradigm if meta_msg else "unknown"
        ti = meta_msg.task_info if meta_msg else None
        base = {"dataset": "EgoPro", "rig": "egopro", "uuid": uid, "task": row["task"],
                "description": ti.task_description if ti else None,
                "environment_id": ti.environment_id if ti else None, "scene_id": ti.scene_id if ti else None,
                "paradigm": paradigm, "segments": segs}
        if paradigm != "natural_human_hand":
            meta = dict(base, usable=False, reason=f"paradigm={paradigm}")
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)
            return meta
        n = {}
        for k in VID:
            raw = os.path.join(TMP, f"{uid}_{k}.{fmt[k]}")
            open(raw, "wb").write(b"".join(pk[k]))
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-threads", "2", "-r", "30", "-i", raw, "-vf", VF[k],
                            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
                            os.path.join(d, f"{k}.mp4")], check=True)
            os.remove(raw)
            r2 = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_read_packets", "-of", "csv=p=0", os.path.join(d, f"{k}.mp4")],
                                capture_output=True, text=True)
            n[k] = int(r2.stdout.strip() or -1)
        T = min(n.values())
        tq = vts["head"][:T]
        Th = nearest(*head_T, tq)
        J = np.full((T, 2, 21, 3), np.nan, np.float32)
        root = np.full((T, 2, 4, 4), np.nan, np.float32)
        B = np.zeros((T, 2), bool)
        for h in (0, 1):
            Hs = nearest(*hands[h], tq)
            bd = nearest([b[0] for b in bad[h]], [b[1] for b in bad[h]], tq)
            for i in range(T):
                B[i, h] = bool(bd[i])
                if Th[i] is None or Hs[i] is None:
                    continue
                inv = np.linalg.inv(Th[i])
                P = np.array([Tj[:3, 3] for Tj in Hs[i]])
                J[i, h] = (np.c_[P, np.ones(21)] @ inv.T)[:, :3]
                root[i, h] = inv @ Hs[i][0]
        np.savez_compressed(os.path.join(d, "hands.npz"), joints_head=J, root_head=root, bad=B)
        can = {}
        for k in VID:
            K = intr[k]["K"]; c = CANVAS[k]
            can[k] = {"fx": K[0] * c["sx"], "fy": K[4] * c["sy"], "cx": K[2] * c["sx"] + c["dx"],
                      "cy": K[5] * c["sy"] + c["dy"], "W": 512, "H": 384}
        json.dump({"source": intr, "canvas": can, "transform": CANVAS}, open(os.path.join(d, "calib.json"), "w"), indent=1)
        t0s = {k: vts[k][0] for k in VID}
        ok = T > 0 and max(n.values()) - T <= 2 and all(abs(t0s[k] - t0s["head"]) < 20_000_000 for k in VID)
        meta = dict(base, n_frames=int(T), fps=30, video_frames=n,
                    start_offset_ms={k: (t0s[k] - t0s["head"]) / 1e6 for k in VID},
                    hand_valid_rate=np.isfinite(J[:, :, 0, 0]).mean(0).round(3).tolist(), usable=bool(ok))
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)
        return meta
    except Exception as e:
        return {"uuid": uid, "usable": False, "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        if os.path.exists(mc):
            os.remove(mc)


def main():
    global OUT, TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="jsonl rows with uuid, task (the prefetcher's selection file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT, TMP = a.out, a.tmp
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    rows = [json.loads(l) for l in open(a.list)]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} episodes", flush=True)
    metas = []
    with Pool(a.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(one, rows)):
            metas.append(m)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} done, usable {sum(x['usable'] for x in metas)}", flush=True)
    ok = [m for m in metas if m["usable"]]
    print(f"{len(ok)}/{len(metas)} usable episodes, {sum(m['n_frames'] for m in ok) / 30 / 3600:.2f} h -> {OUT}")
    for m in [m for m in metas if not m["usable"]][:5]:
        print("unusable:", m.get("uuid"), m.get("reason") or m.get("error"))


if __name__ == "__main__":
    main()
