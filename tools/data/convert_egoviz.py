"""Convert EgoViz-120 clips (MCAP, Hugging Face storage bucket humaidtech/EgoViz-120) into the Ego2Wrist episode format.

Per clip -> <out>/<category>/<chunk_id>/:
  head.mp4 left.mp4 right.mp4   512x384, 30 fps, H.264 CRF 23; source 1920x1200 (16:10) letterboxed to
                                512x320 + 32 px black bars top and bottom. head = ZED left camera.
  hands.npz    joints_head (T,2,21,3) HaWoR joints in the head-left optical frame (OpenPose order, NaN when the
               hand transform is missing), detected (T,2) HaWoR detection flags
  poses.npz    head_odom (T,7) odom->head link (xyz + quat xyzw), T_head_wristcam (T,2,4,4) wrist optical frame
               in the head-left optical frame (HaWoR hand root x the rig's static mount), NaN when missing
  calib.json   K / D / size for head-left and both wrists, static transforms, letterbox
  meta.json    ids, category, title, description, n_frames, usable
Frames are aligned to the head video: for every head image, the nearest transform within 20 ms.
The MCAP is taken from --tmp (tools/data/prefetch_egoviz.py drops complete files there) or, after waiting WAIT_S,
downloaded directly; it is deleted after the clip is written (or on failure). --index is the bucket's index.jsonl
(hf buckets cp hf://buckets/humaidtech/EgoViz-120/index.jsonl ...). The paper uses the categories cleaning_kitchen,
cooking_food, managing_laundry, managing_trash, organizing_items, preparing_cleaning_supplies, preparing_food,
wiping_surfaces.

Usage:
  python tools/data/convert_egoviz.py --categories preparing_food,cooking_food --out $E2W_DATA_ROOT/proc/EgoViz \
      --tmp $E2W_DATA_ROOT/raw/EgoViz_tmp --index $E2W_DATA_ROOT/raw/EgoViz/index.jsonl --workers 8 [--limit 3]
"""
import argparse
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from multiprocessing import Pool

import numpy as np

BASE = "https://huggingface.co/buckets/humaidtech/EgoViz-120/resolve/"
VID = {"head": "/ego/zed_head/left/image/compressed", "left": "/wrist_left/image/compressed",
       "right": "/wrist_right/image/compressed"}
INFO = {"head": "/ego/zed_head/left/camera_info", "left": "/wrist_left/image/camera_info",
        "right": "/wrist_right/image/camera_info"}
VF = "scale=512:320:flags=area,pad=512:384:0:32:black"
OUT = TMP = None
WAIT_S = 3 * 3600


def fetch(path, dst, tries=12):
    """HF rate-limits unauthenticated clients (HTTP 429): back off exponentially, up to 10 min per wait."""
    for k in range(tries):
        try:
            urllib.request.urlretrieve(BASE + urllib.parse.quote(path), dst)
            return
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(min(600, 30 * 2 ** k))


def T_of(tr):
    """geometry_msgs/Transform -> 4x4."""
    q = tr.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return T


def nearest(ts_src, vals, ts_q, tol_ns=20_000_000):
    """For each query time, the value at the nearest source time within tol, else None."""
    if not ts_src:
        return [None] * len(ts_q)
    ts = np.asarray(ts_src)
    order = np.argsort(ts)
    ts = ts[order]
    out = []
    for t in ts_q:
        i = np.searchsorted(ts, t)
        c = [j for j in (i - 1, i) if 0 <= j < len(ts)]
        j = min(c, key=lambda j: abs(ts[j] - t))
        out.append(vals[order[j]] if abs(ts[j] - t) <= tol_ns else None)
    return out


def stamp(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def one(row):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory as PB
    from mcap_ros2.decoder import DecoderFactory as R2
    cid, cat = row["chunk_id"], row["category"]
    d = os.path.join(OUT, cat, cid)
    if os.path.exists(os.path.join(d, "meta.json")):
        return json.load(open(os.path.join(d, "meta.json")))
    os.makedirs(d, exist_ok=True)
    mc = os.path.join(TMP, cid + ".mcap")
    try:
        # the prefetcher (tools/data/prefetch_egoviz.py) drops complete files here; wait for it,
        # falling back to a direct (unauthenticated, rate-limited) download after WAIT_S
        t_wait = time.time()
        while not os.path.exists(mc) and time.time() - t_wait < WAIT_S:
            time.sleep(5)
        if not os.path.exists(mc):
            fetch(row["mcap_path"], mc)
        pk = {k: [] for k in VID}; vts = {k: [] for k in VID}; info = {}; fmt = {}
        tf_dyn = {}; tf_static = {}; kp = {0: ([], []), 1: ([], [])}; det = {0: ([], []), 1: ([], [])}
        odom = ([], [])
        topics = list(VID.values()) + list(INFO.values()) + ["/tf", "/tf_static", "/ego/zed_head/odom",
                  "/hands/left/keypoints", "/hands/right/keypoints", "/hands/left/detected", "/hands/right/detected"]
        with open(mc, "rb") as f:
            r = make_reader(f, decoder_factories=[PB(), R2()])
            for sch, ch, msg, m in r.iter_decoded_messages(topics=topics):
                t = ch.topic
                for k, top in VID.items():
                    if t == top:
                        pk[k].append(m.data); vts[k].append(msg.log_time); fmt[k] = m.format
                for k, top in INFO.items():
                    if t == top and k not in info:
                        info[k] = {"K": list(m.k), "D": list(m.d), "model": m.distortion_model,
                                   "width": m.width, "height": m.height, "frame": m.header.frame_id}
                if t == "/tf":
                    for tr in m.transforms:
                        key = (tr.header.frame_id, tr.child_frame_id)
                        tf_dyn.setdefault(key, ([], []))
                        tf_dyn[key][0].append(stamp(tr.header)); tf_dyn[key][1].append(T_of(tr.transform))
                elif t == "/tf_static":
                    for tr in m.transforms:
                        tf_static[(tr.header.frame_id, tr.child_frame_id)] = T_of(tr.transform)
                elif t == "/ego/zed_head/odom":
                    p = m.pose.pose
                    odom[0].append(stamp(m.header))
                    odom[1].append([p.position.x, p.position.y, p.position.z,
                                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
                elif t.endswith("/keypoints"):
                    h = 0 if "/left/" in t else 1
                    J = next((mk for mk in m.markers if mk.ns == "joints"), None)
                    if J is not None and len(J.points) == 21:
                        kp[h][0].append(stamp(J.header)); kp[h][1].append(np.array([[q.x, q.y, q.z] for q in J.points]))
                elif t.endswith("/detected"):
                    h = 0 if "/left/" in t else 1
                    det[h][0].append(msg.log_time); det[h][1].append(bool(m.data))
        # videos
        n = {}
        for k in VID:
            raw = os.path.join(TMP, f"{cid}_{k}.{fmt[k]}")
            open(raw, "wb").write(b"".join(pk[k]))
            dst = os.path.join(d, f"{k}.mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-threads", "2", "-r", "30", "-i", raw, "-vf", VF,
                            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-an", dst],
                           check=True)
            os.remove(raw)
            r2 = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_read_packets", "-of", "csv=p=0", dst], capture_output=True, text=True)
            n[k] = int(r2.stdout.strip() or -1)
        T = min(n.values())
        tq = vts["head"][:T]
        # geometry in the head-left optical frame
        head_link, head_opt = "zed_head_camera_link", info["head"]["frame"]
        T_link_opt = tf_static[(head_link, head_opt)]
        inv_opt = np.linalg.inv(T_link_opt)
        joints = np.full((T, 2, 21, 3), np.nan, np.float32)
        Tw = np.full((T, 2, 4, 4), np.nan, np.float32)
        detected = np.zeros((T, 2), bool)
        for h, side in ((0, "left"), (1, "right")):
            root = f"{side}_hand_root"
            key = (head_link, root)
            Ts = nearest(*tf_dyn.get(key, ([], [])), tq)
            Ks = nearest(kp[h][0], kp[h][1], tq)
            Ds = nearest(det[h][0], det[h][1], tq)
            wl = f"wrist_{side}_camera_link"
            T_mount = tf_static[(root, wl)] @ tf_static[(wl, info[side]["frame"])]
            for i in range(T):
                detected[i, h] = bool(Ds[i])
                if Ts[i] is None:
                    continue
                Th = inv_opt @ Ts[i]
                Tw[i, h] = Th @ T_mount
                if Ks[i] is not None:
                    P = np.c_[Ks[i], np.ones(21)] @ Th.T
                    joints[i, h] = P[:, :3]
        od = nearest(odom[0], odom[1], tq)
        head_odom = np.array([o if o is not None else [np.nan] * 7 for o in od], np.float32)
        np.savez_compressed(os.path.join(d, "hands.npz"), joints_head=joints, detected=detected)
        np.savez_compressed(os.path.join(d, "poses.npz"), head_odom=head_odom, T_head_wristcam=Tw)
        json.dump({"cams": info, "letterbox": {"scale": 512 / 1920, "pad_top": 32, "pad_left": 0},
                   "static": {f"{a}->{b}": v.tolist() for (a, b), v in tf_static.items()}},
                  open(os.path.join(d, "calib.json"), "w"), indent=1)
        # streams may differ by a trailing frame; accept <= 2 frames difference when all three start together
        t0 = {k: vts[k][0] for k in VID}
        ok = T > 0 and max(n.values()) - T <= 2 and all(abs(t0[k] - t0["head"]) < 20_000_000 for k in VID)
        meta = {"dataset": "EgoViz", "rig": "egoviz", "chunk_id": cid, "category": cat, "title": row.get("title"),
                "description": row.get("description"), "group_id": row.get("group_id"),
                "sequence_id": row.get("sequence_id"), "n_frames": int(T), "fps": 30, "video_frames": n,
                "start_offset_ms": {k: (t0[k] - t0["head"]) / 1e6 for k in VID},
                "hand_valid_rate": detected.mean(0).round(3).tolist(), "usable": bool(ok)}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)
        return meta
    except Exception as e:
        return {"chunk_id": cid, "category": cat, "usable": False, "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        if os.path.exists(mc):
            os.remove(mc)


def main():
    global OUT, TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--index", required=True, help="the bucket's index.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT, TMP = a.out, a.tmp
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    cats = a.categories.split(",")
    rows = [json.loads(l) for l in open(a.index)]
    rows = [r for r in rows if r["category"] in cats]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} clips in {cats}", flush=True)
    metas = []
    with Pool(a.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(one, rows)):
            metas.append(m)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} done, usable {sum(x['usable'] for x in metas)}", flush=True)
    ok = [m for m in metas if m["usable"]]
    print(f"{len(ok)}/{len(metas)} usable clips, {sum(m['n_frames'] for m in ok) / 30 / 3600:.2f} h -> {OUT}")
    for m in [m for m in metas if not m["usable"]][:5]:
        print("unusable:", m.get("chunk_id"), m.get("error"), m.get("video_frames"))


if __name__ == "__main__":
    main()
