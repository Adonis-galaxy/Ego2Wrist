"""Convert one HumanTouch task folder (LeRobot v2.1, 60 fps HEVC; Hugging Face dataset chuqiaoLyu/Xspark-HumanTouch,
one folder per task X001..X010) into the Ego2Wrist episode format.

Per episode -> <out>/<task_folder>/episode_XXXXXX/:
  head.mp4  left.mp4  right.mp4   512x384, 30 fps, H.264 CRF 23. Exactly the even source frames
                                  (0, 2, 4, ...), so row k of every array below is video frame k.
                                  Head is 16:9 -> letterboxed (scaled to 512x288, 48 px black bars);
                                  wrists are 4:3 -> scaled.
  poses.npz     head / wrist_left / wrist_right (T,7) Vive, pose_valid (T,3)
  tactile.npz   raw_left / raw_right (T,460), tactile_valid (T,2), patch_pressure (T,12), patch_valid (T,1)
  skeleton.npz  left/right position (T,25,3), orientation (T,25,4), valid (T,25)   [Manus]
  meta.json     ids, task name (from the episode metadata, NOT the folder name), frame counts, letterbox
An episode is marked usable only if all three videos decode to exactly T frames and the parquet has 2T or 2T-1 rows.

Usage:
  hf download chuqiaoLyu/Xspark-HumanTouch --repo-type dataset --include "X007/*" --local-dir $E2W_DATA_ROOT/raw/HumanTouch
  python tools/data/convert_humantouch.py --src $E2W_DATA_ROOT/raw/HumanTouch/X007 --out $E2W_DATA_ROOT/proc/HumanTouch [--limit 5]
"""
import argparse
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np

CAMS = {"head": "observation.images.cam_head", "left": "observation.images.cam_left_wrist",
        "right": "observation.images.cam_right_wrist"}
VF = {"head": "scale=512:288:flags=area,pad=512:384:0:48:black", "left": "scale=512:384:flags=area",
      "right": "scale=512:384:flags=area"}
SRC = OUT = None


def count_frames(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0", p], capture_output=True, text=True)
    return int(r.stdout.strip() or -1)


def find(pattern):
    """The chunk folder in HumanTouch does not follow episode_index // chunks_size; locate files by name."""
    import glob
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {len(hits)} matches")
    return hits[0]


def one(ep):
    try:
        return _one(ep)
    except Exception as e:
        return {"episode": f"episode_{ep['episode_index']:06d}", "usable": False,
                "error": f"{type(e).__name__}: {e}"[:300], "n_frames": 0, "video_frames": {}, "parquet_rows": -1,
                "src_frames": ep.get("length"), "task_id": None, "task_name": None}


def _one(ep):
    import pyarrow.parquet as pq
    idx, n60 = ep["episode_index"], ep["length"]
    name = f"episode_{idx:06d}"
    d = os.path.join(OUT, name)
    if os.path.exists(os.path.join(d, "meta.json")):
        return json.load(open(os.path.join(d, "meta.json")))
    os.makedirs(d, exist_ok=True)
    T = (n60 + 1) // 2
    frames = {}
    for v, key in CAMS.items():
        src = find(os.path.join(SRC, "videos", "chunk-*", key, f"{name}.mp4"))
        dst = os.path.join(d, f"{v}.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-threads", "2", "-i", src, "-vf",
                        f"select='not(mod(n\\,2))',setpts=N/(30*TB),{VF[v]}", "-vsync", "0",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                        "-an", dst], check=True)
        frames[v] = count_frames(dst)
    t = pq.read_table(find(os.path.join(SRC, "data", "chunk-*", f"{name}.parquet"))).to_pandas()
    rows = len(t)
    sub = t.iloc[::2]
    col = lambda c, dt=np.float32: np.stack(sub[c].to_numpy()).astype(dt)
    np.savez_compressed(os.path.join(d, "poses.npz"),
                        head=col("observation.human.pose.head"), wrist_left=col("observation.human.pose.wrist_left"),
                        wrist_right=col("observation.human.pose.wrist_right"),
                        pose_valid=col("observation.human.pose.valid", bool))
    np.savez_compressed(os.path.join(d, "tactile.npz"),
                        raw_left=col("observation.tactile.raw_left"), raw_right=col("observation.tactile.raw_right"),
                        tactile_valid=col("observation.tactile.valid", bool),
                        patch_pressure=col("observation.tactile.patch_pressure"),
                        patch_valid=col("observation.tactile.patch_pressure_valid", bool))
    sk = {}
    for h in ("left", "right"):
        base = f"observation.human.hand_skeleton.{h}"
        sk[f"{h}_position"] = np.stack([np.stack(x) for x in sub[f"{base}.position"]]).astype(np.float32)
        sk[f"{h}_orientation"] = np.stack([np.stack(x) for x in sub[f"{base}.orientation"]]).astype(np.float32)
        sk[f"{h}_valid"] = np.stack(sub[f"{base}.valid"].to_numpy()).astype(bool)
    np.savez_compressed(os.path.join(d, "skeleton.npz"), **sk)
    em = json.load(open(os.path.join(SRC, "meta", "humantouch", "episodes", f"{name}.json")))
    ok = all(n == T for n in frames.values()) and rows in (n60, n60 - 1) and len(sub) == T
    meta = {"dataset": "HumanTouch", "rig": "humantouch", "folder": os.path.basename(SRC.rstrip("/")),
            "episode": name, "session_id": em["session"]["session_id"], "task_id": em["block"]["task_id"],
            "task_name": em["block"]["task_name"], "quality": em["session"].get("quality_status"),
            "n_frames": T, "fps": 30, "src_fps": 60, "src_frames": n60, "parquet_rows": rows, "video_frames": frames,
            "letterbox": {"head": {"scale": 512 / 1920, "pad_top": 48, "pad_left": 0}},
            "src_res": {"head": [1920, 1080], "left": [1280, 960], "right": [1280, 960]}, "usable": bool(ok),
            # kept verbatim: needed to put the Manus skeleton / Vive poses into the head camera (raw is deleted)
            "camera_calibration": em.get("camera_calibration"), "joint_names": em.get("joint_names"),
            "data_contract": em.get("data_contract"), "semantic_annotations": em.get("semantic_annotations")}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)
    return meta


def main():
    global SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    SRC = a.src
    OUT = os.path.join(a.out, os.path.basename(a.src.rstrip("/")))
    os.makedirs(OUT, exist_ok=True)
    eps = [json.loads(l) for l in open(os.path.join(SRC, "meta", "episodes.jsonl"))]
    if a.limit:
        eps = eps[:a.limit]
    with Pool(a.workers) as pool:
        metas = list(pool.imap_unordered(one, eps))
    ok = [m for m in metas if m["usable"]]
    hrs = sum(m["n_frames"] for m in ok) / 30 / 3600
    tasks = sorted({(m["task_id"], m["task_name"]) for m in metas if m.get("task_id")})
    print(f"{len(ok)}/{len(metas)} usable episodes, {hrs:.2f} h, tasks {tasks}; out {OUT}")
    bad = [m for m in metas if not m["usable"]][:5]
    for m in bad:
        print("unusable:", m["episode"], m.get("error"), m["video_frames"], m["parquet_rows"], m["src_frames"])


if __name__ == "__main__":
    main()
