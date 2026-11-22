"""Scan the raw EgoTouch download and write the episode index used by e2w/data.py.

Per episode: frame count and resolution of chest/left/right (decord), frozen wrist-frame ranges
from quality_issues.json, whether vive_poses carries data, and the official split (split.json).
An episode is usable only if all three videos open, are 640x480, and their frame counts agree
within 2 frames (n_frames = the minimum).

Usage:
  python tools/data/build_index_egotouch.py --root $E2W_DATA_ROOT/raw/EgoTouch --out $E2W_DATA_ROOT/index/egotouch.json
"""
import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool


def probe(ep_dir):
    import decord
    rec = {"frames": {}, "res": {}, "err": None}
    for v in ("chest", "left", "right"):
        p = os.path.join(ep_dir, f"{v}.mp4")
        try:
            vr = decord.VideoReader(p, num_threads=1)
            rec["frames"][v] = len(vr)
            h, w = vr[0].shape[:2]
            rec["res"][v] = [int(w), int(h)]
            rec[f"fps_{v}"] = float(vr.get_avg_fps())
        except Exception as e:  # missing or undecodable
            rec["err"] = f"{v}: {type(e).__name__}: {e}"[:200]
    return ep_dir, rec


def frozen_ranges(ep_dir):
    p = os.path.join(ep_dir, "quality_issues.json")
    out = {"left": [], "right": []}
    if not os.path.exists(p):
        return out
    d = json.load(open(p)).get("duplicate_frames", {})
    for cam, key in (("left", "left_hand_camera"), ("right", "right_hand_camera")):
        out[cam] = [[int(r["start_frame"]), int(r["end_frame"])] for r in d.get(key, [])]
    return out


def vive_present(ep_dir):
    p = os.path.join(ep_dir, "vive_poses.json")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        line = f.readline()
    try:
        return bool(json.loads(line).get("poses"))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--split_json", default=None, help="default: <root>/split.json")
    ap.add_argument("--scenes", default=None, help="comma list to restrict (smoke tests)")
    a = ap.parse_args()

    split_of = {}
    sj = json.load(open(a.split_json or os.path.join(a.root, "split.json")))
    for s, paths in sj.items():
        for p in paths:
            split_of["/".join(p[:-len(".hdf5")].split("/")[-3:])] = s

    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.root, "*", "*", "*", "chest.mp4")))
    if a.scenes:
        dirs = [d for d in dirs if os.path.relpath(d, a.root).split("/")[0] in a.scenes.split(",")]
    with Pool(a.workers) as pool:
        probes = dict(pool.map(probe, dirs, chunksize=8))

    eps, why = [], Counter()
    for d in dirs:
        r = probes[d]
        eid = os.path.relpath(d, a.root)
        scene, task, _ = eid.split("/")
        fr = r["frames"]
        reason = None
        if r["err"]:
            reason = "decode_error"
        elif any(tuple(x) != (640, 480) for x in r["res"].values()):
            reason = "not_640x480"
        elif max(fr.values()) - min(fr.values()) > 2:
            reason = "frame_count_mismatch"
        why[reason or "ok"] += 1
        eps.append({
            "id": eid, "scene": scene, "task": task,
            "split": split_of.get(eid, "none"),
            "n_frames": min(fr.values()) if fr else 0,
            "frames": fr, "res": r["res"], "err": r["err"],
            "frozen": frozen_ranges(d), "vive": vive_present(d),
            "usable": reason is None, "reject": reason,
        })

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"root": a.root, "episodes": eps}, open(a.out, "w"))
    print("episodes:", len(eps), dict(why))
    by = Counter((e["split"], e["usable"]) for e in eps)
    for k in sorted(by):
        print(k, by[k])
    hrs = Counter()
    for e in eps:
        if e["usable"]:
            hrs[e["split"]] += e["n_frames"] / 30 / 3600
    print("usable hours by split:", {k: round(v, 2) for k, v in hrs.items()})


if __name__ == "__main__":
    main()
