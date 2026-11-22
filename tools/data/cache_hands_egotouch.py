"""Cache the head-view WiLoR hand joints that EgoTouch provides (wilor_hands.json per episode) as one .npy per episode
(used by the head-view-crop baseline).

Output: <out>/<Scene>__<task>__<ts>.npy, float32 (T, 2, 21, 3) in WiLoR camera coordinates
(hand 0 = left, 1 = right; OpenPose joint order), NaN where the hand was not detected.
T = the episode's frame count in the index (wilor lines beyond it are dropped, missing lines are NaN).

Usage: python tools/data/cache_hands_egotouch.py --index $E2W_DATA_ROOT/index/egotouch.json --out $E2W_DATA_ROOT/cache/hands_wilor
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

ROOT = None


def one(args):
    ep_id, n = args
    src = os.path.join(ROOT, ep_id, "wilor_hands.json")
    J = np.full((n, 2, 21, 3), np.nan, dtype=np.float32)
    if os.path.exists(src):
        for line in open(src):
            d = json.loads(line)
            i = int(d["frame_index"])
            if 0 <= i < n:
                for h, k in ((0, "left_pos"), (1, "right_pos")):
                    if d.get(k) is not None:
                        J[i, h] = np.asarray(d[k], dtype=np.float32)
    return ep_id, J


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    idx = json.load(open(a.index))
    ROOT = idx["root"]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(e["id"], e["n_frames"]) for e in idx["episodes"] if e["usable"]]
    valid = []
    with Pool(a.workers) as pool:
        for ep_id, J in pool.imap_unordered(one, jobs, chunksize=8):
            np.save(os.path.join(a.out, ep_id.replace("/", "__") + ".npy"), J)
            valid.append(np.isfinite(J[..., 0, 0]).mean(0))
    v = np.mean(valid, 0)
    print(f"cached {len(jobs)} episodes -> {a.out}; mean per-episode detection rate L {v[0]:.3f} R {v[1]:.3f}")


if __name__ == "__main__":
    main()
