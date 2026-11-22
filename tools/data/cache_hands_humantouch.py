"""Ground-truth hand joints for converted HumanTouch episodes, in the head camera (OpenCV) frame (used by the
head-view-crop baseline; the index records the K of the head camera written here).

Manus skeleton (world frame, 25 joints) -> head tracker (Vive head pose) -> head camera via the per-episode
camera_from_head_tracker calibration. The axis convention was determined empirically: Euler order "xyz" on
rotation_rpy_deg, applied as
cam_from_head, with y negated in the head frame before and (x, y) negated in the camera frame after
(Unity -> OpenCV). Projections then land on the correct gloved hands (visual check).

Output: <out>/<folder>__<episode>.npy float32 (T, 2, 21, 3) OpenPose order, NaN when invalid, and a .json with the
pinhole K on the 512x384 letterboxed canvas {"fx","fy","cx","cy","W","H"} for hand_features().

  python tools/data/cache_hands_humantouch.py --proc $E2W_DATA_ROOT/proc/HumanTouch --out $E2W_DATA_ROOT/cache/hands_gt_humantouch
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Manus 25 -> OpenPose 21: wrist, thumb CMC/MCP/IP/tip, then MCP(proximal)/PIP/DIP/tip for each finger
MANUS_TO_OPENPOSE = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
PRE = np.array([1.0, -1.0, 1.0])
POST = np.array([-1.0, -1.0, 1.0])
OUT = None


def quatT(q):
    M = np.eye(4)
    M[:3, :3] = Rot.from_quat(q[3:7]).as_matrix()
    M[:3, 3] = q[:3]
    return M


def one(d):
    m = json.load(open(os.path.join(d, "meta.json")))
    name = f"{m['folder']}__{m['episode']}"
    dst = os.path.join(OUT, name + ".npy")
    if os.path.exists(dst) or not m.get("usable") or not m.get("camera_calibration"):
        return name, None
    cal = m["camera_calibration"]
    ex = cal["Calibration/camera/calibration_extrinsics.json"]
    K = np.array([v for k, v in cal.items() if "calibration_intrinsics" in k][0]["camera_matrix"]["matrix"])
    E = np.eye(4)
    E[:3, :3] = Rot.from_euler("xyz", ex["rotation_rpy_deg"], degrees=True).as_matrix()
    E[:3, 3] = ex["translation_m"]
    sk = np.load(os.path.join(d, "skeleton.npz"))
    po = np.load(os.path.join(d, "poses.npz"))
    T = m["n_frames"]
    J = np.full((T, 2, 21, 3), np.nan, np.float32)
    head_ok = po["pose_valid"][:, 0] if po["pose_valid"].ndim == 2 else np.ones(T, bool)
    for i in range(T):
        if not head_ok[i] or not np.isfinite(po["head"][i]).all():
            continue
        inv = np.linalg.inv(quatT(po["head"][i]))
        for h, side in ((0, "left"), (1, "right")):
            valid = sk[f"{side}_valid"][i][MANUS_TO_OPENPOSE]
            if not valid.all():
                continue
            P = sk[f"{side}_position"][i][MANUS_TO_OPENPOSE]
            ph = (inv @ np.c_[P, np.ones(21)].T).T[:, :3] * PRE
            pc = (E @ np.c_[ph, np.ones(21)].T).T[:, :3] * POST
            if (pc[:, 2] > 0.05).all():
                J[i, h] = pc
    s = 512 / 1920
    np.save(dst, J)
    json.dump({"fx": K[0, 0] * s, "fy": K[1, 1] * s, "cx": K[0, 2] * s, "cy": K[1, 2] * s + 48, "W": 512, "H": 384,
               "source": "manus+vive (ground truth)"}, open(os.path.join(OUT, name + ".json"), "w"))
    return name, np.isfinite(J[:, :, 0, 0]).mean(0)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(a.proc, "X*", "episode_*")) if os.path.exists(os.path.join(d, "meta.json")))
    with Pool(a.workers) as pool:
        res = [r for _, r in pool.imap_unordered(one, dirs, chunksize=8) if r is not None]
    if res:
        v = np.mean(res, 0)
        print(f"cached {len(res)} new episodes -> {OUT}; mean valid rate L {v[0]:.3f} R {v[1]:.3f}")
    else:
        print("nothing new to cache")


if __name__ == "__main__":
    main()
