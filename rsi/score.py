"""Self-improvement step 5: score every synthesized video with HCC (hcc/hcc.py: DINOv3 ViT-7B/16, WiLoR hand detector
with the OWLv2 fallback, region 2.5 x the hand box) and a lens check ("dark" = share of near-black pixels in the 4
corner patches of a wrist view at frames 0, 24, 48; > 0.8 = circular-fisheye look, which the target EgoPro wrist rig
never produces).
Workers claim videos with mkdir under <dir>/locks_score/ (any number of workers, as in rsi/generate.py) and re-scan
<dir>/gen/; a worker leaves when nothing is left (with --follow: once <dir>/GEN_DONE also exists; create it when all
generation workers have finished), or when <dir>/stop_<host>_gpu<N> exists.
-> <dir>/scores/rows_<host>_gpu<N>.csv, one row per (wid, view)
  CUDA_VISIBLE_DEVICES=0 python rsi/score.py --dir $E2W_DATA_ROOT/wild/round1 --gpu_tag 0 \
      --hand_detector $E2W_DATA_ROOT/models/hand_det/detector.pt
"""
import argparse, glob, os, socket, sys, time, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from e2w.paths import HAND_DETECTOR  # noqa: E402
from hcc.hcc import DINOV3_ID, OWL_ID  # noqa: E402


def dark_corner(f):
    import cv2
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    ps = [g[20:60, 20:60], g[20:60, -60:-20], g[-60:-20, 20:60], g[-60:-20, -60:-20]]
    return float(np.mean([(p < 25).mean() for p in ps]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gpu_tag", default="0", help="worker name used in stop files and output names")
    ap.add_argument("--host", default=None, help="host name used in stop files and output names (default: hostname)")
    ap.add_argument("--shards", default="0,1", help="as rsi/generate.py: own half first, the other walked in reverse")
    ap.add_argument("--hand_detector", default=HAND_DETECTOR, help="WiLoR detector.pt")
    ap.add_argument("--dinov3", default=DINOV3_ID)
    ap.add_argument("--owl", default=OWL_ID)
    ap.add_argument("--follow", action="store_true", help="keep waiting for new videos until <dir>/GEN_DONE exists")
    a = ap.parse_args()
    shards = [int(s) for s in a.shards.split(",")]
    import imageio.v3 as iio
    import pandas as pd
    from hcc.hcc import HCC
    d = a.dir
    host = a.host or socket.gethostname().split(".")[0]
    os.makedirs(f"{d}/scores", exist_ok=True); os.makedirs(f"{d}/locks_score", exist_ok=True)
    stop = f"{d}/stop_{host}_gpu{a.gpu_tag}"
    outp = f"{d}/scores/rows_{host}_gpu{a.gpu_tag}.csv"
    scorer = HCC(a.hand_detector, dinov3=a.dinov3, owl=a.owl or None)
    n = 0
    while not os.path.exists(stop):
        done = set()
        for p in glob.glob(f"{d}/scores/rows_*.csv"):
            try:
                done |= set(pd.read_csv(p, usecols=["wid"]).wid)
            except Exception:
                pass
        todo = [os.path.basename(f)[:-4] for f in glob.glob(f"{d}/gen/*.mp4") if not f.endswith(".part.mp4")]
        todo = [w for w in sorted(todo) if w not in done and not os.path.exists(f"{d}/locks_score/{w}")]
        todo = [w for k, s in enumerate(shards) for w in (lambda js: js if k == 0 else js[::-1])(
            [w for w in todo if zlib.crc32(w.encode()) % 2 == s])]
        if not todo:
            if not a.follow or os.path.exists(f"{d}/GEN_DONE"):
                break
            time.sleep(120)
            continue
        for wid in todo[:100]:
            if os.path.exists(stop):
                break
            try:
                os.mkdir(f"{d}/locks_score/{wid}")
            except FileExistsError:
                continue
            x = iio.imread(f"{d}/gen/{wid}.mp4")
            W = x.shape[2] // 3
            head = x[:, :, :W]
            rows = []
            for vi, view in ((1, "left"), (2, "right")):
                v = x[:, :, vi * W:(vi + 1) * W]
                rng = np.random.default_rng(zlib.crc32(f"{wid}{view}".encode()))
                r = scorer(v, head, wid, view, rng)
                rows.append({"wid": wid, "view": view, "dark": round(float(np.mean([dark_corner(v[t]) for t in (0, 24, 48)])), 4), **r})
            pd.DataFrame(rows).to_csv(outp, mode="a", header=not os.path.exists(outp), index=False)
            n += 1
            if n % 50 == 0:
                print(f"{time.strftime('%m-%d %H:%M:%S')} scored {n}", flush=True)
    print(f"done after {n}", flush=True)


if __name__ == "__main__":
    main()
