"""Score every generated video in a folder with HCC (hcc/hcc.py), one CSV row per (video, wrist).

Video layouts (detected from the frame size, or set with --layout):
  panel   one row  [head | left | right]             (rsi/generate.py output; any 3-panel video, e.g. 384 x 1536)
  eval    two rows: top [GT head | GT left | GT right], bottom [head | pred left | pred right]   (e2w.sample mp4s);
          the generated (bottom) wrists are scored against the top-left head video; --real also scores the real
          (top) wrists, written with kind=real
Each panel is a 512 x 384 view (4:3). Videos need >= 49 frames (HCC compares frames 0, 4, ..., 48).
Resumable: videos already in --out are skipped. The per-video HCC is the mean of the two wrists' "hcc" (videos
whose HCC is undefined on a wrist, i.e. fewer than 3 frames with the hand found, get NaN).

  python -m hcc.score_videos --videos <dir> --out <dir>/hcc.csv \
      --hand_detector $E2W_DATA_ROOT/models/hand_det/detector.pt [--dinov3 <local dir or HF id>] [--shard k --n_shards n]
"""
import argparse, glob, os, sys, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from e2w.paths import HAND_DETECTOR  # noqa: E402
from hcc.hcc import DINOV3_ID, OWL_ID, HCC  # noqa: E402


def split_views(x, layout):
    """x [T, H, W, 3] -> (head, {"left": ..., "right": ...}, real wrists or None)"""
    W = x.shape[2] // 3
    if layout == "auto":
        layout = "eval" if x.shape[1] > W else "panel"                 # a 4:3 tile is wider than tall
    if layout == "panel":
        return x[:, :, :W], {"left": x[:, :, W:2 * W], "right": x[:, :, 2 * W:3 * W]}, None
    H = x.shape[1] // 2
    real = {"left": x[:, :H, W:2 * W], "right": x[:, :H, 2 * W:3 * W]}
    return x[:, :H, :W], {"left": x[:, H:, W:2 * W], "right": x[:, H:, 2 * W:3 * W]}, real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of .mp4 files (or a glob pattern)")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--layout", default="auto", choices=["auto", "panel", "eval"])
    ap.add_argument("--real", action="store_true", help="eval layout: also score the real wrist videos")
    ap.add_argument("--hand_detector", default=HAND_DETECTOR, help="WiLoR detector.pt")
    ap.add_argument("--dinov3", default=DINOV3_ID)
    ap.add_argument("--owl", default=OWL_ID, help="'' disables the OWLv2 fallback")
    ap.add_argument("--backbone", default="dinov3_7b", choices=["dinov3_7b", "dinov2_s"])
    ap.add_argument("--roi", type=float, default=2.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    a = ap.parse_args()
    import imageio.v3 as iio
    import pandas as pd
    files = sorted(glob.glob(os.path.join(a.videos, "*.mp4")) if os.path.isdir(a.videos) else glob.glob(a.videos))
    files = [f for f in files if not f.endswith(".part.mp4")][a.shard::a.n_shards]
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out):
        done = set(pd.read_csv(a.out, usecols=["video"]).video)
    todo = [f for f in files if os.path.basename(f)[:-4] not in done]
    print(f"{len(files)} videos, {len(todo)} to score", flush=True)
    if not todo:
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    scorer = HCC(a.hand_detector, dinov3=a.dinov3, owl=a.owl or None, backbone=a.backbone, roi=a.roi)
    for n, f in enumerate(todo):
        name = os.path.basename(f)[:-4]
        head, gen, real = split_views(iio.imread(f), a.layout)
        rows = []
        for kind, views in (("gen", gen), ("real", real if a.real else None)):
            if views is None:
                continue
            for view, v in views.items():
                rng = np.random.default_rng(zlib.crc32(f"{name}{view}".encode()))
                rows.append({"video": name, "kind": kind, "view": view, **scorer(v, head, name, view, rng)})
        pd.DataFrame(rows).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        g = [r["hcc"] for r in rows if r["kind"] == "gen"]
        print(f"[{n + 1}/{len(todo)}] {name}: HCC {np.mean(g):.4f} (L {g[0]:.4f} R {g[1]:.4f})", flush=True)
    df = pd.read_csv(a.out)
    per_video = df[df.kind == "gen"].groupby("video").hcc.mean()
    print(f"done: {len(per_video)} videos, mean HCC {per_video.mean():.4f} ({per_video.isna().sum()} undefined)", flush=True)


if __name__ == "__main__":
    main()
