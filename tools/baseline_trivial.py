"""Non-learned baselines of the main table, on the standard test set (375 clips, the order and windows of e2w.sample),
scored with the same metrics (e2w.sample.metrics: PSNR, SSIM, LPIPS, layout PSNR / SSIM) and written in e2w.sample's
format so tools/paired_report.py can pair them with any model evaluation:
  headcrop    "Head-view crop": the head video cropped around the same-side hand (e2w.multidata.zoom_crops: 4:3, 2.5x
              the median hand size, centre smoothed over 9 frames) and resized to 512 x 384; the hand boxes come from
              the hand joints in the index; when the hand is seen in too few frames, the lower quarter of the head
              view on that side
  other_real  "Another clip's wrist video": the REAL wrist videos of another clip of the same dataset (the derangement
              e2w.sample uses for its shuffle control): realistic but unrelated
-> <out>/<name>/metrics_real_shard0.jsonl (+ crop fallback counts in <out>/SUMMARY.json). CPU only.
The "Untuned Wan2.2" row is e2w.sample with --lora none.
  OMP_NUM_THREADS=8 python tools/baseline_trivial.py --out $E2W_DATA_ROOT/runs/baselines
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from e2w.paths import INDEX, data_path  # noqa: E402

CAPS = {"EgoTouch": 120, "HumanTouch": 80, "EgoViz": 75, "EgoPro": 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--out", default=data_path("runs", "baselines"))
    a = ap.parse_args()
    import cv2
    import lpips
    from e2w.data import load_index
    from e2w.multidata import MultiEvalWindows, hand_boxes, zoom_crops
    from e2w.sample import metrics
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    dev = torch.device("cpu")
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    ds = MultiEvalWindows(load_index(a.index), {"test_seen", "test_unseen"}, max_per_dataset=CAPS, with_hands=False)
    n = len(ds)
    grp = {}
    for i in range(n):
        grp.setdefault(ds.items[i][0]["dataset"], []).append(i)
    partner = [0] * n
    for g in grp.values():
        for k, i in enumerate(g):
            partner[i] = g[(k + len(g) // 2) % len(g)]
    outs = {}
    for name in ("headcrop", "other_real"):
        os.makedirs(f"{a.out}/{name}", exist_ok=True)
        outs[name] = open(f"{a.out}/{name}/metrics_real_shard0.jsonl", "w")
    fallback = {"left": 0, "right": 0}
    for i in range(n):
        it = ds[i]
        ep, start = ds.items[i]
        fr = it["frames"].numpy()                                   # [3, F, H, W, 3]
        H, W = fr.shape[2], fr.shape[3]
        try:
            boxes = hand_boxes(ep, start, fr.shape[1], width=W, height=H)
        except Exception:
            boxes = np.full((fr.shape[1], 2, 4), np.nan, np.float32)
        crops = []
        for k, side in ((0, "left"), (1, "right")):
            z = zoom_crops(fr[0], boxes[:, k], H, W)
            if z is None:                                           # lower quarter of the head view on that side
                fallback[side] += 1
                x0 = 0 if k == 0 else W // 2
                z = np.stack([cv2.resize(f[H // 2:, x0:x0 + W // 2], (W, H), interpolation=cv2.INTER_LINEAR) for f in fr[0]])
            crops.append(z)
        other = ds[partner[i]]["frames"].numpy()
        rec = {"id": it["id"], "start": it["start"], "task": it["task"], "cond": "real", "dataset": it["dataset"]}
        for name, pred in (("headcrop", crops), ("other_real", [other[1], other[2]])):
            r = dict(rec)
            for v, side in ((1, "left"), (2, "right")):
                r[side] = metrics(pred[v - 1], fr[v], lp, dev)
            outs[name].write(json.dumps(r) + "\n"); outs[name].flush()
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{n}", flush=True)
    json.dump({"clips": n, "headcrop_fallback": fallback}, open(f"{a.out}/SUMMARY.json", "w"), indent=1)
    print("done", fallback, flush=True)


if __name__ == "__main__":
    main()
