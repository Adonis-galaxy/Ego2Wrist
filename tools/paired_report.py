"""Clip-paired comparison of two evaluated conditions, per dataset.

  python tools/paired_report.py --pair "ours vs WorldWander" <eval dir A>:real <eval dir B>:real [--pair ...]

Each side is <eval dir>:<cond>, i.e. the metrics_<cond>_shard*.jsonl files written by e2w.sample (or
tools/baseline_trivial.py). Only clips present on both sides are compared (same id and window start). Per dataset
(plus EgoPro test_unseen on its own): n, mean LPIPS of each side, paired LPIPS difference (first - second, both wrists
averaged) with a 95% CI from 10,000 bootstrap resamples of clips, the share of clips where the first side is better,
and the paired layout-PSNR difference. LPIPS lower is better, layout PSNR higher is better; "*" marks a CI that
excludes 0.
"""
import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from e2w.paths import INDEX  # noqa: E402


def load(spec):
    d, cond = spec.rsplit(":", 1)
    rows = {}
    for p in glob.glob(f"{d}/metrics_{cond}_shard*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if "dataset" not in r:  # evals on the single-dataset EgoTouch index (ids without the dataset prefix)
                r["dataset"], r["id"] = "EgoTouch", "EgoTouch/" + r["id"]
            rows[(r["id"], r["start"])] = r
    return rows


def val(r, m):
    return (r["left"][m] + r["right"][m]) / 2 if m in r["left"] else np.nan


def ci(x, n=10000):
    m = np.random.default_rng(0).choice(x, size=(n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def block(name, A, B, keys):
    if not keys:
        return
    a = np.array([val(A[k], "lpips") for k in keys]); b = np.array([val(B[k], "lpips") for k in keys])
    d = a - b; lo, hi = ci(d)
    dl = np.array([val(A[k], "lpsnr") - val(B[k], "lpsnr") for k in keys])
    lay = ""
    if not np.isnan(dl).any():
        llo, lhi = ci(dl)
        lay = f"  layout PSNR {dl.mean():+.2f} dB [{llo:+.2f}, {lhi:+.2f}]{'*' if llo > 0 or lhi < 0 else ''}"
    print(f"  {name:<22} n={len(keys):<4} LPIPS {a.mean():.3f} vs {b.mean():.3f}  diff {d.mean():+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]{'*' if hi < 0 or lo > 0 else ' '}  first better {np.mean(d < 0):.2f}{lay}")


ap = argparse.ArgumentParser()
ap.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "FIRST", "SECOND"), required=True)
ap.add_argument("--index", default=INDEX, help="index with the episode splits (for the EgoPro test_unseen rows)")
a = ap.parse_args()
split = {e["id"]: e["split"] for e in json.load(open(a.index))["episodes"]}
for name, f, s in a.pair:
    A, B = load(f), load(s)
    keys = sorted(set(A) & set(B))
    print(f"\n## {name}   ({f}  vs  {s}; {len(keys)} common clips of {len(A)} / {len(B)})")
    for ds in sorted({A[k]["dataset"] for k in keys}):
        block(ds, A, B, [k for k in keys if A[k]["dataset"] == ds])
        if ds == "EgoPro":
            block("EgoPro test_unseen", A, B, [k for k in keys if A[k]["dataset"] == ds and split.get(k[0]) == "test_unseen"])
    block("all", A, B, keys)
