"""Prefetcher for EgoViz-120 MCAPs (the `hf` CLI, recent enough for storage buckets, with a Hugging Face login).
Downloads <tmp>/<chunk_id>.mcap.part -> renames to .mcap when complete, in the same order tools/data/convert_egoviz.py
processes clips, keeping at most --max_pending finished-but-unconverted files. Clips already converted
(<proc>/<category>/<chunk_id>/meta.json) are skipped. --index is the bucket's index.jsonl.

  python tools/data/prefetch_egoviz.py --categories preparing_food,cooking_food --index $E2W_DATA_ROOT/raw/EgoViz/index.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
from e2w.paths import data_path  # noqa: E402

B = "hf://buckets/humaidtech/EgoViz-120/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True)
    ap.add_argument("--index", required=True, help="the bucket's index.jsonl")
    ap.add_argument("--tmp", default=data_path("raw", "EgoViz_tmp"))
    ap.add_argument("--proc", default=data_path("proc", "EgoViz"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_pending", type=int, default=48)
    a = ap.parse_args()
    cats = a.categories.split(",")
    rows = [json.loads(l) for l in open(a.index)]
    rows = [r for r in rows if r["category"] in cats]
    todo = [r for r in rows if not os.path.exists(os.path.join(a.proc, r["category"], r["chunk_id"], "meta.json"))]
    print(f"{len(todo)} of {len(rows)} clips to fetch", flush=True)

    def pending():
        return len([f for f in os.listdir(a.tmp) if f.endswith(".mcap")])

    def get(r):
        dst = os.path.join(a.tmp, r["chunk_id"] + ".mcap")
        if os.path.exists(dst) or os.path.exists(os.path.join(a.proc, r["category"], r["chunk_id"], "meta.json")):
            return "skip"
        while pending() >= a.max_pending:
            time.sleep(10)
        part = dst + ".part"
        for k in range(8):
            rc = subprocess.run(["hf", "buckets", "cp", B + r["mcap_path"], part], capture_output=True, text=True).returncode
            if rc == 0 and os.path.getsize(part) == r["mcap_bytes"]:
                os.rename(part, dst)
                return "ok"
            time.sleep(min(600, 30 * 2 ** k))
        return "fail"

    done = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(get, todo)):
            done[res] += 1
            if (i + 1) % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(todo)} {done} {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"finished {done}", flush=True)


if __name__ == "__main__":
    main()
