"""Prefetcher for EgoPro MCAPs (the `hf` CLI with a Hugging Face login that has access to the gated bucket
LightwheelAI/EgoPro). Reads a selection jsonl (uuid, path, task, size), downloads <tmp>/<uuid>.mcap.part -> renames to
.mcap when complete, keeps at most --max_pending finished-but-unconverted files (tools/data/convert_egopro.py converts
and deletes them), skips episodes already converted.

The selection is every .mcap file of the bucket (the converter keeps only the natural_human_hand paradigm). With
--listing, the selection is first written from a JSON listing of the bucket, a list of {"path": ..., "size": ...}
objects (paths like "EgoProStandard-body/mcap/<Task Name>/<uuid>.mcap").

  python tools/data/prefetch_egopro.py --listing egopro_bucket.json --selection $E2W_DATA_ROOT/raw/EgoPro/selection.jsonl
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
from e2w.paths import data_path  # noqa: E402

B = "hf://buckets/LightwheelAI/EgoPro/"


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80]


def write_selection(listing, out):
    rows = []
    for x in json.load(open(listing)):
        if x["path"].endswith(".mcap"):
            rows.append({"uuid": os.path.basename(x["path"])[:-5], "path": x["path"], "task": x["path"].split("/")[-2],
                         "size": x["size"]})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        for r in sorted(rows, key=lambda r: r["path"]):
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} MCAPs -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--listing", default="", help="write --selection from this bucket listing first")
    ap.add_argument("--tmp", default=data_path("raw", "EgoPro_tmp"))
    ap.add_argument("--proc", default=data_path("proc", "EgoPro"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_pending", type=int, default=32)
    a = ap.parse_args()
    if a.listing:
        write_selection(a.listing, a.selection)
    rows = [json.loads(l) for l in open(a.selection)]
    os.makedirs(a.tmp, exist_ok=True)
    done_meta = lambda r: os.path.exists(os.path.join(a.proc, slug(r["task"]), r["uuid"], "meta.json"))
    todo = [r for r in rows if not done_meta(r)]
    print(f"{len(todo)} of {len(rows)} to fetch", flush=True)
    pending = lambda: len([f for f in os.listdir(a.tmp) if f.endswith(".mcap")])

    def get(r):
        dst = os.path.join(a.tmp, r["uuid"] + ".mcap")
        if os.path.exists(dst) or done_meta(r):
            return "skip"
        while pending() >= a.max_pending:
            time.sleep(10)
        part = dst + ".part"
        for k in range(8):
            rc = subprocess.run(["hf", "buckets", "cp", B + r["path"], part], capture_output=True, text=True).returncode
            if rc == 0 and os.path.exists(part) and os.path.getsize(part) == r["size"]:
                os.rename(part, dst)
                return "ok"
            time.sleep(min(600, 30 * 2 ** k))
        return "fail"

    done = {"ok": 0, "skip": 0, "fail": 0}; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(get, todo)):
            done[res] += 1
            if (i + 1) % 100 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(todo)} {done} {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"finished {done}", flush=True)


if __name__ == "__main__":
    main()
