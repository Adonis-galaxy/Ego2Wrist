"""Download a diverse subset of Build AI's Egocentric-10K (Hugging Face dataset builddotai/Egocentric-10K; accept its
terms on the Hub and log in with `hf auth login` first): for every factory, --workers_per_factory random workers and
--shards_per_worker random WebDataset tars of each, plus each chosen worker's intrinsics.json. The defaults reproduce
the paper's subset (85 factories x 2 workers x 2 shards, ~340 tars, ~270 GB, 2,916 clips of 1080p30 head-mounted
fisheye video). The seeded selection is written to <out>/MANIFEST.json. Before every file the free space on the target
disk must exceed the file size + --margin_gb, else it stops. Parallel downloads with retries; finished files are
skipped, so it can be rerun to resume.
  python rsi/download_ego10k.py --out $E2W_DATA_ROOT/raw/Egocentric10K
"""
import argparse, collections, json, os, random, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from e2w.paths import data_path  # noqa: E402

REPO = "builddotai/Egocentric-10K"


def say(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=data_path("raw", "Egocentric10K"))
    ap.add_argument("--seed", type=int, default=20261014, help="selection seed of the paper's subset")
    ap.add_argument("--workers_per_factory", type=int, default=2)
    ap.add_argument("--shards_per_worker", type=int, default=2)
    ap.add_argument("--margin_gb", type=float, default=500)
    ap.add_argument("--jobs", type=int, default=4, help="parallel downloads")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    rng = random.Random(a.seed)
    margin = a.margin_gb * 2**30
    os.makedirs(a.out, exist_ok=True)
    info = HfApi().dataset_info(REPO, files_metadata=True)
    size = {s.rfilename: s.size or 0 for s in info.siblings}
    workers = collections.defaultdict(list)
    for f in size:
        if f.endswith(".tar"):
            workers["/".join(f.split("/")[:3])].append(f)
    byfac = collections.defaultdict(list)
    for w in sorted(workers):
        byfac[w.split("/")[0]].append(w)
    pick = []
    for fac in sorted(byfac):
        for w in rng.sample(byfac[fac], min(a.workers_per_factory, len(byfac[fac]))):
            pick += rng.sample(sorted(workers[w]), min(a.shards_per_worker, len(workers[w])))
            intr = f"{w}/intrinsics.json"
            if intr in size:
                pick.append(intr)
    total = sum(size[f] for f in pick)
    json.dump({"repo": REPO, "seed": a.seed, "files": pick, "bytes": total}, open(f"{a.out}/MANIFEST.json", "w"), indent=1)
    say(f"selected {sum(f.endswith('.tar') for f in pick)} shards from {len(byfac)} factories, {total / 2**30:.0f} GB")

    def get(f):
        dst = os.path.join(a.out, f)
        if os.path.exists(dst) and os.path.getsize(dst) == size[f]:
            return "skip"
        if shutil.disk_usage(a.out).free < size[f] + margin:
            return "nospace"
        for att in range(5):
            try:
                hf_hub_download(REPO, f, repo_type="dataset", local_dir=a.out)
                return "ok"
            except Exception as e:
                say(f"retry {att + 1} {f}: {type(e).__name__} {str(e)[:120]}")
                time.sleep(30 * (att + 1))
        return "fail"

    done = collections.Counter()
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, (f, r) in enumerate(zip(pick, ex.map(get, pick))):
            done[r] += 1
            if r == "nospace":
                say(f"stopping: less than file + {a.margin_gb:.0f} GB free on {a.out}"); break
            if (i + 1) % 20 == 0:
                say(f"{i + 1}/{len(pick)} {dict(done)}; free {shutil.disk_usage(a.out).free / 2**40:.2f} TB")
    say(f"done {dict(done)}")
    open(f"{a.out}/DONE", "w").write(json.dumps(dict(done)) + "\n")


if __name__ == "__main__":
    main()
