"""Self-improvement step 6: select the synthetic windows and build the training index of the next model.
  pool   synthesized windows with both wrists scorable by HCC and both passing the lens check (dark < 0.8;
         circular-fisheye looks never occur in the target EgoPro rig) -> window score = mean HCC of the two wrists
  arms   A  paired data only
         B  + every pool window
         C  + a random --frac of the pool (seeded)
         D  + the top --frac of the pool by HCC      (the self-improvement data; D vs C = same amount, other choice)
  index  <index_dir>/<tag>_<arm>.json = the paired index + one 49-frame episode per synthetic window
         (dataset "EgoSyn", split train, "panel" = the head | left | right video, text = caption + rig sentence)
  text   <prompts_out> = the paired prompt embeddings + one per synthetic text (train with --emb_on_cpu)
-> <dir>/ARMS.json (sizes, factories, score statistics, C/D overlap)
Each round adds only its own windows to the paired data and continues from the previous model.
  python rsi/build_round.py --dir $E2W_DATA_ROOT/wild/round1 --tag round1
"""
import argparse, glob, json, os, random, sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from e2w.paths import INDEX, PROMPTS, data_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="round1")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261020, help="seed of the random arm C")
    ap.add_argument("--base_index", default=INDEX, help="paired-data index (tools/data/build_multi_index.py)")
    ap.add_argument("--base_prompts", default=PROMPTS, help="its prompt embeddings (tools/data/encode_prompts.py)")
    ap.add_argument("--index_dir", default=data_path("index"))
    ap.add_argument("--prompts_out", default="", help="default: $E2W_DATA_ROOT/cache/prompt_embeds_<tag>.pt")
    ap.add_argument("--no_embeds", action="store_true")
    a = ap.parse_args()
    d = a.dir
    rows = pd.concat([pd.read_csv(p) for p in glob.glob(f"{d}/scores/rows_*.csv")], ignore_index=True).drop_duplicates(["wid", "view"])
    caps = {json.loads(l)["wid"]: json.loads(l)["text"] for l in open(f"{d}/captions.jsonl")}
    win = {json.loads(l)["wid"]: json.loads(l) for l in open(f"{d}/windows.jsonl")}
    p = rows.pivot_table(index="wid", columns="view", values=["xv_f", "dark"], aggfunc="first")
    p.columns = [f"{m}_{v}" for m, v in p.columns]
    p = p.reset_index()
    n_gen = len(p)
    p["scorable"] = p.xv_f_left.notna() & p.xv_f_right.notna()
    p["lens_ok"] = (p.dark_left < 0.8) & (p.dark_right < 0.8)
    pool = p[p.scorable & p.lens_ok & p.wid.isin(caps) & p.wid.map(lambda w: os.path.exists(f"{d}/gen/{w}.mp4"))].copy()
    pool["score"] = (pool.xv_f_left + pool.xv_f_right) / 2
    pool["factory"] = pool.wid.map(lambda w: win.get(w, {}).get("factory", w.split("_")[0]))
    pool = pool.sort_values("wid").reset_index(drop=True)
    k = int(round(a.frac * len(pool)))
    rng = random.Random(a.seed)
    arms = {"A": [], "B": list(pool.wid), "C": sorted(rng.sample(list(pool.wid), k)),
            "D": list(pool.sort_values("score", ascending=False).wid[:k])}
    base = json.load(open(a.base_index))
    os.makedirs(a.index_dir, exist_ok=True)
    gen_dir = os.path.abspath(f"{d}/gen")
    for arm, wids in arms.items():
        eps = [{"dataset": "EgoSyn", "rig": "egopro", "id": f"EgoSyn/{w}", "panel": f"{gen_dir}/{w}.mp4",
                "videos": {v: f"{gen_dir}/{w}.mp4" for v in ("head", "left", "right")}, "n_frames": 49, "split": "train",
                "text": caps[w], "frozen": {"left": [], "right": []}} for w in wids]
        json.dump({**base, "episodes": base["episodes"] + eps}, open(f"{a.index_dir}/{a.tag}_{arm}.json", "w"))
    sc = pool.set_index("wid").score
    summary = {"generated": n_gen, "scorable": int(p.scorable.sum()), "lens_ok": int(p.lens_ok.sum()), "pool": len(pool),
               "pool_hours": round(len(pool) * 49 / 30 / 3600, 2), "frac": a.frac,
               "arms": {arm: {"windows": len(w), "hours": round(len(w) * 49 / 30 / 3600, 2),
                              "factories": int(pool.set_index("wid").loc[w].factory.nunique()) if w else 0,
                              "score_mean": round(float(sc[w].mean()), 4) if w else None} for arm, w in arms.items()},
               "C_D_overlap": len(set(arms["C"]) & set(arms["D"]))}
    json.dump(summary, open(f"{d}/ARMS.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    if not a.no_embeds:
        import torch
        pe = torch.load(a.base_prompts)
        emb = dict(pe["embeds"])
        for w in pool.wid:
            emb[caps[w]] = torch.load(f"{d}/emb/{w}.pt")
        out = a.prompts_out or data_path("cache", f"prompt_embeds_{a.tag}.pt")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        torch.save({"embeds": emb}, out + ".part")
        os.replace(out + ".part", out)
        print(f"embeddings: {len(emb)} texts -> {out}", flush=True)


if __name__ == "__main__":
    main()
