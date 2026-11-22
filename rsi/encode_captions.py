"""Self-improvement step 3: umT5 embedding of every passing window's text (caption + rig sentence), one file per window
(<dir>/emb/<wid>.pt; tens of thousands of texts would be too big as one dict for every generation worker to load).
Resumable. With --follow it keeps encoding new rows of <dir>/captions.jsonl until <dir>/SCREEN_DONE exists; it writes
<dir>/ENCODE_DONE when finished. One GPU, ~12 GB.
  python rsi/encode_captions.py --dir $E2W_DATA_ROOT/wild/round1
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--batch", type=int, default=4)       # small batches leave room for a generation worker on the GPU
    ap.add_argument("--follow", action="store_true", help="keep waiting for new captions until <dir>/SCREEN_DONE exists")
    a = ap.parse_args()
    from e2w.model import T5_KW, load_text_encoder
    d = a.dir
    os.makedirs(f"{d}/emb", exist_ok=True)
    tok, enc = load_text_encoder()
    enc = enc.cuda()
    n = 0
    while True:
        screen_done = os.path.exists(f"{d}/SCREEN_DONE") or not a.follow
        rows = [json.loads(l) for l in open(f"{d}/captions.jsonl")] if os.path.exists(f"{d}/captions.jsonl") else []
        have = set(os.listdir(f"{d}/emb"))
        todo = [r for r in rows if f"{r['wid']}.pt" not in have]
        if not todo:
            if screen_done:
                break
            time.sleep(120)
            continue
        with torch.no_grad():
            for i in range(0, len(todo), a.batch):
                rs = todo[i:i + a.batch]
                ids = tok([r["text"] for r in rs], padding="max_length", max_length=T5_KW["text_length"], truncation=True,
                          add_special_tokens=True, return_tensors="pt")
                e = enc(ids.input_ids.cuda(), attention_mask=ids.attention_mask.cuda())[0]
                for j, r in enumerate(rs):
                    k = int(ids.attention_mask[j].sum())
                    p = f"{d}/emb/{r['wid']}.pt"
                    torch.save(e[j, :k].to(torch.bfloat16).cpu().clone(), p + ".part")
                    os.replace(p + ".part", p)
                n += len(rs)
        print(f"{time.strftime('%m-%d %H:%M:%S')} encoded {n} (of {len(rows)} captions so far)", flush=True)
    open(f"{d}/ENCODE_DONE", "w").write("done\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
