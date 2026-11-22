"""Encode one umT5 embedding per distinct episode text of an index (plus the empty prompt, used when the text is
dropped) so training and sampling never load T5. One GPU.

  python tools/data/encode_prompts.py --index $E2W_DATA_ROOT/index/paired.json --out $E2W_DATA_ROOT/cache/prompt_embeds_paired.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root

from e2w.data import load_index, prompt_for  # noqa: E402
from e2w.model import T5_KW, load_text_encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    idx = load_index(a.index)
    if idx.get("format") == "multi_v1":          # key = the episode text itself
        tasks = sorted({e["text"] for e in idx["episodes"]})
        prompts = {t: t for t in tasks}
    else:
        tasks = sorted({e["task"] for e in idx["episodes"]})
        prompts = {t: prompt_for(t) for t in tasks}
    prompts[""] = ""
    tok, enc = load_text_encoder()
    enc = enc.cuda()
    out = {}
    with torch.no_grad():
        for key, text in prompts.items():
            ids = tok([text], padding="max_length", max_length=T5_KW["text_length"], truncation=True,
                      add_special_tokens=True, return_tensors="pt")
            n = int(ids.attention_mask.sum())
            e = enc(ids.input_ids.cuda(), attention_mask=ids.attention_mask.cuda())[0]
            out[key] = e[0, :n].to(torch.bfloat16).cpu()
    torch.save({"prompts": prompts, "embeds": out}, a.out)
    print(f"encoded {len(out)} prompts -> {a.out}; example: {prompts[tasks[0]]!r} len={out[tasks[0]].shape}")


if __name__ == "__main__":
    main()
