"""Generate left/right wrist videos from the head video and score them against the real wrist videos.

Conditions (same checkpoint, same initial noise per clip, so they are paired):
  real     the clip's own head video
  shuffle  the head video of another clip of the same dataset (a fixed derangement), everything else identical
  nohead   head removed (grey + mask 0), the branch trained with p_drop_head
If the model uses the head view, `real` must beat both controls clip by clip.

Head guidance: --cfg w applies classifier-free guidance to the head condition only (both branches see the same text),
v = v_nohead + w (v_head - v_nohead); the paper's main table uses --cfg 3, 30 Euler steps.

Per clip and view it writes PSNR / SSIM / LPIPS and layout PSNR / SSIM (1/16 resolution) to
<out>/metrics_<cond>_shard<k>.jsonl and, for --save_mp4 clips spread over the set, a side-by-side mp4
(top: GT head | GT left | GT right, bottom: head | pred L | pred R). Reruns skip clips already in the jsonl.

Standard test set of the paper (375 clips), one of 4 shards:
  python -m e2w.sample --lora <run>/step_N/lora_ema.safetensors --out <dir> --cfg 3 --conds real \
      --max_per_dataset EgoTouch=120,HumanTouch=80,EgoViz=75,EgoPro=100 --shard 0 --n_shards 4
"""
import argparse
import json
import os

import imageio.v3 as iio
import numpy as np
import torch
from safetensors.torch import load_file

from e2w import paths
from e2w.data import EvalWindows, load_index, to_model_range
from e2w.model import (N_VIEWS, add_lora, build_condition, decode_views, encode_views, grey_latent, load_transformer,
                       load_vae, seq_len_of, shift_sigma, token_timesteps)


@torch.no_grad()
def generate(tf, z_head_panel, z_grey, ctx, steps, cfg, seed, drop_head=False):
    """Euler sampling of the flow ODE on the shifted schedule.
    z_head_panel: [1, C, f, h, W] whose head columns hold the conditioning head latent (the wrist columns are ignored).
    cfg: head-guidance weight (1 = no guidance). drop_head: sample the unconditional (no-head) branch."""
    dev, dt = z_head_panel.device, z_head_panel.dtype
    B, C, f, h, W = z_head_panel.shape
    w = W // N_VIEWS
    dh = torch.tensor([drop_head], device=dev)
    y, head = build_condition(z_head_panel, z_grey, dh)
    y_u, head_u = build_condition(z_head_panel, z_grey, torch.tensor([True], device=dev))
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(z_head_panel.shape, generator=g, device=dev, dtype=torch.float32)
    sig = shift_sigma(torch.linspace(1.0, 0.0, steps + 1, device=dev))
    seq = seq_len_of(f, h, W)
    for i in range(steps):
        x[..., :w] = head.float()
        ttok = token_timesteps((sig[i] * 1000.0).view(1), f, h, W)
        with torch.autocast("cuda", dtype=dt):
            v = tf(x=x.to(dt), t=ttok, context=ctx, seq_len=seq, y=y).float()
            if cfg != 1.0 and not drop_head:
                xu = x.clone(); xu[..., :w] = head_u.float()
                vu = tf(x=xu.to(dt), t=ttok, context=ctx, seq_len=seq, y=y_u).float()
                v = vu + cfg * (v - vu)
        x = x + (sig[i + 1] - sig[i]) * v
    x[..., :w] = head.float()
    return x.to(dt)


def to_u8(v):  # [1, 3, F, H, W] in [-1, 1] -> [F, H, W, 3] uint8
    return ((v[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()


def metrics(pred, gt, lp, dev):
    from skimage.metrics import structural_similarity as ssim
    p, g = pred.astype(np.float32) / 255, gt.astype(np.float32) / 255
    mse = ((p - g) ** 2).reshape(len(p), -1).mean(1)
    psnr = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))
    ss = float(np.mean([ssim(a, b, channel_axis=2, data_range=1.0) for a, b in zip(p[::4], g[::4])]))
    tp = torch.from_numpy(p).permute(0, 3, 1, 2).to(dev) * 2 - 1
    tg = torch.from_numpy(g).permute(0, 3, 1, 2).to(dev) * 2 - 1
    with torch.no_grad():
        l = float(torch.cat([lp(tp[i:i + 16], tg[i:i + 16]).flatten() for i in range(0, len(tp), 16)]).mean())
    # layout: 1/16-resolution PSNR / SSIM (coarse orientation and placement, blind to texture)
    import cv2
    ds = lambda X: np.stack([cv2.resize(f, (X.shape[2] // 16, X.shape[1] // 16), interpolation=cv2.INTER_AREA) for f in X])
    pl, gl = ds(p), ds(g)
    lmse = ((pl - gl) ** 2).reshape(len(pl), -1).mean(1)
    lpsnr = float(np.mean(10 * np.log10(1.0 / np.maximum(lmse, 1e-10))))
    lss = float(np.mean([ssim(a, b, channel_axis=2, data_range=1.0, win_size=7) for a, b in zip(pl[::4], gl[::4])]))
    return {"psnr": psnr, "ssim": ss, "lpips": l, "lpsnr": lpsnr, "lssim": lss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=paths.INDEX)
    ap.add_argument("--prompts", default=paths.PROMPTS)
    ap.add_argument("--lora", required=True, help="LoRA weights (lora_ema.safetensors / lora.safetensors), or 'none' "
                    "for the untuned base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test_seen,test_unseen")
    ap.add_argument("--max_items", type=int, default=120, help="single-dataset (EgoTouch) index only")
    ap.add_argument("--conds", default="real,shuffle,nohead")
    ap.add_argument("--max_per_dataset", default="0",
                    help="mixed index only: eval clips per dataset, one number or name=n,... (unlisted datasets: all)")
    ap.add_argument("--datasets", default="", help="mixed index only: comma-separated datasets to evaluate (default: all)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--cfg", type=float, default=1.0, help="head-guidance weight w (paper: 3)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--save_mp4", type=int, default=12, help="-1 = save every clip")
    ap.add_argument("--seed_base", type=int, default=1000, help="initial noise seed = seed_base + clip index")
    ap.add_argument("--only", default="", help="comma-separated clip indices to run, seeds unchanged")
    a = ap.parse_args()

    dev, dt = torch.device("cuda"), torch.bfloat16
    os.makedirs(a.out, exist_ok=True)
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    vae = load_vae().to(dev, dtype=dt)
    tf = load_transformer(dt).to(dev).eval()
    if a.lora == "none":                                    # baseline: the untuned Wan2.2-Fun-5B-InP
        print("no LoRA: base model", flush=True)
    else:
        net = add_lora(tf, rank=a.rank, alpha=a.rank).to(dev)
        net.load_state_dict(load_file(a.lora), strict=True)
        up = sum(float(v.float().pow(2).sum()) for k, v in net.state_dict().items() if "lora_up" in k) ** 0.5
        assert up > 0, "loaded LoRA has all-zero lora_up"
        print(f"loaded {a.lora} |lora_up|={up:.3f}", flush=True)

    index = load_index(a.index)
    if index.get("format") == "multi_v1":
        from e2w.multidata import MultiEvalWindows
        cap = ({k: int(v) for k, v in (x.split("=") for x in a.max_per_dataset.split(","))}
               if "=" in a.max_per_dataset else int(a.max_per_dataset) or None)
        ds = MultiEvalWindows(index, set(a.splits.split(",")), max_per_dataset=cap,
                              datasets=set(a.datasets.split(",")) if a.datasets else None,
                              with_hands=False, height=a.height, width=a.width)
    else:
        ds = EvalWindows(index, set(a.splits.split(",")), max_items=a.max_items)
    n = len(ds)
    # derangement for the shuffle control, within a dataset (a partner from another rig would also swap the rig text)
    grp = {}
    for i in range(n):
        grp.setdefault(ds.items[i][0].get("dataset", "") if hasattr(ds, "items") else "", []).append(i)
    partner = [0] * n
    for g in grp.values():
        for k, i in enumerate(g):
            partner[i] = g[(k + len(g) // 2) % len(g)]
    emb = {k: v.to(dev, dt) for k, v in torch.load(a.prompts)["embeds"].items()}
    z_grey = None
    conds = a.conds.split(",")
    files = {c: open(os.path.join(a.out, f"metrics_{c}_shard{a.shard}.jsonl"), "a") for c in conds}
    done = set()
    for c in conds:
        p = os.path.join(a.out, f"metrics_{c}_shard{a.shard}.jsonl")
        done |= {(c, json.loads(l)["id"]) for l in open(p)} if os.path.getsize(p) else set()
    mine = [i for i in range(n) if i % a.n_shards == a.shard]
    if a.only:
        mine = [i for i in mine if i in {int(x) for x in a.only.split(",")}]
    print(f"{n} eval clips, shard {a.shard}/{a.n_shards}: {len(mine)} clips, conds {conds}", flush=True)
    for i in mine:
        it = ds[i]
        if all((c, it["id"]) in done for c in conds):
            continue
        fr = to_model_range(it["frames"][None], dev, dt)
        z = encode_views(vae, fr)
        if z_grey is None:
            z_grey = grey_latent(vae, fr.shape[3], fr.shape[4], fr.shape[5], dev, dt)
        w = z.shape[-1] // N_VIEWS
        gt = [to_u8(fr[:, v]) for v in range(N_VIEWS)]
        for c in conds:
            if (c, it["id"]) in done:
                continue
            zc = z.clone()
            if c == "shuffle":
                other = ds[partner[i]]
                zc[..., :w] = encode_views(vae, to_model_range(other["frames"][None], dev, dt)[:, :1])
                ctx = [emb[other["task"]]]
            else:
                ctx = [emb[it["task"]]]
            x = generate(tf, zc, z_grey, ctx, a.steps, a.cfg, seed=a.seed_base + i, drop_head=(c == "nohead"))
            views = decode_views(vae, x)
            pred = [to_u8(v) for v in views]
            rec = {"id": it["id"], "start": it["start"], "task": it["task"], "cond": c}
            if "dataset" in it:
                rec["dataset"] = it["dataset"]
            for v, name in ((1, "left"), (2, "right")):
                rec[name] = metrics(pred[v], gt[v], lp, dev)
            files[c].write(json.dumps(rec) + "\n"); files[c].flush()
            if a.save_mp4 < 0 or i % max(1, n // max(1, a.save_mp4 * a.n_shards)) == 0:  # evenly spaced over all scenes
                top = np.concatenate(gt, axis=2)
                bot = np.concatenate([pred[0] if c != "nohead" else np.zeros_like(gt[0]), pred[1], pred[2]], axis=2)
                name = it["id"].replace("/", "__")
                iio.imwrite(os.path.join(a.out, f"{name}_{c}.mp4"), np.concatenate([top, bot], axis=1), fps=30,
                            codec="libx264", quality=8)
            print(f"[{i}] {it['id']} {c}: L lpips {rec['left']['lpips']:.4f} R lpips {rec['right']['lpips']:.4f}",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
