"""LoRA fine-tune of Wan2.2-Fun-5B-InP: head video -> left + right wrist videos (latent panel).

Launch:
  torchrun --nproc_per_node=4 -m e2w.train --index $E2W_DATA_ROOT/index/paired.json --out_dir <run dir> [--steps ...]

Plain torch.distributed: LoRA parameters are not DDP-wrapped (the LoRA modules live in the
network object, not under the transformer), so their gradients are all-reduced by hand as one
flat buffer each step. Batch is 1 clip per GPU and micro-batch; the effective batch is world size x --grad_accum.
Training resumes from the newest <out_dir>/step_* checkpoint when one exists.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Sampler

from e2w import paths
from e2w.data import TrainWindows, load_index, to_model_range
from e2w.model import (N_VIEWS, add_lora, build_condition, encode_views, grey_latent, load_transformer, load_vae,
                       seq_len_of, shift_sigma, token_timesteps)


class StrideSampler(Sampler):
    """Global item g = step * world + rank for step in [start, steps)."""

    def __init__(self, start, steps, rank, world):
        self.start, self.steps, self.rank, self.world = start, steps, rank, world

    def __iter__(self):
        return iter(range(self.start * self.world + self.rank, self.steps * self.world, self.world))

    def __len__(self):
        return max(0, self.steps - self.start)


def log0(rank, msg, path=None):
    if rank == 0:
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if path:
            with open(path, "a") as f:
                f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=paths.INDEX)
    ap.add_argument("--prompts", default=paths.PROMPTS, help="umT5 embeddings keyed by episode text (tools/data/encode_prompts.py)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--p_drop_head", type=float, default=0.1)
    ap.add_argument("--p_drop_text", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--max_steps_this_launch", type=int, default=0, help="stop early (probes); 0 = no limit")
    ap.add_argument("--decay_file", default="", help="a file holding a step number; from that step the LR follows a cosine "
                    "to 0 over --decay_steps, then training saves and stops (lets an external plateau monitor end the run)")
    ap.add_argument("--decay_steps", type=int, default=8000)
    ap.add_argument("--decay_latest", type=int, default=0, help="start the decay here at the latest; 0 = only on the file")
    ap.add_argument("--init_lora", default="", help="start (step 0, fresh optimiser) from these LoRA weights")
    ap.add_argument("--weights", default="", help="mixed index only: dataset sampling weights, e.g. "
                    "EgoPro=0.4,EgoViz=0.24,HumanTouch=0.18,EgoTouch=0.18 (unlisted datasets: 1)")
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimiser step; effective batch = world x this")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the LoRA weights (0 = off); saved as lora_ema.safetensors")
    ap.add_argument("--emb_on_cpu", action="store_true", help="keep text embeddings in CPU memory (many synthetic texts)")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dt = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out_dir, exist_ok=True)
    logp = os.path.join(a.out_dir, "train.log")
    if rank == 0:
        json.dump(vars(a) | {"world": world}, open(os.path.join(a.out_dir, "args.json"), "w"), indent=1)
    torch.manual_seed(a.seed * 1000 + rank)
    gen = torch.Generator(device=dev).manual_seed(a.seed * 1000 + rank)

    vae = load_vae().to(dev, dtype=dt).requires_grad_(False)
    tf = load_transformer(dt).to(dev).requires_grad_(False)
    tf.enable_gradient_checkpointing()
    net = add_lora(tf, rank=a.rank, alpha=a.rank).to(dev)  # LoRA params stay fp32 (autocast in forward)
    params = [p for p in net.parameters()]
    for p in params:
        p.requires_grad_(True)
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))

    start = 0
    decay_start = None
    if a.init_lora and not any(d.startswith("step_") for d in os.listdir(a.out_dir)):
        net.load_state_dict(load_file(a.init_lora), strict=True)
        log0(rank, f"initialised LoRA from {a.init_lora}", logp)
    ck = sorted([d for d in os.listdir(a.out_dir) if d.startswith("step_")], key=lambda d: int(d[5:]))
    if ck:
        last = os.path.join(a.out_dir, ck[-1])
        net.load_state_dict(load_file(os.path.join(last, "lora.safetensors")), strict=True)
        st = torch.load(os.path.join(last, "optim.pt"), map_location="cpu")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"]
        decay_start = st.get("decay_start")
        log0(rank, f"resumed from {last} at step {start}", logp)
    ema = None
    if a.ema > 0 and rank == 0:                                    # EMA of the trainable LoRA weights, rank 0 only
        ema = {n: p.detach().float().clone() for n, p in net.named_parameters()}
        if ck and os.path.exists(os.path.join(a.out_dir, ck[-1], "lora_ema.safetensors")):
            saved = load_file(os.path.join(a.out_dir, ck[-1], "lora_ema.safetensors"))
            ema = {n: saved[n].to(dev).float() if n in saved else v for n, v in ema.items()}
            log0(rank, f"EMA restored from {ck[-1]}", logp)

    index = load_index(a.index)
    if index.get("format") == "multi_v1":
        from e2w.multidata import MultiTrainWindows
        wts = {k: float(v) for k, v in (kv.split("=") for kv in a.weights.split(",") if kv)}
        ds = MultiTrainWindows(index, set(a.splits.split(",")), a.n_frames, seed=a.seed, weights=wts, with_hands=False,
                               height=a.height, width=a.width)
        ds.eps, ds.n_windows = [], sum(v["windows"] for v in ds.summary().values())
        log0(rank, f"mixed data: {ds.summary()}", logp)
    else:
        ds = TrainWindows(index, set(a.splits.split(",")), a.n_frames, a.height, a.width, seed=a.seed)
    dl = DataLoader(ds, batch_size=1, sampler=StrideSampler(start * a.grad_accum, a.steps * a.grad_accum, rank, world),
                    num_workers=a.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    pe = torch.load(a.prompts)
    emb = {k: (v.to(dt) if a.emb_on_cpu else v.to(dev, dt)) for k, v in pe["embeds"].items()}
    z_grey = grey_latent(vae, a.n_frames, a.height, a.width, dev, dt)
    log0(rank, f"world={world} trainable_params={n_par/1e6:.1f}M episodes={len(ds.eps)} windows={ds.n_windows} "
               f"steps {start}->{a.steps}", logp)

    flat = None
    t_last, hist, acc = time.time(), [], []
    step, micro = start, 0
    for batch in dl:
        frames = to_model_range(batch["frames"], dev, dt)          # [B, V, 3, F, H, W]
        assert frames.shape[1] == N_VIEWS and frames.shape[3] == a.n_frames, frames.shape
        z = encode_views(vae, frames)                               # [B, C, f, h, V*w]
        B, C, f, h, W = z.shape
        w = W // N_VIEWS
        drop_head = torch.rand(B, device=dev, generator=gen) < a.p_drop_head
        y, head = build_condition(z, z_grey, drop_head)
        z_in = torch.cat([head, z[..., w:]], dim=-1)                # dropped head -> grey in x too
        sigma = shift_sigma(torch.rand(B, device=dev, generator=gen))
        noise = torch.randn(z.shape, device=dev, generator=gen, dtype=z.dtype)
        sig = sigma.view(B, 1, 1, 1, 1).expand(B, 1, f, h, W).clone()
        sig[..., :w] = 0.0
        xt = ((1 - sig) * z_in.float() + sig * noise.float()).to(dt)
        ttok = token_timesteps(sigma * 1000.0, f, h, W)
        drop_text = torch.rand(B, generator=gen, device=dev) < a.p_drop_text
        ctx = [(emb[""] if drop_text[i] else emb[batch["task"][i]]).to(dev, dt) for i in range(B)]
        with torch.autocast("cuda", dtype=dt):
            pred = tf(x=xt, t=ttok, context=ctx, seq_len=seq_len_of(f, h, W), y=y)
        target = (noise - z).float()
        err = (pred.float() - target) ** 2
        loss = err[..., w:].mean()
        (loss / a.grad_accum).backward()
        acc.append(torch.stack([err[..., w:2 * w].mean(), err[..., 2 * w:].mean(), loss]).detach())
        micro += 1
        if micro % a.grad_accum:
            continue

        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        if flat is None:
            flat = torch.zeros(sum(g.numel() for g in grads), device=dev, dtype=torch.float32)
        o = 0
        for g in grads:
            flat[o:o + g.numel()].copy_(g.view(-1)); o += g.numel()
        dist.all_reduce(flat)
        flat.div_(world)
        o = 0
        for p, g in zip(params, grads):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(flat[o:o + g.numel()].view_as(p)); o += g.numel()
        gnorm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip)

        if step == start:  # guard: the zero-initialised lora_up must receive gradient
            up = sum(float(p.grad.abs().sum()) for n_, p in net.named_parameters() if "lora_up" in n_)
            assert up > 0, "lora_up received no gradient"
            log0(rank, f"guard ok: |grad lora_up| = {up:.3e}", logp)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if ema is not None:
            with torch.no_grad():
                for n, p in net.named_parameters():
                    ema[n].mul_(a.ema).add_(p.detach().float(), alpha=1 - a.ema)
        if (a.decay_file or a.decay_latest) and decay_start is None and step % a.log_every == 0:
            want = a.decay_latest or 10**9
            if a.decay_file and os.path.exists(a.decay_file):
                try:
                    want = min(want, int(open(a.decay_file).read().strip()))
                except ValueError:
                    pass
            wt = torch.tensor([float(want)], device=dev)
            dist.all_reduce(wt, op=dist.ReduceOp.MIN)            # every rank agrees on the same start
            if wt.item() < 10**9 and step >= wt.item():
                decay_start = int(wt.item())
                log0(rank, f"LR decay: cosine to 0 from step {decay_start} over {a.decay_steps} steps", logp)
        if decay_start is not None:
            frac = min(1.0, max(0.0, (step - decay_start) / a.decay_steps))
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * frac))
        stop_now = decay_start is not None and step >= decay_start + a.decay_steps

        lv = torch.stack(acc).mean(0); acc = []
        dist.all_reduce(lv); lv /= world
        hist.append(lv.tolist() + [float(gnorm)])
        if step % a.log_every == 0 or step == start + 1:
            m = np.mean(hist, axis=0); hist = []
            dt_it = (time.time() - t_last) / (a.log_every if step > start + 1 else 1); t_last = time.time()
            mem = torch.cuda.max_memory_allocated(dev) / 2**30
            log0(rank, f"step {step}/{a.steps} loss {m[2]:.4f} (L {m[0]:.4f} R {m[1]:.4f}) gnorm {m[3]:.3f} "
                       f"lr {opt.param_groups[0]['lr']:.2e} {dt_it:.2f}s/it maxmem {mem:.1f}G", logp)
        if rank == 0 and (step % a.ckpt_every == 0 or step == a.steps or stop_now):
            d = os.path.join(a.out_dir, f"step_{step}")
            os.makedirs(d, exist_ok=True)
            sd = {k: v.detach().float().cpu().contiguous() for k, v in net.state_dict().items()}
            up_norm = math.sqrt(sum(float(v.pow(2).sum()) for k, v in sd.items() if "lora_up" in k))
            assert up_norm > 0, "saved LoRA has all-zero lora_up"
            save_file(sd, os.path.join(d, "lora.safetensors"))
            if ema is not None:                                    # same keys as lora.safetensors, EMA values
                save_file({k: (ema[k].detach().cpu().contiguous() if k in ema else v) for k, v in sd.items()},
                          os.path.join(d, "lora_ema.safetensors"))
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "decay_start": decay_start},
                       os.path.join(d, "optim.pt"))
            log0(rank, f"saved {d} (|lora_up| = {up_norm:.3f})", logp)
        if a.max_steps_this_launch and step - start >= a.max_steps_this_launch:
            break
        if stop_now:
            log0(rank, f"LR decay finished at step {step}", logp)
            break
    dist.barrier()
    log0(rank, "done", logp)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
