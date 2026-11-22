"""Self-improvement step 4: synthesize both wrist videos for every screened window with the current model (the paper:
15 Euler steps, no head guidance, seed 9000, one sample per window, the model's raw (non-EMA) LoRA weights).
  jobs     windows with an embedding file (<dir>/emb/<wid>.pt, rsi/encode_captions.py), head video <dir>/heads/<wid>_B.mp4
  workers  any number of workers (one per GPU, on one or several machines sharing <dir>) claim windows with an atomic
           mkdir under <dir>/locks/; windows are split into two halves by crc32(wid) % 2 and --shards gives a worker's
           halves in order of preference (a later half is walked in reverse, so two workers meet in the middle)
  stop     a worker leaves when no job is left (with --follow: once <dir>/ENCODE_DONE also exists), or when the file
           <dir>/stop_<host>_gpu<N> exists
  output   <dir>/gen/<wid>.mp4: head | left | right, 384 x 1536, 30 fps; <dir>/gen/done_<host>_gpu<N>.jsonl (seconds)
Only the head view is VAE-encoded (the sampler reads only the head columns of the panel latent) and only the two wrist
views are decoded; the mp4 is written by a background thread while the GPU works on the next window.
  CUDA_VISIBLE_DEVICES=0 python rsi/generate.py --dir $E2W_DATA_ROOT/wild/round1 --lora <M0>/lora.safetensors --gpu_tag 0
"""
import argparse, os, queue, socket, sys, threading, time, traceback, zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--lora", required=True, help="LoRA weights of the current model")
    ap.add_argument("--gpu_tag", default="0", help="worker name used in stop files and log names")
    ap.add_argument("--host", default=None, help="host name used in stop files and log names (default: hostname)")
    ap.add_argument("--shards", default="0,1")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max_videos", type=int, default=0, help="smoke test: stop after this many videos")
    ap.add_argument("--follow", action="store_true", help="keep waiting for new embeddings until <dir>/ENCODE_DONE exists")
    a = ap.parse_args()
    import imageio.v3 as iio
    from safetensors.torch import load_file
    from e2w.data import to_model_range
    from e2w.model import add_lora, encode_views, grey_latent, load_transformer, load_vae
    from e2w.sample import generate, to_u8
    dev, dt = torch.device("cuda"), torch.bfloat16
    d = a.dir
    host = a.host or socket.gethostname().split(".")[0]
    os.makedirs(f"{d}/gen", exist_ok=True); os.makedirs(f"{d}/locks", exist_ok=True)
    stop = f"{d}/stop_{host}_gpu{a.gpu_tag}"
    shards = [int(s) for s in a.shards.split(",")]
    vae = load_vae().to(dev, dtype=dt)
    tf = load_transformer(dt).to(dev).eval()
    net = add_lora(tf, rank=64, alpha=64).to(dev)
    net.load_state_dict(load_file(a.lora), strict=True)
    z_grey = None
    log = open(f"{d}/gen/done_{host}_gpu{a.gpu_tag}.jsonl", "a")
    n = 0
    wq = queue.Queue(maxsize=2)                         # at most two finished windows wait for their write
    werr = []

    def writer():
        while True:
            item = wq.get()
            if item is None:
                return
            wid, frames, t0 = item
            out = f"{d}/gen/{wid}.mp4"
            try:
                iio.imwrite(out + ".part.mp4", frames, fps=30, codec="libx264", quality=8, output_params=["-threads", "2"])
                os.replace(out + ".part.mp4", out)
                log.write(f'{{"wid": "{wid}", "sec": {time.time() - t0:.1f}}}\n'); log.flush()
            except Exception:                               # the lock stays without output; delete it to retry
                werr.append(wid)
                print(f"write failed {wid}:\n{traceback.format_exc()}"[:2000], flush=True)

    wt = threading.Thread(target=writer, daemon=True); wt.start()

    def jobs():
        wids = sorted(f[:-3] for f in os.listdir(f"{d}/emb") if f.endswith(".pt"))
        out = []
        for k, s in enumerate(shards):
            js = [w for w in wids if zlib.crc32(w.encode()) % 2 == s]
            out += js if k == 0 else js[::-1]
        return [w for w in out if not os.path.exists(f"{d}/gen/{w}.mp4") and not os.path.exists(f"{d}/locks/{w}")]

    try:
        while not os.path.exists(stop) and not (a.max_videos and n >= a.max_videos):
            todo = jobs()
            if not todo:
                if not a.follow or os.path.exists(f"{d}/ENCODE_DONE"):
                    break
                time.sleep(120)
                continue
            for wid in todo[:50]:
                if os.path.exists(stop) or (a.max_videos and n >= a.max_videos):
                    break
                if not os.path.exists(f"{d}/emb/{wid}.pt"):     # removed from the pool since the list was read
                    continue
                try:
                    os.mkdir(f"{d}/locks/{wid}")
                except FileExistsError:
                    continue
                t0 = time.time()
                try:
                    head = iio.imread(f"{d}/heads/{wid}_B.mp4")
                    emb = torch.load(f"{d}/emb/{wid}.pt").to(dev, dt)
                except Exception as e:
                    print(f"skip {wid}: {e!r}"[:300], flush=True)
                    continue
                fr = to_model_range(torch.from_numpy(head[None, None]), dev, dt)      # [1, 1, 3, F, H, W]: head only
                if z_grey is None:
                    z_grey = grey_latent(vae, fr.shape[3], fr.shape[4], fr.shape[5], dev, dt)
                zh = encode_views(vae, fr)
                z = torch.cat([zh, z_grey, z_grey], dim=-1)
                x = generate(tf, z, z_grey, [emb], a.steps, 1.0, seed=a.seed)
                w = zh.shape[-1]
                left, right = (to_u8(vae.decode(x[..., v * w:(v + 1) * w]).sample.clamp(-1, 1)) for v in (1, 2))
                wq.put((wid, np.concatenate([head, left, right], axis=2), t0))
                n += 1
                if n % 25 == 0:
                    print(f"{time.strftime('%m-%d %H:%M:%S')} {n} videos ({time.time() - t0:.1f}s GPU part of the last)", flush=True)
    finally:
        wq.put(None); wt.join()
    print(f"done after {n} videos ({len(werr)} write failures)", flush=True)


if __name__ == "__main__":
    main()
