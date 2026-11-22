"""Self-improvement step 1: cut hand-busy 49-frame windows from downloaded Egocentric-10K tars (rsi/download_ego10k.py).
  per clip   each tar holds several clips (1080p30 H.265, head-mounted fisheye, 16:9); a clip is extracted from its
             tar to a scratch file and deleted after use; candidate windows start every 5 s
  hand check WiLoR's YOLO hand detector (the one HCC uses) on frames +0, +24, +48: a frame counts when it has >= 2 hand
             boxes centred below the top 35% of the image (the wearer's hands enter from below; coworkers' hands are
             usually higher up), one per side (detector classes left + right, or one box in each image half); a window
             qualifies when >= 2 of its 3 frames count and the picture changes (mean abs difference of frames +0 and
             +48 at 96 x 54 grey >= 3 levels; static scenes are not manipulation). All qualifying windows are kept
  head view  centre crop to 4:3 (1440 x 1080, the middle of the fisheye) scaled to 512 x 384
             -> <out>/heads/<wid>_B.mp4, and 4 key frames (+0, +16, +32, +48, 768 x 432) for the vision-language
             screen -> <out>/keyframes/<wid>_<k>.jpg
-> <out>/windows.jsonl (one row per window), <out>/clips_done.txt (resumable), <out>/PREP_DONE when finished.
CPU only; workers pin their thread pools to one thread.
  python rsi/extract_windows.py --raw $E2W_DATA_ROOT/raw/Egocentric10K --out $E2W_DATA_ROOT/wild/round1 --procs 16
"""
import argparse, glob, json, os, random, shutil, sys, tarfile

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")    # FFmpeg decoding defaults to one thread per core
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from e2w.paths import HAND_DETECTOR, data_path  # noqa: E402

T = 49
YOLO_ = None


def read(cap, start, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            return None
        out.append(f)                                    # BGR
    return out


def fmt_b(f):
    """centre crop to 4:3, scaled to the model's 512 x 384 head view"""
    h, w = f.shape[:2]
    cw = h * 4 // 3
    x0 = (w - cw) // 2
    return cv2.resize(f[:, x0:x0 + cw], (512, 384), interpolation=cv2.INTER_AREA)


def jobs_list(raw):
    out = []
    for t in sorted(glob.glob(f"{raw}/factory_*/workers/*/*.tar")):
        with tarfile.open(t) as tf:
            for m in sorted(x.name for x in tf.getmembers() if x.name.endswith(".mp4")):
                out.append((t, m))
    return out


def clip_windows(job):
    global YOLO_
    tar, member, out, detector = job
    import imageio.v3 as iio
    import torch
    torch.set_num_threads(1); cv2.setNumThreads(1)
    if YOLO_ is None:
        from ultralytics import YOLO
        YOLO_ = YOLO(detector)
    clip = os.path.basename(member)[:-4]
    tmp = f"{out}/scratch/{clip}.mp4"
    with tarfile.open(tar) as tf, tf.extractfile(member) as s, open(tmp, "wb") as d:
        shutil.copyfileobj(s, d, 1 << 24)
    try:
        cap = cv2.VideoCapture(tmp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        starts = list(range(30, n - T - 1, 150))
        want = {st + o for st in starts for o in (0, 24, 48)}
        small = {}
        for i in range(max(want) + 1 if want else 0):   # pass 1: sequential decode, keep only the probe frames
            if i in want:
                ok, f = cap.read()
                if not ok:
                    break
                small[i] = cv2.resize(f, (640, 360))
            elif not cap.grab():
                break
        keep = []
        for st in starts:
            if not all(st + o in small for o in (0, 24, 48)):
                continue
            fr = [small[st + o] for o in (0, 24, 48)]
            res = YOLO_.predict(fr, conf=0.3, verbose=False)
            both = 0
            for r in res:
                bx = [(float((b[0] + b[2]) / 2), YOLO_.names[int(c)]) for b, c in zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist())
                      if (b[1] + b[3]) / 2 > 0.35] if r.boxes is not None else []
                both += len(bx) >= 2 and ({"left", "right"} <= {nm for _, nm in bx}
                                          or (any(c < 0.5 for c, _ in bx) and any(c >= 0.5 for c, _ in bx)))
            g = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(float) for x in fr]
            motion = float(np.abs(g[0] - g[2]).mean())
            if both >= 2 and motion >= 3:
                keep.append((st, both, motion))
        small.clear()
        rows = []
        fac = tar.split("/")[-4]
        worker = tar.split("/")[-2]
        for st, both, motion in keep:                    # pass 2: one seek per kept window
            wid = f"{clip}_{st:05d}"
            fr = read(cap, st, T)
            if fr is None:
                continue
            v = np.stack([cv2.cvtColor(fmt_b(x), cv2.COLOR_BGR2RGB) for x in fr])
            iio.imwrite(f"{out}/heads/{wid}_B.mp4", v, fps=30, codec="libx264", quality=9, output_params=["-threads", "1"])
            for k, o in enumerate((0, 16, 32, 48)):
                cv2.imwrite(f"{out}/keyframes/{wid}_{k}.jpg", cv2.resize(fr[o], (768, 432), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
            rows.append({"wid": wid, "clip": clip, "factory": fac, "worker": worker, "start": st, "hands_both": both,
                         "motion": round(motion, 2)})
        cap.release()
        return clip, rows, len(starts)
    finally:
        os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=data_path("raw", "Egocentric10K"), help="download dir of rsi/download_ego10k.py")
    ap.add_argument("--out", default=data_path("wild", "round1"))
    ap.add_argument("--hand_detector", default=HAND_DETECTOR, help="WiLoR detector.pt")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N clips")
    a = ap.parse_args()
    from multiprocessing import get_context
    for d in ("heads", "keyframes", "scratch"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    for f in glob.glob(f"{a.out}/scratch/*.mp4"):                    # left by an interrupted run
        os.remove(f)
    done_p, rows_p = f"{a.out}/clips_done.txt", f"{a.out}/windows.jsonl"
    done = set(open(done_p).read().split()) if os.path.exists(done_p) else set()
    have = {json.loads(l)["wid"] for l in open(rows_p)} if os.path.exists(rows_p) else set()
    jobs = jobs_list(a.raw)
    random.Random(20261020).shuffle(jobs)                 # factories interleaved: early output already diverse
    if a.limit:
        jobs = jobs[:a.limit]
    jobs = [(t, m, a.out, a.hand_detector) for t, m in jobs if os.path.basename(m)[:-4] not in done]
    print(f"{len(jobs)} clips to process", flush=True)
    k = n_win = n_cand = 0
    with get_context("spawn").Pool(a.procs) as pool, open(rows_p, "a") as fo, open(done_p, "a") as fd:
        for clip, rows, nc in pool.imap_unordered(clip_windows, jobs):
            for r in rows:
                if r["wid"] not in have:
                    fo.write(json.dumps(r) + "\n"); have.add(r["wid"]); n_win += 1
            fo.flush(); fd.write(clip + "\n"); fd.flush()
            k += 1; n_cand += nc
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} clips, {n_cand} candidates, {n_win} windows kept", flush=True)
    if not a.limit:
        open(f"{a.out}/PREP_DONE", "w").write("done\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
