"""Mixed-dataset windows (EgoPro + EgoTouch + EgoViz + HumanTouch, plus synthetic in-the-wild clips of the
self-improvement rounds) from the unified index (tools/data/build_multi_index.py, rsi/build_round.py).

Same item layout as e2w.data.TrainWindows / EvalWindows: frames [3 views, F, 384, 512, 3] uint8 in the order
(head, left, right), "task" = the episode text (the key into the prompt-embedding cache), "hands" (optional) = the
73-d per-frame hand features of e2w.data.hand_features computed with the episode's own camera K.

Hand sources differ per dataset and are made comparable here (used by the head-view-crop baseline and by the optional
hand features):
  wilor          EgoTouch-provided WiLoR joints (virtual K, not metric); NaN when undetected
  gt_humantouch  Manus + Vive ground truth in the head camera; a hand counts as observed only when its wrist
                 projects inside the (un-letterboxed) head image, as an estimator would only see it then
  hawor_egoviz   HaWoR joints in the head camera; frames HaWoR flags as undetected are dropped
  vendor_egopro  the dataset's 21-joint hand poses in the head camera; bad-frame flags and off-screen hands dropped
Training samples a dataset by --weights, then a window uniformly within it.
"""
import os
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from e2w.data import hand_features, valid_starts

MV = ("head", "left", "right")


def read_window_multi(ep, start, n_frames, height=384, width=512, idx=None):
    import decord
    idx = list(range(start, start + n_frames)) if idx is None else list(idx)
    if ep.get("panel"):                                   # synthetic clip: one video, head | left | right
        x = decord.VideoReader(ep["panel"], width=3 * width, height=height, num_threads=1).get_batch(idx).asnumpy()
        return np.stack([x[:, :, k * width:(k + 1) * width] for k in range(3)])
    out = []
    for v in MV:
        vr = decord.VideoReader(ep["videos"][v], width=width, height=height, num_threads=1)
        out.append(vr.get_batch(idx).asnumpy())
    return np.stack(out)


@lru_cache(maxsize=64)
def _npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def zoom_crops(head, box, height=384, width=512, rng=None):
    """head [F, H, W, 3] uint8; box [F, 4] (cx, cy, w, h) of one hand in head pixels, NaN where unseen -> [F, height,
    width, 3] uint8 crop that follows the hand: 4:3, height 2.5x the window's median hand size (96..height px), centre
    interpolated over unseen frames and smoothed over 9 frames; mid-grey outside the image. None if the hand is seen in
    fewer than max(3, F/8) frames. rng: scale x U(0.85, 1.15), centre shifted by U(-0.1, 0.1) x the size, once per
    window, to mimic a detector's boxes. Used by the head-view-crop baseline (tools/baseline_trivial.py)."""
    import cv2
    ok = np.isfinite(box[:, 0])
    if ok.sum() < max(3, len(box) // 8):
        return None
    t = np.arange(len(box))
    ker = np.ones(9) / 9
    cx = np.convolve(np.pad(np.interp(t, t[ok], box[ok, 0]), 4, mode="edge"), ker, "valid")
    cy = np.convolve(np.pad(np.interp(t, t[ok], box[ok, 1]), 4, mode="edge"), ker, "valid")
    s = 2.5 * float(np.median(np.maximum(box[ok, 2], box[ok, 3])))
    if rng is not None:
        s *= rng.uniform(0.85, 1.15)
        cx = cx + rng.uniform(-0.1, 0.1) * s
        cy = cy + rng.uniform(-0.1, 0.1) * s
    ch = float(np.clip(s, 96, height)); cw = ch * width / height
    out = np.empty((len(head), height, width, 3), np.uint8)
    for f in range(len(head)):
        M = np.array([[width / cw, 0, -(cx[f] - cw / 2) * width / cw], [0, height / ch, -(cy[f] - ch / 2) * height / ch]], np.float32)
        out[f] = cv2.warpAffine(np.ascontiguousarray(head[f]), M, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    return out


def read_hands_multi(ep, start, n_frames, idx=None):
    if idx is not None:
        span = idx[-1] - start + 1
        return read_hands_multi(ep, start, span)[np.asarray(idx) - start]
    J, Kt = read_hand_joints(ep, start, n_frames)
    return hand_features(J, Kt)


def hand_boxes(ep, start, n_frames, idx=None, width=512, height=384):
    """Per frame and hand (0 = left, 1 = right) the 2-D box of the projected hand joints in the head video as stored
    (width x height): [F, 2, 4] = (cx, cy, w, h) in pixels, NaN where the hand is not visible."""
    if idx is not None:
        span = idx[-1] - start + 1
        return hand_boxes(ep, start, span, width=width, height=height)[np.asarray(idx) - start]
    J, (fx, fy, cx, cy, W, H) = read_hand_joints(ep, start, n_frames)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (fx * J[..., 0] / J[..., 2] + cx) * width / W
        v = (fy * J[..., 1] / J[..., 2] + cy) * height / H
    ok = np.isfinite(u) & np.isfinite(v) & (J[..., 2] > 0)
    out = np.full(J.shape[:2] + (4,), np.nan, np.float32)
    for f in range(J.shape[0]):
        for k in range(J.shape[1]):
            m = ok[f, k]
            if m.sum() >= 5:
                x0, x1 = np.clip(u[f, k, m].min(), 0, width), np.clip(u[f, k, m].max(), 0, width)
                y0, y1 = np.clip(v[f, k, m].min(), 0, height), np.clip(v[f, k, m].max(), 0, height)
                if x1 - x0 > 2 and y1 - y0 > 2:
                    out[f, k] = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    return out


def read_hand_joints(ep, start, n_frames):
    """-> (J [F, 2, 21, 3] in the head camera, NaN where invalid; (fx, fy, cx, cy, W, H) of that camera)"""
    h = ep["hands"]
    K = h["K"]
    Kt = (K["fx"], K["fy"], K["cx"], K["cy"], K["W"], K["H"])
    if h["kind"] == "wilor":
        J = np.asarray(np.load(h["path"], mmap_mode="r")[start:start + n_frames], dtype=np.float32)
    elif h["kind"] == "gt_humantouch":
        J = np.asarray(np.load(h["path"], mmap_mode="r")[start:start + n_frames], dtype=np.float32).copy()
        w = J[:, :, 0]
        z = w[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = K["fx"] * w[..., 0] / z + K["cx"]
            v = K["fy"] * w[..., 1] / z + K["cy"]
        pad = h.get("pad_top", 0)
        seen = (z > 0.1) & (u >= 0) & (u < K["W"]) & (v >= pad) & (v < K["H"] - pad)
        J[~seen] = np.nan
    elif h["kind"] == "vendor_egopro":
        z = _npz(h["path"])
        J = z["joints_head"][start:start + n_frames].astype(np.float32).copy()
        J[z["bad"][start:start + n_frames]] = np.nan
        w = J[:, :, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = K["fx"] * w[..., 0] / w[..., 2] + K["cx"]
            v = K["fy"] * w[..., 1] / w[..., 2] + K["cy"]
        seen = (w[..., 2] > 0.1) & (u >= 0) & (u < K["W"]) & (v >= 0) & (v < K["H"])   # vendor poses exist off-screen too
        J[~seen] = np.nan
    elif h["kind"] == "hawor_egoviz":
        z = _npz(h["path"])
        J = z["joints_head"][start:start + n_frames].astype(np.float32).copy()
        J[~z["detected"][start:start + n_frames]] = np.nan
    else:
        raise ValueError(h["kind"])
    return J, Kt


class MultiTrainWindows(Dataset):
    def __init__(self, index, splits, n_frames=49, seed=0, length=10**8, weights=None, with_hands=False,
                 height=384, width=512):
        eps = [e for e in index["episodes"] if e["split"] in splits]
        self.by = {}
        for e in eps:
            s = valid_starts(e, n_frames)
            if len(s):
                self.by.setdefault(e["dataset"], []).append((e, s))
        self.names = sorted(self.by)
        w = np.array([(weights or {}).get(n, 1.0) for n in self.names], dtype=np.float64)
        self.p = w / w.sum()
        self.cum = {n: np.cumsum([len(s) for _, s in self.by[n]]) for n in self.names}
        self.n_frames, self.seed, self.length, self.with_hands = n_frames, seed, length, with_hands
        self.height, self.width = height, width

    def summary(self):
        return {n: {"episodes": len(self.by[n]), "windows": int(self.cum[n][-1]), "p": float(p)}
                for n, p in zip(self.names, self.p)}

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        name = self.names[rng.choice(len(self.names), p=self.p)]
        cum = self.cum[name]
        wi = int(rng.integers(cum[-1]))
        k = int(np.searchsorted(cum, wi, side="right"))
        ep, starts = self.by[name][k]
        start = int(starts[wi - (int(cum[k - 1]) if k else 0)])
        fr = read_window_multi(ep, start, self.n_frames, self.height, self.width)
        out = {"task": ep["text"], "id": ep["id"], "start": start, "dataset": name, "frames": torch.from_numpy(fr)}
        if self.with_hands:
            out["hands"] = torch.from_numpy(read_hands_multi(ep, start, self.n_frames))
        return out


class MultiEvalWindows(Dataset):
    """One fixed window per episode (valid start nearest the middle); per dataset at most max_per_dataset
    episodes, evenly spaced over the id-sorted list."""

    def __init__(self, index, splits, n_frames=49, max_per_dataset=None, datasets=None, with_hands=False, height=384,
                 width=512):
        items = {}
        for e in index["episodes"]:
            if e["split"] not in splits or (datasets and e["dataset"] not in datasets):
                continue
            s = valid_starts(e, n_frames)
            if len(s):
                mid = (e["n_frames"] - n_frames) // 2
                items.setdefault(e["dataset"], []).append((e, int(s[np.argmin(np.abs(s - mid))])))
        self.items = []
        for n in sorted(items):
            L = sorted(items[n], key=lambda x: x[0]["id"])
            cap = max_per_dataset.get(n) if isinstance(max_per_dataset, dict) else max_per_dataset
            if cap and len(L) > cap:
                sel = np.linspace(0, len(L) - 1, cap).round().astype(int)
                L = [L[j] for j in sorted(set(sel.tolist()))]
            self.items += L
        self.n_frames, self.with_hands = n_frames, with_hands
        self.height, self.width = height, width

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, start = self.items[i]
        fr = read_window_multi(ep, start, self.n_frames, self.height, self.width)
        out = {"task": ep["text"], "id": ep["id"], "start": start, "dataset": ep["dataset"], "frames": torch.from_numpy(fr)}
        if self.with_hands:
            out["hands"] = torch.from_numpy(read_hands_multi(ep, start, self.n_frames))
        return out
