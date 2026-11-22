"""Shared data utilities, and the single-dataset loader for EgoTouch head + two-wrist clips read straight from the raw
episode mp4s (the mixed-dataset loader used for the paper is e2w/multidata.py).

Each sample is one 49-frame window (30 fps) taken at the same frame indices from chest.mp4 (EgoTouch's name for the
head-mounted camera), left.mp4 and right.mp4. There is no clip-extraction step: the index
(tools/data/build_index_egotouch.py) records per-episode frame counts and the frozen-frame ranges flagged in
quality_issues.json, and windows that overlap a frozen range are never drawn.

Views are returned in the fixed order VIEWS = (head, left, right).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

VIEWS = ("chest", "left", "right")


def load_index(path):
    with open(path) as f:
        return json.load(f)


def valid_starts(ep, n_frames):
    """Start frames s such that [s, s + n_frames) avoids every frozen range of both wrists."""
    n = ep["n_frames"]
    if n < n_frames:
        return np.zeros(0, dtype=np.int64)
    ok = np.ones(n - n_frames + 1, dtype=bool)
    for cam in ("left", "right"):
        for a, b in ep["frozen"].get(cam, []):
            lo = max(0, a - n_frames + 1)
            hi = min(len(ok) - 1, b)
            if lo <= hi:
                ok[lo:hi + 1] = False
    return np.nonzero(ok)[0]


def read_window(ep_dir, start, n_frames, height, width):
    """uint8 array [V, F, H, W, 3] for the three views, same frame indices."""
    import decord
    idx = list(range(start, start + n_frames))
    out = []
    for v in VIEWS:
        vr = decord.VideoReader(os.path.join(ep_dir, f"{v}.mp4"), width=width, height=height, num_threads=1)
        out.append(vr.get_batch(idx).asnumpy())
    return np.stack(out)


HAND_FEAT = 73        # 6 rot + 2 image uv + 1 rel. log-depth + 63 articulation + 1 valid
WILOR_K = (12500.0, 12500.0, 320.0, 240.0, 640.0, 480.0)  # virtual K of the WiLoR export (fx, fy, cx, cy, W, H)


def hand_features(J, K=WILOR_K):
    """J: (F, 2, 21, 3) head-camera WiLoR joints (NaN = not detected) -> (F, 2, HAND_FEAT) float32.

    Per hand and frame: hand-frame orientation (6D: wrist->middle-MCP axis and the in-palm axis
    index-MCP minus pinky-MCP), wrist position in the head image (u, v in [-1, 1]), log wrist depth
    relative to the clip median (WiLoR's depth scale is not metric), joints relative to the wrist in
    the hand frame scaled by hand length (63), and a valid flag. Undetected hands are all zeros."""
    fx, fy, cx, cy, W, H = K
    F_ = J.shape[0]
    out = np.zeros((F_, 2, HAND_FEAT), dtype=np.float32)
    ok = np.isfinite(J).all(axis=(2, 3)) & (J[:, :, 0, 2] > 1e-3) if np.isfinite(J).any() else np.zeros(J.shape[:2], bool)
    if not ok.any():
        return out
    Jv = np.nan_to_num(J)
    w = Jv[:, :, 0]                                   # (F, 2, 3)
    a = Jv[:, :, 9] - w
    L = np.linalg.norm(a, axis=-1, keepdims=True)
    ok &= L[..., 0] > 1e-6
    a = a / np.maximum(L, 1e-6)
    b = Jv[:, :, 5] - Jv[:, :, 17]
    b = b - (b * a).sum(-1, keepdims=True) * a
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    ok &= nb[..., 0] > 1e-6
    b = b / np.maximum(nb, 1e-6)
    n = np.cross(a, b)
    R = np.stack([a, b, n], axis=-1)                  # (F, 2, 3, 3), columns = hand axes
    rel = np.einsum("fhji,fhkj->fhki", R, Jv - w[:, :, None]) / np.maximum(L[..., None], 1e-6)
    z = np.maximum(w[..., 2], 1e-3)
    u = (fx * w[..., 0] / z + cx) / W * 2 - 1
    v = (fy * w[..., 1] / z + cy) / H * 2 - 1
    lz = np.log(z)
    for h in range(2):
        if ok[:, h].any():
            lz[:, h] -= np.median(lz[ok[:, h], h])
    feat = np.concatenate([a, b, u[..., None], v[..., None], lz[..., None], rel.reshape(F_, 2, 63),
                           np.ones((F_, 2, 1), np.float32)], axis=-1).astype(np.float32)
    out[ok] = feat[ok]
    return out


def read_hands(hands_dir, ep_id, start, n_frames):
    J = np.load(os.path.join(hands_dir, ep_id.replace("/", "__") + ".npy"), mmap_mode="r")
    return hand_features(np.asarray(J[start:start + n_frames], dtype=np.float32))


def prompt_for(task):
    return f"A person {task.replace('_', ' ')}, seen from a head-mounted camera and two wrist-mounted cameras."


class TrainWindows(Dataset):
    """Uniform over all valid windows of the split. Item i is a deterministic function of (seed, i),
    so a run is reproducible and ranks draw disjoint items by taking i = rank, rank + world, ..."""

    def __init__(self, index, splits, n_frames=49, height=384, width=512, seed=0, length=10**8, hands_dir=None):
        self.hands_dir = hands_dir
        self.eps = [e for e in index["episodes"] if e["split"] in splits and e["usable"]]
        self.starts = [valid_starts(e, n_frames) for e in self.eps]
        keep = [i for i, s in enumerate(self.starts) if len(s) > 0]
        self.eps = [self.eps[i] for i in keep]
        self.starts = [self.starts[i] for i in keep]
        counts = np.array([len(s) for s in self.starts], dtype=np.float64)
        self.cum = np.cumsum(counts)
        self.n_windows = int(self.cum[-1])
        self.root = index["root"]
        self.n_frames, self.height, self.width = n_frames, height, width
        self.seed, self.length = seed, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        w = int(rng.integers(self.n_windows))
        e = int(np.searchsorted(self.cum, w, side="right"))
        off = w - (int(self.cum[e - 1]) if e > 0 else 0)
        ep = self.eps[e]
        start = int(self.starts[e][off])
        frames = read_window(os.path.join(self.root, ep["id"]), start, self.n_frames, self.height, self.width)
        out = {"frames": torch.from_numpy(frames), "task": ep["task"], "id": ep["id"], "start": start}
        if self.hands_dir:
            out["hands"] = torch.from_numpy(read_hands(self.hands_dir, ep["id"], start, self.n_frames))
        return out


class EvalWindows(Dataset):
    """One fixed window per episode (the valid start closest to the episode middle)."""

    def __init__(self, index, splits, n_frames=49, height=384, width=512, max_items=None, hands_dir=None):
        self.hands_dir = hands_dir
        self.items = []
        for e in index["episodes"]:
            if e["split"] not in splits or not e["usable"]:
                continue
            s = valid_starts(e, n_frames)
            if len(s) == 0:
                continue
            mid = (e["n_frames"] - n_frames) // 2
            self.items.append((e, int(s[np.argmin(np.abs(s - mid))])))
        self.items.sort(key=lambda x: x[0]["id"])
        if max_items is not None:
            # evenly spaced subset, deterministic
            sel = np.linspace(0, len(self.items) - 1, min(max_items, len(self.items))).round().astype(int)
            self.items = [self.items[i] for i in sorted(set(sel.tolist()))]
        self.root = index["root"]
        self.n_frames, self.height, self.width = n_frames, height, width

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, start = self.items[i]
        frames = read_window(os.path.join(self.root, ep["id"]), start, self.n_frames, self.height, self.width)
        out = {"frames": torch.from_numpy(frames), "task": ep["task"], "id": ep["id"], "start": start}
        if self.hands_dir:
            out["hands"] = torch.from_numpy(read_hands(self.hands_dir, ep["id"], start, self.n_frames))
        return out


def to_model_range(frames_u8, device, dtype):
    """[B, V, F, H, W, 3] uint8 -> [B, V, 3, F, H, W] in [-1, 1]."""
    x = frames_u8.to(device, non_blocking=True).permute(0, 1, 5, 2, 3, 4).to(torch.float32)
    return (x / 127.5 - 1.0).to(dtype)
