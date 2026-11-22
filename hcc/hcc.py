"""Hand-centric cross-view consistency (HCC): a score of a generated wrist video that needs no wrist ground truth.

The wrist view is matched only against the part of the head view that the wrist camera can see: a square region
around the same-side hand, which shows the hand, the object it handles and the nearby scene.

Per frame t in 0, 4, ..., 48 (13 frames):
  head region    WiLoR's YOLO hand detector on the head frame, box of the SAME side as the wrist camera (class left /
                 right, conf >= 0.3; else the most confident hand of either class in that half of the image), with an
                 OWLv2 fallback (text queries for a hand / a black glove, score >= 0.15) for frames the detector misses
                 (gloved hands); the region is a square of 2.5 x the longer side of the box, centred on it; a frame
                 counts if the region covers >= 4 patches
  features       DINOv3 ViT-7B/16 patch tokens of the last layer after the final norm, unit length, at 384 x 512
                 (24 x 32 patches); the wrist "centre" C is the central half of the wrist frame in each direction
Scores (means over the frames that count; NaN if fewer than 3 frames count):
  hcc / xv_f     HCC = mean of xv_prec and xv_rec  (the paper's score; the sample score is the mean over both wrists)
  xv_prec        precision: wrist centre patches, best cosine match among the head-region patches
  xv_rec         recall: head-region patches, best cosine match among all wrist patches
  xv_full        "whole frame" variant: wrist centre patches, best match over ALL head patches
Auxiliary diagnostics (not part of HCC):
  t_adv          temporal test: a wrist frame's region match at the SAME time minus its mean match at the other times
  t_align        share of wrist frames whose best-matching head time is within one sampled frame of their own time
  mnn_n          mutual nearest neighbours between all wrist patches and head-region patches with cosine >= 0.5
  knn_keep       for each MNN pair, share of its 4 nearest MNN neighbours in the wrist view whose head partners are also
                 among the head partner's 4 nearest (layout preserved up to any scale; needs >= 6 pairs)
  aff_inl        RANSAC affine wrist -> head coordinates over MNN pairs, inlier share
  refl           reflection test: share of frames whose best affine map is a mirror (det < 0)
  verif          share of frames whose same-side hand was found in the head view
  wh_x, wh_y     mean centre of the most confident hand in the wrist frame (normalised); wh_rate = share of frames
                 with one

Usage:
  scorer = HCC(dinov3="facebook/dinov3-vit7b16-pretrain-lvd1689m", hand_detector="detector.pt")
  row = scorer(wrist_frames, head_frames, key="clip_0", side="left", rng=np.random.default_rng(0))
  row["hcc"]
Frames are uint8 arrays [T >= 49, H, W, 3] (RGB); the head frames should be the 512 x 384 head view.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

T = 49
KS = list(range(0, T, 4))                    # the 13 compared frames
OWL_Q = ["a hand wearing a black glove", "a human hand", "a black glove"]
DINOV3_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
OWL_ID = "google/owlv2-base-patch16-ensemble"


def affine_ransac(src, dst, rng, iters=200, thr=1.5):
    """src, dst: (n, 2) patch coordinates -> (inlier share, det of the linear part) of the best affine map."""
    n = len(src)
    if n < 6:
        return np.nan, np.nan
    A = np.c_[src, np.ones(n)]
    best, best_in = None, -1
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        try:
            M = np.linalg.solve(A[i], dst[i])                       # (3, 2)
        except np.linalg.LinAlgError:
            continue
        inl = np.linalg.norm(A @ M - dst, axis=1) < thr
        if inl.sum() > best_in:
            best_in, best = inl.sum(), inl
    if best is None or best.sum() < 3:
        return float(best_in) / n if best_in > 0 else 0.0, np.nan
    M, *_ = np.linalg.lstsq(A[best], dst[best], rcond=None)
    return best.sum() / n, float(np.linalg.det(M[:2].T))


class HCC:
    """backbone: "dinov3_7b" (the paper) or "dinov2_s" (DINOv2 ViT-S/14 from torch.hub at 336 x 448; a light variant
    that was NOT used for the paper's numbers). dinov3 / owl: Hugging Face ids or local snapshot directories
    (transformers >= 4.56 for DINOv3). owl=None disables the OWLv2 fallback. hand_detector: WiLoR's detector.pt."""

    def __init__(self, hand_detector, dinov3=DINOV3_ID, owl=OWL_ID, backbone="dinov3_7b", roi=2.5, res=None,
                 device="cuda"):
        from ultralytics import YOLO
        self.backbone, self.dev = backbone, device
        self.patch = 16 if backbone.startswith("dinov3") else 14
        self.res = int(res or (384 if self.patch == 16 else 336))      # input height; width = 4/3 of it
        self.GH, self.GW = self.res // self.patch, (self.res * 4 // 3) // self.patch
        self.roi_f = float(roi)                                         # region side / longer hand-box side
        if backbone == "dinov3_7b":
            from transformers import AutoModel
            self.dino = AutoModel.from_pretrained(dinov3, torch_dtype=torch.bfloat16).to(device).eval()
            self.nreg = self.dino.config.num_register_tokens
        elif backbone == "dinov2_s":
            self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).to(device).eval()
        else:
            raise ValueError(backbone)
        self.yolo = YOLO(hand_detector)
        self.owl = None
        if owl:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.owl_proc = Owlv2Processor.from_pretrained(owl)
            self.owl = Owlv2ForObjectDetection.from_pretrained(owl).to(device).eval()
        GH, GW = self.GH, self.GW
        yy, xx = np.meshgrid(np.arange(GH), np.arange(GW), indexing="ij")
        self.pc = np.stack([yy.ravel(), xx.ravel()], 1).astype(float)          # patch centres (row, col)
        self.centre = ((yy >= GH // 4) & (yy < GH - GH // 4) & (xx >= GW // 4) & (xx < GW - GW // 4)).ravel()
        self.head_cache = {}

    @torch.no_grad()
    def owl_boxes(self, frames, half):
        """OWLv2 open-vocabulary fallback: per frame the best box (score >= 0.15) for OWL_Q in the wanted image half"""
        ims = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        H, W = frames.shape[1], frames.shape[2]; S = max(H, W)              # OWLv2 pads to a square (bottom / right)
        out = []
        for i in range(0, len(ims), 8):
            inp = self.owl_proc(text=[OWL_Q] * len(ims[i:i + 8]), images=ims[i:i + 8], return_tensors="pt").to(self.dev)
            res = self.owl_proc.post_process_object_detection(self.owl(**inp), threshold=0.15,
                                                              target_sizes=torch.tensor([[S, S]] * len(ims[i:i + 8])).to(self.dev))
            for r in res:
                best = None
                for sc, b in zip(r["scores"].tolist(), r["boxes"].tolist()):
                    x0, y0, x1, y1 = b
                    bb = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)
                    if half(bb) and (best is None or sc > best[0]):
                        best = (sc, bb)
                out.append(best[1] if best else None)
        return out

    @torch.no_grad()
    def feats(self, frames):
        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().to(self.dev) / 255
        x = F.interpolate(x, size=(self.GH * self.patch, self.GW * self.patch), mode="bilinear", antialias=True,
                          align_corners=False)
        m = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        if self.backbone == "dinov3_7b":                             # [CLS, registers, patches] after the final norm
            out = torch.cat([self.dino(pixel_values=((x[i:i + 4] - m) / s).to(torch.bfloat16)).last_hidden_state
                             for i in range(0, len(x), 4)])
            return F.normalize(out[:, 1 + self.nreg:].float(), dim=-1)
        return F.normalize(self.dino.forward_features((x - m) / s)["x_norm_patchtokens"].float(), dim=-1)

    def hands(self, frames, side=None):
        """per frame: (cx, cy, w, h) normalised box of the most confident hand of class `side` (any if None), or None"""
        res = self.yolo.predict(list(np.ascontiguousarray(frames[..., ::-1])), conf=0.3, verbose=False)
        out = []
        for p in res:
            b = p.boxes
            best = None
            if b is not None and len(b):
                for k in range(len(b)):
                    if (side is None or self.yolo.names[int(b.cls[k])] == side) and (best is None or float(b.conf[k]) > best[0]):
                        x0, y0, x1, y1 = b.xyxyn[k].tolist()
                        best = (float(b.conf[k]), ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
            out.append(best[1] if best else None)
        return out

    def head(self, key, head, side):
        """head-frame patch features and, per compared frame, the boolean patch mask of the hand region (or None);
        cached per (key, side), so both wrists of a clip share the head features"""
        k = (key, side)
        if k not in self.head_cache:
            if len(self.head_cache) > 64:
                self.head_cache.clear()
            ph = self.feats(head[KS])
            boxes = self.hands(head[KS], side)
            anyb = self.hands(head[KS])                                    # fallback: handedness is often wrong in
            half = (lambda b: b[0] < 0.5) if side == "left" else (lambda b: b[0] >= 0.5)   # egocentric views
            boxes = [b if b is not None else (a if a is not None and half(a) else None) for b, a in zip(boxes, anyb)]
            if self.owl is not None and any(b is None for b in boxes):    # gloved hands and other misses
                ob = self.owl_boxes(head[KS], half)
                boxes = [b if b is not None else o for b, o in zip(boxes, ob)]
            rois = []
            for bx in boxes:
                if bx is None:
                    rois.append(None)
                    continue
                cx, cy, w, h = bx
                Hp, Wp = head.shape[1], head.shape[2]
                hs = self.roi_f / 2 * max(w * Wp, h * Hp)                  # region: square, roi_f x the longer box side
                m = ((np.abs((self.pc[:, 1] + .5) / self.GW - cx) * Wp <= hs)
                     & (np.abs((self.pc[:, 0] + .5) / self.GH - cy) * Hp <= hs))
                rois.append(m if m.sum() >= 4 else None)
            self.head_cache[k] = (ph, rois)
        return self.head_cache[k]

    @torch.no_grad()
    def __call__(self, v, head, key, side, rng):
        """v: wrist video [T, H, W, 3] uint8; head: head video of the same window; key: any id of the head video (cache);
        side: "left" | "right"; rng: numpy Generator (RANSAC of the auxiliary affine test). -> dict of scores."""
        assert len(v) >= T and len(head) >= T, (v.shape, head.shape)
        GH = self.GH
        ph, rois = self.head(key, head, side)
        pw = self.feats(v[KS])
        r = {"verif": float(np.mean([m is not None for m in rois]))}
        S_all = torch.einsum("tpc,tqc->tpq", pw, ph).cpu().numpy()          # (t, wrist patches, head patches)
        # temporal: wrist frame t against the head region at every time u (centre patches, best match within the region)
        valid = [u for u, m in enumerate(rois) if m is not None]
        if len(valid) >= 3:
            cen = pw[:, torch.from_numpy(self.centre).to(pw.device)]
            P = np.full((len(KS), len(KS)), np.nan)
            for u in valid:
                hm = ph[u][torch.from_numpy(rois[u]).to(ph.device)]
                P[:, u] = torch.einsum("tpc,qc->tpq", cen, hm).amax(-1).mean(-1).cpu().numpy()
            adv = [P[t, t] - np.nanmean(np.delete(P[t], t)) for t in valid if np.sum(~np.isnan(np.delete(P[t], t))) >= 2]
            r["t_adv"] = float(np.mean(adv)) if len(adv) >= 3 else np.nan
            r["t_align"] = float(np.mean([abs(int(np.nanargmax(P[t])) - t) <= 1 for t in valid]))
        else:
            r["t_adv"] = r["t_align"] = np.nan
        r["xv_full"] = float(S_all[:, self.centre].max(-1).mean())
        prec, rec, mnn, knn, inl, refl = [], [], [], [], [], []
        for t, m in enumerate(rois):
            if m is None:
                continue
            S = S_all[t][:, m]                                              # (wrist patches, region patches)
            prec.append(S[self.centre].max(1).mean())
            rec.append(S.max(0).mean())
            j = S.argmax(1); i = S.argmax(0)
            pairs = [(a, j[a]) for a in range(S.shape[0]) if i[j[a]] == a and S[a, j[a]] >= 0.5]
            mnn.append(len(pairs))
            if len(pairs) >= 6:
                hw = self.pc[[a for a, _ in pairs]]
                hh = self.pc[np.where(m)[0][[b for _, b in pairs]]]
                dw = np.linalg.norm(hw[:, None] - hw[None], axis=-1); dh = np.linalg.norm(hh[:, None] - hh[None], axis=-1)
                np.fill_diagonal(dw, np.inf); np.fill_diagonal(dh, np.inf)
                k = min(4, len(pairs) - 1)
                nw = np.argsort(dw, 1)[:, :k]; nh = np.argsort(dh, 1)[:, :k]
                knn.append(np.mean([len(set(nw[q]) & set(nh[q])) / k for q in range(len(pairs))]))
                a_in, det = affine_ransac(hw[:, ::-1], hh[:, ::-1], rng, thr=1.5 * GH / 12)   # (x, y) order
                inl.append(a_in)
                if not np.isnan(det):
                    refl.append(float(det < 0))
        ok = len(prec) >= 3
        nanm = lambda x: float(np.mean(x)) if ok and len(x) else np.nan
        r.update({"xv_prec": nanm(prec), "xv_rec": nanm(rec), "mnn_n": nanm(mnn), "knn_keep": nanm(knn),
                  "aff_inl": nanm(inl), "refl": nanm(refl)})
        r["xv_f"] = (r["xv_prec"] + r["xv_rec"]) / 2 if ok else np.nan
        r["hcc"] = r["xv_f"]
        wb = [b for b in self.hands(v[KS]) if b is not None]      # own hand in the wrist view, either class
        r["wh_rate"] = len(wb) / len(KS)
        r["wh_x"] = float(np.mean([b[0] for b in wb])) if wb else np.nan
        r["wh_y"] = float(np.mean([b[1] for b in wb])) if wb else np.nan
        return r


XVC = HCC   # name used by earlier scripts
