"""Unified episode index over EgoPro + EgoTouch + EgoViz + HumanTouch for mixed training (e2w/multidata.py).

Every episode: {dataset, rig, id, videos{head,left,right}, src_wh (decode size for raw EgoTouch, else None),
n_frames, split, text, hands{kind, path, K}, frozen{left,right}}.
Videos are 30 fps; the loader resizes / reads them at 512x384 (HumanTouch and EgoViz are already letterboxed).

Splits (whole recording units never straddle splits):
  EgoTouch    official split.json (as in the EgoTouch index of tools/data/build_index_egotouch.py)
  HumanTouch  tasks named in --ht_heldout -> test_unseen; other tasks by session hash: 3% test_seen, 2% val, rest train
  EgoViz      by sequence_id hash: 4% test_seen, 2% val, rest train
  EgoPro      by environment_id hash: 6% of environments -> test_unseen (new places); the rest by uuid hash:
              2% test_seen, 2% val, rest train
--splits_json (configs/splits.json reproduces the paper): episodes listed there get the listed split; an episode that
is not listed goes to train if the rule above puts it there, and to "heldout" (never read) otherwise. The paper's
evaluation sets were fixed on a first part of EgoPro; EgoPro episodes converted later went to train when the rule
said train and to "heldout" otherwise, so the file is needed to get exactly the paper's test clips.
Text = the task / clip description plus a rig phrase, so the text also tells the model which wrist rig to render.

  python tools/data/build_multi_index.py --out $E2W_DATA_ROOT/index/paired.json --splits_json configs/splits.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
from e2w.paths import data_path  # noqa: E402

RIG_PHRASE = {
    "egotouch": "fisheye wrist cameras at the base of the palm, gloved hands",
    "humantouch": "wide-angle wrist cameras under the wrist looking at the palm, gloved hands",
    "egoviz": "wide-angle cameras on the forearms looking toward the bare hands",
    "egopro": "wide-angle cameras on wrist straps under the wrists looking toward the bare hands",
}


def h100(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100


def text_for(desc, rig):
    desc = desc.strip().rstrip(".")
    return f"{desc}. Seen from a head-mounted camera and two {RIG_PHRASE[rig]}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--egotouch_index", default=data_path("index", "egotouch.json"),
                    help="tools/data/build_index_egotouch.py output ('' = no EgoTouch)")
    ap.add_argument("--egotouch_hands", default=data_path("cache", "hands_wilor"))
    ap.add_argument("--ht_proc", default=data_path("proc", "HumanTouch"))
    ap.add_argument("--ht_hands", default=data_path("cache", "hands_gt_humantouch"))
    ap.add_argument("--ht_heldout", default="X010", help="comma-separated HumanTouch task_ids held out as test_unseen")
    ap.add_argument("--ev_proc", default=data_path("proc", "EgoViz"))
    ap.add_argument("--egopro_proc", default=data_path("proc", "EgoPro"), help="'' = no EgoPro")
    ap.add_argument("--splits_json", default="", help="fixed splits of listed episodes (configs/splits.json)")
    a = ap.parse_args()
    fixed = json.load(open(a.splits_json))["splits"] if a.splits_json else None

    def split_of(eid, rule):
        if fixed is None:
            return rule
        return fixed.get(eid, "train" if rule == "train" else "heldout")

    eps = []
    if a.egotouch_index:
        et = json.load(open(a.egotouch_index))
        for e in et["episodes"]:
            if not e["usable"] or e["split"] == "none":
                continue
            eid = "EgoTouch/" + e["id"]
            eps.append({"dataset": "EgoTouch", "rig": "egotouch", "id": eid,
                        "videos": {v: os.path.join(et["root"], e["id"], f"{f}.mp4") for v, f in
                                   (("head", "chest"), ("left", "left"), ("right", "right"))},
                        "src_wh": [640, 480], "n_frames": e["n_frames"], "split": split_of(eid, e["split"]),
                        "text": text_for(f"A person {e['task'].replace('_', ' ')}", "egotouch"),
                        "hands": {"kind": "wilor", "path": os.path.join(a.egotouch_hands, e["id"].replace("/", "__") + ".npy"),
                                  "K": {"fx": 12500.0, "fy": 12500.0, "cx": 320.0, "cy": 240.0, "W": 640, "H": 480}},
                        "frozen": e["frozen"]})

    held = set(t for t in a.ht_heldout.split(",") if t)
    for mp in sorted(glob.glob(os.path.join(a.ht_proc, "X*", "episode_*", "meta.json"))):
        m = json.load(open(mp))
        d = os.path.dirname(mp)
        hp = os.path.join(a.ht_hands, f"{m['folder']}__{m['episode']}.npy")
        if not m.get("usable"):
            continue
        if m["task_id"] in held:
            split = "test_unseen"
        else:
            k = h100(m["session_id"])
            split = "test_seen" if k < 3 else "val" if k < 5 else "train"
        eid = f"HumanTouch/{m['folder']}/{m['episode']}"
        K = json.load(open(hp[:-4] + ".json")) if os.path.exists(hp) else None
        eps.append({"dataset": "HumanTouch", "rig": "humantouch", "id": eid,
                    "videos": {v: os.path.join(d, f"{v}.mp4") for v in ("head", "left", "right")}, "src_wh": None,
                    "n_frames": m["n_frames"], "split": split_of(eid, split), "task_id": m["task_id"],
                    "text": text_for(f"A person does the task: {m['task_name']}", "humantouch"),
                    "hands": {"kind": "gt_humantouch", "path": hp, "K": K, "pad_top": 48} if K else None,
                    "frozen": {"left": [], "right": []}})

    for mp in sorted(glob.glob(os.path.join(a.ev_proc, "*", "*", "meta.json"))):
        m = json.load(open(mp))
        if not m.get("usable"):
            continue
        d = os.path.dirname(mp)
        cal = json.load(open(os.path.join(d, "calib.json")))
        Kh = cal["cams"]["head"]["K"]
        s, pad = cal["letterbox"]["scale"], cal["letterbox"]["pad_top"]
        k = h100(m.get("sequence_id") or m["chunk_id"])
        split = "test_seen" if k < 4 else "val" if k < 6 else "train"
        eid = f"EgoViz/{m['category']}/{m['chunk_id']}"
        eps.append({"dataset": "EgoViz", "rig": "egoviz", "id": eid,
                    "videos": {v: os.path.join(d, f"{v}.mp4") for v in ("head", "left", "right")}, "src_wh": None,
                    "n_frames": m["n_frames"], "split": split_of(eid, split), "category": m["category"],
                    "text": text_for(m.get("description") or m.get("title") or m["category"].replace("_", " "), "egoviz"),
                    "hands": {"kind": "hawor_egoviz", "path": os.path.join(d, "hands.npz"),
                              "K": {"fx": Kh[0] * s, "fy": Kh[4] * s, "cx": Kh[2] * s, "cy": Kh[5] * s + pad, "W": 512, "H": 384},
                              "pad_top": pad},
                    "frozen": {"left": [], "right": []}})

    if a.egopro_proc:
        for mp in sorted(glob.glob(os.path.join(a.egopro_proc, "*", "*", "meta.json"))):
            m = json.load(open(mp))
            if not m.get("usable"):
                continue
            d = os.path.dirname(mp)
            c = json.load(open(os.path.join(d, "calib.json")))["canvas"]["head"]
            env = m.get("environment_id") or "?"
            if h100("env:" + env) < 6:
                split = "test_unseen"
            else:
                k = h100(m["uuid"])
                split = "test_seen" if k < 2 else "val" if k < 4 else "train"
            eid = f"EgoPro/{m['uuid']}"
            eps.append({"dataset": "EgoPro", "rig": "egopro", "id": eid,
                        "videos": {v: os.path.join(d, f"{v}.mp4") for v in ("head", "left", "right")}, "src_wh": None,
                        "n_frames": m["n_frames"], "split": split_of(eid, split), "environment_id": env,
                        "text": text_for(m.get("description") or m["task"], "egopro"),
                        "hands": {"kind": "vendor_egopro", "path": os.path.join(d, "hands.npz"),
                                  "K": {k_: c[k_] for k_ in ("fx", "fy", "cx", "cy", "W", "H")}},
                        "frozen": {"left": [], "right": []}})

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"format": "multi_v1", "episodes": eps}, open(a.out, "w"))
    from collections import defaultdict
    hrs = defaultdict(float); cnt = defaultdict(int)
    for e in eps:
        hrs[(e["dataset"], e["split"])] += e["n_frames"] / 30 / 3600
        cnt[(e["dataset"], e["split"])] += 1
    for k in sorted(hrs):
        print(f"{k[0]:10s} {k[1]:11s} {cnt[k]:6d} episodes {hrs[k]:7.2f} h")
    if fixed is not None:
        have = {e["id"] for e in eps}
        miss = [i for i in fixed if i not in have]
        print(f"{len(fixed) - len(miss)} of {len(fixed)} episodes of {a.splits_json} found"
              + (f"; missing e.g. {miss[:3]} (not converted or not usable): evaluation sets will differ" if miss else ""))
    print(f"unique texts: {len(set(e['text'] for e in eps))} -> {a.out}")


if __name__ == "__main__":
    main()
