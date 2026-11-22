"""Self-improvement step 2: screen and caption every window in ONE vision-language call (Gemini, rsi/gemini.py) from the
4 key frames written by rsi/extract_windows.py.
  screen   the camera wearer's own hands only; pass = both own hands visible in >= 3 of the 4 key frames, both bare or
           dark-gloved (what the paired training data covers), handling an object, caption of >= 5 words
  caption  one sentence in the style of the training captions; text for the model = caption + RIG (the EgoPro rig
           sentence, rsi/gemini.py)
-> <dir>/screen.jsonl (every window) and <dir>/captions.jsonl (passing windows: wid, caption, text); resumable.
With --follow it keeps reading <dir>/windows.jsonl while extraction is still running and stops once <dir>/PREP_DONE
exists and every window is screened; it writes <dir>/SCREEN_DONE when finished.
  GEMINI_API_KEY=... python rsi/screen_caption.py --dir $E2W_DATA_ROOT/wild/round1 --workers 16
"""
import argparse, base64, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from rsi.gemini import MODEL, RIG, api_key, call  # noqa: E402

PROMPT = """These 4 images are frames, in time order over 1.6 seconds, from a head-mounted fisheye camera worn by a factory worker (the camera wearer).
Answer ONLY about the camera wearer's OWN hands, except for "others_hands" and "caption". The wearer's own forearms and hands enter the image from the BOTTOM edge and are close to the camera. Hands of other people (coworkers) are farther away and belong to bodies visible in the scene.
1. own_hands: for each of the 4 frames, how many of the wearer's own hands are visible (0, 1 or 2). A hand counts as visible when most of the hand (palm or back of the hand and fingers) is inside the image.
2. left and right: what the wearer's own left hand and own right hand wear. Use exactly one of: "bare" (skin of the fingers visible; thin finger cots or tape still count as bare), "dark_glove" (black or dark grey glove), "light_glove" (white or light-coloured glove), "colored_glove" (glove of any other colour), "not_visible".
3. manipulating: is the wearer handling an object with the hands (true or false).
4. others_hands: are hands of other people visible (true or false).
5. caption: ONE English sentence (8 to 25 words) describing what the wearer's hands are doing: the action, the objects and where they go, in the style of: "The worker picks up dough, fills it with meat, folds it, and places the finished dumpling on a tray." Describe only what is visible; do not mention the camera, the video or the frames.
Return JSON only: {"own_hands": [2, 2, 2, 2], "left": "bare", "right": "bare", "manipulating": true, "others_hands": false, "caption": "The worker ..."}"""
OK_WEAR = {"bare", "dark_glove"}


def one(d, wid, model):
    try:
        parts = [{"text": PROMPT}] + [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(
            open(f"{d}/keyframes/{wid}_{k}.jpg", "rb").read()).decode()}} for k in range(4)]
    except FileNotFoundError:
        return None
    r = call(parts, model=model)
    if not r or "own_hands" not in r:
        return None
    try:
        oh = [int(x) for x in r["own_hands"]][:4]
    except (TypeError, ValueError):
        return None
    cap = " ".join(str(r.get("caption", "")).split())
    if cap and not cap.endswith("."):
        cap += "."
    ok = (sum(x >= 2 for x in oh) >= 3 and r.get("left") in OK_WEAR and r.get("right") in OK_WEAR
          and bool(r.get("manipulating")) and len(cap.split()) >= 5)
    return {"wid": wid, "own_hands": oh, "left": r.get("left"), "right": r.get("right"), "manipulating": r.get("manipulating"),
            "others_hands": r.get("others_hands"), "caption": cap, "pass": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--workers", type=int, default=16, help="concurrent requests")
    ap.add_argument("--model", default=MODEL, help="Gemini model (default: $GEMINI_MODEL or gemini-4-argon)")
    ap.add_argument("--follow", action="store_true", help="keep waiting for new windows until <dir>/PREP_DONE exists")
    a = ap.parse_args()
    api_key()
    d = a.dir
    sp, cp = f"{d}/screen.jsonl", f"{d}/captions.jsonl"
    seen = {json.loads(l)["wid"] for l in open(sp)} if os.path.exists(sp) else set()
    fails = {}
    t0, n0 = time.time(), len(seen)
    with ThreadPoolExecutor(a.workers) as ex, open(sp, "a") as fs, open(cp, "a") as fc:
        while True:
            prep_done = os.path.exists(f"{d}/PREP_DONE") or not a.follow
            wids = [json.loads(l)["wid"] for l in open(f"{d}/windows.jsonl")] if os.path.exists(f"{d}/windows.jsonl") else []
            todo = [w for w in wids if w not in seen and fails.get(w, 0) < 3]
            if not todo:
                if prep_done:
                    break
                time.sleep(60)
                continue
            for w, r in zip(todo, ex.map(lambda w: one(d, w, a.model), todo)):
                if r is None:
                    fails[w] = fails.get(w, 0) + 1
                    continue
                seen.add(w)
                fs.write(json.dumps(r) + "\n")
                if r["pass"]:
                    fc.write(json.dumps({"wid": w, "caption": r["caption"], "text": f"{r['caption']} {RIG}"}) + "\n")
            fs.flush(); fc.flush()
            rate = (len(seen) - n0) / max(1, time.time() - t0)
            print(f"{time.strftime('%m-%d %H:%M:%S')} screened {len(seen)} of {len(wids)} windows ({rate:.2f}/s), "
                  f"{sum(1 for v in fails.values() if v >= 3)} given up", flush=True)
    open(f"{d}/SCREEN_DONE", "w").write("done\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
