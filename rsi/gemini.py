"""Minimal Gemini client for screening and captioning in-the-wild windows (REST generateContent, JSON response).

  GEMINI_API_KEY   required, the API key
  GEMINI_MODEL     optional, the model name (default gemini-4-argon, the model used for the paper)

RIG is the sentence appended to every in-the-wild caption: the rig sentence of the EgoPro training text (bare hands,
wide-angle wrist cameras on straps), i.e. the wrist rig the synthetic data is generated for.
"""
import json, os, random, time, urllib.error, urllib.request

MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-argon")
RIG = "Seen from a head-mounted camera and two wide-angle cameras on wrist straps under the wrists looking toward the bare hands."
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def api_key():
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k:
        raise SystemExit("set GEMINI_API_KEY")
    return k


def call(parts, model=None, temperature=0.2, tries=8):
    """parts: Gemini content parts (text / inline_data). Returns the parsed JSON object of the answer, or None."""
    body = json.dumps({"contents": [{"parts": parts}],
                       "generationConfig": {"responseMimeType": "application/json", "temperature": temperature}}).encode()
    key = api_key()
    for att in range(tries):
        req = urllib.request.Request(URL.format(model=model or MODEL), data=body,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
            return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
            code = getattr(e, "code", None)
            time.sleep(min(60, 2 ** att + random.random() * 2) if code in (429, 500, 503, None) else 5)
    return None
