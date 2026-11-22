"""Default locations of data, models and caches.

Everything lives under one root, E2W_DATA_ROOT (default: ./data, relative to the current directory):

  $E2W_DATA_ROOT/models/Wan2.2-Fun-5B-InP        base model (override with E2W_BASE_MODEL)
  $E2W_DATA_ROOT/models/hand_det/detector.pt     WiLoR hand detector (HCC and in-the-wild window extraction)
  $E2W_DATA_ROOT/raw/...                         downloaded datasets
  $E2W_DATA_ROOT/proc/...                        converted paired datasets (tools/data/convert_*.py)
  $E2W_DATA_ROOT/index/paired.json               episode index (tools/data/build_multi_index.py)
  $E2W_DATA_ROOT/cache/...                       prompt embeddings, hand joints
  $E2W_DATA_ROOT/runs/...                        checkpoints and evaluations
  $E2W_DATA_ROOT/wild/...                        in-the-wild windows of the self-improvement rounds

Every script also takes explicit command-line paths; these are only the defaults.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("E2W_DATA_ROOT", os.path.abspath("data"))


def data_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


BASE_MODEL = os.environ.get("E2W_BASE_MODEL", data_path("models", "Wan2.2-Fun-5B-InP"))
HAND_DETECTOR = data_path("models", "hand_det", "detector.pt")
INDEX = data_path("index", "paired.json")
PROMPTS = data_path("cache", "prompt_embeds_paired.pt")
