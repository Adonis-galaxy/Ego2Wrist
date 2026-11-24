# Ego2Wrist: Bootstrapping Wrist-View Synthesis from Egocentric Video

Ego2Wrist synthesizes the two wrist-camera videos of a person from a head-camera video. Given 49 frames of
head-mounted egocentric video and a one-line task description, it generates the left and right wrist videos
(384x512 each). The model is a rank-64 LoRA on Wan2.2-Fun-5B-InP that denoises a latent panel
[head | left wrist | right wrist]: the head latent is given, the wrist latents are generated, and every wrist token
attends to every head token and to the other wrist. Because in-the-wild egocentric video has no wrist cameras, we also
introduce HCC (hand-centric cross-view consistency), a score that needs no wrist ground truth: it compares the
generated wrist view with the region around the same-side hand in the head view. HCC lets the model improve itself on
unlabeled video: it synthesizes wrist views for windows of Egocentric-10K, keeps the top 30% by HCC, adds them to the
paired training data and continues training (recursive self-improvement).

**Project page:** https://adonis-galaxy.github.io/Ego2Wrist/ · **Paper:** [arXiv:2611.29850](https://arxiv.org/abs/2611.29850) · **Weights:** [Hugging Face](https://huggingface.co/adonis126/Ego2Wrist)

This repository contains the code for the three parts of the paper: the panel diffusion model (training, sampling,
evaluation), HCC, and the self-improvement loop, plus data conversion and indexing for the four paired datasets and
Egocentric-10K, the trivial baselines and the paired bootstrap report of the main table.

## Main result

Standard test set (375 clips of 49 frames), LPIPS of both wrists against the real wrist videos (lower is better):

| Method | EgoPro | EgoTouch | EgoViz | HumanTouch | All |
|---|---|---|---|---|---|
| Untuned Wan2.2-Fun-5B-InP | 0.841 | 0.997 | 0.891 | 0.932 | 0.920 |
| Head-view crop | 0.744 | 0.812 | 0.792 | 0.752 | 0.777 |
| Real wrist video of another clip | 0.650 | 0.461 | 0.604 | 0.657 | 0.582 |
| WorldWander, fine-tuned on our data | 0.548 | 0.415 | 0.577 | 0.542 | 0.510 |
| Ego2Wrist, head guidance w=1 | 0.425 | 0.264 | 0.496 | 0.360 | 0.374 |
| **Ego2Wrist, head guidance w=3** | **0.419** | **0.248** | **0.496** | **0.349** | **0.365** |

## Repository layout

| Path | Content |
|---|---|
| `e2w/model.py` | latent panel, inpainting condition, per-token timesteps, model loading, LoRA |
| `e2w/train.py` | LoRA training (`python -m e2w.train`, torchrun) |
| `e2w/sample.py` | sampling with head guidance and evaluation (PSNR / SSIM / LPIPS / layout PSNR) (`python -m e2w.sample`) |
| `e2w/data.py`, `e2w/multidata.py` | window loaders for the mixed index |
| `e2w/paths.py` | default locations under `E2W_DATA_ROOT` |
| `hcc/hcc.py` | HCC (`HCC` class) |
| `hcc/score_videos.py` | HCC of every video in a folder |
| `rsi/` | self-improvement round: download, window extraction, screening and captioning, text encoding, generation, scoring, selection |
| `tools/data/` | download helpers, converters and index builder of the paired datasets, prompt encoding |
| `tools/paired_report.py` | clip-paired comparison with bootstrap confidence intervals |
| `tools/baseline_trivial.py` | head-view-crop and another-clip baselines |
| `configs/splits.json` | the paper's splits of all non-training episodes (reproduces the evaluation sets) |
| `third_party/training/` | Wan2.2 model code vendored from DWM (see `third_party/README.md`) |

## Installation

Python 3.10, one or more NVIDIA GPUs with 48 GB (training uses one 49-frame clip per GPU with gradient checkpointing).

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118   # or your CUDA
pip install -r requirements.txt
export E2W_DATA_ROOT=/path/to/data     # default: ./data
```

Run all commands from the repository root. `e2w/model.py` adds `third_party/` to the Python path itself.

HCC uses DINOv3 through `transformers>=4.56`, while the panel model was trained and evaluated with
`transformers==4.51.3`. We ran HCC in a separate environment (or with a newer `transformers` first on `PYTHONPATH`);
a single environment with a newer `transformers` is likely to work for both but was not tested.

Models:

| Model | Used by | Where to put it |
|---|---|---|
| [alibaba-pai/Wan2.2-Fun-5B-InP](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP) | everything | `$E2W_DATA_ROOT/models/Wan2.2-Fun-5B-InP` (or set `E2W_BASE_MODEL`) |
| WiLoR hand detector [`detector.pt`](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt) | HCC, window extraction | `$E2W_DATA_ROOT/models/hand_det/detector.pt` |
| [facebook/dinov3-vit7b16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) (gated) | HCC | Hugging Face cache, or pass a local directory with `--dinov3` |
| [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) | HCC (fallback for gloved hands) | Hugging Face cache, or `--owl <dir>` |

If loading `detector.pt` fails under torch 2.6 with a `weights_only` error, set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

## Data

### Layout

```
$E2W_DATA_ROOT/
  models/                     base model, hand detector
  raw/                        downloads (EgoTouch, HumanTouch, EgoPro_tmp, EgoViz_tmp, Egocentric10K)
  proc/EgoPro/<task>/<uuid>/            head.mp4 left.mp4 right.mp4 hands.npz calib.json meta.json
  proc/EgoViz/<category>/<chunk_id>/    head.mp4 left.mp4 right.mp4 hands.npz poses.npz calib.json meta.json
  proc/HumanTouch/<task>/episode_*/     head.mp4 left.mp4 right.mp4 poses.npz skeleton.npz tactile.npz meta.json
  index/egotouch.json         EgoTouch episode index (EgoTouch videos are read from raw/EgoTouch directly)
  index/paired.json           the mixed index used for training and evaluation
  cache/prompt_embeds_paired.pt
  cache/hands_wilor/, cache/hands_gt_humantouch/    head-view hand joints (head-view-crop baseline)
  runs/                       checkpoints and evaluations
  wild/round1/ ...            self-improvement rounds
```

All converted videos are 30 fps, 512x384 per view, with black bars where the aspect ratio differs.

### Paired datasets

Download each dataset under its own terms (see the dataset pages; EgoPro and some others are gated).

```bash
# EgoTouch (Hugging Face dataset zhouzhoujy/EgoTouch) -> raw/EgoTouch/<scene>/<task>/<episode>/{chest,left,right}.mp4
# (+ quality_issues.json per episode, split.json at the root)
python tools/data/build_index_egotouch.py --root $E2W_DATA_ROOT/raw/EgoTouch --out $E2W_DATA_ROOT/index/egotouch.json

# HumanTouch (Hugging Face dataset chuqiaoLyu/Xspark-HumanTouch), one task folder at a time, X001 ... X010
hf download chuqiaoLyu/Xspark-HumanTouch --repo-type dataset --include "X001/*" --local-dir $E2W_DATA_ROOT/raw/HumanTouch
python tools/data/convert_humantouch.py --src $E2W_DATA_ROOT/raw/HumanTouch/X001 --out $E2W_DATA_ROOT/proc/HumanTouch

# EgoViz-120 (storage bucket humaidtech/EgoViz-120); the paper uses 8 categories (see tools/data/convert_egoviz.py)
hf buckets cp hf://buckets/humaidtech/EgoViz-120/index.jsonl $E2W_DATA_ROOT/raw/EgoViz/index.jsonl
CATS=cleaning_kitchen,cooking_food,managing_laundry,managing_trash,organizing_items,preparing_cleaning_supplies,preparing_food,wiping_surfaces
python tools/data/prefetch_egoviz.py --categories $CATS --index $E2W_DATA_ROOT/raw/EgoViz/index.jsonl &
python tools/data/convert_egoviz.py --categories $CATS --index $E2W_DATA_ROOT/raw/EgoViz/index.jsonl \
    --out $E2W_DATA_ROOT/proc/EgoViz --tmp $E2W_DATA_ROOT/raw/EgoViz_tmp

# EgoPro (gated storage bucket LightwheelAI/EgoPro); listing = JSON list of {"path", "size"} of the bucket's files
python tools/data/prefetch_egopro.py --listing egopro_bucket.json --selection $E2W_DATA_ROOT/raw/EgoPro/selection.jsonl &
python tools/data/convert_egopro.py --list $E2W_DATA_ROOT/raw/EgoPro/selection.jsonl \
    --out $E2W_DATA_ROOT/proc/EgoPro --tmp $E2W_DATA_ROOT/raw/EgoPro_tmp
```

The converters are resumable (episodes with a `meta.json` are skipped) and delete each MCAP after conversion.

### Index and text embeddings

```bash
# head-view hand joints (only needed for the head-view-crop baseline)
python tools/data/cache_hands_egotouch.py --index $E2W_DATA_ROOT/index/egotouch.json --out $E2W_DATA_ROOT/cache/hands_wilor
python tools/data/cache_hands_humantouch.py --proc $E2W_DATA_ROOT/proc/HumanTouch --out $E2W_DATA_ROOT/cache/hands_gt_humantouch

python tools/data/build_multi_index.py --out $E2W_DATA_ROOT/index/paired.json --splits_json configs/splits.json
python tools/data/encode_prompts.py --index $E2W_DATA_ROOT/index/paired.json --out $E2W_DATA_ROOT/cache/prompt_embeds_paired.pt
```

Splits keep whole recording units apart (EgoTouch: official split; HumanTouch: task X010 is the unseen-task test
split, other tasks split by session; EgoViz: by recording sequence; EgoPro: 6% of environments form the unseen test
split, the rest split by episode). `configs/splits.json` holds the paper's split of every non-training episode; with
it, the index reproduces the paper's training set and evaluation clips exactly (the script reports episodes of the
file that it could not find). EgoPro episodes that are neither training nor in the file go to the `heldout` split,
which is used only for the extended EgoPro evaluation below. Each episode's text is its task description followed by
a sentence that names the camera rig.

## Training

`e2w.train` trains the LoRA (rank 64, alpha 64) on all linear layers of the transformer with AdamW
(betas 0.9 / 0.95, weight decay 0.01), gradient clipping at 1, head and text dropout of 0.1 each, flow matching with
shift 12, one 49-frame clip per GPU and micro-batch. The batch is `world_size x --grad_accum`. A run resumes from the
newest `step_*` checkpoint in `--out_dir`; `--init_lora` starts a new stage from earlier weights with a fresh
optimizer; `--decay_latest` / `--decay_steps` end it with a cosine decay; `--ema 0.9995` also saves
`lora_ema.safetensors`.

The final model was trained in consecutive stages, each starting from the previous one (dataset weights
EgoPro 0.4, EgoViz 0.24, HumanTouch 0.18, EgoTouch 0.18 throughout):

| Stage | Steps | Batch | LR | Warm-up | Cosine decay | Notes |
|---|---|---|---|---|---|---|
| 1 | 12,000 | 4 | 1e-4 | 200 | none | |
| 2 | 12,000 -> 60,000 | 6 | 1e-4 | (continues stage 1 with its optimizer) | from 52,000 over 8,000 | |
| 3 | 4,000 | 24 | 5e-5 | 200 | from 2,000 over 2,000 | |
| 4 | 53,500 | 24 | 5e-5 | 200 | from 49,000 over 4,500 | 288x384 until step 44,500, then 384x512; EMA 0.9995 |
| 5 | 12,550 | 24 | 5e-5 | 500 | from 8,050 over 4,500 | starts from stage 4's raw (non-EMA) weights; EMA 0.9995 |

Stages 1 to 3 and the first 10,500 steps of stage 4 used only the part of EgoPro available at the time. The final
stage, for example on 4 GPUs:

```bash
W=EgoPro=0.4,EgoViz=0.24,HumanTouch=0.18,EgoTouch=0.18
torchrun --nproc_per_node=4 -m e2w.train --out_dir $E2W_DATA_ROOT/runs/final --init_lora <stage 4>/lora.safetensors \
    --steps 12550 --grad_accum 6 --lr 5e-5 --warmup 500 --ckpt_every 500 --seed 606 --ema 0.9995 \
    --weights $W --decay_latest 8050 --decay_steps 4500
```

## Sampling and evaluation

`e2w.sample` takes the LoRA weights as an argument (`--lora`), samples 30 Euler steps with fixed noise per clip
(seed 1000 + clip index, so all compared models start from the same noise), and writes per-clip metrics. Head
guidance (`--cfg`) applies classifier-free guidance to the head condition only; the main table uses `w=3`. Standard
test set, 4 shards (one per GPU):

```bash
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k python -m e2w.sample --lora <ckpt>/lora_ema.safetensors --out $E2W_DATA_ROOT/runs/eval_w3 \
      --cfg 3 --conds real --max_per_dataset EgoTouch=120,HumanTouch=80,EgoViz=75,EgoPro=100 --shard $k --n_shards 4 &
done; wait
```

`--conds real,shuffle,nohead` adds two controls (another clip's head video, no head video). `--lora none` evaluates
the untuned base model. The extended EgoPro set (600 clips, used for the self-improvement rounds) is
`--splits test_unseen,heldout --datasets EgoPro --max_per_dataset 600`.

Baselines and paired comparison:

```bash
python tools/baseline_trivial.py --out $E2W_DATA_ROOT/runs/baselines          # head-view crop, another clip's wrist
python tools/paired_report.py --pair "w=3 vs head crop" $E2W_DATA_ROOT/runs/eval_w3:real $E2W_DATA_ROOT/runs/baselines/headcrop:real
```

The report gives, per dataset, the mean LPIPS of both sides, the paired difference with a 95% bootstrap interval over
clips (10,000 resamples), the share of clips on which the first side is better, and the paired layout-PSNR difference
(PSNR after shrinking frames by 16).

WorldWander baseline: the code depends on the WorldWander repository and is not included. We fine-tuned its
three2one mode (head video in context, one wrist video generated, side drawn per sample) from its released
`three2one_realworld` LoRA with its public recipe on our training split and dataset weights: 8,001 steps at batch 16,
learning rate 1e-4 (its rule: 5e-5 x sqrt(GPUs x accumulation)), its flow-matching loss and 10% text dropout, our
384x512, 49-frame windows, the prompt "Transform it into the {left|right} wrist-camera perspective." followed by our
task text. At test time each wrist is sampled on its own from the same head window with its sampler (UniPC, 50 steps,
flow shift 5, guidance 5), and scored with `e2w.sample.metrics`.

## HCC on a folder of generated videos

```bash
python -m hcc.score_videos --videos <folder of .mp4> --out <folder>/hcc.csv \
    --hand_detector $E2W_DATA_ROOT/models/hand_det/detector.pt
```

Videos are either a single row `[head | left | right]` (as written by `rsi/generate.py`) or the two-row videos of
`e2w.sample` (`--save_mp4`; add `--real` to also score the real wrist videos); each view is 512x384 and at least
49 frames long. The CSV has one row per video and wrist; the column `hcc` is the score of a wrist (NaN when the hand
is found in fewer than 3 of the 13 frames), and the score of a video is the mean over its two wrists. Precision
(`xv_prec`), recall (`xv_rec`) and auxiliary diagnostics are also written; see `hcc/hcc.py`. In Python:

```python
from hcc.hcc import HCC
scorer = HCC(hand_detector="detector.pt")              # DINOv3 ViT-7B/16 + WiLoR detector + OWLv2 fallback
row = scorer(wrist_frames, head_frames, key="clip0", side="left", rng=np.random.default_rng(0))
print(row["hcc"])
```

## One round of recursive self-improvement

Each round starts from the current model M (round 1: the final model M0), generates with its raw weights
(`lora.safetensors`) and continues training from them. All steps are resumable; steps 4 and 5 can run with several
workers on one or more machines that share the round directory.

```bash
D=$E2W_DATA_ROOT/wild/round1; M=<ckpt dir of the current model>
# 1. download the Egocentric-10K subset (accept the dataset terms and `hf auth login` first) and cut hand-busy windows
python rsi/download_ego10k.py --out $E2W_DATA_ROOT/raw/Egocentric10K
python rsi/extract_windows.py --raw $E2W_DATA_ROOT/raw/Egocentric10K --out $D --procs 16
# 2. screen (both own hands visible, bare or dark gloves, handling an object) and caption in one Gemini call
export GEMINI_API_KEY=...            # optional: GEMINI_MODEL (default gemini-4-argon)
python rsi/screen_caption.py --dir $D
# 3. text embeddings
python rsi/encode_captions.py --dir $D
# 4. synthesize both wrists for every window (15 steps, no guidance, one sample per window); one worker per GPU
CUDA_VISIBLE_DEVICES=0 python rsi/generate.py --dir $D --lora $M/lora.safetensors --gpu_tag 0
# 5. HCC + lens check
CUDA_VISIBLE_DEVICES=0 python rsi/score.py --dir $D --gpu_tag 0
# 6. keep the top 30% by HCC -> index/round1_D.json and cache/prompt_embeds_round1.pt
python rsi/build_round.py --dir $D --tag round1
# 7. continue training: synthetic windows are a fifth source sampled 25% of the time, batch about 30
W=EgoPro=0.3,EgoViz=0.18,HumanTouch=0.135,EgoTouch=0.135,EgoSyn=0.25
torchrun --nproc_per_node=4 -m e2w.train --index $E2W_DATA_ROOT/index/round1_D.json \
    --prompts $E2W_DATA_ROOT/cache/prompt_embeds_round1.pt --emb_on_cpu --init_lora $M/lora.safetensors \
    --out_dir $E2W_DATA_ROOT/runs/round1 --steps 4000 --lr 2e-5 --warmup 200 --grad_accum 8 --seed 911 \
    --ckpt_every 500 --ema 0.9995 --weights $W --decay_latest 2000 --decay_steps 2000
# 8. evaluate the EMA weights of the last step (standard set and extended EgoPro set, see above)
```

Steps 2 to 5 can overlap: run them with `--follow` and they wait for the previous step's marker file
(`PREP_DONE`, `SCREEN_DONE`, `ENCODE_DONE`, and `GEN_DONE`, which you create once all generation workers are done).
`build_round.py` also writes the comparison arms of the paper (A: paired data only, B: all windows, C: a random 30%).
The next round uses new windows (new Egocentric-10K workers or clips), generates with the raw weights of the new model
and builds its index from the paired data plus only its own selected windows.

## Checkpoints

We release the final model M0 (the model of the main table) as LoRA weights (rank 64, alpha 64, on all linear layers
of the transformer) for the base model Wan2.2-Fun-5B-InP, which must be downloaded separately. Each file is a
safetensors state dict of the LoRA network (about 753 MB, fp32) and is loaded with `--lora`. To run your own
self-improvement rounds, start from the raw M0 weights (see "One round of recursive self-improvement").

| Model | File | Description | Download |
|---|---|---|---|
| M0 (main table) | `ego2wrist_m0_ema.safetensors` | the final model, EMA weights; the main-table rows "Ego2Wrist, w=1 / w=3" | [download](https://huggingface.co/adonis126/Ego2Wrist/resolve/main/ego2wrist_m0_ema.safetensors) |
| M0, raw | `ego2wrist_m0_raw.safetensors` | the same step without averaging; the starting point and generator of self-improvement round 1 | [download](https://huggingface.co/adonis126/Ego2Wrist/resolve/main/ego2wrist_m0_raw.safetensors) |

```bash
python -m e2w.sample --lora ego2wrist_m0_ema.safetensors --out <dir> --cfg 3 --conds real \
    --max_per_dataset EgoTouch=120,HumanTouch=80,EgoViz=75,EgoPro=100
```

Both files are on [Hugging Face](https://huggingface.co/adonis126/Ego2Wrist) (`hf download adonis126/Ego2Wrist`). sha256: `0b172bc1…3950af3d` (EMA), `5b8df31c…22b6a85` (raw).

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). `third_party/training` is an
unmodified subset of [DWM](https://github.com/snuvclab/dwm), also under Apache-2.0
([third_party/training/LICENSE](third_party/training/LICENSE)). The released model weights are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), because part of the training data is licensed for
non-commercial use only. HCC uses DINOv3 and the WiLoR hand detector, which are downloaded separately and are subject
to their own licenses (the DINOv3 License and CC BY-NC-ND 4.0); each dataset is subject to its own terms.

## Authors

Ziyao Zeng<sup>1</sup>, Xiatao Sun<sup>1</sup>, Hao Wang<sup>2</sup>, Ziyu Guo<sup>2</sup>, Youheng Yao<sup>1</sup>,
Jingcheng Ni<sup>3</sup>, Shenyuan Gao<sup>4</sup>, Fengyu Yang<sup>1</sup>, Renrui Zhang<sup>5</sup>, Bangya Liu<sup>6</sup>,
Daniel Rakita<sup>1</sup>

<sup>1</sup>Yale University, <sup>2</sup>Google DeepMind, <sup>3</sup>Brown University, <sup>4</sup>NVIDIA,
<sup>5</sup>Independent Researcher, <sup>6</sup>2077 AI

## Citation

```bibtex
@article{zeng2026ego2wrist,
  title   = {Ego2Wrist: Bootstrapping Wrist-View Synthesis from Egocentric Video},
  author  = {Zeng, Ziyao and Sun, Xiatao and Wang, Hao and Guo, Ziyu and Yao, Youheng and Ni, Jingcheng and Gao, Shenyuan and Yang, Fengyu and Zhang, Renrui and Liu, Bangya and Rakita, Daniel},
  journal = {arXiv preprint arXiv:2611.29850},
  year    = {2026}
}
```
