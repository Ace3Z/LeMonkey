# SmolVLA + VL co-train — quickstart for a single AWS GPU node

RT-2 §3.2 style co-training: every `vl_ratio+1`-th step is a VQA batch with CE loss on SmolVLM2's LM head; the rest are robot batches with SmolVLA's standard flow-matching action loss. Both gradients flow into the same VLM body, which is what keeps the celeb-name prior alive (the failure mode where sequential VLM→action fine-tunes produce a positional-shortcut policy).

This is the SmolVLA-450M sibling of `eval_3/scripts/track_2/lerobot_train_with_vl_cotrain.py` (Pi0.5-3B). Unlike that one, the script here is **end-to-end integrated** — its training loop runs as-is on a single GPU node.

---

## 0. HANDOVER — read this first if you're an agent picking this up on a fresh cluster instance

**Goal:** fix the diagnosed failure where the SmolVLA policy ignores the prompt
celeb name and always places the coke on the same workspace position
("positional shortcut" / attention sink-lock). We do this with a single
co-training run carrying up to three losses:

- **L_action** — SmolVLA flow-matching on teleop episodes (robot frames). Unchanged.
- **L_name** — VQA CE on celebrity images (name the face). Keeps the celeb-name
  prior alive so it can't drift while the policy learns to act.
- **L_attn (KLAL)** — KL attention-supervision: pushes the name-token→image-patch
  attention toward the prompted celeb's face bbox. The head-on fix for the
  "looks at the wrong face" routing failure. (arXiv:2511.12738; SmolVLM2 port in
  `smolvla_klal.py`.)

`loss = L_action  on robot steps;  vqa + klal_lam·klal  on VL steps` (a VL step
hits every `vl_ratio+1`-th step). Both flow into the shared SmolVLM2 backbone.

**What to run, in order** — see "KLAL ablation arms" below for the exact commands:
1. **A — full fine-tune + KLAL** (paper's setup, λ=1). Run FIRST; sets the ceiling.
2. **B — wide LoRA + KLAL** (λ≈0.1). Run SECOND; "can a cheap adapter match full?"
3. **C — q/k-only LoRA + KLAL** (λ≈0.1). Run THIRD / parallel if compute spare.
   Plus a no-KLAL **cheap bbox-as-text baseline** for comparison.
Always **smoke-test (200 steps) before any long run** (gates below). Start with
arm A's smoke — it has no LoRA injection, fewest moving parts.

**Inference reality (drives the design):** at deploy the model gets only the
**text name** in the prompt ("Place the coke on Barack Obama") + the single
scene camera. No reference image. So the VL stream uses text-name identity (not
the `reference_image_path` column), and KLAL anchors on name-tokens.

### Datasets & models

| Role | Repo | Notes |
|---|---|---|
| Robot teleop (L_action) | `HBOrtiz/so101_eval3_track3_v3_baseline` | 9,394 eps, 3 celebs (Swift/Obama/LeCun), text-only prompts |
| VL pairs (L_name + L_attn) | `HBOrtiz/eval3_objectvla_vl_pairs` | 176,670 rows, 192 celebs. Cols: `image_path` (scene), `prompt`, `target`, `bbox_xyxy_norm` [x1,y1,x2,y2] (face box on the scene), `caption_type` (`qa_grounded`=bare name, `location_explicit`=bbox-as-text), `celeb_slug`, `pid`, `reference_image_path` (UNUSED — inference is text-only) |
| Warm VLM backbone | `HBOrtiz/smolvlm2_lora_celebs` | SmolVLM2-500M with celeb LoRA merged (~60% name acc). Warm-start all KLAL arms from this. |
| SmolVLA base policy | `lerobot/smolvla_base` | 450M; the action-expert + VLM scaffold |

## 1. Environment setup (fresh cluster / Brev / AWS instance)

These are the hard-won versions — wrong Python or ffmpeg will waste hours.
**HuggingFace Pro** strongly recommended (free-tier rate limits stall the 14 GB
dataset pulls).

```bash
# Conda env — Python 3.12 (lerobot requires >=3.12; 3.10 fails to install it)
conda create -n lerobot python=3.12 -y && conda activate lerobot

# ffmpeg 7.1.1 from conda-forge (torchcodec needs it; 8.x / 4.x mismatch)
conda install -y -c conda-forge "ffmpeg=7.1.1"

# libstdc++ fix so torchcodec loads (CXXABI mismatch otherwise) — persistent
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' \
    > $CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# PyTorch for CUDA 12.8 (matches H100 driver). Adjust cu-tag to your driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# lerobot FROM THE SUBMODULE (the PyPI build routes pyav through a missing
# torchvision.VideoReader; the pinned submodule routes it correctly).
cd ~/LeMonkey && git submodule update --init --recursive
cd third_party/lerobot && pip install -e ".[smolvla]" && cd ~/LeMonkey

# Remove hf_xet — its range-get HEAD checks hang snapshot_download mid-run.
pip uninstall -y hf_xet

# File-descriptor limit (dataloader opens many mp4s; default 1024 too low)
ulimit -n 65535 && echo 'ulimit -n 65535' >> ~/.bashrc

# Auth
hf auth login --token <YOUR_HF_TOKEN>   # needs read+write to HBOrtiz/*
wandb login <YOUR_WANDB_TOKEN>          # optional (this script logs to stdout)

# Verify
python -c "import torch, lerobot, torchcodec; print('cuda', torch.cuda.is_available(), '| lerobot', lerobot.__file__)"
```

`lerobot.__file__` MUST point inside `third_party/lerobot/` (editable submodule),
not site-packages. If a pull stalls on HF rate limits, pre-`hf download` the
robot + VL datasets and `HBOrtiz/smolvlm2_lora_celebs` with `--max-workers 2`.

## TL;DR

```bash
export HF_TOKEN=hf_...
# Smoke first (200 steps, small batch, no push):
STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2 bash launch.sh
# Then the real run:
PUSH_REPO=HBOrtiz/smolvla_eval3_cotrain_10to1 bash launch.sh
```

## What the script does

1. Loads `SmolVLAPolicy` from `lerobot/smolvla_base` (or a warm VLM) with `train_expert_only=False` (VLM body trainable) and `freeze_vision_encoder=True` (SigLIP frozen).
2. Builds **two** dataloaders:
   - Robot: `LeRobotDataset` over `HBOrtiz/so101_eval3_track3_v3_baseline`.
   - VL: custom dataset over `HBOrtiz/eval3_objectvla_vl_pairs` parquet (~177k face-VQA pairs).
3. Alternates batches: `step % (vl_ratio+1) == 0` → VL batch (CE loss via `vlm.forward(labels=...)`); else → robot batch (`policy.forward(batch)`).
4. Single AdamW optimizer over all trainable params, gradient clip 10.0.
5. Periodic checkpoints + final HF push.

## Pre-flight (must be green before launching)

| Check | Command |
|---|---|
| `HF_TOKEN` exported (read+write) | `echo $HF_TOKEN \| head -c 8` |
| GPU available | `nvidia-smi` |
| `lerobot` import works | `python -c "import lerobot.policies.smolvla.modeling_smolvla as m; print(m.SmolVLAPolicy.name)"` |
| Robot dataset loads | `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('HBOrtiz/so101_eval3_track3_v3_baseline'); print(len(d))"` |
| VL manifest loads | `python -c "from huggingface_hub import hf_hub_download; p=hf_hub_download('HBOrtiz/eval3_objectvla_vl_pairs','manifest.parquet',repo_type='dataset'); import pandas as pd; print(len(pd.read_parquet(p)))"` |

## Smoke-test gates (200 steps, must all pass)

Run:
```bash
STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2 LOG_EVERY=1 bash launch.sh 2>&1 | tee smoke.log
```

Then grep:

| Gate | What to check | Pass condition |
|---|---|---|
| Both losses fire | `grep -c flow_loss smoke.log` and `grep -c vqa_loss smoke.log` | Both > 0; ratio ≈ vl_ratio:1 |
| No silent NaN | `grep "non-finite" smoke.log` | No matches (or only 1-2 early steps) |
| VRAM headroom | watch `nvidia-smi` during the run | < 90% of total VRAM |
| Loss trending down | compare step 10 vs step 190 `flow_loss` | step 190 ≤ ~70% of step 10 |
| VL loss not stuck | step 11 vs step 187 `vqa_loss` (they hit every 11 steps) | step 187 ≤ ~85% of step 11 |

**If a gate fails, do not launch the 24h run.** Diagnose first.

## The two ratios worth running in parallel

ObjectVLA used 10:1 robot:VL on tabletop objects. Celeb-face discrimination at wrist-cam angle is plausibly harder. Hedge:

```bash
# Terminal / instance 1:
VL_RATIO=10 OUT_DIR=outputs/cotrain_10to1 PUSH_REPO=HBOrtiz/smolvla_cotrain_10to1 bash launch.sh

# Terminal / instance 2 (parallel):
VL_RATIO=5  OUT_DIR=outputs/cotrain_5to1  PUSH_REPO=HBOrtiz/smolvla_cotrain_5to1  bash launch.sh
```

After both finish (~6-8 h each on H100), Strix-test the prompt-sensitivity gate on each — pick the winner.

## Verifying the fix at training time

The whole point of cotrain is to fix the positional-shortcut failure mode. To verify mid-training:

```bash
# Pull an intermediate checkpoint:
huggingface-cli download <repo>/<run> --revision step_15000 --local-dir /tmp/ckpt
# On Strix: load policy, fix scene, run twice with different celeb prompts.
# If the target photo changes → cotrain worked. If not → still broken.
```

If still broken at step 15k, the candidate fixes (in order):
1. Lower `vl_ratio` (more VQA pressure) — relaunch with `VL_RATIO=3`.
2. Raise `vl_batch_size` (more VQA samples per VQA step) — relaunch with `VL_BATCH_SIZE=16`.
3. Switch starting point to `HansOrtiz/smolvlm2_celeb_warm` (warm VLM has VGGFace2 LoRA merged) — relaunch with `PRETRAINED=HansOrtiz/smolvlm2_celeb_warm` (note: warm VLM must Strix-probe cleanly first).

## Known caveats

- The VL collator runs the SmolVLM2 processor twice per batch (prompt-only + prompt+target) to get a correct image-token-aware prompt length for label masking. Slight per-step overhead, but cheap compared to the VLM forward.
- If `vqa_loss` doesn't drop after step 100, dump 2-3 labels rows and verify the `-100` boundary lands right where the target starts. Edge-case: if the SmolVLM2 tokenizer ever produces a different number of leading specials for the prompt-only call vs the prompt+target call, the boundary shifts by 1-2 tokens — visible as `vqa_loss` plateauing at a slightly elevated floor.
- `compile_model` is OFF by default. `torch.compile` over SmolVLA's custom forward path has been flaky in some lerobot versions — turn it on only after smoke passes.
- AMP is bf16 by default (matches SmolVLM2's bf16 weights). fp16 will likely NaN-out the loss; don't switch.
- The script saves checkpoints every `save_freq` steps in `lerobot` format **plus** the preprocessor/postprocessor JSON (normalization stats + tokenizer config). All three must be pushed together — otherwise Strix-side inference cannot reproduce normalization. The training script does this automatically.
- This script has been **written and statically reviewed but NOT smoke-tested** on a real GPU. CLAUDE.md §7 rigour bar: someone must run the 200-step smoke before trusting it for a 24h run. The reviewer specifically flagged two crash-by-step-1 bugs in the original draft (preprocessor not applied to robot batches; VL label masking off by ~80-170 image tokens); both are fixed in the current version but verify on smoke.
- Action-head dim mismatch (low-probability): when loading `lerobot/smolvla_base`, if the checkpoint's `action_in_proj`/`action_out_proj` dimensions disagree with the config's `max_action_dim`, lerobot's `strict=False` silently keeps random-init projections. The first few steps' `flow_loss` will be very large if this happens. Watch step 0-10.

## L_attn — KLAL attention supervision (added on hans-smolvla-cotrain-klal)

The base script co-trains `L_action` (flow) + `L_name` (VQA CE). On top of that
we optionally add `L_attn` — a KL attention-supervision loss that pushes the
name-token→image-patch attention toward the prompted celeb's face bbox. This is
the SmolVLM2 port of the Pi0.5 KLAL (`eval_3/aug/m2_klal.py`); see
`smolvla_klal.py` for the verified SmolVLM2 specifics (8×8 pixel-shuffle grid,
Llama RoPE, GQA 15/5, image_token_id 49190, capture layers ≤15).

Total on a VL step: `loss = vqa + klal_lam · klal`. KLAL recomputes attention
from hooked q_proj/k_proj outputs, so it runs under SDPA (no eager-attention /
`output_attentions` needed).

### KLAL ablation arms (run on the cluster)

KLAL fine-tunes against the bare-name captions (`qa_grounded`) so every label
position is a name token. The `--lora_scope` flag selects the VLM fine-tune
arm. Warm-start all arms from the merged celeb VLM so naming isn't re-learned
from zero (`PRETRAINED=HBOrtiz/smolvlm2_lora_celebs`).

**A — full fine-tune + KLAL** (the KLAL paper's exact setup; run FIRST — it's
the validated recipe and sets the ceiling. λ=1 matches the paper because the
full param set absorbs the signal):
```bash
PRETRAINED=HBOrtiz/smolvlm2_lora_celebs \
CAPTION_FILTER=qa_grounded USE_KLAL=1 KLAL_LAM=1.0 LORA_SCOPE=full \
PUSH_REPO=HBOrtiz/smolvla_eval3_klal_full OUT_DIR=outputs/klal_full \
bash launch.sh
```

**B — wide LoRA + KLAL** (q/k/v/o+MLP; run SECOND — "can a cheap adapter match
full?". λ smaller because few params absorb the KL signal):
```bash
PRETRAINED=HBOrtiz/smolvlm2_lora_celebs \
CAPTION_FILTER=qa_grounded USE_KLAL=1 KLAL_LAM=0.1 LORA_SCOPE=wide LORA_R=32 \
PUSH_REPO=HBOrtiz/smolvla_eval3_klal_wide OUT_DIR=outputs/klal_wide \
bash launch.sh
```

**C — Q/K-only LoRA + KLAL** (attention-routing lever; run THIRD/parallel):
```bash
PRETRAINED=HBOrtiz/smolvlm2_lora_celebs \
CAPTION_FILTER=qa_grounded USE_KLAL=1 KLAL_LAM=0.1 LORA_SCOPE=qk LORA_R=32 \
PUSH_REPO=HBOrtiz/smolvla_eval3_klal_qk OUT_DIR=outputs/klal_qk \
bash launch.sh
```

**Cheap baseline (no KLAL) — ObjectVLA bbox-as-text** for comparison:
```bash
CAPTION_FILTER=location_explicit LORA_SCOPE=full \
PUSH_REPO=HBOrtiz/smolvla_eval3_bboxtext OUT_DIR=outputs/bboxtext \
bash launch.sh
```

Notes:
- `KLAL_LAYERS=all` (default) supervises all 16 text layers, matching the paper
  (Eq.4 averages over all layers). Narrow with a csv (e.g. `6,9,12,15`) to save
  compute if needed.
- λ: full arm uses 1.0 (paper); LoRA arms start 0.1 and sweep up. Watch the
  `klal=` term — it should fall over the first few hundred steps.

### KLAL smoke gates (in addition to the base gates above)
- `grep klal smoke.log` shows a finite, decreasing `klal=` value (not NaN, not stuck).
- Capture layers must be in [0,15] or the hookset raises at construction.
- If `klal=0.0000` every step → bboxes aren't reaching the loss (check the
  manifest has `bbox_xyxy_norm` and `--caption_filter=qa_grounded` kept rows).

> **Review status (CLAUDE.md §9):** `smolvla_klal.py` is non-trivial. The pure
> geometry (bbox→8×8 grid→Gaussian, empty-bbox→no-supervision) is unit-checked
> on CPU. The attention recompute (RoPE capture, GQA expand, causal-column
> renorm) is **statically written, not yet smoke-tested on a GPU** and has not
> had a second-reviewer pass — do both before a long run.

## File layout

```
eval_3/scripts/smolvla_cotrain/
├── cotrain.py        # the training script (L_action + L_name + optional L_attn)
├── smolvla_klal.py   # KLAL attention-supervision loss, SmolVLM2 port
├── launch.sh         # env-var-driven launcher
└── README.md         # this file
```
