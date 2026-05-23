# SmolVLA + VL co-train - quickstart for a single AWS GPU node

RT-2 §3.2 style co-training: every `vl_ratio+1`-th step is a VQA batch with CE loss on SmolVLM2's LM head; the rest are robot batches with SmolVLA's standard flow-matching action loss. Both gradients flow into the same VLM body, which is what keeps the celeb-name prior alive (the failure mode where sequential VLM→action fine-tunes produce a positional-shortcut policy).

This is the SmolVLA-450M sibling of `eval_3/scripts/pi05_vl_cotrain/lerobot_train_with_vl_cotrain.py` (Pi0.5-3B). Unlike that one, the script here is **end-to-end integrated** - its training loop runs as-is on a single GPU node.

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
   - Robot: `LeRobotDataset` over `HBOrtiz/so101_eval3_cotrain`.
   - VL: custom dataset over `HBOrtiz/so101_eval3_broad_grounding` parquet (~177k face-VQA pairs).
3. Alternates batches: `step % (vl_ratio+1) == 0` → VL batch (CE loss via `vlm.forward(labels=...)`); else → robot batch (`policy.forward(batch)`).
4. Single AdamW optimizer over all trainable params, gradient clip 10.0.
5. Periodic checkpoints + final HF push.

## Pre-flight (must be green before launching)

| Check | Command |
|---|---|
| `HF_TOKEN` exported (read+write) | `echo $HF_TOKEN \| head -c 8` |
| GPU available | `nvidia-smi` |
| `lerobot` import works | `python -c "import lerobot.policies.smolvla.modeling_smolvla as m; print(m.SmolVLAPolicy.name)"` |
| Robot dataset loads | `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('HBOrtiz/so101_eval3_cotrain'); print(len(d))"` |
| VL manifest loads | `python -c "from huggingface_hub import hf_hub_download; p=hf_hub_download('HBOrtiz/so101_eval3_broad_grounding','manifest.parquet',repo_type='dataset'); import pandas as pd; print(len(pd.read_parquet(p)))"` |

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

After both finish (~6-8 h each on H100), Strix-test the prompt-sensitivity gate on each - pick the winner.

## Verifying the fix at training time

The whole point of cotrain is to fix the positional-shortcut failure mode. To verify mid-training:

```bash
# Pull an intermediate checkpoint:
huggingface-cli download <repo>/<run> --revision step_15000 --local-dir /tmp/ckpt
# On Strix: load policy, fix scene, run twice with different celeb prompts.
# If the target photo changes → cotrain worked. If not → still broken.
```

If still broken at step 15k, the candidate fixes (in order):
1. Lower `vl_ratio` (more VQA pressure) - relaunch with `VL_RATIO=3`.
2. Raise `vl_batch_size` (more VQA samples per VQA step) - relaunch with `VL_BATCH_SIZE=16`.
3. Switch starting point to `HansOrtiz/smolvlm2_celeb_warm` (warm VLM has VGGFace2 LoRA merged) - relaunch with `PRETRAINED=HansOrtiz/smolvlm2_celeb_warm` (note: warm VLM must Strix-probe cleanly first).

## Known caveats

- The VL collator runs the SmolVLM2 processor twice per batch (prompt-only + prompt+target) to get a correct image-token-aware prompt length for label masking. Slight per-step overhead, but cheap compared to the VLM forward.
- If `vqa_loss` doesn't drop after step 100, dump 2-3 labels rows and verify the `-100` boundary lands right where the target starts. Edge-case: if the SmolVLM2 tokenizer ever produces a different number of leading specials for the prompt-only call vs the prompt+target call, the boundary shifts by 1-2 tokens - visible as `vqa_loss` plateauing at a slightly elevated floor.
- `compile_model` is OFF by default. `torch.compile` over SmolVLA's custom forward path has been flaky in some lerobot versions - turn it on only after smoke passes.
- AMP is bf16 by default (matches SmolVLM2's bf16 weights). fp16 will likely NaN-out the loss; don't switch.
- The script saves checkpoints every `save_freq` steps in `lerobot` format **plus** the preprocessor/postprocessor JSON (normalization stats + tokenizer config). All three must be pushed together - otherwise Strix-side inference cannot reproduce normalization. The training script does this automatically.
- This script has been **written and statically reviewed but NOT smoke-tested** on a real GPU. CLAUDE.md §7 rigour bar: someone must run the 200-step smoke before trusting it for a 24h run. The reviewer specifically flagged two crash-by-step-1 bugs in the original draft (preprocessor not applied to robot batches; VL label masking off by ~80-170 image tokens); both are fixed in the current version but verify on smoke.
- Action-head dim mismatch (low-probability): when loading `lerobot/smolvla_base`, if the checkpoint's `action_in_proj`/`action_out_proj` dimensions disagree with the config's `max_action_dim`, lerobot's `strict=False` silently keeps random-init projections. The first few steps' `flow_loss` will be very large if this happens. Watch step 0-10.

## File layout

```
eval_3/scripts/smolvla_cotrain/
├── cotrain.py    # the training script
├── launch.sh     # env-var-driven launcher
└── README.md     # this file
```
