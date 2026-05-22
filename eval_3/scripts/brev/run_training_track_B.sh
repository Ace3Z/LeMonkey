#!/usr/bin/env bash
# Track B — Pi0.5 (PaliGemma-2B + Gemma-300M expert) on Brev RTX PRO 6000 Blackwell.
#
# See eval_3/tracks/TRACK_B.md for the full recipe + 3-agent validation findings.
#
# PRE-FLIGHT (must be true before running this):
#   1. Quantile stats recomputed on the merged dataset:
#        python third_party/lerobot/src/lerobot/scripts/augment_dataset_quantile_stats.py \
#            --repo-id local/eval3_track3_v3 \
#            --root /home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_merged \
#            --overwrite
#   2. Dataset re-pushed to HF (so HBOrtiz/so101_eval3_track3_v3_baseline has the
#      corrected meta/stats.json):
#        python eval_3/scripts/push_dataset_to_hf.py \
#            --local /home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_merged \
#            --repo HBOrtiz/so101_eval3_track3_v3_baseline
#   3. Brev VM has the conda 'lemonkey' env + this repo synced.

set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/pi05_track_B}"
# Pi0.5 reads exact-quantile stats from the dedicated pi05 dataset repo
# (the SmolVLA baseline uses approximate quantiles in stats.json which is
# fine for MEAN_STD normalization but NOT for Pi0.5's QUANTILES mode).
DATASET="${DATASET:-HBOrtiz/so101_eval3_track3_v3_pi05}"
PUSH_REPO="${PUSH_REPO:-HBOrtiz/pi05_eval3_track_B}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LR="${LR:-1e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
# Full Gemma transformer block: 4 attention projections + 3 gated-MLP projections.
# Standard LLaMA/Gemma LoRA practice. Wider coverage = more capacity for the
# adapters to absorb celeb-discriminative features in the VLM.
LORA_TARGETS='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'

echo "==> Track B (Pi0.5) launching"
echo "    dataset : $DATASET"
echo "    output  : $OUT_DIR"
echo "    push to : $PUSH_REPO"
echo "    steps   : $STEPS"
echo "    batch   : $BATCH_SIZE"
echo "    lr      : $LR"
echo "    lora_r  : $LORA_R"
echo

lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=True \
    --policy.train_expert_only=False \
    --policy.empty_cameras=2 \
    --policy.optimizer_lr="$LR" \
    --policy.gradient_checkpointing=True \
    --policy.compile_model=True \
    \
    --peft.method_type=LORA \
    --peft.target_modules="$LORA_TARGETS" \
    --peft.r="$LORA_R" \
    --peft.lora_alpha="$LORA_ALPHA" \
    --peft.lora_dropout=0.05 \
    \
    --dataset.repo_id="$DATASET" \
    --dataset.rename_map='{"observation.images.camera1":"observation.images.right_wrist_0_rgb"}' \
    \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --output_dir="$OUT_DIR" \
    --policy.push_to_hub=True \
    --policy.repo_id="$PUSH_REPO"

echo
echo "==> Track B training done. Pushed to $PUSH_REPO."
