#!/usr/bin/env bash
# Pi0.5 VL cotrain — ObjectVLA bbox-grounded VQA co-train on Pi0.5 (brev_instance2).
#
# Mirrors train_pi05.sh + adds:
#   - --vl_dataset.manifest  (Darius's VL pairs, HBOrtiz/so101_eval3_broad_grounding)
#   - --vl_ratio=10          (10 robot batches : 1 VL batch — ObjectVLA published)
#   - --policy.pretrained_path=HBOrtiz/pi05_paligemma_celeb_warm_v2  (Enhancement B-1)
#   - --dataset.episodes_file (Enhancement B-2 keep_list — when ready)
#   - --dataset.sample_weights (Enhancement B-3 hardneg weights — when ready)
#   - --dataset.curriculum_switch_step=5000 (Enhancement B-5)
#   - --peft.layer_rank_config (Enhancement B-4 — per-layer LoRA rank)
#   - --train.use_ema (Enhancement B-7)
#
#
# PRE-FLIGHT (must be true before running this):
#   1. Darius has pushed the VL pairs manifest HBOrtiz/so101_eval3_broad_grounding.
#   2. Roham has delivered per-frame bboxes for 200-celeb dataset.
#   3. ArcFace audit pipeline has run:
#        python eval_3/scripts/pi05_vl_cotrain/arcface_audit_200celeb.py
#        python eval_3/scripts/pi05_vl_cotrain/build_keep_list_and_weights.py
#        → keep_episodes.txt + hardneg_weights.npy exist
#   4. Brev VM has conda 'lemonkey' env, transformers + lerobot + peft installed.
#   5. Smoke test passed (run with STEPS=200 first — see the ObjectVLA spec).

set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/pi05_vl_cotrain_objectvla}"
DATASET="${DATASET:-HBOrtiz/so101_eval3_aug_v3_200celebs}"
VL_MANIFEST="${VL_MANIFEST:-HBOrtiz/so101_eval3_broad_grounding}"
VL_RATIO="${VL_RATIO:-10}"
PUSH_REPO="${PUSH_REPO:-HBOrtiz/so101_pi05_eval3}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-1e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGETS='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'

# Enhancement B-1: warm-PG starting point (NOT cold pi05_base).
PRETRAINED="${PRETRAINED:-HBOrtiz/pi05_paligemma_celeb_warm_v2}"

# Enhancement B-2/B-3 artifacts (paths relative to repo root).
KEEP_LIST="${KEEP_LIST:-eval_3/scripts/pi05_vl_cotrain/keep_episodes.txt}"
SAMPLE_WEIGHTS="${SAMPLE_WEIGHTS:-eval_3/scripts/pi05_vl_cotrain/hardneg_weights.npy}"

# Enhancement B-4: per-layer LoRA rank config.
LAYER_RANK_CONFIG="${LAYER_RANK_CONFIG:-eval_3/scripts/pi05_vl_cotrain/layer_rank.json}"

# Enhancement B-5: curriculum learning switch step.
CURRICULUM_SWITCH="${CURRICULUM_SWITCH:-5000}"

# Enhancement B-7: EMA shadow weights.
USE_EMA="${USE_EMA:-true}"
EMA_ALPHA="${EMA_ALPHA:-0.999}"

# Pre-flight artifact checks — emit [WARN] not abort (some enhancements optional).
if [ ! -f "$KEEP_LIST" ]; then
    echo "[WARN] keep_list missing: expected=$KEEP_LIST, got=missing, fallback=launch without B-2 filter" >&2
    KEEP_LIST_FLAG=""
else
    KEEP_LIST_FLAG="--dataset.episodes_file=$KEEP_LIST"
fi

if [ ! -f "$SAMPLE_WEIGHTS" ]; then
    echo "[WARN] hardneg_weights missing: expected=$SAMPLE_WEIGHTS, got=missing, fallback=launch without B-3" >&2
    SAMPLE_WEIGHTS_FLAG=""
else
    SAMPLE_WEIGHTS_FLAG="--dataset.sample_weights=$SAMPLE_WEIGHTS"
fi

if [ ! -f "$LAYER_RANK_CONFIG" ]; then
    echo "[WARN] layer_rank_config missing: expected=$LAYER_RANK_CONFIG, got=missing, fallback=uniform r=$LORA_R" >&2
    LAYER_RANK_FLAG="--peft.r=$LORA_R"
else
    LAYER_RANK_FLAG="--peft.layer_rank_config=$LAYER_RANK_CONFIG"
fi

echo "==> Pi0.5 VL cotrain (Pi0.5 ObjectVLA enhanced) launching"
echo "    pretrained: $PRETRAINED"
echo "    dataset   : $DATASET"
echo "    vl_manif  : $VL_MANIFEST"
echo "    vl_ratio  : $VL_RATIO"
echo "    output    : $OUT_DIR"
echo "    push to   : $PUSH_REPO"
echo "    steps     : $STEPS  (curriculum switch at: $CURRICULUM_SWITCH)"
echo "    batch     : $BATCH_SIZE"
echo "    lr        : $LR"
echo "    keep_list : ${KEEP_LIST_FLAG:-(none)}"
echo "    weights   : ${SAMPLE_WEIGHTS_FLAG:-(none)}"
echo "    layer_rank: $LAYER_RANK_FLAG"
echo "    ema       : $USE_EMA (alpha=$EMA_ALPHA)"
echo

python -u eval_3/scripts/pi05_vl_cotrain/lerobot_train_with_vl_cotrain.py \
    --policy.type=pi05 \
    --policy.pretrained_path="$PRETRAINED" \
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
    $LAYER_RANK_FLAG \
    --peft.lora_alpha="$LORA_ALPHA" \
    --peft.lora_dropout=0.05 \
    \
    --dataset.repo_id="$DATASET" \
    --dataset.rename_map='{"observation.images.camera1":"observation.images.right_wrist_0_rgb"}' \
    $KEEP_LIST_FLAG \
    $SAMPLE_WEIGHTS_FLAG \
    --dataset.curriculum_switch_step="$CURRICULUM_SWITCH" \
    \
    --vl_dataset.manifest="$VL_MANIFEST" \
    --vl_ratio="$VL_RATIO" \
    \
    --train.use_ema="$USE_EMA" \
    --train.ema_alpha="$EMA_ALPHA" \
    \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --output_dir="$OUT_DIR" \
    --policy.push_to_hub=True \
    --policy.repo_id="$PUSH_REPO"

echo
echo "==> Pi0.5 VL cotrain training done. Pushed to $PUSH_REPO."
