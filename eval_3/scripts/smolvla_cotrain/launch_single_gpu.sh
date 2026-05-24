#!/usr/bin/env bash
# SmolVLA + VL co-train launch - single-node AWS / Brev / generic CUDA box.
#
# Override any value via env vars before invoking, e.g.:
#   VL_RATIO=5 STEPS=20000 PUSH_REPO=HBOrtiz/my_run bash launch_single_gpu.sh
#
# PRE-FLIGHT (must be true):
#   1. HF_TOKEN exported (read+write) - needed for HBOrtiz/* repos
#   2. conda env with lerobot, transformers, peft, torch (CUDA), pandas, PIL installed
#   3. At least one CUDA GPU visible (`nvidia-smi`)
#   4. The local lerobot at $(python -c 'import lerobot; print(lerobot.__path__)')
#      includes the SmolVLAPolicy class (smolvla policy registered)
#
# SMOKE TEST FIRST: run with STEPS=200 BATCH_SIZE=4 to confirm both losses fire
# and there's no OOM. See README.md gates.

set -euo pipefail

# ---- Defaults (override via env) ----------------------------------------------

ROBOT_DATASET="${ROBOT_DATASET:-HBOrtiz/so101_eval3_cotrain}"
VL_MANIFEST="${VL_MANIFEST:-HBOrtiz/so101_eval3_cotrain_grounding}"
VL_IMAGE_ROOT="${VL_IMAGE_ROOT:-}"          # leave empty to auto-download
PRETRAINED="${PRETRAINED:-lerobot/smolvla_base}"   # or HBOrtiz/smolvlm2_celeb_warm
VLM_OVERRIDE="${VLM_OVERRIDE:-}"            # set to a warm VLM repo to swap inner VLM

STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VL_BATCH_SIZE="${VL_BATCH_SIZE:-8}"
VL_RATIO="${VL_RATIO:-10}"
LR="${LR:-5e-5}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-2}"          # cotrain default: pad 2 missing slots
                                              # so SmolVLA's 3-cam image_features is
                                              # camera1 + 2 empty pads (no reference)

OUT_DIR="${OUT_DIR:-outputs/smolvla_cotrain_${VL_RATIO}to1}"
PUSH_REPO="${PUSH_REPO:-}"                  # leave empty to skip HF push

# ---- KLAL + LoRA (celeb-routing enhancement; off unless ENABLE_*=1) ----------
ENABLE_LORA="${ENABLE_LORA:-0}"             # 1 => freeze VLM base, adapt via LoRA
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LAYERS="${LORA_LAYERS:-all}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

ENABLE_KLAL="${ENABLE_KLAL:-0}"             # 1 => add KLAL attention loss
KLAL_LAMBDA="${KLAL_LAMBDA:-1.0}"
KLAL_LAYERS="${KLAL_LAYERS:-10,12,14}"      # must be a subset of LoRA layers
KLAL_SIGMA="${KLAL_SIGMA:-1.0}"
# KLAL's attention target is built from the VL dataset's quad_corners_norm
# column - no external bbox source is needed.

# ---- Pre-flight ---------------------------------------------------------------

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[WARN] HF_TOKEN not set - HF download/push will fail if any repo is private" >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found - running on CPU will be unusably slow" >&2
else
    echo "==> GPU(s) available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

python -c "import lerobot.policies.smolvla.modeling_smolvla" \
    || { echo "[ERROR] cannot import lerobot.policies.smolvla.modeling_smolvla - check env" >&2; exit 1; }

# ---- Launch -------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$REPO_ROOT/eval_3/scripts/smolvla_cotrain/train_smolvla_cotrain.py"

echo "==> SmolVLA cotrain launching"
echo "    robot      : $ROBOT_DATASET"
echo "    vl         : $VL_MANIFEST"
echo "    pretrained : $PRETRAINED"
[ -n "$VLM_OVERRIDE" ] && echo "    vlm override: $VLM_OVERRIDE"
echo "    steps      : $STEPS"
echo "    bs / vl_bs : $BATCH_SIZE / $VL_BATCH_SIZE"
echo "    vl_ratio   : $VL_RATIO (=> VL batch every $((VL_RATIO + 1))-th step)"
echo "    lr         : $LR"
echo "    output     : $OUT_DIR"
echo "    push       : ${PUSH_REPO:-(skip)}"
echo "    lora       : $([ "$ENABLE_LORA" = "1" ] && echo "on (r=$LORA_R a=$LORA_ALPHA layers=$LORA_LAYERS)" || echo "off (full VLM fine-tune)")"
echo "    klal       : $([ "$ENABLE_KLAL" = "1" ] && echo "on (lambda=$KLAL_LAMBDA layers=$KLAL_LAYERS sigma=$KLAL_SIGMA)" || echo "off")"
echo

CMD=( python -u "$SCRIPT"
      --robot_dataset="$ROBOT_DATASET"
      --vl_manifest="$VL_MANIFEST"
      --pretrained_path="$PRETRAINED"
      --steps="$STEPS"
      --batch_size="$BATCH_SIZE"
      --vl_batch_size="$VL_BATCH_SIZE"
      --vl_ratio="$VL_RATIO"
      --lr="$LR"
      --save_freq="$SAVE_FREQ"
      --log_every="$LOG_EVERY"
      --num_workers="$NUM_WORKERS"
      --seed="$SEED"
      --dtype="$DTYPE"
      --empty_cameras="$EMPTY_CAMERAS"
      --output_dir="$OUT_DIR" )

[ -n "$VL_IMAGE_ROOT" ] && CMD+=( --vl_image_root="$VL_IMAGE_ROOT" )
[ -n "$VLM_OVERRIDE"  ] && CMD+=( --vlm_model_name="$VLM_OVERRIDE" )
[ -n "$PUSH_REPO"     ] && CMD+=( --push_to_hub_repo="$PUSH_REPO" )

if [ "$ENABLE_LORA" = "1" ]; then
    CMD+=( --enable_lora
           --lora_r="$LORA_R"
           --lora_alpha="$LORA_ALPHA"
           --lora_dropout="$LORA_DROPOUT"
           --lora_layers="$LORA_LAYERS"
           --lora_target_modules="$LORA_TARGET_MODULES" )
fi

if [ "$ENABLE_KLAL" = "1" ]; then
    CMD+=( --enable_klal
           --klal_lambda="$KLAL_LAMBDA"
           --klal_layers="$KLAL_LAYERS"
           --klal_sigma="$KLAL_SIGMA" )
fi

"${CMD[@]}"
