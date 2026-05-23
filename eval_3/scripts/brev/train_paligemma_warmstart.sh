#!/usr/bin/env bash
# Brev - PaliGemma VQA warm-start (LoRA on Pi0.5's PaliGemma, VGGFace2 data).
#
# Designed to run on a SECOND Brev H100 80GB VM, parallel to the Pi0.5
# vanilla LoRA run. See eval_3/scripts/warmstart/train_paligemma_vqa.py for
# the recipe + architecture rationale.
#
# PRE-FLIGHT (on the Brev VM):
#   1. Conda env `lemonkey` with lerobot + peft + datasets installed
#   2. VGGFace2 manifest at $MANIFEST_PATH (build via prepare_vggface2_vqa.py
#      using Hans's VGGFace2 raw dir).
#   3. HF token at secrets/huggingface/token_hbortiz.

set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/paligemma_celeb_warm}"
PUSH_REPO="${PUSH_REPO:-HBOrtiz/pi05_paligemma_celeb_warm}"
MANIFEST_PATH="${MANIFEST_PATH:-$HOME/LeMonkey/datasets/vggface2_vqa_train.parquet}"
PRETRAINED_PI05="${PRETRAINED_PI05:-lerobot/pi05_base}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# Activate conda env (nohup'd shells are non-login; need explicit source)
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" != "lemonkey" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate lemonkey
    else
        echo "[FATAL] conda.sh missing - install miniconda + create lemonkey env first" >&2
        exit 2
    fi
fi

# Resolve HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -f "$HOME/LeMonkey/secrets/huggingface/token_hbortiz" ]]; then
        HF_TOKEN="$(cat "$HOME/LeMonkey/secrets/huggingface/token_hbortiz")"
    fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[WARN] HF_TOKEN unset: expected=token_hbortiz, got=missing, fallback=anonymous (push will fail)" >&2
else
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "[FATAL] manifest missing: $MANIFEST_PATH" >&2
    echo "        Build it first via:" >&2
    echo "        python eval_3/scripts/warmstart/prepare_vggface2_vqa.py \\" >&2
    echo "             --vggface2-root /path/to/vggface2/train \\" >&2
    echo "             --names-csv     /path/to/identity_meta.csv \\" >&2
    echo "             --scraped-root  ~/LeMonkey/datasets/eval3_celebs/scraped \\" >&2
    echo "             --out           $MANIFEST_PATH" >&2
    exit 2
fi

echo "==> PaliGemma VQA warm-start launching"
echo "    manifest      : $MANIFEST_PATH"
echo "    pretrained    : $PRETRAINED_PI05"
echo "    output_dir    : $OUT_DIR"
echo "    push_to       : $PUSH_REPO"
echo "    epochs        : $EPOCHS"
echo "    batch (eff)   : $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "    lr            : $LR"
echo "    lora r/alpha  : $LORA_R / $LORA_ALPHA"
echo

cd "$HOME/LeMonkey"

python eval_3/scripts/warmstart/train_paligemma_vqa.py \
    --manifest "$MANIFEST_PATH" \
    --pretrained-pi05 "$PRETRAINED_PI05" \
    --output-dir "$OUT_DIR" \
    --push-repo "$PUSH_REPO" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA"

echo
echo "==> Done. Warmed Pi0.5 pushed to $PUSH_REPO."
echo "    For the Pi0.5 re-launch, use --policy.pretrained_path=$PUSH_REPO"
