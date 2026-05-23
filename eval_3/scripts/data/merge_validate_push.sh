#!/usr/bin/env bash
# Post-augmentation dataset pipeline:
#   1. Merge base teleops with the augmented variants into a LeRobot v3 dataset
#   2. Validate the merged dataset against the LeRobot v3 schema
#   3. Push to HF
#
# Usage:
#   bash eval_3/scripts/data/merge_validate_push.sh          # full pipeline
#   bash eval_3/scripts/data/merge_validate_push.sh merge    # only merge
#   bash eval_3/scripts/data/merge_validate_push.sh validate # only validate (assumes merged exists)
#   bash eval_3/scripts/data/merge_validate_push.sh push     # only push (assumes validated)

set -euo pipefail

STEP="${1:-all}"

# Activate conda env (override with CONDA_ENV=...)
CONDA_ENV="${CONDA_ENV:-lemonkey}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
cd "$HOME/LeMonkey"

BASE_ROOT="${BASE_ROOT:-$HOME/LeMonkey/datasets/eval3}"
AUG_ROOT="${AUG_ROOT:-$HOME/LeMonkey/datasets/eval3_aug_v3_200celebs}"
MERGED_DST="${MERGED_DST:-$HOME/LeMonkey/datasets/eval3_aug_v3_200celebs_merged}"
HF_REPO="${HF_REPO:-HBOrtiz/so101_eval3_aug_v3_200celebs}"
TOKEN_FILE="${TOKEN_FILE:-$HOME/LeMonkey/secrets/huggingface/token_hbortiz}"

# Pin thread counts for any worker pool used by pyarrow / hf_hub
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

if [[ "$STEP" == "all" || "$STEP" == "merge" ]]; then
    echo "==> [1/3] merging base + aug"
    echo "    base: $BASE_ROOT"
    echo "    aug : $AUG_ROOT"
    echo "    dst : $MERGED_DST"
    n_aug=$(find "$AUG_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*__var*" 2>/dev/null | wc -l)
    n_base=$(find "$BASE_ROOT" -mindepth 1 -maxdepth 1 -type d -name "quick_*" 2>/dev/null | wc -l)
    echo "    base eps found : $n_base"
    echo "    aug variants   : $n_aug"
    t_start=$(date +%s)
    # Broad-pipeline variants use the "__var" suffix (not "__t3_").
    python eval_3/scripts/data/merge_episodes.py \
        --base-root "$BASE_ROOT" \
        --aug-root  "$AUG_ROOT" \
        --aug-pattern "__var" \
        --dst       "$MERGED_DST"
    echo "==> merge done in $(($(date +%s) - t_start))s"
    echo
fi

if [[ "$STEP" == "all" || "$STEP" == "validate" ]]; then
    echo "==> [2/3] validating merged dataset against LeRobot v3 schema"
    python eval_3/scripts/data/validate_v3_schema.py --root "$MERGED_DST" || {
        echo "[FATAL] validation failed; not pushing" >&2
        exit 1
    }
    echo "==> validation passed"
    echo
fi

if [[ "$STEP" == "all" || "$STEP" == "push" ]]; then
    echo "==> [3/3] pushing to HF repo: $HF_REPO"
    if [[ ! -f "$TOKEN_FILE" ]]; then
        echo "[FATAL] HF token missing: $TOKEN_FILE" >&2
        exit 2
    fi
    python eval_3/scripts/data/push_dataset_to_hf.py \
        --local "$MERGED_DST" \
        --repo  "$HF_REPO" \
        --token-file "$TOKEN_FILE"
    echo "==> push done: https://huggingface.co/datasets/$HF_REPO"
fi
