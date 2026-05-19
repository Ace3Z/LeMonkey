#!/usr/bin/env bash
# Edna — 200-celeb v3 augmentation production launcher.
#
# Launches NUM_WORKERS parallel python processes, each handling a stripe of
# the base teleop list. Each worker writes its own _run_summary_w<id>.json
# so they don't collide. Per-worker logs land in $OUT_ROOT/_logs/.
#
# Usage:
#   bash eval_3/scripts/edna/run_aug_v3_200celebs.sh           # full run (10k variants, 64 workers, ~2h)
#   bash eval_3/scripts/edna/run_aug_v3_200celebs.sh smoke     # smoke (4 variants, 2 workers, ~2 min)
#
# Env-vars:
#   OUT_ROOT      output dir (default: ~/LeMonkey/datasets/eval3_aug_v3_200celebs)
#   NUM_VARIANTS variants per base ep   (default 56 → ~50 target appearances per celeb)
#   NUM_WORKERS  parallel workers        (default 64 → ≈3 bases per worker)
#   SEED         RNG seed                (default 42)
#
# Background usage:
#   nohup bash eval_3/scripts/edna/run_aug_v3_200celebs.sh > ~/aug_v3.log 2>&1 &
#   echo $! > ~/aug_v3.pid
#   tail -f ~/aug_v3.log

set -euo pipefail

MODE="${1:-full}"

# Activate conda env
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" != "aug" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate aug
    else
        echo "[FATAL] conda.sh missing at \$HOME/miniconda3/etc/profile.d/conda.sh" >&2
        exit 2
    fi
fi

cd "$HOME/LeMonkey"

# Defaults
case "$MODE" in
    smoke)
        OUT_ROOT="${OUT_ROOT:-/tmp/aug_v3_smoke_$(date +%s)}"
        NUM_VARIANTS="${NUM_VARIANTS:-2}"
        NUM_WORKERS="${NUM_WORKERS:-2}"
        ;;
    full)
        OUT_ROOT="${OUT_ROOT:-$HOME/LeMonkey/datasets/eval3_aug_v3_200celebs}"
        NUM_VARIANTS="${NUM_VARIANTS:-56}"
        NUM_WORKERS="${NUM_WORKERS:-64}"
        ;;
    *)
        echo "Usage: $0 [smoke|full]" >&2
        exit 1
        ;;
esac

SEED="${SEED:-42}"
BASE_ROOT="${BASE_ROOT:-$HOME/LeMonkey/datasets/eval3}"
PHOTO_BANK="${PHOTO_BANK:-$HOME/LeMonkey/datasets/eval3_celebs/scraped}"

if [[ ! -d "$BASE_ROOT" ]]; then
    echo "[FATAL] base teleop root missing: $BASE_ROOT" >&2
    exit 2
fi
if [[ ! -d "$PHOTO_BANK" ]]; then
    echo "[FATAL] photo bank missing: $PHOTO_BANK" >&2
    exit 2
fi

N_BASES=$(find "$BASE_ROOT" -maxdepth 1 -name "quick_*" -type d | wc -l)
N_CELEBS=$(find "$PHOTO_BANK" -mindepth 1 -maxdepth 1 -type d | wc -l)

mkdir -p "$OUT_ROOT" "$OUT_ROOT/_logs"

echo "==> aug v3 launcher ($MODE)"
echo "    base teleops : $N_BASES (from $BASE_ROOT)"
echo "    celebs       : $N_CELEBS (from $PHOTO_BANK)"
echo "    out_root     : $OUT_ROOT"
echo "    num_variants : $NUM_VARIANTS per base = ~$((N_BASES * NUM_VARIANTS)) total variants planned"
echo "    workers      : $NUM_WORKERS  (≈$((N_BASES / NUM_WORKERS)) bases per worker)"
echo "    seed         : $SEED"
echo

t_start=$(date +%s)

# For smoke mode, only feed the first 2 base teleops; in full mode use --root.
if [[ "$MODE" == "smoke" ]]; then
    FIRST_TWO=$(ls -d "$BASE_ROOT"/quick_* | head -2)
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        # shellcheck disable=SC2086
        python eval_3/aug/generate_aug_v3.py \
            --episode-dirs $FIRST_TWO \
            --photo-bank "$PHOTO_BANK" \
            --out-root "$OUT_ROOT" \
            --num-variants "$NUM_VARIANTS" \
            --num-workers "$NUM_WORKERS" --worker-id "$i" \
            --seed "$SEED" \
            > "$OUT_ROOT/_logs/worker_$(printf '%02d' "$i").log" 2>&1 &
    done
else
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        python eval_3/aug/generate_aug_v3.py \
            --root "$BASE_ROOT" \
            --photo-bank "$PHOTO_BANK" \
            --out-root "$OUT_ROOT" \
            --num-variants "$NUM_VARIANTS" \
            --num-workers "$NUM_WORKERS" --worker-id "$i" \
            --seed "$SEED" \
            > "$OUT_ROOT/_logs/worker_$(printf '%02d' "$i").log" 2>&1 &
    done
fi

# Track PIDs so failures don't go silent
PIDS=( $(jobs -p) )
echo "    launched PIDs: ${PIDS[*]}"
echo "    monitor with:  tail -f $OUT_ROOT/_logs/worker_00.log"
echo

wait
t_end=$(date +%s)
elapsed=$((t_end - t_start))

n_variants=$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "_logs" | wc -l)
n_mp4=$(find "$OUT_ROOT" -name "*.mp4" | wc -l)
n_errors=$(grep -lE "error|Traceback" "$OUT_ROOT"/_run_summary_w*.json 2>/dev/null | wc -l)

echo "==> done in ${elapsed}s ($((elapsed / 60)) min)"
echo "    variants     : $n_variants  (mp4 count $n_mp4)"
echo "    workers with errors in summary: $n_errors"
if (( n_errors > 0 )); then
    echo "    [WARN] some workers reported errors — inspect $OUT_ROOT/_run_summary_w*.json"
fi
echo
echo "Next: merge with eval_3/scripts/merge_track3_custom.py + push to HF if you want it as a Track-B dataset."
