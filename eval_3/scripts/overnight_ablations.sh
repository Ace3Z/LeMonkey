#!/usr/bin/env bash
#
# overnight_ablations.sh — run multiple LoRA configs back-to-back, eval each,
# summarize results. Designed to run unattended overnight on the H100.
#
# Continues on failure (set -u, not -e) so one busted run doesn't kill the rest.
# All logs land in $EVAL3_ROOT/ablation_logs/.
#
# Run inside tmux so you can detach and sleep:
#   tmux new -s ablate
#   bash eval_3/scripts/overnight_ablations.sh
#   # Ctrl+B then D
#
# Wake up, attach:
#   tmux attach -t ablate
#   cat $EVAL3_ROOT/ablation_logs/summary.tsv | column -t -s $'\t'

set -u

# === Config ===
: "${EVAL3_ROOT:=/home/shadeform/datasets/eval3_celebs}"
REPO_DIR="${REPO_DIR:-$HOME/LeMonkey}"
LOG_DIR="$EVAL3_ROOT/ablation_logs"
SUMMARY="$LOG_DIR/summary.tsv"

mkdir -p "$LOG_DIR"
echo "[$(date +%H:%M:%S)] EVAL3_ROOT=$EVAL3_ROOT"
echo "[$(date +%H:%M:%S)] logs → $LOG_DIR"

# Write a fresh summary header (preserve old as .bak if it exists)
[ -f "$SUMMARY" ] && cp "$SUMMARY" "$SUMMARY.bak.$(date +%s)"
printf "name\trank\tepochs\tlr\tacc\tid_acc\ttrain_min\teval_min\tstatus\n" > "$SUMMARY"

cd "$REPO_DIR"

run_ablation() {
    local NAME="$1"
    local RANK="$2"
    local EPOCHS="$3"
    local LR="${4:-1e-4}"
    local OUT="$EVAL3_ROOT/lora_celeb_$NAME"
    local TRAIN_LOG="$LOG_DIR/${NAME}_train.log"
    local EVAL_LOG="$LOG_DIR/${NAME}_eval.log"
    local EVAL_JSONL="$LOG_DIR/${NAME}_predictions.jsonl"

    echo ""
    echo "================================================================="
    echo "[$(date +%H:%M:%S)] START $NAME  (rank=$RANK  epochs=$EPOCHS  lr=$LR)"
    echo "================================================================="

    # --- TRAIN ---
    local T0=$(date +%s)
    python eval_3/scripts/train_smolvlm2_lora.py \
        --data-root "$EVAL3_ROOT/manifests" \
        --out-dir "$OUT" \
        --epochs "$EPOCHS" \
        --lora-rank "$RANK" \
        --lr "$LR" \
        --batch-size 2 --grad-accum 8 \
        --run-name "$NAME" \
        > "$TRAIN_LOG" 2>&1
    local TRAIN_EXIT=$?
    local T1=$(date +%s)
    local TRAIN_MIN=$(( (T1 - T0) / 60 ))

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ $NAME TRAIN FAILED (exit $TRAIN_EXIT, ${TRAIN_MIN}m)"
        printf "%s\t%s\t%s\t%s\t-\t-\t%s\t-\tTRAIN_FAIL\n" \
            "$NAME" "$RANK" "$EPOCHS" "$LR" "$TRAIN_MIN" >> "$SUMMARY"
        return
    fi
    echo "[$(date +%H:%M:%S)] ✓ $NAME train done in ${TRAIN_MIN}m"

    # --- EVAL ---
    local E0=$(date +%s)
    python eval_3/scripts/eval_lora_train_id_accuracy.py \
        --adapter "$OUT" \
        --data-root "$EVAL3_ROOT" \
        --n-identities 50 --n-imgs-per-id 2 \
        --seed 0 \
        --out-jsonl "$EVAL_JSONL" \
        > "$EVAL_LOG" 2>&1
    local EVAL_EXIT=$?
    local E1=$(date +%s)
    local EVAL_MIN=$(( (E1 - E0) / 60 ))

    if [ $EVAL_EXIT -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ $NAME EVAL FAILED (exit $EVAL_EXIT)"
        printf "%s\t%s\t%s\t%s\t-\t-\t%s\t%s\tEVAL_FAIL\n" \
            "$NAME" "$RANK" "$EPOCHS" "$LR" "$TRAIN_MIN" "$EVAL_MIN" >> "$SUMMARY"
        return
    fi

    local ACC=$(grep -oE 'accuracy:[[:space:]]+[0-9]+/[0-9]+ = [0-9.]+%' "$EVAL_LOG" | grep -oE '[0-9.]+%$' || echo "?")
    local ID_ACC=$(grep -oE 'identities w/ ≥1 correct:.+= [0-9.]+%' "$EVAL_LOG" | grep -oE '[0-9.]+%$' || echo "?")
    echo "[$(date +%H:%M:%S)] ✓ $NAME eval: acc=$ACC  id_acc=$ID_ACC  (${EVAL_MIN}m eval)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tOK\n" \
        "$NAME" "$RANK" "$EPOCHS" "$LR" "$ACC" "$ID_ACC" "$TRAIN_MIN" "$EVAL_MIN" >> "$SUMMARY"
}

# =========================================================================
# Ablation matrix — ordered by likely impact / shortest first.
# Total time estimate on H100 (effective batch 16):
#   r=64  e=10 : ~75m  ─ tests rank capacity
#   r=16  e=20 : ~140m ─ tests epoch / training time
#   r=64  e=20 : ~150m ─ combines both winners
#   r=128 e=10 : ~85m  ─ tests if more rank still helps
# TOTAL: ~7.5h. If start at 02:00, done around 09:30.
# Drop the last run if you want a safer 6h budget.
# =========================================================================

run_ablation "r64_e10"  64  10 1e-4
run_ablation "r16_e20"  16  20 1e-4
run_ablation "r64_e20"  64  20 1e-4
run_ablation "r128_e10" 128 10 1e-4

echo ""
echo "================================================================="
echo "[$(date +%H:%M:%S)] ALL DONE"
echo "================================================================="
echo ""
echo "Summary:"
cat "$SUMMARY" | column -t -s $'\t'
echo ""
echo "Don't forget to delete the H100:  brev delete hans-vlm-finetune"
