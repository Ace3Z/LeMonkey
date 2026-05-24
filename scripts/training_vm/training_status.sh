#!/usr/bin/env bash
# One-shot snapshot of a lerobot-train run's status: systemd service state,
# any matching process, GPU utilisation, latest progress line from the log,
# the last 5 INFO/WARN/ERR events, and the saved checkpoint list.
#
# Usage (all four bits required):
#   LOG=$HOME/outputs/train/<run>.log \
#   UNIT=lerobot-train-eval3 \
#   CHECKPOINT_DIR=$HOME/outputs/train/<run>/checkpoints \
#       bash scripts/training_vm/training_status.sh
#
# LOG may also be passed as argv[1] for convenience.

LOG="${1:-${LOG:-}}"
UNIT="${UNIT:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

if [ -z "$LOG" ] || [ -z "$UNIT" ] || [ -z "$CHECKPOINT_DIR" ]; then
  echo "[FATAL] LOG, UNIT, and CHECKPOINT_DIR are all required." >&2
  echo "        See the header of this script for the invocation pattern." >&2
  exit 2
fi

echo "=== systemd service ==="
systemctl --user is-active "${UNIT}.service" 2>&1 | head -1
echo
echo "=== process ==="
ps -ef | grep -E "lerobot-train" | grep -v grep | head -3 || echo "  no training process running"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader 2>&1

echo
echo "=== latest training progress (from $LOG) ==="
if [ ! -f "$LOG" ]; then
  echo "  log file does not exist"
  exit 0
fi

tr '\r' '\n' < "$LOG" | grep -E "Training: +[0-9]+%|step [0-9]+/" | tail -1

echo
echo "=== last 5 INFO/WARN/ERR events ==="
tr '\r' '\n' < "$LOG" | grep -E "INFO|\[WARN\]|ERROR|Traceback|saved|complete|Finished" | tail -5

echo
echo "=== checkpoints saved ==="
ls -d "$CHECKPOINT_DIR"/*/ 2>/dev/null | head -10 || echo "  no checkpoints yet"
