#!/usr/bin/env bash
# One-shot snapshot of training status (eval2 default). Run: ~/training_status.sh
LOG="${1:-$HOME/outputs/train/smolvla_eval2.log}"

echo "=== systemd service ==="
systemctl --user is-active lerobot-train-eval2.service 2>&1 | head -1
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
ls -d ~/outputs/train/smolvla_eval2/checkpoints/*/ 2>/dev/null | head -10 || echo "  no checkpoints yet"
