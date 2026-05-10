#!/usr/bin/env bash
# Interactive SmolVLA rollout runner.
# Asks for a fresh prompt before each rollout. Type 'q' to quit.
#
# Usage:
#   ./run_rollout.sh                 # uses default checkpoint (v2 025000)
#   ./run_rollout.sh 020000          # use a different v2 checkpoint
# Press right-arrow during a rollout to end it early.
set -euo pipefail

CKPT_STEP="${1:-025000}"
POLICY="/home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1_v2/checkpoints/${CKPT_STEP}/pretrained_model"
ROLLOUT_DIR="/home/lemonkey/LeMonkey/eval_1/rollouts"
HERE="$(dirname "$(readlink -f "$0")")"
PYBIN="/home/lemonkey/miniconda3/envs/lemonkey/bin/python"
HOME_POSE="/tmp/run_rollout_home.json"
HOME_DRIVE_S=2.0

if [ ! -d "$POLICY" ]; then
  echo "ERROR: checkpoint not found: $POLICY" >&2
  exit 1
fi

mkdir -p "$ROLLOUT_DIR"
echo "Using checkpoint: $POLICY"
echo "Rollouts will be saved under: $ROLLOUT_DIR"
echo

i=1
while true; do
  echo "=================================================="
  echo "Rollout #$i"
  read -r -p "Prompt (or 'q' to quit): " PROMPT
  case "$PROMPT" in
    q|Q|quit|exit) echo "Bye."; exit 0 ;;
    "") echo "Empty prompt — skipping."; continue ;;
  esac

  TS=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="run_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo "→ Running: $PROMPT"
  echo "→ Saving to: $RUN_PATH"
  echo

  "$PYBIN" "$HERE/auto_home.py" capture "$HOME_POSE"

  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id="local/eval_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$PROMPT" \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --policy.path="$POLICY"

  "$PYBIN" "$HERE/auto_home.py" drive "$HOME_POSE" "$HOME_DRIVE_S"

  echo
  echo "✓ Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
