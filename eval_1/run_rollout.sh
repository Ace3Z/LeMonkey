#!/usr/bin/env bash
# Interactive SmolVLA rollout runner.
# Asks for a fresh prompt before each rollout. Type 'q' to quit.
#
# Usage:
#   ./run_rollout.sh                 # uses default checkpoint (020000)
#   ./run_rollout.sh 015000          # use a different checkpoint
set -euo pipefail

CKPT_STEP="${1:-020000}"
POLICY="/home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/${CKPT_STEP}/pretrained_model"
ROLLOUT_DIR="/home/lemonkey/LeMonkey/eval_1/rollouts"

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

  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id="local/eval_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$PROMPT" \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --policy.path="$POLICY"

  echo
  echo "✓ Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
