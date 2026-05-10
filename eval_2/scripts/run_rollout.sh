#!/usr/bin/env bash
# Interactive Eval 2 rollout runner — type any prompt, run it.
# Mirrors eval_1/scripts/run_rollout.sh but points at the camera-frame v2 model.
#
# Usage:
#   ./run_rollout.sh                 # default checkpoint (v2 025000)
#   ./run_rollout.sh 020000          # try a different intermediate ckpt
# Press right-arrow during a rollout to end it early.
# Type 'q' at the prompt to quit.
#
# Convention reminder: the model was trained with CAMERA-FRAME spatial language
# — "leftmost" means the bowl on the IMAGE LEFT side. Look at the camera feed.
set -euo pipefail

CKPT_STEP="${1:-025000}"
POLICY="/home/lemonkey/LeMonkey/eval_2/train/smolvla_eval2_v2/checkpoints/${CKPT_STEP}/pretrained_model"
ROLLOUT_DIR="/home/lemonkey/LeMonkey/eval_2/rollouts"
HERE="$(dirname "$(readlink -f "$0")")"
PYBIN="/home/lemonkey/miniconda3/envs/lemonkey/bin/python"
AUTO_HOME="/home/lemonkey/LeMonkey/eval_1/scripts/auto_home.py"
HOME_POSE="/tmp/run_rollout_eval2_typed_home.json"
HOME_DRIVE_S=2.0

if [ ! -d "$POLICY" ]; then
  echo "ERROR: checkpoint not found: $POLICY" >&2
  exit 1
fi

mkdir -p "$ROLLOUT_DIR"
echo "Using checkpoint: $POLICY"
echo "Rollouts will be saved under: $ROLLOUT_DIR"
echo "Convention      : CAMERA-FRAME (read 'left'/'right' as image-left/right)"
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

  # Strip the eval-day "from the robot perspective" qualifier (PROJECT.md §2)
  # before sending to the policy — the model was not trained on that phrase.
  RAW_PROMPT="$PROMPT"
  PROMPT=$("$PYBIN" "$HERE/filter_prompt.py" "$PROMPT")
  if [ "$PROMPT" != "$RAW_PROMPT" ]; then
    echo "→ Original: $RAW_PROMPT"
    echo "→ Filtered: $PROMPT"
  fi

  TS=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="typed_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo "→ Running: $PROMPT"
  echo "→ Saving to: $RUN_PATH"
  echo

  "$PYBIN" "$AUTO_HOME" capture "$HOME_POSE"

  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
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

  "$PYBIN" "$AUTO_HOME" drive "$HOME_POSE" "$HOME_DRIVE_S"

  echo
  echo "✓ Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
