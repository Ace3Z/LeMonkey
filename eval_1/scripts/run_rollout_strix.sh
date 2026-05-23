#!/usr/bin/env bash
# Interactive Eval 1 rollout runner - adapted from run_rollout.sh for the
# Strix laptop (rohamzn@rohamzn-ROG-Strix-G533ZX-G533ZX).
#
# Differences vs the Thor original (eval_1/scripts/run_rollout.sh):
#   - paths: /home/lemonkey/... -> /home/rohamzn/ETH_Uni/...
#   - python: ~/miniconda3 -> ~/anaconda3
#   - --policy.path points at the FINAL checkpoint at the repo root
#     (HBOrtiz/so101_smolvla_eval1 was downloaded with only root-level
#     files; the checkpoints/ subdir is empty here)
#   - wrapped in `sg dialout -c` so the new dialout group membership
#     applies (Claude Code's parent shell is still on the old group set)
#
# Usage:
#   ./run_rollout_strix.sh                 # default final checkpoint
# Press right-arrow during a rollout to end it early. Type 'q' at the prompt to quit.
set -euo pipefail

POLICY="/home/rohamzn/ETH_Uni/LeMonkey/eval_1/train/so101_smolvla_eval1"
ROLLOUT_DIR="/home/rohamzn/ETH_Uni/LeMonkey/eval_1/rollouts"
HERE="$(dirname "$(readlink -f "$0")")"
PYBIN="/home/rohamzn/anaconda3/envs/lemonkey/bin/python"
HOME_POSE="/tmp/run_rollout_home.json"
HOME_DRIVE_S=2.0

# Make the lemonkey env's `rerun` viewer binary discoverable inside the
# `sg dialout -c` subshell. Without this, lerobot-record's --display_data=true
# crashes with "Failed to find Rerun Viewer executable in PATH".
export PATH=/home/rohamzn/anaconda3/envs/lemonkey/bin:$PATH

if [ ! -d "$POLICY" ]; then
  echo "ERROR: policy dir not found: $POLICY" >&2
  exit 1
fi

mkdir -p "$ROLLOUT_DIR"
echo "Using policy : $POLICY"
echo "Rollouts go  : $ROLLOUT_DIR"
echo

i=1
while true; do
  echo "=================================================="
  echo "Rollout #$i"
  read -r -p "Prompt (or 'q' to quit): " PROMPT
  case "$PROMPT" in
    q|Q|quit|exit) echo "Bye."; exit 0 ;;
    "") echo "Empty prompt - skipping."; continue ;;
  esac

  TS=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="run_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo "-> Running: $PROMPT"
  echo "-> Saving to: $RUN_PATH"
  echo

  sg dialout -c "\"$PYBIN\" \"$HERE/auto_home.py\" capture \"$HOME_POSE\""

  sg dialout -c "
    /home/rohamzn/anaconda3/envs/lemonkey/bin/lerobot-record \
      --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
      --robot.cameras='{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' \
      --display_data=true \
      --dataset.repo_id='local/eval_$RUN_NAME' \
      --dataset.root='$RUN_PATH' \
      --dataset.num_episodes=1 \
      --dataset.episode_time_s=40 \
      --dataset.reset_time_s=10 \
      --dataset.single_task='$PROMPT' \
      --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
      --dataset.push_to_hub=false \
      --policy.path='$POLICY'
  "

  sg dialout -c "\"$PYBIN\" \"$HERE/auto_home.py\" drive \"$HOME_POSE\" \"$HOME_DRIVE_S\""

  echo
  echo "[OK] Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
