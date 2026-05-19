#!/usr/bin/env bash
# macOS sibling of run_rollout.sh. Same behavior, just retargeted at:
#   - the lemonkey conda env under ~/miniforge3 (not /home/lemonkey/miniconda3)
#   - the SO-101 follower on /dev/cu.usbmodemXXXX (not /dev/so101-follower)
#   - a USB camera by integer index (not /dev/video0)
#   - a checkpoint on the T7 SSD (not ~/LeMonkey/eval_1/train/...)
#
# All four are overridable via env vars. Defaults:
#   FOLLOWER_PORT  — first /dev/cu.usbmodem* found
#   CAMERA_INDEX   — 0
#   CKPT           — /Volumes/T7/LeMonkey/models/smolvla_eval1_v2/checkpoints/${CKPT_STEP}/pretrained_model
#   ROLLOUT_DIR    — $HOME/LeMonkey/eval_1/rollouts
#
# Usage:
#   ./run_rollout_mac.sh                  # default ckpt step 025000
#   ./run_rollout_mac.sh 020000           # different checkpoint step
#   FOLLOWER_PORT=/dev/cu.usbmodem1234 CAMERA_INDEX=1 ./run_rollout_mac.sh
set -euo pipefail

CKPT_STEP="${1:-025000}"
CKPT="${CKPT:-/Volumes/T7/LeMonkey/models/smolvla_eval1_v2/checkpoints/${CKPT_STEP}/pretrained_model}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$HOME/LeMonkey/eval_1/rollouts}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PYBIN="${PYBIN:-$HOME/miniforge3/envs/lemonkey/bin/python}"
HOME_POSE="/tmp/run_rollout_home.json"
HOME_DRIVE_S=2.0

# Auto-discover follower port if not set
if [ -z "${FOLLOWER_PORT:-}" ]; then
  FOLLOWER_PORT=$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)
  if [ -z "$FOLLOWER_PORT" ]; then
    echo "ERROR: no /dev/cu.usbmodem* found. Plug in the SO-101 follower, or set FOLLOWER_PORT." >&2
    exit 1
  fi
  echo "[INFO] auto-detected FOLLOWER_PORT=$FOLLOWER_PORT (set FOLLOWER_PORT=... to override)"
fi
CAMERA_INDEX="${CAMERA_INDEX:-0}"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "       Set CKPT=/path/to/pretrained_model or download via:" >&2
  echo "         huggingface-cli download HBOrtiz/smolvla_eval1_v2 --local-dir /Volumes/T7/LeMonkey/models/smolvla_eval1_v2 --local-dir-use-symlinks False" >&2
  exit 1
fi
if [ ! -x "$PYBIN" ]; then
  echo "ERROR: python not found at $PYBIN (set PYBIN=... to override)" >&2
  exit 1
fi

export SO101_FOLLOWER_PORT="$FOLLOWER_PORT"  # auto_home.py reads this

mkdir -p "$ROLLOUT_DIR"
echo "Using checkpoint:   $CKPT"
echo "Follower port:      $FOLLOWER_PORT"
echo "Camera index:       $CAMERA_INDEX"
echo "Rollouts saved to:  $ROLLOUT_DIR"
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

  "$PYBIN" -m lerobot.scripts.lerobot_record \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=my_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id="local/eval_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$PROMPT" \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --policy.path="$CKPT"

  "$PYBIN" "$HERE/auto_home.py" drive "$HOME_POSE" "$HOME_DRIVE_S"

  echo
  echo "✓ Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
