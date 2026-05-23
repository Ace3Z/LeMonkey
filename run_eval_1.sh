#!/usr/bin/env bash
# Eval 1 - interactive SmolVLA rollout runner (HBOrtiz/so101_smolvla_eval1).
#
# SmolVLA-450M trained for direct color-conditioned pick-and-place: pick up the
# banana, place it in the bowl named by the prompt ("Put the banana in the
# <colour> colored bowl."). Single-camera contract: wrist USB cam on
# /dev/video0 at 480x640 / 30 fps.
#
# Downloads the final 25k checkpoint from HF on first use.
#
# Usage (with the `lemonkey` conda env active, from anywhere in the repo):
#   ./run_eval_1.sh                       # default = HF root (final 25k)
#   ./run_eval_1.sh checkpoints/020000    # earlier intermediate
#   ./run_eval_1.sh /local/dir            # custom local pretrained dir
# Type 'q' at the prompt to quit. Press right-arrow during a rollout to end early.
set -euo pipefail

REPO_ID="HBOrtiz/so101_smolvla_eval1"
ARG="${1:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_HOME="$REPO_ROOT/scripts/auto_home.py"
ROLLOUT_DIR="$REPO_ROOT/eval_1/rollouts"
# Prefer the shipped policy/ copy (submission zip layout). Falls back to a
# user cache if policy/ is not present.
if [ -d "$REPO_ROOT/policy/so101_smolvla_eval1" ]; then
  CACHE="$REPO_ROOT/policy/so101_smolvla_eval1"
else
  CACHE="$HOME/.cache/lemonkey_eval/so101_smolvla_eval1"
fi
HOME_POSE=/tmp/run_eval_1_home.json
HOME_DRIVE_S=2.0

# HF offline avoids the SmolVLM2 chat-template HEAD probe that otherwise hits
# HF's anonymous rate limit. The checkpoint is cached locally on first use.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -n "$ARG" ] && [ -d "$ARG" ] && [ -f "$ARG/model.safetensors" ]; then
  POLICY_PATH="$ARG"
elif [ -z "$ARG" ]; then
  POLICY_PATH="$CACHE"
  if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
    echo "==> downloading ${REPO_ID} (root) -> $CACHE"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    hf download "$REPO_ID" --exclude "checkpoints/*" --local-dir "$CACHE"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  fi
else
  POLICY_PATH="$CACHE/$ARG"
  if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
    echo "==> downloading ${REPO_ID} ($ARG) -> $CACHE"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    hf download "$REPO_ID" --include "$ARG/*" --local-dir "$CACHE"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  fi
fi

if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
  echo "ERROR: no model.safetensors at $POLICY_PATH" >&2
  exit 1
fi

mkdir -p "$ROLLOUT_DIR"
echo "Eval 1 - so101_smolvla_eval1 - interactive rollout"
echo "Policy : $POLICY_PATH"
echo "Saving : $ROLLOUT_DIR"
echo "Prompt : 'Put the banana in the <colour> colored bowl.'"
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
  RUN_NAME="eval1_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo "-> Running: $PROMPT"
  echo "-> Saving to: $RUN_PATH"

  python "$AUTO_HOME" capture "$HOME_POSE" || true

  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
    --robot.cameras='{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' \
    --display_data=true \
    --dataset.repo_id="local/eval_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$PROMPT" \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --policy.path="$POLICY_PATH"

  python "$AUTO_HOME" drive "$HOME_POSE" "$HOME_DRIVE_S" || true

  echo
  echo "[done] Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
