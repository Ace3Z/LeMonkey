#!/usr/bin/env bash
# Eval 3 rollout runner — HBOrtiz/so101_smolvla_eval3_broad.
#
# The broad / out-of-distribution Eval 3 policy: SmolVLA-450M trained on the
# 192-celebrity dataset (so101_eval3_broad), 30k steps. Final 25k checkpoint at
# the HF repo root; intermediates under checkpoints/{005000..025000}/.
#
# Single-camera inference: supply only camera1; the unused camera slot is auto
# zero-padded via the policy's empty_cameras=1 setting.
#
# Usage (run with the `lemonkey` conda env active, from anywhere in the repo):
#   ./run_rollout_smolvla_broad.sh                      # default = root (final 25k)
#   ./run_rollout_smolvla_broad.sh checkpoints/020000   # an earlier checkpoint
#   ./run_rollout_smolvla_broad.sh /local/dir           # a custom local pretrained dir
set -euo pipefail

REPO_ID="HBOrtiz/so101_smolvla_eval3_broad"
ARG="${1:-}"

# Repo-relative paths (this script lives at eval_3/scripts/).
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AUTO_HOME="$REPO_ROOT/scripts/auto_home.py"
ROLLOUT_DIR="$REPO_ROOT/eval_3/rollouts"
if [ -d "$REPO_ROOT/policy/so101_smolvla_eval3_broad" ]; then
  CACHE="$REPO_ROOT/policy/so101_smolvla_eval3_broad"
else
  CACHE="$HOME/.cache/eval3_rollout/so101_smolvla_eval3_broad"
fi
HOME_POSE=/tmp/run_rollout_smolvla_broad_home.json
HOME_DRIVE_S=2.0

# HF offline: skips the SmolVLM2 chat-template HEAD probe that otherwise hits
# HF's anonymous rate limit and stalls every rollout by ~8 min. (Safe to keep.)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Resolve $ARG → a local pretrained dir.
if [ -n "$ARG" ] && [ -d "$ARG" ] && [ -f "$ARG/model.safetensors" ]; then
  POLICY_PATH="$ARG"
elif [ -z "$ARG" ]; then
  # Default: final 25k at the repo root.
  POLICY_PATH="$CACHE"
  if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
    echo "==> downloading ${REPO_ID} (root) → $CACHE"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    hf download "$REPO_ID" --exclude "checkpoints/*" --local-dir "$CACHE"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  fi
else
  # $ARG is something like 'checkpoints/020000'.
  POLICY_PATH="$CACHE/$ARG"
  if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
    echo "==> downloading ${REPO_ID} ($ARG) → $CACHE"
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
echo "Eval 3 — so101_smolvla_eval3_broad (192-celebrity model) — interactive rollout"
echo "Policy : $POLICY_PATH"
echo "Saving : $ROLLOUT_DIR"
echo "Schema : camera1 (live); camera2/3 + empty_camera_0 auto-padded (empty_cameras=1)"
echo "Prompt : 'Put the coke on <celeb_name>.'"
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
  RUN_NAME="smolvla_broad_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo "→ Running: $PROMPT"
  echo "→ Saving to: $RUN_PATH"

  python "$AUTO_HOME" capture "$HOME_POSE" || true

  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
    --robot.cameras='{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}' \
    --display_data=true \
    --dataset.repo_id="local/eval_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$PROMPT" \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --policy.path="$POLICY_PATH"

  python "$AUTO_HOME" drive "$HOME_POSE" "$HOME_DRIVE_S" || true

  echo
  echo "✓ Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
