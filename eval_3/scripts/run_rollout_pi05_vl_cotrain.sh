#!/usr/bin/env bash
# Interactive Pi0.5 VL cotrain rollout runner — Pi0.5 + ObjectVLA enhanced.
#
# Loads `HBOrtiz/so101_pi05_eval3` (Pi0.5 VL cotrain bbox-grounded
# VQA co-train enhanced spec) from Hugging Face and feeds the typed
# prompt to the policy through lerobot-record.
#
# Adapted from eval_3/scripts/run_rollout.sh (SmolVLA cotrain, SmolVLA, commit
# 68e3ecd) — same control flow, Pi0.5-specific defaults.
#
# Eval-day input contract:
#   observation.images.camera1 + observation.state + task
#   (no reference image, no asset table at inference)
#
# Pi0.5 expects 4 cameras; we feed only `camera1` and rely on the policy's
# `--policy.empty_cameras=N` (baked into the checkpoint) to zero-pad the
# rest. The `--dataset.rename_map` flag matches training-time naming
# (`observation.images.camera1` → `observation.images.right_wrist_0_rgb`).
#
# Usage:
#   ./run_rollout_pi05_vl_cotrain.sh                       # default revision: main
#   ./run_rollout_pi05_vl_cotrain.sh main                  # explicit main branch
#   ./run_rollout_pi05_vl_cotrain.sh <commit-sha>          # pin to a specific HF revision
#   ./run_rollout_pi05_vl_cotrain.sh /path/to/pretrained   # local pretrained_model dir
#
# Trained celebs: ~193 from the scraped bank + 3 IID (Swift/Obama/LeCun).
# Recommended prompt: 'Place the can on the photo of <Name>.' or
#                     'Place the coke on <Name>.'
# Either prompt form should work — Pi0.5 VL cotrain trains on both (6 prompt patterns
# per eval_3/scripts/pi05_vl_cotrain/task_index_to_centroid.json).
#
# Press right-arrow during a rollout to end it early.
# Type 'q' at the prompt to quit.

set -euo pipefail

REPO_ID="${REPO_ID:-HBOrtiz/so101_pi05_eval3}"
ARG="${1:-main}"

HERE="$(dirname "$(readlink -f "$0")")"
PYBIN="${PYBIN:-/home/lemonkey/miniconda3/envs/lemonkey/bin/python}"
[ -x "$PYBIN" ] || PYBIN="$(command -v python)"
AUTO_HOME="${AUTO_HOME:-/home/lemonkey/LeMonkey/scripts/auto_home.py}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/home/lemonkey/LeMonkey/eval_3/rollouts}"
HOME_POSE="/tmp/run_rollout_eval3_pi05_vl_cotrain_home.json"
HOME_DRIVE_S=2.0

# Resolve $ARG → local pretrained_model dir.
if [ -d "$ARG" ] && [ -f "$ARG/model.safetensors" ]; then
  POLICY_PATH="$ARG"
  POLICY_DESC="local: $ARG"
else
  CACHE_BASE="${HOME}/.cache/eval3_rollout"
  POLICY_PATH="$CACHE_BASE/pi05_vl_cotrain_$(echo "$ARG" | tr / _)"
  mkdir -p "$CACHE_BASE"
  if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
    echo "==> downloading ${REPO_ID}@${ARG} → ${POLICY_PATH}"
    huggingface-cli download "$REPO_ID" --revision "$ARG" --local-dir "$POLICY_PATH" \
      || hf download "$REPO_ID" --revision "$ARG" --local-dir "$POLICY_PATH"
  fi
  POLICY_DESC="${REPO_ID}@${ARG} (cached at ${POLICY_PATH})"
fi

if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
  echo "[ERROR] no model.safetensors at $POLICY_PATH" >&2
  echo "[ERROR] expected=downloaded Pi0.5 checkpoint, got=missing" >&2
  exit 1
fi

mkdir -p "$ROLLOUT_DIR"
echo "Eval 3 (Pi0.5 VL cotrain — Pi0.5 ObjectVLA enhanced) — interactive rollout"
echo "Using checkpoint:  $POLICY_DESC"
echo "Rollouts saved to: $ROLLOUT_DIR"
echo
echo "Prompt format:     'Place the can on the photo of <Name>.'"
echo "                   'Place the coke on <Name>.'  (also OK)"
echo "Trained celebs:    ~193 from scraped bank + Swift/Obama/LeCun"
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
  RUN_NAME="pi05_vl_cotrain_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo "→ Running: $PROMPT"
  echo "→ Saving to: $RUN_PATH"
  echo

  if [ -f "$AUTO_HOME" ]; then
    "$PYBIN" "$AUTO_HOME" capture "$HOME_POSE" || true
  fi

  # Pi0.5 uses PaliGemma (not SmolVLM), so the SmolVLM boundaries patch in
  # lerobot_record_with_patch.py does NOT apply. Use plain lerobot-record.
  # If Pi0.5 hits its own transformers compatibility issue at inference time
  # (e.g., the dict-attention-mask risk from TRACK_B_WARMSTART.md §6),
  # we'll add a Pi0.5-specific patch wrapper here.
  #
  # Camera renaming: training used --dataset.rename_map so the policy expects
  # `right_wrist_0_rgb`. We supply the same map at inference for consistency.
  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id="local/eval3_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$PROMPT" \
    --dataset.rename_map='{"observation.images.camera1":"observation.images.right_wrist_0_rgb"}' \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --policy.path="$POLICY_PATH"

  if [ -f "$AUTO_HOME" ]; then
    "$PYBIN" "$AUTO_HOME" drive "$HOME_POSE" "$HOME_DRIVE_S" || true
  fi

  echo
  echo "✓ Rollout #$i complete: $RUN_PATH"
  echo
  i=$((i+1))
done
