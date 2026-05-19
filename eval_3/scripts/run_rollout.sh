#!/usr/bin/env bash
# Interactive Eval 3 rollout runner — type any prompt, run it.
#
# Loads `HBOrtiz/smolvla_eval3_track_D_m2_mahbod` (Track D, M2 ArcFace
# distillation) from Hugging Face and feeds the typed prompt to the policy
# through lerobot-record. Mirrors eval_2/scripts/run_rollout.sh.
#
# Eval-day input contract (TODO.md §inference recipe, post-TA-ruling):
#   observation.images.camera1 + observation.state + task
#   (no reference image, no asset table at inference)
#
# Usage:
#   ./run_rollout.sh                        # default revision: step-10000
#   ./run_rollout.sh step-10000             # explicit revision name
#   ./run_rollout.sh step-5000              # earlier intermediate
#   ./run_rollout.sh main                   # final checkpoint (once training finishes)
#   ./run_rollout.sh /path/to/pretrained    # local pretrained_model dir
#
# Trained celebs: Taylor Swift, Barack Obama, Yann LeCun
# Recommended prompt: 'Place the coke on <celeb_name>.'
#
# Press right-arrow during a rollout to end it early.
# Type 'q' at the prompt to quit.

set -euo pipefail

REPO_ID="HBOrtiz/smolvla_eval3_track_D_m2_mahbod"
ARG="${1:-step-10000}"

HERE="$(dirname "$(readlink -f "$0")")"
PYBIN="${PYBIN:-/home/lemonkey/miniconda3/envs/lemonkey/bin/python}"
[ -x "$PYBIN" ] || PYBIN="$(command -v python)"
AUTO_HOME="${AUTO_HOME:-/home/lemonkey/LeMonkey/eval_1/scripts/auto_home.py}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/home/lemonkey/LeMonkey/eval_3/rollouts}"
HOME_POSE="/tmp/run_rollout_eval3_typed_home.json"
HOME_DRIVE_S=2.0

# Resolve $ARG → local pretrained_model dir.
if [ -d "$ARG" ] && [ -f "$ARG/model.safetensors" ]; then
  POLICY_PATH="$ARG"
  POLICY_DESC="local: $ARG"
else
  CACHE_BASE="${HOME}/.cache/eval3_rollout"
  POLICY_PATH="$CACHE_BASE/$(echo "$ARG" | tr / _)"
  mkdir -p "$CACHE_BASE"
  if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
    echo "==> downloading ${REPO_ID}@${ARG} → ${POLICY_PATH}"
    huggingface-cli download "$REPO_ID" --revision "$ARG" --local-dir "$POLICY_PATH"
  fi
  POLICY_DESC="${REPO_ID}@${ARG} (cached at ${POLICY_PATH})"
fi

if [ ! -f "$POLICY_PATH/model.safetensors" ]; then
  echo "ERROR: no model.safetensors at $POLICY_PATH" >&2
  exit 1
fi

mkdir -p "$ROLLOUT_DIR"
echo "Eval 3 (Track D, M2 ArcFace) — interactive rollout"
echo "Using checkpoint:  $POLICY_DESC"
echo "Rollouts saved to: $ROLLOUT_DIR"
echo
echo "Prompt format:     'Place the coke on <celeb_name>.'"
echo "Trained celebs:    Taylor Swift, Barack Obama, Yann LeCun"
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
  RUN_NAME="typed_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo "→ Running: $PROMPT"
  echo "→ Saving to: $RUN_PATH"
  echo

  if [ -f "$AUTO_HOME" ]; then
    "$PYBIN" "$AUTO_HOME" capture "$HOME_POSE" || true
  fi

  # Run lerobot-record via our wrapper so the SmolVLM inference patch is
  # applied (transformers==4.55.0 boundaries-on-CPU bug). Same CLI as the
  # `lerobot-record` entry point.
  "$PYBIN" "$HERE/lerobot_record_with_patch.py" \
    --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id="local/eval3_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="$PROMPT" \
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
