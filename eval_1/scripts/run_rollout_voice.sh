#!/usr/bin/env bash
# Voice-driven SmolVLA rollout runner.
#
# Flow per rollout:
#   1. ENTER -> start recording
#   2. ENTER -> stop recording
#   3. Whisper transcribes
#   4. Confirm: y=use, r=re-record, t=type manually, q=quit
#   5. Run lerobot-record with the confirmed prompt
#
# Usage:
#   ./run_rollout_voice.sh                # default checkpoint v2 025000
#   ./run_rollout_voice.sh 020000         # different v2 checkpoint
# Press right-arrow during a rollout to end it early.
set -euo pipefail

CKPT_STEP="${1:-025000}"
POLICY="/home/lemonkey/LeMonkey/eval_1/train/so101_smolvla_eval1/checkpoints/${CKPT_STEP}/pretrained_model"
ROLLOUT_DIR="/home/lemonkey/LeMonkey/eval_1/rollouts"
HERE="$(dirname "$(readlink -f "$0")")"
TRANSCRIBE="$HERE/voice_transcribe.py"
PYBIN="/home/lemonkey/miniconda3/envs/lemonkey/bin/python"
WAV="/tmp/voice_prompt.wav"
MIC="plughw:1,0"
HOME_POSE="/tmp/run_rollout_home.json"
HOME_DRIVE_S=2.0

if [ ! -d "$POLICY" ]; then
  echo "ERROR: checkpoint not found: $POLICY" >&2
  exit 1
fi
if [ ! -x "$TRANSCRIBE" ]; then
  chmod +x "$TRANSCRIBE" 2>/dev/null || true
fi

mkdir -p "$ROLLOUT_DIR"
echo "Checkpoint:  $POLICY"
echo "Rollouts to: $ROLLOUT_DIR"
echo "Microphone:  $MIC"
echo

i=1
while true; do
  echo "=================================================="
  echo "Rollout #$i"

  PROMPT=""
  while [ -z "$PROMPT" ]; do
    read -r -p "ENTER to record / 't' to type / 'q' to quit: " ACTION
    case "$ACTION" in
      q|Q|quit|exit) echo "Bye."; exit 0 ;;
      t|T) read -r -p "Type prompt: " PROMPT; break ;;
      "") ;;  # fall through to record
      *) continue ;;
    esac

    rm -f "$WAV"
    echo "🎙  Recording... press ENTER to stop"
    arecord -q -D "$MIC" -f S16_LE -r 16000 -c 1 "$WAV" &
    REC_PID=$!
    read -r _DUMMY
    kill "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true

    if [ ! -s "$WAV" ]; then
      echo "(empty recording, try again)"
      continue
    fi

    echo "🤖 Transcribing..."
    PROMPT="$("$PYBIN" "$TRANSCRIBE" "$WAV" 2>/tmp/voice_transcribe.err || true)"
    if [ -z "$PROMPT" ]; then
      echo "(no speech detected; try again)"
      cat /tmp/voice_transcribe.err >&2 2>/dev/null || true
      continue
    fi

    echo
    echo "📝 Heard: \"$PROMPT\""
    read -r -p "[y]es use this / [r]etry / [t]ype manually / [q]uit: " CHOICE
    case "$CHOICE" in
      y|Y|"") break ;;
      r|R) PROMPT=""; continue ;;
      t|T) read -r -p "Type prompt: " PROMPT; break ;;
      q|Q) exit 0 ;;
      *) PROMPT=""; continue ;;
    esac
  done

  TS=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="run_${i}_${TS}"
  RUN_PATH="$ROLLOUT_DIR/$RUN_NAME"

  echo
  echo "→ Prompt:    \"$PROMPT\""
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
