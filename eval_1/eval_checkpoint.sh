#!/usr/bin/env bash
# Structured per-checkpoint evaluation harness for SmolVLA on SO-101.
#
# Runs N rollouts (default 9 = 3 per color), asks you for success/fail after
# each, logs everything to a CSV under eval_1/evals/<ckpt>_<timestamp>.csv,
# and prints a summary at the end.
#
# Usage:
#   ./eval_checkpoint.sh                    # default: ckpt 020000, 9 rollouts
#   ./eval_checkpoint.sh 015000             # different ckpt, 9 rollouts
#   ./eval_checkpoint.sh 020000 6           # 6 rollouts (2 per color)
#   ./eval_checkpoint.sh 020000 9 voice     # use voice input for prompts
set -euo pipefail

CKPT="${1:-020000}"
N="${2:-9}"
INPUT_MODE="${3:-type}"   # "type" or "voice"

POLICY="/home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/${CKPT}/pretrained_model"
OUT_BASE="/home/lemonkey/LeMonkey/eval_1/evals"
ROLL_BASE="/home/lemonkey/LeMonkey/eval_1/rollouts"
HERE="$(dirname "$(readlink -f "$0")")"
PYBIN="/home/lemonkey/miniconda3/envs/lerobot/bin/python"
WAV="/tmp/voice_prompt.wav"
MIC="plughw:1,0"

if [ ! -d "$POLICY" ]; then
  echo "ERROR: checkpoint not found: $POLICY" >&2; exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
SESS="ckpt${CKPT}_${TS}"
CSV="$OUT_BASE/$SESS.csv"
mkdir -p "$OUT_BASE" "$ROLL_BASE"

echo "============================================================"
echo "  SmolVLA checkpoint evaluation"
echo "  checkpoint : $CKPT"
echo "  rollouts   : $N"
echo "  input mode : $INPUT_MODE"
echo "  log file   : $CSV"
echo "============================================================"
echo

# CSV header
echo "rollout,color_target,prompt,success,banana_pos_offset_cm,notes,run_path" > "$CSV"

# Cycle colors evenly across N rollouts
COLORS=(blue red green)
declare -a ROUND_COLORS
for ((i=0; i<N; i++)); do
  ROUND_COLORS[i]="${COLORS[$((i % 3))]}"
done

for ((r=1; r<=N; r++)); do
  COLOR="${ROUND_COLORS[$((r-1))]}"
  echo "------------------------------------------------------------"
  echo "Rollout $r / $N — TARGET COLOR: ${COLOR}"
  echo "------------------------------------------------------------"

  # Get prompt
  PROMPT=""
  while [ -z "$PROMPT" ]; do
    if [ "$INPUT_MODE" = "voice" ]; then
      read -r -p "ENTER to record / 't' to type / 's' to skip rollout: " A
      case "$A" in
        s|S) echo "skipped"; PROMPT="__SKIP__"; break ;;
        t|T) read -r -p "Type prompt (target=${COLOR}): " PROMPT; break ;;
        "")
          rm -f "$WAV"
          echo "🎙  Recording... ENTER to stop"
          arecord -q -D "$MIC" -f S16_LE -r 16000 -c 1 "$WAV" &
          RP=$!
          read -r _
          kill "$RP" 2>/dev/null || true
          wait "$RP" 2>/dev/null || true
          [ ! -s "$WAV" ] && { echo "(empty recording)"; continue; }
          PROMPT="$("$PYBIN" "$HERE/voice_transcribe.py" "$WAV" 2>/dev/null || true)"
          [ -z "$PROMPT" ] && { echo "(no speech detected)"; continue; }
          echo "📝 Heard: \"$PROMPT\""
          read -r -p "[y]es / [r]etry / [t]ype: " C
          case "$C" in y|Y|"") break ;; t|T) read -r -p "Type prompt: " PROMPT; break ;; *) PROMPT=""; continue ;; esac
          ;;
        *) ;;
      esac
    else
      read -r -p "Prompt (target=${COLOR}, suggestion: 'Put the banana in the ${COLOR} colored bowl.'): " PROMPT
    fi
  done

  if [ "$PROMPT" = "__SKIP__" ]; then
    echo "$r,$COLOR,SKIPPED,skipped,,,," >> "$CSV"
    continue
  fi

  read -r -p "Banana position (e.g. 'home', '+3cm x', '-2cm y'): " POS
  POS="${POS:-home}"

  RUN_NAME="${SESS}_r${r}_${COLOR}"
  RUN_PATH="$ROLL_BASE/$RUN_NAME"

  echo
  echo "→ Running. Make sure banana + bowls are in position. ENTER when ready..."
  read -r _

  set +e
  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id="local/eval_$RUN_NAME" \
    --dataset.root="$RUN_PATH" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="$PROMPT" \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --policy.path="$POLICY"
  RC=$?
  set -e
  if [ $RC -ne 0 ]; then
    echo "(rollout failed with exit $RC — recording as 0)"
    echo "$r,$COLOR,\"$PROMPT\",0,$POS,run-failed,$RUN_PATH" >> "$CSV"
    continue
  fi

  echo
  read -r -p "Success? [y/n] (banana fully INSIDE the ${COLOR} bowl?): " S
  read -r -p "Notes (optional, ENTER to skip): " NOTE
  case "$S" in y|Y) RES=1 ;; *) RES=0 ;; esac

  # Escape commas in prompt/notes
  PROMPT_ESC="${PROMPT//,/;}"; NOTE_ESC="${NOTE//,/;}"
  echo "$r,$COLOR,\"$PROMPT_ESC\",$RES,$POS,\"$NOTE_ESC\",$RUN_PATH" >> "$CSV"
  echo "  → recorded: $([ $RES -eq 1 ] && echo SUCCESS || echo FAIL)"
done

# Summary
echo
echo "============================================================"
echo "  RESULTS — $SESS"
echo "============================================================"
"$PYBIN" - "$CSV" << 'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
done = [r for r in rows if r['success'] not in ('','skipped')]
n = len(done)
ok = sum(1 for r in done if r['success']=='1')
print(f"  Total rollouts: {n}")
print(f"  Successes:      {ok} / {n}  ({100*ok/n if n else 0:.0f}%)")
by_color = defaultdict(lambda: [0,0])
for r in done:
    by_color[r['color_target']][0] += int(r['success'])
    by_color[r['color_target']][1] += 1
print(f"  By color:")
for c, (s, t) in sorted(by_color.items()):
    print(f"    {c:6s} {s}/{t}  ({100*s/t if t else 0:.0f}%)")
print(f"  CSV: {sys.argv[1]}")
PY
