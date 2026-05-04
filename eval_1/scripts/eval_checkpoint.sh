#!/usr/bin/env bash
# Structured per-checkpoint evaluation harness.
#
# Runs 30 rollouts (10 per color, shuffled) using a curated mix of:
#   - 5 IN-DISTRIBUTION prompts per color (verbatim from training data)
#   - 5 OUT-OF-DISTRIBUTION prompts per color (paraphrased, never seen)
#
# For each rollout the script:
#   1. Shows the target color, prompt type, and exact prompt
#   2. Waits for ENTER once you've positioned banana + bowls
#   3. Runs lerobot-record with that prompt
#   4. Asks Success? [y/n]
#
# Logs to eval_1/evals/ckpt<step>_<ts>.csv with prompt_type column so
# compare_evals.py can split in-dist vs OOD performance.
#
# Usage:
#   ./eval_checkpoint.sh                       # v1 base, ckpt 020000, random seed
#   ./eval_checkpoint.sh 015000                # v1 base, different ckpt
#   ./eval_checkpoint.sh 020000 42             # v1 base, fixed seed (reproducible shuffle)
#   MODEL=v2 ./eval_checkpoint.sh 025000       # v2 base (smolvla_eval1_v2)
#   MODEL=v2 ./eval_checkpoint.sh 020000 42    # v2 base at intermediate step, fixed seed
set -euo pipefail

CKPT="${1:-020000}"
SEED="${2:-$$}"
MODEL="${MODEL:-v1}"

case "$MODEL" in
  v1) MODEL_DIR="smolvla_eval1" ;;
  v2) MODEL_DIR="smolvla_eval1_v2" ;;
  *) echo "ERROR: MODEL must be v1 or v2 (got: $MODEL)" >&2; exit 1 ;;
esac

POLICY="/home/lemonkey/LeMonkey/eval_1/train/${MODEL_DIR}/checkpoints/${CKPT}/pretrained_model"
OUT_BASE="/home/lemonkey/LeMonkey/eval_1/evals"
ROLL_BASE="/home/lemonkey/LeMonkey/eval_1/rollouts"
PYBIN="/home/lemonkey/miniconda3/envs/lerobot/bin/python"

if [ ! -d "$POLICY" ]; then
  echo "ERROR: checkpoint not found: $POLICY" >&2; exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
SESS="${MODEL}_ckpt${CKPT}_${TS}"
CSV="$OUT_BASE/$SESS.csv"
mkdir -p "$OUT_BASE" "$ROLL_BASE"

# Generate the shuffled 30-prompt list (10/color, 5 trained + 5 untrained each)
PROMPT_LIST=$("$PYBIN" - "$SEED" <<'EOF'
import random, sys
random.seed(int(sys.argv[1]))

# 5 verbatim training-data phrasings (what the policy SAW during training)
trained = [
    "Put the banana in the {} colored bowl.",
    "Put the banana in the {} bowl",
    "Place the banana in the {} bowl",
    "pick the banana and put it in the {} bowl",
    "Place the banana in the {} colored bowl",
]

# 5 unseen but plausible paraphrases (out-of-distribution)
untrained = [
    "Move the banana to the {} bowl",
    "Drop the banana in the {} bowl",
    "Take the banana and put it in the {} bowl",
    "Put it into the {} bowl",
    "Banana goes in the {} bowl",
]

# Build all 30 prompts (3 colors × 10) and shuffle ALL of them so colors
# are interleaved — no 10-of-the-same-color streaks.
items = []
for color in ["blue", "red", "green"]:
    for t in trained:
        items.append((color, "trained",   t.format(color)))
    for t in untrained:
        items.append((color, "untrained", t.format(color)))
random.shuffle(items)

for color, kind, prompt in items:
    print(f"{color}\t{kind}\t{prompt}")
EOF
)

TOTAL=$(echo "$PROMPT_LIST" | wc -l)

echo "============================================================"
echo "  SmolVLA checkpoint evaluation"
echo "  model      : $MODEL  ($MODEL_DIR)"
echo "  checkpoint : $CKPT"
echo "  rollouts   : $TOTAL  (10/color, 5 trained + 5 untrained, shuffled)"
echo "  seed       : $SEED   (pass as 2nd arg to reproduce this order)"
echo "  log file   : $CSV"
echo "============================================================"
echo

# CSV header
echo "rollout,color_target,prompt_type,prompt,success,duration_s,duration_min,notes,run_path" > "$CSV"

i=1
while IFS=$'\t' read -r COLOR KIND PROMPT; do
  [ -z "$COLOR" ] && continue

  echo
  echo "╔══════════════════════════════════════════════════════════╗"
  printf "║ ROLLOUT %2d / %d                                          ║\n" "$i" "$TOTAL"
  echo "╠══════════════════════════════════════════════════════════╣"
  printf "║ Target color : %-42s║\n" "$COLOR"
  printf "║ Prompt type  : %-42s║\n" "$KIND"
  echo "║ Prompt       :                                           ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo
  echo "    \"$PROMPT\""
  echo

  read -r -p "Position banana + bowls, ENTER to RUN / 's' to skip / 'q' to quit: " A < /dev/tty
  case "$A" in
    q|Q) echo "aborted by user."; break ;;
    s|S)
      echo "$i,$COLOR,$KIND,\"${PROMPT//,/;}\",skipped,,,," >> "$CSV"
      i=$((i+1)); continue ;;
  esac

  RUN_NAME="${SESS}_r${i}_${COLOR}"
  RUN_PATH="$ROLL_BASE/$RUN_NAME"

  START_S=$(date +%s)
  set +e
  lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
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
  END_S=$(date +%s)
  DUR_S=$((END_S - START_S))
  DUR_MIN=$(awk "BEGIN {printf \"%.2f\", $DUR_S/60}")

  echo
  echo "  ⏱  rollout duration: ${DUR_S}s (${DUR_MIN} min)"
  if [ $RC -ne 0 ]; then
    echo "  ⚠️  rollout exited with code $RC (the run may have completed partially)"
  fi

  # Always ask y/n, validated, no matter what happened above
  echo
  echo "▶ Was the banana FULLY INSIDE the $COLOR bowl at the end?"
  RES=""
  while [ -z "$RES" ]; do
    read -r -p "  Success? [y/n]: " S < /dev/tty || { echo; echo "  (no terminal input; aborting)"; exit 130; }
    case "$S" in
      y|Y|yes|YES|Yes) RES=1 ;;
      n|N|no|NO|No)    RES=0 ;;
      *) echo "  please answer 'y' or 'n'" ;;
    esac
  done
  read -r -p "  Notes (ENTER to skip): " NOTE < /dev/tty
  if [ $RC -ne 0 ]; then
    NOTE="rc=$RC; $NOTE"
  fi

  echo "$i,$COLOR,$KIND,\"${PROMPT//,/;}\",$RES,$DUR_S,$DUR_MIN,\"${NOTE//,/;}\",$RUN_PATH" >> "$CSV"
  echo "  → recorded: $([ $RES -eq 1 ] && echo SUCCESS || echo FAIL)  duration=${DUR_S}s"
  i=$((i+1))
done <<< "$PROMPT_LIST"

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
ok = sum(1 for r in done if r['success'] == '1')
print(f"  Total: {ok}/{n}  ({100*ok/n if n else 0:.0f}%)")
print()
print("  By color:")
by_color = defaultdict(lambda: [0,0])
for r in done:
    by_color[r['color_target']][0] += int(r['success'])
    by_color[r['color_target']][1] += 1
for c, (s, t) in sorted(by_color.items()):
    print(f"    {c:6s} {s}/{t}  ({100*s/t if t else 0:.0f}%)")
print()
print("  By prompt type (in-dist vs OOD generalization):")
by_kind = defaultdict(lambda: [0,0])
for r in done:
    by_kind[r['prompt_type']][0] += int(r['success'])
    by_kind[r['prompt_type']][1] += 1
for k, (s, t) in sorted(by_kind.items()):
    print(f"    {k:9s} {s}/{t}  ({100*s/t if t else 0:.0f}%)")
print()
durs = [int(r['duration_s']) for r in done if r.get('duration_s','').isdigit()]
if durs:
    total_s = sum(durs)
    print(f"  Timing:")
    print(f"    rollouts measured: {len(durs)}")
    print(f"    avg per rollout :  {sum(durs)/len(durs):.1f} s ({sum(durs)/len(durs)/60:.2f} min)")
    print(f"    min / max       :  {min(durs)} s / {max(durs)} s")
    print(f"    total (rollouts):  {total_s} s ({total_s/60:.1f} min)")
print()
print(f"  CSV: {sys.argv[1]}")
PY
