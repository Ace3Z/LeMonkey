#!/bin/bash
# Guided teleop recording session for Eval 3.
#
# Research-validated plan (cross-checked against 6+ sources, see RESEARCH below):
#   3 celebs × 3 can positions (LEFT/MIDDLE/RIGHT) × 17 episodes = 153 demos
#   ~26 min recording per celeb × 3 = ~78 min total + setup/breaks ≈ 2 hr
#
# Math: 17 episodes/position × 3 positions = 51 demos/celeb (matches the
#   LeRobot SmolVLA 50-demo/task published floor).
#
# RESEARCH SOURCES (all 3 agents converged on this plan):
#   • LeRobot SmolVLA docs        — 50/task floor
#   • SmolVLA paper (2506.01844)  — 10×5 positions structure on SO-101
#   • Gando SO-101 reproduction   — 75 demos with consistent grasp > 81 mixed
#   • Pi0 LIBERO few-shot         — 40 trajectory floor for positive success
#   • Lin et al. ICLR 2025        — K=50 per env-object pair before plateau
#   • Pumacay 2023 (gen gap)      — position diversity > density (unanimous)
#   • Pumacay + LIBERO-Plus 2025  — camera viewpoint is #1 fragility axis
#
# NOTE on 3 vs 5 positions: SmolVLA tutorial uses 5; we use 3 because the user's
# workspace only physically supports 3 can starting positions (left / middle /
# right). We compensate by increasing episodes/position from 10 → 17 so total
# demos/celeb still hits the 50-demo floor. Effective position diversity is
# lower than the published baseline — flag this as a known risk.
#
# LOCKED (must NOT change during the session):
#   • Wrist camera pose (camera mounted on arm, looking AT the workspace)
#   • Lighting (close blinds, fixed bulb)
#   • Gripper grasp strategy (Gando: consistency beats variety here)
#   • Table surface
#   • Robot home pose
#   • Photo layout (Swift–Obama–LeCun left→right; printed PORTRAIT orientation)
#
# VARIED (diversity axes we explicitly want):
#   • Can starting position — LEFT / MIDDLE / RIGHT (3 positions per celeb)
#   • Target celebrity — alternates per phase (Swift → Obama → LeCun)
#   • (photo content) — varied later via the v9.3 inpainting pipeline

set -euo pipefail

# ────────────── CONFIG ──────────────
ROOT_DEFAULT="$HOME/LeMonkey/datasets/eval3"
EPISODES_PER_POSITION=17       # 17 × 3 positions = 51/celeb (matches LeRobot 50/task floor)
EPISODE_TIME_S=20
RESET_TIME_S=10
LAYOUT="SOL"                   # Swift–Obama–LeCun left→right (printed-photo physical order)
TARGETS=(swift obama lecun)
POSITIONS=(
  "LEFT    — leftmost third of the workspace, IN FRONT of the camera"
  "MIDDLE  — center of the workspace, IN FRONT of the camera"
  "RIGHT   — rightmost third of the workspace, IN FRONT of the camera"
)
POSITION_KEYS=("L" "M" "R")
TOTAL_EPISODES=$(( EPISODES_PER_POSITION * ${#POSITIONS[@]} * ${#TARGETS[@]} ))
EPISODES_PER_CELEB=$(( EPISODES_PER_POSITION * ${#POSITIONS[@]} ))

ROOT="${1:-$ROOT_DEFAULT}"
mkdir -p "$ROOT"

# ────────────── PRE-FLIGHT VALIDATION ──────────────
echo
echo "════════════════════════════════════════════════════════════"
echo "  EVAL 3 GUIDED TELEOP RECORDING"
echo "  Output: $ROOT"
echo "  Plan: 3 celebs × 3 positions × $EPISODES_PER_POSITION episodes = $TOTAL_EPISODES demos"
echo "        ($EPISODES_PER_CELEB demos per celeb — matches LeRobot 50/task floor)"
echo "════════════════════════════════════════════════════════════"
echo

errors=0
echo "[1/5] Pre-flight checks…"

# 1. Conda env
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "lemonkey" ]]; then
  echo "  ✗ Conda env not active. Run:"
  echo "      conda activate lemonkey"
  errors=$((errors + 1))
else
  echo "  ✓ conda env = $CONDA_DEFAULT_ENV"
fi

# 2. Arm ports (udev rules should be in place from earlier sessions)
for port in /dev/so101-leader /dev/so101-follower; do
  if [[ -e "$port" ]]; then
    echo "  ✓ $port present"
  else
    echo "  ✗ $port not found — check USB connections / udev rules"
    errors=$((errors + 1))
  fi
done

# 3. Camera
if [[ -e /dev/video0 ]]; then
  echo "  ✓ /dev/video0 present"
else
  echo "  ✗ /dev/video0 not found — check USB camera"
  errors=$((errors + 1))
fi

# 4. Calibration files
CAL_DIR="$HOME/.cache/huggingface/lerobot/calibration"
for arm in my_leader my_follower; do
  found=$(find "$CAL_DIR" -name "$arm*.json" 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    echo "  ✓ calibration found: $(basename "$found")"
  else
    echo "  ✗ calibration missing for $arm in $CAL_DIR"
    errors=$((errors + 1))
  fi
done

# 5. record_eval3_quick.py present
RECORDER="$HOME/LeMonkey/eval_3/scripts/record_eval3_quick.py"
if [[ -f "$RECORDER" ]]; then
  echo "  ✓ recorder script: $RECORDER"
else
  echo "  ✗ recorder script missing: $RECORDER"
  errors=$((errors + 1))
fi

# 6. Photo bank for sidecar reference-photo (1 per celeb, used only as label)
BANK="$HOME/LeMonkey/datasets/eval3_celebs/web"
for celeb in "${TARGETS[@]}"; do
  ref=$(ls -1 "$BANK/$celeb"/*.jpg 2>/dev/null | head -1)
  if [[ -n "$ref" ]]; then
    echo "  ✓ ref photo for $celeb: $(basename "$ref")"
  else
    echo "  ✗ no photo bank entry for $celeb under $BANK/$celeb"
    errors=$((errors + 1))
  fi
done

if (( errors > 0 )); then
  echo
  echo "❌ $errors pre-flight error(s). Fix them before recording. Exiting."
  exit 1
fi
echo "  ✓ all pre-flight checks passed"
echo

# ────────────── STATIC SETUP INSTRUCTIONS ──────────────
cat <<'BANNER'
[2/5] STATIC SETUP — do this ONCE, then DO NOT change for the whole session.
─────────────────────────────────────────────────────────────────────────────

  Camera is mounted on the ARM and looks IN FRONT of it (toward workspace).
  Operator stands wherever the leader arm reaches; physical layout below is
  shown FROM THE CAMERA'S VIEWPOINT — i.e. what the camera sees:

   ┌──────────────────────────────────────────────┐
   │              CAMERA VIEW                      │
   │                                              │
   │     ┌────────────────────────────────┐       │
   │     │   3 PRINTED PHOTOS (BACK ROW)  │       │
   │     │                                │       │
   │     │     [S]     [O]     [L]        │       │
   │     │      ↑       ↑       ↑         │       │
   │     │    SWIFT   OBAMA   LECUN        │      │
   │     │                                │       │
   │     │                                │       │
   │     │    CAN starts in ONE of:       │       │
   │     │                                │       │
   │     │   ●LEFT   ●MIDDLE   ●RIGHT     │       │
   │     │   (in front of the photos)     │       │
   │     │                                │       │
   │     └────────────────────────────────┘       │
   │                                              │
   │  ←camera-LEFT     camera-MIDDLE   camera-RIGHT→
   └──────────────────────────────────────────────┘

✓ 3 PRINTED PHOTOS, PORTRAIT orientation (taller than wide), all upright.
✓ Photo order LEFT→RIGHT in the camera view: SWIFT, OBAMA, LECUN.
✓ Photos ~10 cm apart, roughly co-linear, in the BACK row of the workspace.
✓ Camera mounted on the arm — DO NOT bump or re-aim it during the session.
✓ Close blinds; lighting locked.
✓ Coca-Cola can is the same can throughout.

3 CAN STARTING POSITIONS (rotate through them per phase):
BANNER
for i in 0 1 2; do
  echo "   ${POSITION_KEYS[$i]}: ${POSITIONS[$i]}"
done
cat <<'BANNER'

  All can positions are IN FRONT of the photo row (between photos and operator).

WHEN THINGS CHANGE during the session — the script will tell you:
   • PHOTOS                  → place ONCE at start, then NEVER move them
   • CAN POSITION            → script prompts you between every 17-episode batch
                                (3 batches per celeb = 3 position changes/celeb)
   • TARGET CELEB / PROMPT   → script prompts you between every 51-episode phase
                                (2 phase changes total: Swift→Obama, Obama→LeCun)
   • CAMERA / LIGHTING / GRIPPER / TABLE / ROBOT HOME → never (locked)

WHY THIS LAYOUT (cross-checked sources):
   • Position diversity > demo count   — Lin et al. ICLR 2025
   • Camera locked                     — Pumacay #1 fragility axis (-45.9 pp)
   • Lighting locked                   — Pumacay #4 axis, cheap to control
   • Photos portrait-oriented          — v9.4 auto-rotate handles mismatch
                                          but portrait input is cleaner
   • 3 positions × 17 eps = 51/celeb   — matches the LeRobot 50/task floor

Press ENTER when the static setup is done (or Ctrl-C to abort).
BANNER
read -r _

# ────────────── RECORDING PHASES ──────────────
echo
echo "[3/5] Recording 3 phases (one per target celebrity)…"
echo

start_time=$(date +%s)
total_episodes_done=0

phase_num=0
for celeb in "${TARGETS[@]}"; do
  phase_num=$((phase_num + 1))
  ref_photo=$(ls -1 "$BANK/$celeb"/*.jpg 2>/dev/null | head -1)

  echo
  echo "═══════════════════════════════════════════════════════════════"
  echo "  PHASE $phase_num / ${#TARGETS[@]} — change TARGET CELEB to: $celeb"
  echo "  Prompt sent to policy: \"Place the coke on $celeb.\""
  echo "  Will record 3 positions × $EPISODES_PER_POSITION episodes = $EPISODES_PER_CELEB demos"
  echo "═══════════════════════════════════════════════════════════════"
  if (( phase_num > 1 )); then
    echo "  NOTE: photos STAY in the same SOL layout from phase 1. Do not move them."
    echo "        Only the TARGET (= which photo you should place the can on) changes."
    read -p "  Press ENTER when ready to start phase $phase_num (target = $celeb)... " _
  fi

  for i in 0 1 2; do
    pkey="${POSITION_KEYS[$i]}"
    pdesc="${POSITIONS[$i]}"
    echo
    echo "───────────────────────────────────────"
    echo "  PHASE $phase_num — Position ${pkey} ($((i+1))/3): $pdesc"
    echo "  Will record $EPISODES_PER_POSITION episodes here."
    echo "───────────────────────────────────────"
    echo "  ACTION: place the COCA-COLA CAN at position $pkey on the table."
    echo "          (in front of the photos, $pdesc)"
    echo "          DO NOT touch the photos or the camera."
    echo "          Confirm the leader arm is in resting position."
    read -p "  Press ENTER to start the $EPISODES_PER_POSITION-episode batch (or 'q' to quit)... " ans
    if [[ "$ans" == "q" ]]; then echo "  quitting at user request"; exit 0; fi

    # Per-batch recorder loop with error recovery + manual delete option.
    # The Python recorder (record_eval3_quick.py) already supports an in-loop
    # 'd' key to drop the LAST completed episode between episodes. The bash
    # wrapper handles the OUTER cases: the recorder crashed mid-episode (e.g.
    # motor lost power, USB hiccup) — we count what's actually on disk and
    # offer retry/skip/delete-partial.
    while true; do
      n_before=$(ls -1d "$ROOT"/quick_${celeb}_${LAYOUT}_*/ 2>/dev/null | wc -l)
      set +e
      EVAL3_CAN_POSITION="$pkey" python "$RECORDER" \
        --target "$celeb" \
        --layout "$LAYOUT" \
        --reference-photo "$ref_photo" \
        --num-episodes "$EPISODES_PER_POSITION" \
        --episode-time-s "$EPISODE_TIME_S" \
        --reset-time-s "$RESET_TIME_S" \
        --root "$ROOT"
      rc=$?
      set -e
      n_after=$(ls -1d "$ROOT"/quick_${celeb}_${LAYOUT}_*/ 2>/dev/null | wc -l)
      recorded_this_batch=$(( n_after - n_before ))

      if (( rc == 0 )); then
        # Clean exit — the python recorder might have quit early via 'q'.
        # Trust whatever ended up on disk.
        total_episodes_done=$((total_episodes_done + recorded_this_batch))
        elapsed=$(( $(date +%s) - start_time ))
        echo "  ✓ Batch done. Recorded $recorded_this_batch episodes. Total so far: $total_episodes_done / $TOTAL_EPISODES (elapsed ${elapsed}s)"
        break
      fi

      # rc != 0 — recorder crashed. Inspect on-disk state and offer options.
      echo
      echo "  ⚠ Recorder exited rc=$rc."
      echo "    Episodes completed this batch (saved to disk): $recorded_this_batch"
      echo "    Episodes intended this batch: $EPISODES_PER_POSITION"

      # Identify the most-recent run dir — likely a partial if rc != 0.
      latest_dir=$(ls -1td "$ROOT"/quick_${celeb}_${LAYOUT}_*/ 2>/dev/null | head -1)
      if [[ -n "$latest_dir" ]]; then
        # A "partial" dir typically has empty data/ or videos/ subdirs.
        if [[ ! -s "$latest_dir/data/chunk-000/file-000.parquet" ]] 2>/dev/null \
           || [[ -z "$(ls -A "$latest_dir/videos" 2>/dev/null)" ]]; then
          partial_dir="$latest_dir"
        else
          partial_dir=""
        fi
      else
        partial_dir=""
      fi

      cat <<MENU

  ─── recovery menu (batch: $celeb / $pkey) ──────────────
    r)  RETRY the entire $EPISODES_PER_POSITION-episode batch
        (will append new episode dirs — already-recorded $recorded_this_batch are kept)
    d)  DELETE the latest partial directory and RETRY
        partial detected: ${partial_dir:-none}
    s)  SKIP this batch (move on; accept $recorded_this_batch < $EPISODES_PER_POSITION)
    p)  PAUSE — fix the issue (e.g. plug in arm power), then ENTER to retry
    q)  QUIT the recording session entirely
  ───────────────────────────────────────────────────────
MENU
      read -p "  Choice [r/d/s/p/q]: " choice
      case "${choice,,}" in
        r) echo "  → retrying batch…"; continue ;;
        d)
          if [[ -n "$partial_dir" ]]; then
            echo "  → deleting partial: $partial_dir"
            rm -rf "$partial_dir"
          else
            echo "  → no partial dir to delete"
          fi
          echo "  → retrying batch…"; continue ;;
        s)
          total_episodes_done=$((total_episodes_done + recorded_this_batch))
          echo "  → skipping. Total so far: $total_episodes_done / $TOTAL_EPISODES"
          break ;;
        p)
          read -p "  → paused. Fix the issue, then press ENTER to retry… " _
          continue ;;
        q) echo "  → quitting"; exit 0 ;;
        *) echo "  unknown choice — defaulting to pause"
          read -p "  → press ENTER to retry… " _
          continue ;;
      esac
    done
  done

  echo
  echo "  ✓ Phase $celeb complete ($EPISODES_PER_CELEB demos)"
done

# ────────────── POST-SESSION ──────────────
elapsed=$(( $(date +%s) - start_time ))
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ ALL DONE: $total_episodes_done demos recorded in ${elapsed}s"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "[4/5] Sanity check — listing recorded episodes…"
n_actual=$(ls -1d "$ROOT"/quick_*/ 2>/dev/null | wc -l)
echo "  Found $n_actual episode dirs under $ROOT"
if (( n_actual < TOTAL_EPISODES )); then
  echo "  ⚠ Less than $TOTAL_EPISODES — some episodes were skipped/deleted; that's OK if intentional."
fi

cat <<EOF

[5/5] Next steps (augmentation + verification):

  Run the v9.3 augmentation pipeline to materialize the augmented dataset:

    cd ~/LeMonkey
    python eval_3/aug/2_detect_static.py --root $ROOT --force
    python eval_3/aug/4_inpaint_video.py --root $ROOT --num-variants 5

  This will produce 5 augmented variants per real demo = up to 750 effective
  training videos. Per research:
    • cap aug at N=5/demo (Lin et al. plateau analysis)
    • augmentation is a MULTIPLIER on real data, never a substitute
    • watch for failure signals during eval:
        - succeeds on aug but fails on real photo → spectral artifact lock
        - OOD < TOY/IID by >30 pp → spurious face-region cue
        - seam-line at photo border affects grasp → boundary cue

  Then train SmolVLA-450M with Path A on the combined dataset (real + aug).

EOF
