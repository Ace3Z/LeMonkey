#!/usr/bin/env bash
# SmolVLA + VL co-train launch — single-node AWS / Brev / generic CUDA box.
#
# Override any value via env vars before invoking, e.g.:
#   VL_RATIO=5 STEPS=20000 PUSH_REPO=HBOrtiz/my_run bash launch.sh
#
# PRE-FLIGHT (must be true):
#   1. HF_TOKEN exported (read+write) — needed for HBOrtiz/* repos
#   2. conda env with lerobot, transformers, peft, torch (CUDA), pandas, PIL installed
#   3. At least one CUDA GPU visible (`nvidia-smi`)
#   4. The local lerobot at $(python -c 'import lerobot; print(lerobot.__path__)')
#      includes the SmolVLAPolicy class (smolvla policy registered)
#
# SMOKE TEST FIRST: run with STEPS=200 BATCH_SIZE=4 to confirm both losses fire
# and there's no OOM. See README.md gates.

set -euo pipefail

# ---- Defaults (override via env) ----------------------------------------------

ROBOT_DATASET="${ROBOT_DATASET:-HBOrtiz/so101_eval3_track3_v3_baseline}"
VL_MANIFEST="${VL_MANIFEST:-HBOrtiz/eval3_objectvla_vl_pairs}"
VL_IMAGE_ROOT="${VL_IMAGE_ROOT:-}"          # leave empty to auto-download
PRETRAINED="${PRETRAINED:-lerobot/smolvla_base}"   # or HansOrtiz/smolvlm2_celeb_warm
VLM_OVERRIDE="${VLM_OVERRIDE:-}"            # set to a warm VLM repo to swap inner VLM

STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VL_BATCH_SIZE="${VL_BATCH_SIZE:-8}"
VL_RATIO="${VL_RATIO:-10}"
LR="${LR:-5e-5}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"

OUT_DIR="${OUT_DIR:-outputs/smolvla_cotrain_${VL_RATIO}to1}"
PUSH_REPO="${PUSH_REPO:-}"                  # leave empty to skip HF push

# ---- Offline-mode detection ---------------------------------------------------
# snapshot_download enumerates every file in a repo via the HF API on every
# call — even when the data is already cached — triggering rate limits and
# multi-minute stalls.  Setting HF_HUB_OFFLINE=1 makes huggingface_hub skip
# all network I/O and serve from local cache only.  We set it automatically
# once both datasets have been pre-downloaded.

_LEROBOT_HUB="${HOME}/.cache/huggingface/lerobot/hub"
_HF_HUB="${HOME}/.cache/huggingface/hub"

# Robot dataset: lerobot uses its own cache dir
_ROBOT_SLUG="datasets--$(echo "$ROBOT_DATASET" | sed 's|/|--|g')"
_ROBOT_SNAP_DIR="$_LEROBOT_HUB/$_ROBOT_SLUG/snapshots"
_ROBOT_SNAP=$(ls -1 "$_ROBOT_SNAP_DIR" 2>/dev/null | head -1)
_ROBOT_CACHED=false
if [ -n "$_ROBOT_SNAP" ] && [ -d "$_ROBOT_SNAP_DIR/$_ROBOT_SNAP/videos" ]; then
    _ROBOT_CACHED=true
fi

# VL dataset: standard HF hub cache
_VL_SLUG="datasets--$(echo "${VL_MANIFEST%%/*}--$(echo "$VL_MANIFEST" | cut -d/ -f2)" | sed 's|/|--|g')"
_VL_SLUG="datasets--$(echo "$VL_MANIFEST" | sed 's|/|--|g')"
_VL_SNAP_DIR="$_HF_HUB/$_VL_SLUG/snapshots"
_VL_SNAP=$(ls -1 "$_VL_SNAP_DIR" 2>/dev/null | head -1)
_VL_CACHED=false
if [ -n "$_VL_SNAP" ] && [ -f "$_VL_SNAP_DIR/$_VL_SNAP/manifest.parquet" ]; then
    _VL_CACHED=true
    # Pass local paths directly — VLPairsDataset accepts a local parquet path
    # and image root, bypassing any HF download code entirely.
    if [ -z "${VL_MANIFEST##*/*}" ] && [ -z "${VL_MANIFEST##HBOrtiz*}" ]; then
        VL_MANIFEST="$_VL_SNAP_DIR/$_VL_SNAP/manifest.parquet"
        echo "==> VL manifest cached locally: $VL_MANIFEST"
    fi
    if [ -z "${VL_IMAGE_ROOT:-}" ] && [ -d "$_VL_SNAP_DIR/$_VL_SNAP/images" ]; then
        VL_IMAGE_ROOT="$_VL_SNAP_DIR/$_VL_SNAP/images"
        echo "==> VL images cached locally: $VL_IMAGE_ROOT"
    fi
fi

if [ "$_ROBOT_CACHED" = true ] && [ "$_VL_CACHED" = true ]; then
    export HF_HUB_OFFLINE=1
    echo "==> Both datasets cached — HF_HUB_OFFLINE=1 (zero network I/O)"
elif [ "$_ROBOT_CACHED" = false ]; then
    echo "[WARN] Robot dataset not cached. Run predl_vl.sh equivalent for robot dataset first." >&2
fi

# Episode coverage is handled inside cotrain.py (_episodes_with_complete_files):
# it filters to episodes whose data+video files are all present in the cache,
# which is correct even when the partial download has non-contiguous gaps (a
# file-count heuristic here cannot — it once capped to 938 and then crashed on
# the missing reference/file-934.mp4 inside that range). Set ROBOT_MAX_EPISODES
# explicitly only to deliberately shrink the run (e.g. a fast smoke test).

# ---- Pre-flight ---------------------------------------------------------------

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[WARN] HF_TOKEN not set — HF download/push will fail if any repo is private" >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found — running on CPU will be unusably slow" >&2
else
    echo "==> GPU(s) available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# Count GPUs via torch so the value honours CUDA_VISIBLE_DEVICES (nvidia-smi
# does not) — otherwise torchrun would spawn ranks for GPUs torch can't see.
_NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

# How many GPUs to train on. Auto = all visible; override with NUM_GPUS.
# >1 launches cotrain.py under torchrun (DDP); 1 runs a plain single process.
NUM_GPUS="${NUM_GPUS:-$_NGPU}"
[ "${NUM_GPUS:-1}" -lt 1 ] && NUM_GPUS=1

python -c "import lerobot.policies.smolvla.modeling_smolvla" \
    || { echo "[ERROR] cannot import lerobot.policies.smolvla.modeling_smolvla — check env" >&2; exit 1; }

# ---- Launch -------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$REPO_ROOT/eval_3/scripts/smolvla_cotrain/cotrain.py"

echo "==> SmolVLA cotrain launching"
echo "    robot      : $ROBOT_DATASET"
echo "    vl         : $VL_MANIFEST"
echo "    pretrained : $PRETRAINED"
[ -n "$VLM_OVERRIDE" ] && echo "    vlm override: $VLM_OVERRIDE"
echo "    steps      : $STEPS"
echo "    bs / vl_bs : $BATCH_SIZE / $VL_BATCH_SIZE (global, split across $NUM_GPUS GPU(s))"
echo "    vl_ratio   : $VL_RATIO (=> VL batch every $((VL_RATIO + 1))-th step)"
echo "    lr         : $LR"
echo "    gpus       : $NUM_GPUS $([ "$NUM_GPUS" -gt 1 ] && echo '(DDP via torchrun)' || echo '(single process)')"
echo "    output     : $OUT_DIR"
echo "    push       : ${PUSH_REPO:-(skip)}"
echo

# torchrun for multi-GPU (DDP), plain python for single-GPU. PYTHONUNBUFFERED
# keeps torchrun child stdout line-buffered for live log tailing.
export PYTHONUNBUFFERED=1
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER=( torchrun --standalone --nproc_per_node="$NUM_GPUS" )
else
    LAUNCHER=( python -u )
fi

CMD=( "${LAUNCHER[@]}" "$SCRIPT"
      --robot_dataset="$ROBOT_DATASET"
      --vl_manifest="$VL_MANIFEST"
      --pretrained_path="$PRETRAINED"
      --steps="$STEPS"
      --batch_size="$BATCH_SIZE"
      --vl_batch_size="$VL_BATCH_SIZE"
      --vl_ratio="$VL_RATIO"
      --lr="$LR"
      --save_freq="$SAVE_FREQ"
      --log_every="$LOG_EVERY"
      --num_workers="$NUM_WORKERS"
      --seed="$SEED"
      --dtype="$DTYPE"
      --output_dir="$OUT_DIR" )

[ -n "${ROBOT_MAX_EPISODES:-}" ] && CMD+=( --robot_max_episodes="$ROBOT_MAX_EPISODES" )
[ -n "$VL_IMAGE_ROOT" ]         && CMD+=( --vl_image_root="$VL_IMAGE_ROOT" )
[ -n "$VLM_OVERRIDE"  ]         && CMD+=( --vlm_model_name="$VLM_OVERRIDE" )
[ -n "$PUSH_REPO"     ]         && CMD+=( --push_to_hub_repo="$PUSH_REPO" )

"${CMD[@]}"
