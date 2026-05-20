#!/usr/bin/env bash
# run_cluster.sh — multi-GPU KLAL + LoRA co-training launch.
#
# One command for the whole run: autodetects every GPU on the node, extracts
# the bundled KLAL data, and launches `cotrain.py` under torchrun with manual
# data-parallel gradient all-reduce. 25k steps, checkpoint + HF push every 5k.
#
# Prereqs (see RUN_ON_CLUSTER.md for the full step-by-step):
#   1. repo cloned, branch dev/mahbod/kl-divergence checked out
#   2. the python env is active and has lerobot[smolvla,dataset,av-dep] installed
#   3. HF_TOKEN  exported — a token with WRITE access (checkpoints are pushed)
#   4. PUSH_REPO exported — the HF model repo to push checkpoints to
#
# Run from anywhere:
#   HF_TOKEN=hf_... PUSH_REPO=youruser/smolvla_klal_lora_25k \
#       bash eval_3/scripts/smolvla_cotrain/run_cluster.sh
#
# Single node, several GPUs. For a multi-node job, set the torchrun rendezvous
# args yourself instead of --standalone.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# ── HuggingFace token (needed to push checkpoints every 5k steps) ────────────
# Provide it EITHER via the HF_TOKEN env var, OR in a local token file
# ($HERE/.hf_token — gitignored, so the token is never committed). Do NOT
# hard-code a token into this script: it would be pushed to the git remote,
# and HuggingFace auto-revokes tokens it finds in public repos.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HERE/.hf_token}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    HF_TOKEN="$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE")"
    echo "==> HF_TOKEN loaded from $HF_TOKEN_FILE"
fi
: "${HF_TOKEN:?provide HF_TOKEN — export it, or write it into $HERE/.hf_token (gitignored)}"
: "${PUSH_REPO:?set PUSH_REPO — the HF model repo for checkpoints, e.g. youruser/smolvla_klal_lora_25k}"
export HF_TOKEN

# ── tunables (override via env) ──────────────────────────────────────────────
STEPS="${STEPS:-25000}"               # total training steps
SAVE_FREQ="${SAVE_FREQ:-5000}"        # checkpoint + HF push every N steps
BATCH_SIZE="${BATCH_SIZE:-200}"       # robot batch PER GPU  (~115 GB on a 141 GB card)
VL_BATCH_SIZE="${VL_BATCH_SIZE:-100}" # VL batch PER GPU     (keep at BATCH_SIZE/2)
VL_RATIO="${VL_RATIO:-10}"            # 10:1 robot:VL (ObjectVLA recipe)
LR="${LR:-5e-5}"
NUM_WORKERS="${NUM_WORKERS:-16}"      # dataloader workers per GPU process
ROBOT_DATASET="${ROBOT_DATASET:-HBOrtiz/so101_eval3_track3_v3_baseline}"
VL_MANIFEST="${VL_MANIFEST:-HBOrtiz/eval3_track3_vl_pairs}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_base}"
OUT_DIR="${OUT_DIR:-outputs/smolvla_klal_lora_25k}"
# KLAL + LoRA
KLAL_LAYERS="${KLAL_LAYERS:-10,12,14}"
KLAL_LAMBDA="${KLAL_LAMBDA:-1.0}"
KLAL_SIGMA="${KLAL_SIGMA:-1.0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# ── GPU autodetect ───────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found — this run needs CUDA GPUs" >&2
    exit 1
fi
NGPU="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [ "${NGPU:-0}" -lt 1 ]; then
    echo "[ERROR] no GPUs detected by nvidia-smi" >&2
    exit 1
fi
echo "==> $NGPU GPU(s) detected:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── extract the bundled KLAL data (once) ─────────────────────────────────────
M2_DIR="$HERE/m2_klal_data"
M2B="$M2_DIR/m2bundle"
if [ ! -d "$M2B" ]; then
    echo "==> extracting KLAL data bundle (m2_klal_data.tar.zst) ..."
    mkdir -p "$M2_DIR"
    tar --use-compress-program=unzstd -xf "$HERE/m2_klal_data.tar.zst" -C "$M2_DIR"
fi
for f in face_labels celeb_embeddings.json aug episode_mapping.json; do
    [ -e "$M2B/$f" ] || { echo "[ERROR] KLAL bundle missing $f" >&2; exit 1; }
done
echo "==> KLAL data ready at $M2B"

# ── preflight ────────────────────────────────────────────────────────────────
python -c "import lerobot.policies.smolvla.modeling_smolvla" 2>/dev/null \
    || { echo "[ERROR] cannot import lerobot SmolVLA — is the env active / installed? See RUN_ON_CLUSTER.md" >&2; exit 1; }
command -v torchrun >/dev/null 2>&1 \
    || { echo "[ERROR] torchrun not on PATH — is the python env active?" >&2; exit 1; }

# ── launch ───────────────────────────────────────────────────────────────────
echo
echo "==> launching $NGPU-GPU co-training"
echo "    steps      : $STEPS   (checkpoint + push every $SAVE_FREQ)"
echo "    batch/GPU  : robot $BATCH_SIZE / vl $VL_BATCH_SIZE   (effective robot batch: $((BATCH_SIZE * NGPU)))"
echo "    datasets   : $ROBOT_DATASET  +  $VL_MANIFEST"
echo "    push to    : $PUSH_REPO"
echo "    KLAL       : layers=$KLAL_LAYERS lambda=$KLAL_LAMBDA   LoRA r=$LORA_R"
echo "    (first run downloads ~15 GB of datasets — this is normal, be patient)"
echo

cd "$REPO_ROOT"
exec torchrun --standalone --nproc_per_node="$NGPU" \
    eval_3/scripts/smolvla_cotrain/cotrain.py \
    --robot_dataset="$ROBOT_DATASET" \
    --vl_manifest="$VL_MANIFEST" \
    --pretrained_path="$PRETRAINED" \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --batch_size="$BATCH_SIZE" \
    --vl_batch_size="$VL_BATCH_SIZE" \
    --vl_ratio="$VL_RATIO" \
    --lr="$LR" \
    --num_workers="$NUM_WORKERS" \
    --output_dir="$OUT_DIR" \
    --push_to_hub_repo="$PUSH_REPO" \
    --enable_lora --lora_r="$LORA_R" --lora_alpha="$LORA_ALPHA" \
    --enable_klal --klal_layers="$KLAL_LAYERS" \
    --klal_lambda="$KLAL_LAMBDA" --klal_sigma="$KLAL_SIGMA" \
    --face_labels_dir="$M2B/face_labels" \
    --celeb_manifest="$M2B/celeb_embeddings.json" \
    --aug_root="$M2B/aug" \
    --episode_mapping="$M2B/episode_mapping.json"
