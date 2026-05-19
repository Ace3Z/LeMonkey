#!/usr/bin/env bash
# Track E (Pi0.5 + M2 + KLAL) — TOY dataset: 9.4k episodes, 3 IID celebs.
#
# Designed to run on H100 80GB. Maximizes throughput:
#   --batch_size=48  (H100 vs Track B's bs=24 for A100 conservatism)
#   --num_workers=24 (data-bound on AV1; 28 cores total, leave 4 for OS+main)
#   --save_freq=2000 (per user instruction)
#   --compile_model=False (REQUIRED — hooks break with compile)
#
# Disk budget: 97 GB on a-toy-pi05. We auto-push each new checkpoint to
# HF then delete local (see autopush_checkpoints.sh). Keeps disk bounded.
#
# Output repo: HBOrtiz/pi05_eval3_track_E_m2_mahbod
# Branch revisions saved: step-2000, step-4000, ..., step-30000, main

set -euo pipefail

cd ~/LeMonkey
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pi05 2>/dev/null || conda activate lemonkey 2>/dev/null || \
    echo "[warn] could not activate conda env"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# ─── M2 toolkit + 200-celeb-extended ArcFace centroids ────────────────
M2_TOOLKIT_DIR=${M2_TOOLKIT_DIR:-~/eval3_m2_toolkit}
if [ ! -d "$M2_TOOLKIT_DIR" ]; then
  echo "==> pulling M2 toolkit from HF"
  python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='HBOrtiz/eval3_m2_arcface_toolkit',
                  repo_type='dataset',
                  local_dir=os.path.expanduser('$M2_TOOLKIT_DIR'))
"
fi

# Build episode_index → variant mapping for the 3-celeb dataset.
if [ ! -f "$M2_TOOLKIT_DIR/episode_mapping_3celeb.json" ]; then
  echo "==> building episode_mapping_3celeb.json"
  python eval_3/aug/m2_episode_mapping.py \
    --base-root ~/LeMonkey/datasets/eval3 \
    --aug-root  ~/LeMonkey/datasets/eval3_track3_aug \
    --output    "$M2_TOOLKIT_DIR/episode_mapping_3celeb.json"
fi

export M2_FACE_LABELS_DIR="$M2_TOOLKIT_DIR/face_labels"
export M2_MANIFEST_PATH="$M2_TOOLKIT_DIR/celeb_embeddings.json"
export M2_AUG_ROOT=~/LeMonkey/datasets/eval3_track3_aug
export M2_EPISODE_MAPPING="$M2_TOOLKIT_DIR/episode_mapping_3celeb.json"
export M2_DATASET_REPO_ID="HBOrtiz/so101_eval3_track3_v3_baseline"
export M2_LAMBDA=${M2_LAMBDA:-0.2}
export M2_CAPTURE_LAYER=${M2_CAPTURE_LAYER:-10}
export M2_LOG_EVERY=${M2_LOG_EVERY:-100}
export KLAL_LAMBDA=${KLAL_LAMBDA:-1.0}
export KLAL_LAYERS=${KLAL_LAYERS:-"6,10,14,17"}
export KLAL_SIGMA_PATCHES=${KLAL_SIGMA_PATCHES:-1.5}

echo "==> Track E config (3-celeb TOY):"
echo "    M2_LAMBDA=$M2_LAMBDA  capture_layer=$M2_CAPTURE_LAYER"
echo "    KLAL_LAMBDA=$KLAL_LAMBDA  layers=$KLAL_LAYERS  sigma=$KLAL_SIGMA_PATCHES"

# ─── Output paths ──────────────────────────────────────────────────────
mkdir -p $HOME/outputs/train
LOG=$HOME/outputs/train/pi05_track_E_m2_3celeb.log
OUTPUT_DIR=$HOME/outputs/train/pi05_track_E_m2_3celeb
echo "==> log:    $LOG"
echo "==> output: $OUTPUT_DIR"
echo "==> started at: $(date)"

# ─── Launch ────────────────────────────────────────────────────────────
python -u eval_3/scripts/lerobot_train_with_m2_pi05.py \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.freeze_vision_encoder=True \
  --policy.train_expert_only=False \
  --policy.empty_cameras=3 \
  --policy.optimizer_lr=1e-5 \
  --policy.compile_model=False \
  --policy.device=cuda \
  --policy.gradient_checkpointing=True \
  --policy.push_to_hub=True \
  --policy.repo_id=HBOrtiz/pi05_eval3_track_E_m2_mahbod \
  --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \
  --dataset.video_backend=pyav \
  --dataset.revision=v3.0 \
  --batch_size=48 \
  --steps=30000 \
  --save_freq=2000 \
  --num_workers=24 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_track_E_m2_3celeb \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "==> finished at: $(date)"
echo "==> checkpoint repo: HBOrtiz/pi05_eval3_track_E_m2_mahbod"
