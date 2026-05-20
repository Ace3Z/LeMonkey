#!/usr/bin/env bash
# Track E (Pi0.5 + M2 + KLAL) — 200-celeb dataset variant.
#
# Same recipe as run_training_track_E_pi05_3celeb.sh but reads
# HBOrtiz/so101_eval3_aug_v3_200celebs (10k episodes, 200 celebs).
# Aimed at testing whether the broader-celeb diversity forces stronger
# name→face binding than the 3-celeb run.
#
# Requires:
#   - The 200-celeb ArcFace centroid manifest must cover all 200 celebs
#     (run `extend_arcface_to_200celebs.py` first if not).
#   - The episode_mapping for the 200-celeb dataset must be built (the
#     `m2_episode_mapping.py` script + the local 200-celeb aug variants).
#   - The 200-celeb dataset's augmentation.json files must be available
#     at $M2_AUG_ROOT_200CELEB (either downloaded or scp'd from local).
#
# Output repo: HBOrtiz/pi05_eval3_track_E_m2_200celeb_mahbod

set -euo pipefail

cd ~/LeMonkey
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pi05 2>/dev/null || conda activate lemonkey 2>/dev/null || \
    echo "[warn] could not activate conda env"

export PYTORCH_ALLOC_CONF=expandable_segments:True

M2_TOOLKIT_DIR=${M2_TOOLKIT_DIR:-~/eval3_m2_toolkit}

# 200-celeb-specific paths.
M2_AUG_ROOT_200CELEB=${M2_AUG_ROOT_200CELEB:-~/LeMonkey/datasets/eval3_aug_v3_200celebs}
EPISODE_MAPPING_200CELEB=${EPISODE_MAPPING_200CELEB:-"$M2_TOOLKIT_DIR/episode_mapping_200celeb.json"}
MANIFEST_200CELEB=${MANIFEST_200CELEB:-"$M2_TOOLKIT_DIR/celeb_embeddings_200.json"}

# Build episode_index → variant mapping for the 200-celeb dataset.
if [ ! -f "$EPISODE_MAPPING_200CELEB" ]; then
  echo "==> building episode_mapping_200celeb.json"
  python eval_3/aug/m2_episode_mapping.py \
    --base-root ~/LeMonkey/datasets/eval3 \
    --aug-root  "$M2_AUG_ROOT_200CELEB" \
    --output    "$EPISODE_MAPPING_200CELEB"
fi

# Verify 200-celeb manifest exists.
if [ ! -f "$MANIFEST_200CELEB" ]; then
  echo "[error] $MANIFEST_200CELEB not found"
  echo "       Run: python eval_3/aug/extend_arcface_to_200celebs.py"
  echo "         --celeb-bank ~/Downloads/eval3_celebs/track3_bank"
  echo "         --output     $MANIFEST_200CELEB"
  exit 2
fi

export M2_FACE_LABELS_DIR="$M2_TOOLKIT_DIR/face_labels"
export M2_MANIFEST_PATH="$MANIFEST_200CELEB"
export M2_AUG_ROOT="$M2_AUG_ROOT_200CELEB"
export M2_EPISODE_MAPPING="$EPISODE_MAPPING_200CELEB"
export M2_DATASET_REPO_ID="HBOrtiz/so101_eval3_aug_v3_200celebs"
# M2_LAMBDA=1.0 (not BlindVLA's 0.2): Pi0.5 trains at lr=1e-5, 5x below the
# 5e-5 SmolVLA used where M2 reached mean_cos 0.88 at lambda=0.2; 5x lambda
# restores M2's effective step. Verified 2026-05-20: mean_cos +0.01 -> +0.56
# over 800 steps (dead flat at lambda=0.2).
export M2_LAMBDA=${M2_LAMBDA:-1.0}
export M2_CAPTURE_LAYER=${M2_CAPTURE_LAYER:-10}
export M2_LOG_EVERY=${M2_LOG_EVERY:-100}
export KLAL_LAMBDA=${KLAL_LAMBDA:-1.0}
# KLAL_LAYERS drops layer 6: the partial-freeze freezes LM layers 0-9, so
# KLAL@6 trained zero params (dead weight diluting the loss). 10/14/17 are
# the supervised AND trainable layers.
export KLAL_LAYERS=${KLAL_LAYERS:-"10,14,17"}
export KLAL_SIGMA_PATCHES=${KLAL_SIGMA_PATCHES:-1.5}

echo "==> Track E config (200-celeb):"
echo "    M2_LAMBDA=$M2_LAMBDA  capture_layer=$M2_CAPTURE_LAYER"
echo "    KLAL_LAMBDA=$KLAL_LAMBDA  layers=$KLAL_LAYERS  sigma=$KLAL_SIGMA_PATCHES"
echo "    DATASET=$M2_DATASET_REPO_ID"

mkdir -p $HOME/outputs/train
LOG=$HOME/outputs/train/pi05_track_E_m2_200celeb.log
OUTPUT_DIR=$HOME/outputs/train/pi05_track_E_m2_200celeb
echo "==> log:    $LOG"
echo "==> output: $OUTPUT_DIR"
echo "==> started at: $(date)"

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
  --policy.repo_id=HBOrtiz/pi05_eval3_track_E_m2_200celeb_mahbod \
  --dataset.repo_id="$M2_DATASET_REPO_ID" \
  --dataset.video_backend=pyav \
  --batch_size=48 \
  --steps=30000 \
  --save_freq=2000 \
  --num_workers=24 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi05_track_E_m2_200celeb \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "==> finished at: $(date)"
echo "==> checkpoint repo: HBOrtiz/pi05_eval3_track_E_m2_200celeb_mahbod"
