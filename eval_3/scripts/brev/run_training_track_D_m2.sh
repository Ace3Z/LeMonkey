#!/usr/bin/env bash
# Track D (M2 ArcFace cosine distillation) launch script for Brev.
#
# Mirrors Hans's Track A spec from TODO.md (SmolVLA + HansOrtiz/smolvlm2_celeb_warm)
# with three M2-specific changes:
#   - --policy.train_expert_only=False  → unfreezes VLM layers (M2 needs gradients)
#   - apply_m2_partial_freeze inside the wrapper re-freezes layers 0-8 → only
#     layers 9-15 of SmolLM2 receive grads. Preserves Hans's warm prior in the
#     early layers while M2 supervision shapes the mid-late representations.
#   - calls eval_3/scripts/lerobot_train_with_m2.py instead of lerobot-train
#     directly; the launcher monkey-patches make_policy to wrap with M2.
#
# Pre-flight (assumes the Brev VM already has:
#   - the LeMonkey repo at ~/LeMonkey
#   - conda env `lemonkey` with lerobot installed
#   - datasets/eval3 base teleop dirs at ~/LeMonkey/datasets/eval3
#   - datasets/eval3_track3_aug aug variants at ~/LeMonkey/datasets/eval3_track3_aug
#   - HF_TOKEN exported (or in .env at ~/LeMonkey/.env))
#
# Expected wall time: 5-8 h on a single A100 80GB (30k steps, bs=64; M2 backward
# pass roughly 2× Track A's cost due to ~10× more trainable params).
#
# Output checkpoint: HBOrtiz/smolvla_eval3_track_D_m2_mahbod
#   (distinct from Hans's HBOrtiz/smolvla_eval3_track_A so we never clobber it).

set -euo pipefail

cd ~/LeMonkey

# Activate conda env (assumes `lemonkey` is provisioned).
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate lemonkey 2>/dev/null || echo "[warn] could not activate conda; assuming env already active"

# Memory-allocator tweak — mirrors run_training.sh.
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── M2-specific config ────────────────────────────────────────────────────
# Toolkit data: pulled from HF (HBOrtiz/eval3_m2_arcface_toolkit) on first run.
M2_TOOLKIT_DIR=${M2_TOOLKIT_DIR:-~/eval3_m2_toolkit}
if [ ! -d "$M2_TOOLKIT_DIR" ]; then
  echo "==> pulling M2 toolkit data from HF"
  python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='HBOrtiz/eval3_m2_arcface_toolkit',
    repo_type='dataset',
    local_dir=os.path.expanduser('$M2_TOOLKIT_DIR'),
)
"
fi

# Build episode_index → variant mapping (depends on local dataset dirs).
if [ ! -f "$M2_TOOLKIT_DIR/episode_mapping.json" ]; then
  echo "==> building episode_mapping.json"
  python eval_3/aug/m2_episode_mapping.py \
    --base-root ~/LeMonkey/datasets/eval3 \
    --aug-root ~/LeMonkey/datasets/eval3_track3_aug \
    --output "$M2_TOOLKIT_DIR/episode_mapping.json"
fi

export M2_FACE_LABELS_DIR="$M2_TOOLKIT_DIR/face_labels"
export M2_MANIFEST_PATH="$M2_TOOLKIT_DIR/celeb_embeddings.json"
export M2_AUG_ROOT=~/LeMonkey/datasets/eval3_track3_aug
export M2_EPISODE_MAPPING="$M2_TOOLKIT_DIR/episode_mapping.json"
export M2_LAMBDA=${M2_LAMBDA:-0.2}
export M2_CAPTURE_LAYER=${M2_CAPTURE_LAYER:-9}
export M2_LOG_EVERY=${M2_LOG_EVERY:-100}

echo "==> M2 config:"
echo "    face_labels=$M2_FACE_LABELS_DIR"
echo "    manifest=$M2_MANIFEST_PATH"
echo "    aug_root=$M2_AUG_ROOT"
echo "    episode_mapping=$M2_EPISODE_MAPPING"
echo "    lambda=$M2_LAMBDA  capture_layer=$M2_CAPTURE_LAYER"

# ── Training launch ───────────────────────────────────────────────────────
mkdir -p ~/outputs/train
LOG=~/outputs/train/smolvla_track_D_m2.log
echo "==> training log: $LOG"
echo "==> started at: $(date)"

python -u eval_3/scripts/lerobot_train_with_m2.py \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.vlm_model_name=HansOrtiz/smolvlm2_celeb_warm \
  --policy.freeze_vision_encoder=True \
  --policy.train_expert_only=False \
  --policy.empty_cameras=1 \
  --policy.optimizer_lr=5e-5 \
  --policy.compile_model=False \
  --policy.device=cuda \
  --policy.push_to_hub=True \
  --policy.repo_id=HBOrtiz/smolvla_eval3_track_D_m2_mahbod \
  --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --steps=30000 \
  --save_freq=5000 \
  --num_workers=8 \
  --output_dir=~/outputs/train/smolvla_track_D_m2 \
  --job_name=smolvla_track_D_m2 \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "==> finished at: $(date)"
echo "==> checkpoint pushed to: HBOrtiz/smolvla_eval3_track_D_m2_mahbod"
