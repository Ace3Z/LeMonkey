#!/usr/bin/env bash
# The actual lerobot-train command for Eval 3 (π0.5 / Coke-on-celebrity).
# Run this directly OR via start_training.sh (which wraps it in systemd).
#
# Recipe diffs vs eval_2 (SmolVLA → π0.5):
#   - base policy : lerobot/pi05_base (PaliGemma-3B + 300M flow-matching expert)
#   - dataset     : merged eval_3 dataset (~150 ep celebrity demos — TBD)
#   - batch_size  : 32 (down from 192; π0.5 ≈ 6 × SmolVLA VRAM. Drop to 16 if OOM.)
#   - steps       : 30000 (larger model + smaller batch → more steps)
#   - output_dir  : ~/outputs/train/pi05_eval3/
#   - defaults left untouched (relying on lerobot/pi05_base config):
#       train_expert_only=true, freeze_vision_encoder=true,
#       optimizer_lr=2.5e-5, num_inference_steps=10
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lemonkey
cd ~/LeMonkey

mkdir -p ~/outputs/train

LOG=~/outputs/train/pi05_eval3.log
echo "==> training log: $LOG"
echo "==> started at: $(date)"
echo

python -u "$(which lerobot-train)" \
  --policy.path=lerobot/pi05_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --dataset.repo_id=local/so101_eval3_all \
  --dataset.root=/home/shadeform/LeMonkey/datasets/eval3_merged \
  --dataset.image_transforms.enable=true \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=/home/shadeform/outputs/train/pi05_eval3 \
  --job_name=pi05_eval3 \
  --policy.device=cuda \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "==> finished at: $(date)"
