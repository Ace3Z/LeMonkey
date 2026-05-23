#!/usr/bin/env bash
# The actual lerobot-train command for Eval 2 (compositional VLA).
# Run this directly OR via start_training.sh (which wraps it in systemd).
#
# Recipe diffs vs eval_1/v2:
#   - base policy: lerobot/smolvla_base (NOT a v2 warm-start - see eval_2/README.md)
#   - dataset    : merged 180-ep so101_eval2 (compositional prompts)
#   - no --rename_map (eval2 datasets already have observation.images.camera1)
#   - output_dir : ~/outputs/train/so101_smolvla_eval2/
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lemonkey
cd ~/LeMonkey

mkdir -p ~/outputs/train

LOG=~/outputs/train/so101_smolvla_eval2.log
echo "==> training log: $LOG"
echo "==> started at: $(date)"
echo

python -u "$(which lerobot-train)" \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --dataset.repo_id=local/so101_eval2 \
  --dataset.root=/home/shadeform/LeMonkey/datasets/eval2_merged \
  --dataset.image_transforms.enable=true \
  --batch_size=192 \
  --steps=25000 \
  --save_freq=5000 \
  --output_dir=/home/shadeform/outputs/train/so101_smolvla_eval2 \
  --job_name=so101_smolvla_eval2 \
  --policy.device=cuda \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "==> finished at: $(date)"
