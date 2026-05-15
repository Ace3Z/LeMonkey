#!/usr/bin/env bash
# The actual lerobot-train command for Eval 3 (SmolVLA, image-as-prompt
# Coke-on-celebrity task). Run this directly OR via start_training.sh
# (which wraps it in systemd).
#
# Recipe rationale — see eval_3/aug/STRATEGY_v3.md §6 and the 3-agent
# parallel research cross-check 2026-05-15 (Interleave-VLA 2505.02152,
# Pi0.5-KI 2505.23705, "Don't Blind Your VLA" 2510.25616, SmolVLA paper
# 2506.01844, canonical configuration_smolvla.py).
#
# Key diffs vs eval_2's recipe:
#   - dual image input        (observation.images.{camera1,reference})
#   - empty_cameras=0          (we supply exactly the 2 cameras we declare,
#                                vs eval_2's 1 supplied / 3 expected → 2)
#   - add_image_special_tokens=true   (BOI/EOI separators between the
#                                cameras so the LM decoder can tell them apart;
#                                mirrors Interleave-VLA §A.1)
#   - train_expert_only=false  (CRITICAL — frozen VLM yields ~0% on
#                                face-matching, per all 3 papers above)
#   - freeze_vision_encoder=false  (SigLIP must adapt for face matching
#                                across reference photo ↔ printed portrait;
#                                Blind-VLA Table 2 +24% semantic / +12% vision)
#   - optimizer_lr=5e-5        (half the LeRobot default 1e-4; protects
#                                pretrained features when unfreezing both
#                                VLM and SigLIP — Interleave-VLA recipe)
#   - batch_size=96            (start safe; H100 80GB has headroom for 128
#                                with gradient checkpointing + bfloat16, but
#                                unfrozen SigLIP eats activation memory)
#   - steps=30000              (cosine endpoint at canonical scheduler_decay_steps)
#   - dataset.image_transforms.enable=true  (no flips; affine ±5° kept
#                                because prompts never reference position)
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lemonkey
cd ~/LeMonkey

mkdir -p ~/outputs/train

LOG=~/outputs/train/smolvla_eval3.log
echo "==> training log: $LOG"
echo "==> started at: $(date)"
echo

python -u "$(which lerobot-train)" \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.add_image_special_tokens=true \
  --policy.empty_cameras=0 \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --dataset.repo_id=local/so101_eval3_all \
  --dataset.root=/home/shadeform/LeMonkey/datasets/eval3_merged \
  --dataset.image_transforms.enable=true \
  --batch_size=96 \
  --steps=30000 \
  --save_freq=5000 \
  --num_workers=8 \
  --output_dir=/home/shadeform/outputs/train/smolvla_eval3 \
  --job_name=smolvla_eval3 \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "==> finished at: $(date)"
