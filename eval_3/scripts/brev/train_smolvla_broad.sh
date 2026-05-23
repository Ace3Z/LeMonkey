#!/usr/bin/env bash
# The actual lerobot-train command for Eval 3 (SmolVLA, image-as-prompt
# Coke-on-celebrity task). Run this directly OR via start_training.sh
# (which wraps it in systemd).
#
# Recipe rationale - see eval_3/aug/STRATEGY.md §6 and the 3-agent
# parallel research cross-check 2026-05-15 (Interleave-VLA 2505.02152,
# Pi0.5-KI 2505.23705, "Don't Blind Your VLA" 2510.25616, SmolVLA paper
# 2506.01844, canonical configuration_smolvla.py).
#
# Key diffs vs eval_2's recipe:
#   - dual image input         (camera1 = wrist webcam; camera2 = reference photo
#                                stream after --rename_map from
#                                observation.images.reference. empty_cameras=1
#                                fills the unused camera3 slot. The SmolVLA
#                                base policy hard-expects 3 cameras
#                                (camera1/2/3); the rename + empty_cameras=1
#                                give it exactly that.)
#   - add_image_special_tokens=true   (BOI/EOI separators between the
#                                cameras so the LM decoder can tell them apart;
#                                mirrors Interleave-VLA §A.1)
#   - train_expert_only=false  (CRITICAL - frozen VLM yields ~0% on
#                                face-matching, per all 3 papers above)
#   - freeze_vision_encoder=false  (SigLIP must adapt for face matching
#                                across reference photo ↔ printed portrait;
#                                Blind-VLA Table 2 +24% semantic / +12% vision)
#   - optimizer_lr=5e-5        (half the LeRobot default 1e-4; protects
#                                pretrained features when unfreezing both
#                                VLM and SigLIP - Interleave-VLA recipe)
#   - use_amp=true             (bf16 mixed precision. SmolVLAConfig has no
#                                `gradient_checkpointing` or `dtype` flag, so
#                                AMP is the available memory-saving knob.)
#   - batch_size=64            (Measured empirically 2026-05-15 on RTX PRO 6000
#                                97 GB: bs=64 uses 82.8/97 GB (85 %) with GPU
#                                util at 100 %. bs=80 OOMs. The unfrozen
#                                VLM+SigLIP activation memory is the bottleneck.
#                                Step rate ≈ 1.0 s/step at bs=64 -> 30k steps
#                                ≈ 8.3 h total.)
#   - steps=30000              (cosine endpoint at canonical scheduler_decay_steps)
#   - dataset.image_transforms.enable=true  (no flips; affine ±5° kept
#                                because prompts never reference position)
#   - PYTORCH_ALLOC_CONF=expandable_segments:True
#                              (mitigates allocator fragmentation that surfaces
#                                near the VRAM ceiling. PyTorch emitted this
#                                exact recommendation in the bs=96 OOM probe.)
#   - dataset.video_backend=pyav  (Replaces lerobot's default torchcodec.
#                                Empirically 2026-05-15: torchcodec leaks ~35 GB
#                                per DataLoader worker over ~30 min of training
#                                because decoder contexts aren't freed across
#                                episode boundaries when iterating 8390 unique
#                                mp4 files. dmesg showed single pt_data_worker
#                                hitting anon-rss=34.9 GB with num_workers=4 and
#                                17.9 GB with num_workers=8 - leak is per-worker.
#                                pyav is older + libav-based + well-tested.)
#   - num_workers=8            (Bumped back to 8 with pyav backend in place. The
#                                leak was per-worker in torchcodec, so with pyav
#                                no-leak expected -> 8 workers max-parallelism is
#                                fine. If pyav also leaks, drop stepwise: 8->4->2.)
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lemonkey
cd ~/LeMonkey

# Memory-allocator tweak - see header.
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p ~/outputs/train

LOG=~/outputs/train/so101_smolvla_eval3_broad.log
echo "==> training log: $LOG"
echo "==> started at: $(date)"
echo

python -u "$(which lerobot-train)" \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.add_image_special_tokens=true \
  --policy.empty_cameras=1 \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.use_amp=true \
  --policy.device=cuda \
  --dataset.repo_id=local/so101_eval3_broad \
  --dataset.root=/home/shadeform/LeMonkey/datasets/eval3_merged \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.reference": "observation.images.camera2"}' \
  --batch_size=64 \
  --steps=30000 \
  --save_freq=5000 \
  --num_workers=8 \
  --output_dir=/home/shadeform/outputs/train/so101_smolvla_eval3_broad \
  --job_name=so101_smolvla_eval3_broad \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "==> finished at: $(date)"
