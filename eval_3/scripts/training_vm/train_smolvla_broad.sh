#!/usr/bin/env bash
# SmolVLA + VL co-train on the BROAD (192-celebrity) dataset.
#
# Same trainer as the IID cotrain recipe (`train_smolvla_cotrain.py` via
# `scripts/smolvla_cotrain/launch_multi_gpu.sh`), pointed at the broad
# robot dataset + broad VL grounding pairs, at ObjectVLA's default 10:1
# robot:VL ratio (vs the IID cotrain's 5:1) to reflect the harder
# OOD-celeb generalisation task.
#
# This script is a thin shim over the shared multi-GPU launcher: it just
# overrides the dataset / VL-manifest / VL-ratio / output-dir defaults to
# the broad recipe and exec's launch_multi_gpu.sh. Every other tunable
# (BATCH_SIZE, STEPS, LR, KLAL_LAYERS, LORA_R, ...) inherits the cotrain
# launcher's default unless explicitly overridden in the environment.
#
# Usage:
#   HF_TOKEN=hf_... PUSH_REPO=youruser/so101_smolvla_eval3_broad \
#       bash eval_3/scripts/training_vm/train_smolvla_broad.sh
#
# Or via the shared systemd-wrap launcher (survives SSH disconnect):
#   UNIT=lerobot-train-eval3-broad \
#   DESCRIPTION="LeRobot SmolVLA Eval 3 broad cotrain (192 celebs, 10:1)" \
#   TRAIN_SCRIPT=$REPO_ROOT/eval_3/scripts/training_vm/train_smolvla_broad.sh \
#   LOG_FILE=$HOME/outputs/train/so101_smolvla_eval3_broad.log \
#   LIMIT_NOFILE=524288 \
#       bash $REPO_ROOT/scripts/training_vm/start_training.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# Broad-specific defaults (override the cotrain launcher's IID defaults).
# Anything not set here inherits from launch_multi_gpu.sh.

# Robot + VL data pair for the 192-celebrity broad recipe.
export ROBOT_DATASET="${ROBOT_DATASET:-HBOrtiz/so101_eval3_broad}"
export VL_MANIFEST="${VL_MANIFEST:-HBOrtiz/so101_eval3_broad_grounding}"

# ObjectVLA's 10:1 robot:VL default for the broader OOD task (vs the IID
# cotrain's 5:1). Override via VL_RATIO=... if you want to ablate.
export VL_RATIO="${VL_RATIO:-10}"

# Output dir reflects the broad recipe; STEPS comes from launch_multi_gpu.sh
# (default 50000) so the suffix stays accurate when STEPS is overridden.
export OUT_DIR="${OUT_DIR:-outputs/smolvla_broad_klal_lora_${STEPS:-50000}}"

# Default push target if the operator doesn't override it. The shared
# launcher [FATAL]s if PUSH_REPO is unset.
export PUSH_REPO="${PUSH_REPO:-HBOrtiz/so101_smolvla_eval3_broad}"

# Hand off to the shared multi-GPU launcher. Same trainer (train_smolvla_cotrain.py),
# same preflight gates, same KLAL + LoRA defaults, same systemd-friendly
# torchrun invocation - just with broad data + 10:1 ratio under it.
exec bash "$REPO_ROOT/eval_3/scripts/smolvla_cotrain/launch_multi_gpu.sh"
