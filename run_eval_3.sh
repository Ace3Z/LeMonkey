#!/usr/bin/env bash
# Eval 3 - interactive SmolVLA rollout runner.
#
# Two deployed policies, selectable via a flag:
#
#   ./run_eval_3.sh                   # default: in-distribution cotrain
#                                     #   -> HBOrtiz/so101_smolvla_eval3_cotrain
#                                     #      (SmolVLA + robot + VL grounding, 5:1)
#   ./run_eval_3.sh --broad           # broad / out-of-distribution
#                                     #   -> HBOrtiz/so101_smolvla_eval3_broad
#                                     #      (192-celebrity cotrain)
#
# To pick a non-default checkpoint, pass it as a second argument:
#   ./run_eval_3.sh --cotrain step_020000
#   ./run_eval_3.sh --broad   checkpoints/020000
#
# Prompt: "Put the coke on <celebrity_name>."
# Single-camera contract: wrist USB cam on /dev/video0; unused slots are auto
# zero-padded via the policy's empty_cameras setting.
set -euo pipefail

MODE="cotrain"
CKPT=""
for arg in "$@"; do
  case "$arg" in
    --cotrain) MODE="cotrain" ;;
    --broad)   MODE="broad" ;;
    *)         CKPT="$arg" ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
  cotrain)
    exec "$REPO_ROOT/eval_3/scripts/rollout/smolvla_cotrain.sh" ${CKPT:+"$CKPT"}
    ;;
  broad)
    exec "$REPO_ROOT/eval_3/scripts/rollout/smolvla_broad.sh" ${CKPT:+"$CKPT"}
    ;;
esac
