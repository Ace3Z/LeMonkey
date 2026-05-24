#!/usr/bin/env bash
# Sync code + dataset to a fresh training VM. Run from the dev box.
#
# What gets synced:
#   - the LeMonkey git repo (excluding heavy artefacts: third_party/sam2,
#     datasets/, outputs/, eval_{1,2}/train/, .git internals)
#   - the merged eval3 dataset (datasets/eval3_merged/, 7 GB)
#   - the HBOrtiz HF token so the policy push at end-of-training works
#
# Usage:
#   eval_3/scripts/training_vm/sync_to_vm.sh user@<vm-host>:~/LeMonkey
#
# Tested on the training VM / Shadeform / generic CUDA VMs.
#
# After sync, on the VM:
#   bash ~/LeMonkey/eval_3/scripts/training_vm/setup_pi05.sh        # idempotent env install for Pi0.5
#   # or: bash ~/LeMonkey/eval_3/scripts/training_vm/setup_paligemma_warmstart.sh   # for the PaliGemma warm-start path
#   # launch training via the shared systemd-wrap launcher (see eval_3/scripts/training_vm/README.md for the full invocation):
#   UNIT=lerobot-train-eval3 TRAIN_SCRIPT=~/LeMonkey/eval_3/scripts/training_vm/train_pi05.sh ... bash ~/LeMonkey/scripts/training_vm/start_training.sh
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <user@host:path>" >&2
  echo "Example: $0 user@<vm-host>:~/LeMonkey" >&2
  exit 2
fi

DST="$1"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "==> repo root: $REPO_ROOT"
echo "==> destination: $DST"
echo

# 1. Sync the code repo (everything except heavy artefacts)
echo "==> [1/3] syncing code repo ..."
rsync -avP --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude 'third_party/sam2/' \
  --exclude 'third_party/lerobot/.git/' \
  --exclude 'datasets/' \
  --exclude 'outputs/' \
  --exclude 'eval_1/train/' --exclude 'eval_2/train/' --exclude 'eval_3/train/' \
  --exclude 'eval_1/rollouts/' --exclude 'eval_2/rollouts/' --exclude 'eval_3/rollouts/' \
  --exclude '*.tar.gz' --exclude '*.zip' \
  --exclude 'wandb/' --exclude 'logs/' \
  "$REPO_ROOT/" "$DST/"

# 2. Sync the merged dataset (7 GB)
echo
echo "==> [2/3] syncing merged dataset ..."
rsync -avP --delete \
  "$REPO_ROOT/datasets/eval3_merged/" "$DST/datasets/eval3_merged/"

# 3. Sync the HBOrtiz HF token so end-of-training push works
echo
echo "==> [3/3] syncing HBOrtiz HF token ..."
rsync -avP "$REPO_ROOT/secrets/huggingface/token_hbortiz" \
  "$DST/secrets/huggingface/token_hbortiz"

echo
echo "==> sync complete."
echo
echo "Next steps on the VM:"
echo "  bash ~/LeMonkey/eval_3/scripts/training_vm/setup_pi05.sh   # or setup_paligemma_warmstart.sh"
echo "  # then launch training via the shared launcher; see eval_3/scripts/training_vm/README.md for the full env-var invocation."
