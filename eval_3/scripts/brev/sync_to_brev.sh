#!/usr/bin/env bash
# Sync code + dataset to a fresh Brev VM. Run from the dev box.
#
# What gets synced:
#   - the LeMonkey git repo (excluding heavy artefacts: third_party/sam2,
#     datasets/, outputs/, eval_{1,2}/train/, .git internals)
#   - the merged eval3 dataset (datasets/eval3_merged/, 7 GB)
#   - the HBOrtiz HF token so the policy push at end-of-training works
#
# Usage:
#   eval_3/scripts/brev/sync_to_brev.sh shadeform@<brev-host>:~/LeMonkey
#
# After sync, on Brev:
#   bash ~/LeMonkey/eval_1/scripts/brev_setup.sh        # idempotent env install
#   bash ~/LeMonkey/eval_3/scripts/brev/start_training.sh  # launch training
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <user@host:path>" >&2
  echo "Example: $0 shadeform@brev-h100-foo:~/LeMonkey" >&2
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
echo "Next steps on Brev:"
echo "  bash ~/LeMonkey/eval_1/scripts/brev_setup.sh"
echo "  bash ~/LeMonkey/eval_3/scripts/brev/start_training.sh"
