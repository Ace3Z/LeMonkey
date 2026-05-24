#!/usr/bin/env bash
# Brev-side env setup for Pi0.5 (Pi0.5 + LoRA).
#
# Differs from scripts/brev_setup_smolvla.sh: that script installed
# `lerobot[smolvla]==0.5.1` from PyPI; Pi0.5 needs Pi0.5 + PEFT support
# which we have in the vendored fork at third_party/lerobot/. So this
# script does `pip install -e third_party/lerobot[smolvla,pi]` instead.
#
# Idempotent. Run on Brev after rsync.
set -euo pipefail

ENV_NAME=lemonkey

echo "=== [0/5] GPU detect ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -3 || echo "[WARN] no GPUs detected"
echo

# ── 1. Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
  echo "=== [1/5] Installing miniconda ==="
  cd /tmp
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
  rm Miniconda3-latest-Linux-x86_64.sh
  cd -
else
  echo "=== [1/5] Miniconda already installed ==="
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ── 2. Conda env
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "=== [2/5] conda env '$ENV_NAME' already exists - skipping create ==="
else
  echo "=== [2/5] Creating conda env '$ENV_NAME' (python 3.12) ==="
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

# Sanity: python is the right one
PYBIN="$(which python)"
case "$PYBIN" in
  *"miniconda3/envs/$ENV_NAME"*) ;;
  *) echo "[WARN] python is not from the lemonkey env: $PYBIN" >&2 ;;
esac

# ── 3. pip install lerobot from our local fork (editable)
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LEROBOT_DIR="$REPO_ROOT/third_party/lerobot"
if [ ! -d "$LEROBOT_DIR" ]; then
  echo "[FATAL] third_party/lerobot not found at $LEROBOT_DIR - did rsync run?" >&2
  exit 1
fi
echo "=== [3/5] pip install -e $LEROBOT_DIR[smolvla,pi]  (editable, our fork) ==="
pip install --quiet --upgrade pip
pip install --quiet -e "$LEROBOT_DIR[smolvla,pi]" 2>&1 | tail -5

# Apply our lerobot patches (groot @strict + untagged-dataset fallback).
# See third_party/lerobot_patches/README.md for rationale.
bash "$REPO_ROOT/third_party/lerobot_patches/apply.sh"

# Extras we use in our scripts that aren't in lerobot[smolvla,pi]
echo "=== [4/5] pip install pandas safetensors peft ==="
pip install --quiet pandas safetensors peft 2>&1 | tail -3

# ── 5. Verify
echo
echo "=== [5/5] Verification ==="
python -c "
import sys, torch, inspect
print(f'  python  : {sys.version.split()[0]}')
print(f'  torch   : {torch.__version__}  (cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"n/a\"})')
import lerobot
print(f'  lerobot : {getattr(lerobot, \"__version__\", \"editable\")}  ({inspect.getfile(lerobot)})')
import transformers; print(f'  transformers: {transformers.__version__}')
import peft;        print(f'  peft        : {peft.__version__}')
import pandas;      print(f'  pandas      : {pandas.__version__}')
# Pi0.5-critical imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.default import PeftConfig
print('  [OK] Pi0.5 policy + config + PeftConfig importable')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('  [OK] LeRobotDataset importable')
"

# ── 6. HF auth (one-time)
echo
echo "=== HF auth ==="
TOKEN_FILE="$REPO_ROOT/secrets/huggingface/token_hbortiz"
if [ ! -f "$TOKEN_FILE" ]; then
  echo "[WARN] token file missing: $TOKEN_FILE - re-run sync_to_brev.sh from dev box"
else
  if hf auth whoami >/dev/null 2>&1; then
    echo "  [OK] already logged in: $(hf auth whoami 2>&1 | head -1)"
  else
    echo "  Logging in with token from $TOKEN_FILE ..."
    hf auth login --token "$(cat "$TOKEN_FILE")"
    hf auth whoami 2>&1 | head -2
  fi
fi

echo
echo "=== Pi0.5 setup complete ==="
echo "  Activate in new shell:"
echo "    source ~/miniconda3/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "  Launch training:"
echo "    cd ~/LeMonkey && bash eval_3/scripts/brev/train_pi05.sh"
