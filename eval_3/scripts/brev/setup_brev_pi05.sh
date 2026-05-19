#!/usr/bin/env bash
# Provision a fresh Brev VM for Pi0.5 + M2 + KLAL training.
#
# Run as: bash setup_brev_pi05.sh
#
# Idempotent — safe to re-run if a step failed.
# Expects HF_TOKEN to be available in ~/LeMonkey/.env (or env var).

set -euo pipefail

cd ~

# ─── 1. Miniconda ──────────────────────────────────────────────────
if [ ! -d ~/miniconda3 ]; then
  echo "==> installing miniconda3"
  curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p ~/miniconda3
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ─── 2. pi05 env (mirror of the SmolVLA `lemonkey` env) ─────────────
if ! conda env list | grep -q '^pi05 '; then
  echo "==> creating conda env pi05 (python 3.12)"
  conda create -y -n pi05 python=3.12
fi
conda activate pi05
python --version

# ─── 3. Repo (LeMonkey) ─────────────────────────────────────────────
if [ ! -d ~/LeMonkey ]; then
  echo "==> cloning LeMonkey via SSH (requires forwarded agent: ssh -A ...)"
  ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
  git clone --recurse-submodules git@github.com:Ace3Z/LeMonkey.git ~/LeMonkey
fi
cd ~/LeMonkey
git fetch origin
git checkout dev/m2-arcface-toolkit
git pull --ff-only origin dev/m2-arcface-toolkit
git submodule update --init --recursive

# ─── 4. PyTorch (CUDA 12.8 for H100 driver 5xx) ─────────────────────
echo "==> installing PyTorch cu128"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ─── 5. lerobot (editable from submodule) + the working version combo ─
echo "==> installing lerobot + deps"
pip install -e third_party/lerobot
pip install \
    "transformers==4.55.0" \
    "huggingface-hub==0.34.0" \
    "datasets" \
    "av" \
    "pyarrow" \
    "num2words" \
    "pyserial" \
    "pillow" \
    "draccus" \
    "safetensors"

# ─── 6. HF token ────────────────────────────────────────────────────
if [ ! -f ~/LeMonkey/.env ]; then
  echo "[setup] No ~/LeMonkey/.env — create one with: HF_TOKEN=hf_..."
  echo "        (the training script + autopush watcher both need it)"
fi

# ─── 7. Pre-download the M2 toolkit + Pi0.5 base ────────────────────
set -a; source ~/LeMonkey/.env 2>/dev/null; set +a

mkdir -p ~/eval3_m2_toolkit
echo "==> snapshot_download M2 toolkit"
python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='HBOrtiz/eval3_m2_arcface_toolkit', repo_type='dataset',
                  local_dir=os.path.expanduser('~/eval3_m2_toolkit'))
"

echo "==> warming Pi0.5 base + PaliGemma weights cache"
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lerobot/pi05_base', repo_type='model')
"
# google/paligemma-3b-pt-224 is used by the G2 probe (separate generate API).
python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download(repo_id='google/paligemma-3b-pt-224', repo_type='model')
except Exception as e:
    print('[warn] paligemma-3b-pt-224 download:', e)
"

# ─── 8. Verify CUDA + import sanity ─────────────────────────────────
python -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
import transformers, lerobot, av
print(f'transformers={transformers.__version__}  lerobot={getattr(lerobot, \"__version__\", \"src\")}')
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print('PI05Policy import OK')
"

echo
echo "==> setup complete. Next steps:"
echo "  1. Make sure ~/LeMonkey/.env has HF_TOKEN=..."
echo "  2. (Optional) Pre-cache the 3-celeb dataset + aug variants:"
echo "       see eval_3/scripts/brev/RUNBOOK.md"
echo "  3. Run gating: python eval_3/scripts/probe_paligemma_celeb_vqa.py"
echo "                python eval_3/scripts/attention_map_probe_pi05.py"
echo "  4. Launch training: bash eval_3/scripts/brev/run_training_track_E_pi05_3celeb.sh"
