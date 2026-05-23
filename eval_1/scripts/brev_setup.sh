#!/usr/bin/env bash
# Bootstrap script for a fresh Brev H100 instance.
# Idempotent - safe to re-run. Logs every fallback per the no-silent-fallbacks rule.
#
# What it does:
#   1. Install Miniconda if missing
#   2. Create / activate the `lemonkey` conda env (python 3.12)
#   3. pip install lerobot[smolvla] + a couple of tiny extras we need
#   4. Verify imports + GPU
#
# Run on the Brev VM after rsync:
#   bash ~/LeMonkey/eval_1/scripts/brev_setup.sh
set -euo pipefail

# ─── 0. Sanity: must be on a CUDA host ───────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found - script assumed a CUDA host. Continuing anyway." >&2
fi
echo "=== GPU detect ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -3 || echo "[WARN] no GPUs"
echo ""

# ─── 0b. ffmpeg (provides libavutil etc. - needed by torchcodec) ─────────────
# lerobot reads video frames from the LeRobotDataset via torchcodec, which
# ctypes-loads libavutil.so.X from the system. Ubuntu 22.04 base images don't
# ship ffmpeg, so torchcodec.decoders.VideoDecoder fails on first dataset access
# with: 'libavutil.so.60: cannot open shared object file' (and similar for 59..56).
if ldconfig -p 2>/dev/null | grep -q libavutil; then
  echo "=== ffmpeg already installed (libavutil found) - skipping ==="
else
  echo "=== Installing ffmpeg (provides libavutil for torchcodec) ==="
  sudo apt-get update -qq
  sudo apt-get install -y ffmpeg
fi
echo ""

# ─── 1. Miniconda ────────────────────────────────────────────────────────────
CONDA_DIR="$HOME/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
  echo "=== Installing Miniconda to $CONDA_DIR ==="
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -u -p "$CONDA_DIR"
  rm /tmp/miniconda.sh

  # First-time bootstrap: hook conda into bash + don't auto-activate base.
  # (Per SETUP.md §1.)
  "$CONDA_DIR/bin/conda" init bash >/dev/null 2>&1 || \
    echo "[WARN] conda init bash failed - manual ~/.bashrc edit may be required" >&2
  "$CONDA_DIR/bin/conda" config --set auto_activate_base false || \
    echo "[WARN] could not set auto_activate_base=false" >&2
else
  echo "=== Miniconda already present at $CONDA_DIR - skipping install ==="
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

# Anaconda's default channels require ToS acceptance on conda 24+
# (Without this, `conda create` fails with CondaToSNonInteractiveError.)
echo "=== Accepting Anaconda channel ToS (idempotent) ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || \
  echo "[WARN] could not accept ToS for pkgs/main - channel may not be in use" >&2
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || \
  echo "[WARN] could not accept ToS for pkgs/r - channel may not be in use" >&2

# ─── 2. Conda env "lemonkey" ─────────────────────────────────────────────────
ENV_NAME=lemonkey
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "=== conda env '$ENV_NAME' already exists - skipping create ==="
else
  echo "=== Creating conda env '$ENV_NAME' (python 3.12) ==="
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

# Quick sanity: python is the right one
PYBIN="$(which python)"
case "$PYBIN" in
  *"miniconda3/envs/$ENV_NAME"*) ;;
  *)
    echo "[WARN] python is not from the lemonkey env: $PYBIN" >&2
    ;;
esac

# ─── 3. pip install lerobot[smolvla] + extras ────────────────────────────────
# Version-pinned to match SETUP.md §3 (lerobot==0.5.1). DON'T use the
# third_party/lerobot submodule in this repo - it's missing
# `lerobot.datasets` and friends (see SETUP.md §12).
echo "=== pip install lerobot[smolvla]==0.5.1 ==="
pip install --quiet --upgrade pip
pip install --quiet 'lerobot[smolvla]==0.5.1' 2>&1 | tail -5

# Extras we use in our residual scripts that aren't pulled by lerobot[smolvla]
echo "=== pip install pandas safetensors (for our scripts) ==="
pip install --quiet pandas safetensors 2>&1 | tail -3

# ─── 4. Verify ───────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python -c "
import sys, torch
print(f'  python  : {sys.version.split()[0]}')
print(f'  torch   : {torch.__version__}  (cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"n/a\"})')
import lerobot, transformers, pandas, safetensors
print(f'  lerobot : {lerobot.__version__ if hasattr(lerobot, \"__version__\") else \"installed\"}')
print(f'  transformers: {transformers.__version__}')
print(f'  pandas      : {pandas.__version__}')
print(f'  safetensors : installed')
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('  [OK] SmolVLA policy importable')
"

# ─── 5. HF auth reminder ─────────────────────────────────────────────────────
echo ""
echo "=== Next: HF auth ==="
if hf auth whoami >/dev/null 2>&1; then
  WHO=$(hf auth whoami 2>&1 | head -1)
  echo "  [OK] already logged in: $WHO"
else
  echo "  ⚠️  not logged in. Run:"
  echo "      hf auth login    # paste your write token"
fi

echo ""
echo "=== Setup complete ==="
echo "  To activate the env in a new shell:"
echo "    source ~/miniconda3/etc/profile.d/conda.sh && conda activate lemonkey"
