#!/usr/bin/env bash
# Bootstrap a fresh Brev instance for Eval 3 (π0.5 / PaliGemma-3B).
# Idempotent — safe to re-run.
#
# Differences vs eval_1/scripts/brev_setup.sh:
#   - installs `lerobot[pi0]==0.5.1` instead of `lerobot[smolvla]==0.5.1`
#     (the [pi0] extra pulls the PaliGemma deps used by π0 *and* π0.5).
#   - verifies the π0.5 policy import (PI05Policy) instead of SmolVLA.
#
# Run on the Brev VM after rsync:
#   bash ~/LeMonkey/eval_3/scripts/brev/setup_pi05.sh
set -euo pipefail

# ─── 0. Sanity: must be on a CUDA host ───────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found — script assumed a CUDA host. Continuing anyway." >&2
fi
echo "=== GPU detect ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -3 || echo "[WARN] no GPUs"
echo ""

# ─── 0b. ffmpeg (libavutil for torchcodec) ───────────────────────────────────
if ldconfig -p 2>/dev/null | grep -q libavutil; then
  echo "=== ffmpeg already installed (libavutil found) — skipping ==="
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
  "$CONDA_DIR/bin/conda" init bash >/dev/null 2>&1 || \
    echo "[WARN] conda init bash failed — manual ~/.bashrc edit may be required" >&2
  "$CONDA_DIR/bin/conda" config --set auto_activate_base false || \
    echo "[WARN] could not set auto_activate_base=false" >&2
else
  echo "=== Miniconda already present at $CONDA_DIR — skipping install ==="
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "=== Accepting Anaconda channel ToS (idempotent) ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || \
  echo "[WARN] could not accept ToS for pkgs/main — channel may not be in use" >&2
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || \
  echo "[WARN] could not accept ToS for pkgs/r — channel may not be in use" >&2

# ─── 2. Conda env "lemonkey" (shared with eval_1/2) ──────────────────────────
ENV_NAME=lemonkey
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "=== conda env '$ENV_NAME' already exists — skipping create ==="
else
  echo "=== Creating conda env '$ENV_NAME' (python 3.12) ==="
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

PYBIN="$(which python)"
case "$PYBIN" in
  *"miniconda3/envs/$ENV_NAME"*) ;;
  *)
    echo "[WARN] python is not from the lemonkey env: $PYBIN" >&2
    ;;
esac

# ─── 3. pip install lerobot[pi0]==0.5.1 + extras ─────────────────────────────
# The [pi0] extra also covers π0.5 — same PaliGemma backbone, same deps.
# DON'T use the third_party/lerobot submodule (see eval_1/scripts/brev_setup.sh).
echo "=== pip install lerobot[pi0]==0.5.1 ==="
pip install --quiet --upgrade pip
pip install --quiet 'lerobot[pi0]==0.5.1' 2>&1 | tail -5

echo "=== pip install pandas safetensors ==="
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
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print('  ✓ π0.5 policy importable')
"

# ─── 5. HF auth reminder ─────────────────────────────────────────────────────
echo ""
echo "=== Next: HF auth ==="
if hf auth whoami >/dev/null 2>&1; then
  WHO=$(hf auth whoami 2>&1 | head -1)
  echo "  ✓ already logged in: $WHO"
else
  echo "  ⚠️  not logged in. Run:"
  echo "      hf auth login    # paste your write token"
fi

echo ""
echo "=== Setup complete ==="
echo "  To activate the env in a new shell:"
echo "    source ~/miniconda3/etc/profile.d/conda.sh && conda activate lemonkey"
