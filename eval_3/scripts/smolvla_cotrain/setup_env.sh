#!/usr/bin/env bash
# setup_env.sh - one-time bootstrap of the lemonkey conda env for SmolVLA cotrain.
#
# Run once from repo root:
#   bash eval_3/scripts/smolvla_cotrain/setup_env.sh
#
# After this completes:
#   conda activate lemonkey
#   export HF_TOKEN=hf_...
#   cd ~/LeMonkey
#   STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2 LOG_EVERY=1 \
#       bash eval_3/scripts/smolvla_cotrain/launch.sh 2>&1 | tee smoke.log
#
# GPU: 2× NVIDIA RTX PRO 6000 Blackwell (97 GB each), driver CUDA 13.0.
# PyTorch: 2.11.0+cu128 (CUDA 12.8 bundle; latest wheel that fits
#   lerobot's torch>=2.7,<2.12.0 constraint and runs on Blackwell sm_120).
# lerobot: v0.5.2-ish HEAD of main (the pinned commit 7d8914c8 was GC'd
#   from origin; HEAD at dfdc48a7 includes the same SmolVLA policy + a
#   VideoDecoderCache OOM fix - better than the pin, not worse).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "==> repo root: $REPO_ROOT"

# ── Source conda ────────────────────────────────────────────────────────────
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
    echo "[ERROR] conda not found at $CONDA_SH - adjust CONDA_SH in this script" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate lemonkey
echo "==> activated conda env: $(conda info --envs | grep '^\*' | awk '{print $1}')"

# ── 1. PyTorch (CUDA 12.8 bundle) ───────────────────────────────────────────
echo
echo "==> [1/4] installing PyTorch 2.11.0+cu128 + torchvision 0.26.0+cu128 ..."
echo "    (Blackwell sm_120 support; within lerobot torch>=2.7,<2.12.0 constraint)"
pip install \
    torch==2.11.0+cu128 \
    torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ── 2. lerobot + extras ─────────────────────────────────────────────────────
echo
echo "==> [2/4] installing lerobot[smolvla,dataset,av-dep] from third_party/ ..."
echo "    smolvla : SmolVLAPolicy + transformers>=5.4 + accelerate"
echo "    dataset : LeRobotDataset + pandas + pyarrow + torchcodec"
echo "    av-dep  : av>=15 (PyAV - video backend used at training time)"
pip install -e "$REPO_ROOT/third_party/lerobot[smolvla,dataset,av-dep]"

# Apply our lerobot patches (groot @strict + untagged-dataset fallback).
# See third_party/lerobot_patches/README.md for rationale.
bash "$REPO_ROOT/third_party/lerobot_patches/apply.sh"

# ── 3. zstandard (VL image tar.zst extraction) ──────────────────────────────
echo
echo "==> [3/4] installing zstandard (needed if VL images arrive as images.tar.zst) ..."
pip install zstandard

# ── 4. Verify ────────────────────────────────────────────────────────────────
echo
echo "==> [4/4] running verification checks ..."

python - <<'PYEOF'
import sys

# torch + CUDA
import torch
assert torch.cuda.is_available(), "CUDA not available - check driver / CUDA toolkit"
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")

# SmolVLAPolicy importable
import lerobot.policies.smolvla.modeling_smolvla as m
print(f"  SmolVLAPolicy.name = {m.SmolVLAPolicy.name}")

# LeRobotDataset importable
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print(f"  LeRobotDataset imported OK")

# pandas + PIL
import pandas as pd
from PIL import Image
print(f"  pandas {pd.__version__}  Pillow {Image.__version__}")

# huggingface_hub
import huggingface_hub
print(f"  huggingface_hub {huggingface_hub.__version__}")

print("\nAll checks passed.")
PYEOF

echo
echo "================================================================="
echo " lemonkey env is ready."
echo ""
echo " Next steps:"
echo "   conda activate lemonkey"
echo "   export HF_TOKEN=hf_..."
echo ""
echo "   # 200-step smoke test:"
echo "   cd ~/LeMonkey"
echo "   STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2 LOG_EVERY=1 \\"
echo "       bash eval_3/scripts/smolvla_cotrain/launch.sh 2>&1 | tee smoke.log"
echo ""
echo "   # Check the 5 gates in README.md §Smoke-test gates, then:"
echo "   PUSH_REPO=HBOrtiz/smolvla_eval3_cotrain_10to1 \\"
echo "       bash eval_3/scripts/smolvla_cotrain/launch.sh"
echo "================================================================="
