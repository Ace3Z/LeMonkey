#!/usr/bin/env bash
# Brev-side env setup for the PaliGemma VQA warm-start (Pi0.5 fallback).
#
# Differs from setup_pi05.sh in two ways:
#   1. Forces cu128 PyTorch for Blackwell sm_120 GPUs (the RTX PRO 6000
#      VMs from Massed Compute / Brev). The default torch wheel pulled by
#      `pip install lerobot` is pre-cu128 and reports torch.cuda.is_available()
#      = False on Blackwell.
#   2. Installs `datasets` + `pillow` explicitly (needed by the VQA training
#      script's HF Dataset loader + image collator).
#
# Idempotent. Run on Brev after rsync.

set -euo pipefail

ENV_NAME=lemonkey

echo "=== [0/6] GPU detect ==="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "    detected: $GPU_NAME"
NEEDS_CU128=0
case "$GPU_NAME" in
    *Blackwell*|*RTX\ PRO\ 6000*|*RTX\ 5090*|*B100*|*B200*)
        echo "    -> Blackwell architecture (sm_120) - will force cu128 PyTorch"
        NEEDS_CU128=1
        ;;
    *H100*|*A100*|*L40*|*A10*)
        echo "    -> Hopper/Ampere (sm_90/sm_80) - stock PyTorch fine"
        ;;
    *)
        echo "    [WARN] unknown GPU model - assuming stock PyTorch works; verify cuda.is_available() below"
        ;;
esac
echo

# ── 1. Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
    echo "=== [1/6] Installing miniconda ==="
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
    rm Miniconda3-latest-Linux-x86_64.sh
    cd -
else
    echo "=== [1/6] Miniconda already installed ==="
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ── 2. Conda env
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "=== [2/6] conda env '$ENV_NAME' exists ==="
else
    echo "=== [2/6] creating conda env '$ENV_NAME' (python 3.12) ==="
    conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

# ── 3. lerobot editable install from our fork
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LEROBOT_DIR="$REPO_ROOT/third_party/lerobot"
if [ ! -d "$LEROBOT_DIR" ]; then
    echo "[FATAL] third_party/lerobot not found at $LEROBOT_DIR - did rsync run?" >&2
    exit 1
fi
echo "=== [3/6] pip install -e $LEROBOT_DIR[smolvla,pi] ==="
pip install --quiet --upgrade pip
pip install --quiet -e "$LEROBOT_DIR[smolvla,pi]" 2>&1 | tail -3

# Apply our lerobot patches (groot @strict + untagged-dataset fallback).
# See third_party/lerobot_patches/README.md for rationale.
bash "$REPO_ROOT/third_party/lerobot_patches/apply.sh"

# ── 4. Force cu128 PyTorch on Blackwell (must happen AFTER lerobot install
#       since lerobot pulls a torch version that's pre-cu128).
if [ "$NEEDS_CU128" = "1" ]; then
    echo "=== [4/6] forcing cu128 PyTorch for Blackwell sm_120 ==="
    pip install --quiet --upgrade torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
else
    echo "=== [4/6] keeping default PyTorch (non-Blackwell GPU) ==="
fi

# ── 5. VQA-specific extras
echo "=== [5/6] pip install peft, datasets, pandas, safetensors, Pillow ==="
pip install --quiet peft datasets pandas safetensors Pillow 2>&1 | tail -3

# ── 6. Verify
echo
echo "=== [6/6] Verification ==="
python <<'PY'
import sys, inspect
import torch
print(f"  python  : {sys.version.split()[0]}")
print(f"  torch   : {torch.__version__}")
print(f"  cuda    : available={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
if not torch.cuda.is_available():
    print("  [FATAL] torch.cuda.is_available()=False - re-run with cu128 install")
    sys.exit(1)
import lerobot, transformers, peft, datasets, PIL
print(f"  lerobot     : {getattr(lerobot, '__version__', 'editable')}  ({inspect.getfile(lerobot)})")
print(f"  transformers: {transformers.__version__}")
print(f"  peft        : {peft.__version__}")
print(f"  datasets    : {datasets.__version__}")
print(f"  Pillow      : {PIL.__version__}")
# VQA-critical imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
print("  [OK] PI05Policy importable")
from transformers import AutoProcessor, Trainer
print("  [OK] AutoProcessor + Trainer importable")
from peft import LoraConfig, get_peft_model, TaskType
print("  [OK] peft.LoraConfig + get_peft_model importable")
PY

# ── 7. HF auth
echo
echo "=== HF auth ==="
TOKEN_FILE="$REPO_ROOT/secrets/huggingface/token_hbortiz"
if [ ! -f "$TOKEN_FILE" ]; then
    echo "[WARN] token file missing: $TOKEN_FILE - re-run rsync from dev box"
else
    if hf auth whoami >/dev/null 2>&1; then
        echo "  [OK] already logged in: $(hf auth whoami 2>&1 | head -1)"
    else
        echo "  logging in with token from $TOKEN_FILE ..."
        hf auth login --token "$(cat "$TOKEN_FILE")"
        hf auth whoami 2>&1 | head -2
    fi
fi

echo
echo "=== the training VM (VQA warm-start) setup complete ==="
echo "Next:"
echo "  1. Obtain VGGFace2 raw images (via the dataset's official request page) or use CASIA-WebFace from HF."
echo "  2. Build manifest via eval_3/scripts/warmstart/prepare_vggface2_vqa.py"
echo "  3. Smoke test: bash eval_3/scripts/brev/train_paligemma_warmstart.sh after setting"
echo "     MANIFEST_PATH=... in env, with python eval_3/scripts/warmstart/train_paligemma_vqa.py --smoke"
echo "  4. Full run: bash eval_3/scripts/brev/train_paligemma_warmstart.sh"
