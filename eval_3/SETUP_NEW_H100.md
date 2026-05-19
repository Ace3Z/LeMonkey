# Setup a fresh H100 (or any GPU box) for SmolVLA training

End-to-end recipe to go from a clean Ubuntu cloud VM to a working SmolVLA
training run on Eval 3 datasets. Captures everything we learned the painful
way over a multi-day debugging session — read it before doing anything if
you don't want to repeat all of those mistakes.

> **Important context**: this recipe assumes you have **HuggingFace Pro**
> ($9/mo). Without it, HF rate limits will kill you on dataset downloads.
> Buy Pro first.

---

## Required versions (all must match — these are non-negotiable)

| Component | Required | Why |
|---|---|---|
| Python | **3.12** | LeRobot's pyproject.toml requires `>=3.12`. 3.10 silently fails to install lerobot from source. |
| ffmpeg | **7.1.1** (conda-forge) | torchcodec auto-detects ffmpeg version; 8.x and 4.x both cause issues. |
| PyTorch | **2.10+ with cu128** | Matches H100 driver (CUDA 12.8). Wrong CUDA version → driver mismatch errors. |
| LeRobot | **from `third_party/lerobot` submodule** (pinned commit `7d8914c`) | The PyPI version routes `pyav` through `torchvision.io.VideoReader` which doesn't ship with cu128 torchvision wheels. Submodule version routes correctly. |
| HF tier | **Pro** | Free tier rate limits (1000 req/5min XET, 5000 req/5min resolvers) make 14GB+ dataset downloads impossible. |

---

## Block 1 — Install miniconda (skip if already installed)

```bash
cd ~
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
source $HOME/miniconda3/etc/profile.d/conda.sh
echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
```

## Block 2 — Conda ToS + fresh env

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Wipe any existing env (idempotent)
conda env remove -n lerobot -y 2>/dev/null

# Create with Python 3.12 (NOT 3.10 — lerobot needs 3.12)
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

## Block 3 — ffmpeg via conda-forge (PINNED to 7.1.1)

```bash
conda install -y -c conda-forge "ffmpeg=7.1.1"
```

## Block 4 — Persistent LD_LIBRARY_PATH fix (libstdc++ issue)

ffmpeg 7 from conda-forge needs a newer libstdc++ than the system has. Force
conda's libstdc++ to be used:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh << 'EOF'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF

# Apply now so the current shell session has it
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

After this, every `conda activate lerobot` auto-exports this. No more
`CXXABI_1.3.15 not found` errors on torchcodec import.

## Block 5 — PyTorch for CUDA 12.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Block 6 — Clone LeMonkey + init lerobot submodule

```bash
# Set up GitHub SSH key first (or use HTTPS + PAT)
ssh-keygen -t ed25519 -C "h100-vm" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# → Add this key at https://github.com/settings/keys
ssh -T git@github.com  # accept fingerprint

# Clone LeMonkey
git clone git@github.com:Ace3Z/LeMonkey.git
cd LeMonkey
git checkout hans-smolvlm-finetuning

# Init the lerobot submodule (must be empty or this errors — wipe if exists)
rm -rf third_party/lerobot
git submodule update --init --recursive
ls third_party/lerobot/   # MUST show files, not be empty

# Install lerobot from submodule (NOT from pip — pip version routes pyav wrong)
cd third_party/lerobot
pip install -e ".[smolvla]"
cd ~/LeMonkey
```

## Block 7 — Remove hf_xet (causes snapshot_download hangs)

XET storage backend has stalls during HEAD validation. Force regular HTTPS:

```bash
pip uninstall -y hf_xet
```

## Block 8 — Verify everything

```bash
python -c "
import torch, lerobot, torchcodec
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0))
print('lerobot:', lerobot.__version__, '| path:', lerobot.__file__)
print('torchcodec:', torchcodec.__version__)
"
```

Expected output:
- `torch: 2.10+ | cuda: True | device: NVIDIA H100 ...`
- `lerobot path: .../LeMonkey/third_party/lerobot/src/lerobot/__init__.py` (**NOT** site-packages)
- `torchcodec: 0.x.x` with no error

If any line errors, fix it before proceeding.

## Block 9 — Auth to HuggingFace + Wandb

```bash
hf auth login --token <YOUR_HF_TOKEN>   # needs read+write to HBOrtiz/* (or your namespace)
hf auth whoami    # verifies login (should print HBOrtiz or whoever)

wandb login <YOUR_WANDB_TOKEN>
```

## Block 10 — Raise file descriptor limit (PERSISTENT)

PyTorch DataLoader with 4+ workers × many mp4 files blows past the default
1024 fd limit. Raise it permanently:

```bash
echo 'ulimit -n 65535' >> ~/.bashrc
ulimit -n 65535  # apply to current shell too
```

---

## Pre-flight before launching training

### Pre-download the dataset directly into lerobot's cache

LeRobot uses its OWN cache at `~/.cache/huggingface/lerobot/hub/` (different
from the standard HF cache at `~/.cache/huggingface/hub/`). Pre-downloading
to the wrong place wastes bandwidth and re-downloads at training time.

The dataset HEAD-check is the most fragile part of training startup. Pre-pull
fully BEFORE starting training so lerobot just verifies cached files.

```bash
# Pick your dataset:
#   - HBOrtiz/so101_eval3_track3_v3_baseline (3-celeb, 14 GB, the canonical training dataset)
#   - HBOrtiz/so101_eval3_all (200-celeb, 7.5 GB, includes ref-only prompts that need filtering)

# For toy (3 celebs):
hf download HBOrtiz/so101_eval3_track3_v3_baseline --repo-type dataset --max-workers 4

# Pre-pull the policy + VLM too
hf download lerobot/smolvla_base --max-workers 4
hf download HBOrtiz/smolvlm2_toy_celebs --max-workers 4  # or smolvlm2_lora_celebs for broad
```

### If your dataset is `so101_eval3_all` (broad), generate keep-indices first

```bash
mkdir -p $HOME/datasets/eval3_celebs
export EVAL3_ROOT=$HOME/datasets/eval3_celebs
python eval_3/scripts/filter_eval3_all_episodes.py \
    --out $EVAL3_ROOT/keep_indices_eval3_all_textonly.json
# → drops 595 ref-only episodes (TA-banned at inference)
```

---

## Training command — TOY VLM + 3-celeb dataset

```bash
rm -rf ~/LeMonkey/outputs/smolvla_toy_v0
tmux new -s smolvla_toy

# Inside tmux:
ulimit -n 65535
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=10
source ~/.bashrc && conda activate lerobot && cd ~/LeMonkey

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.vlm_model_name=HBOrtiz/smolvlm2_toy_celebs \
    --policy.repo_id=HBOrtiz/smolvla_toy_v0 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.add_image_special_tokens=true \
    --policy.empty_cameras=1 \
    --policy.optimizer_lr=5e-5 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.use_amp=true \
    --policy.device=cuda \
    --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --rename_map='{"observation.images.reference": "observation.images.camera2"}' \
    --batch_size=32 \
    --steps=20000 \
    --save_freq=5000 \
    --num_workers=4 \
    --output_dir=outputs/smolvla_toy_v0 \
    --job_name=smolvla_toy_v0 \
    --wandb.enable=true \
    --wandb.project=lemonkey-eval3-smolvla
```

## Training command — BROAD VLM + 200-celeb dataset (with filter)

```bash
rm -rf ~/LeMonkey/outputs/smolvla_broad_v0
tmux new -s smolvla_broad

# Inside tmux:
ulimit -n 65535
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=10
source ~/.bashrc && conda activate lerobot && cd ~/LeMonkey
export EVAL3_ROOT=$HOME/datasets/eval3_celebs
INDICES=$(cat $EVAL3_ROOT/keep_indices_eval3_all_textonly.json)

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.vlm_model_name=HBOrtiz/smolvlm2_lora_celebs \
    --policy.repo_id=HBOrtiz/smolvla_broad_v0 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.add_image_special_tokens=true \
    --policy.empty_cameras=1 \
    --policy.optimizer_lr=5e-5 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.use_amp=true \
    --policy.device=cuda \
    --dataset.repo_id=HBOrtiz/so101_eval3_all \
    --dataset.episodes="$INDICES" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --rename_map='{"observation.images.reference": "observation.images.camera2"}' \
    --batch_size=32 \
    --steps=20000 \
    --save_freq=5000 \
    --num_workers=4 \
    --output_dir=outputs/smolvla_broad_v0 \
    --job_name=smolvla_broad_v0 \
    --wandb.enable=true \
    --wandb.project=lemonkey-eval3-smolvla
```

**Note on the broad run:** the `--dataset.episodes` flag with 3,600 indices
triggers a slow `fnmatch` phase inside `huggingface_hub.snapshot_download`
that takes ~30-40 min CPU-bound before training starts. No way around it
without cleaning the dataset upstream (see the deprecated section in
`docs/EVAL_3_DATASETS.md`).

---

## Critical flag rationale (from team's `eval_3/scripts/brev/run_training.sh`)

| Flag | Why |
|---|---|
| `--policy.train_expert_only=false` | CRITICAL — frozen VLM yields ~0% accuracy on face-matching (per Interleave-VLA, Pi0.5-KI, "Don't Blind Your VLA" papers). |
| `--policy.freeze_vision_encoder=false` | SigLIP must adapt for face matching across reference photo ↔ printed portrait. |
| `--policy.add_image_special_tokens=true` | BOI/EOI separators between cameras so the LM decoder can tell them apart. |
| `--policy.empty_cameras=1` | SmolVLA expects 3 cameras (camera1/2/3); with our 2 dataset cameras + this flag, the 3rd is zero-padded. |
| `--policy.optimizer_lr=5e-5` | Half of LeRobot's default 1e-4 — protects pretrained features when unfreezing both VLM and SigLIP. |
| `--policy.use_amp=true` | bf16 mixed precision (memory-saving knob since SmolVLAConfig has no gradient_checkpointing flag). |
| `--rename_map={"observation.images.reference": "observation.images.camera2"}` | Maps dataset's `reference` channel to the policy's `camera2` slot. |
| `--dataset.video_backend=pyav` | torchcodec leaks ~35 GB per worker over ~30 min — confirmed by team. pyav has no such issue. |
| `--batch_size=32` | H100 PCIe = 80 GB VRAM. Team measured bs=64 on 97 GB (RTX PRO 6000); 64 OOMs on 80 GB. |
| `--num_workers=4` | bs=32 doesn't need 8 workers. 4 is enough and avoids race conditions on the dataset reader. |
| `--steps=20000` | ~5-6 hours wall time on H100. Team used 30k for production; 20k is fine for fine-tuning. |

---

## Common gotchas (debug menu)

| Symptom | Cause | Fix |
|---|---|---|
| `Package 'lerobot' requires a different Python: 3.10.X not in '>=3.12'` | Env created with Python 3.10 | Recreate env with `python=3.12` (Block 2) |
| `OSError: libavutil.so.56: cannot open shared object file` | ffmpeg not installed in env | `conda install -c conda-forge "ffmpeg=7.1.1"` |
| `OSError: libstdc++.so.6: version CXXABI_1.3.15 not found` | System libstdc++ too old | LD_LIBRARY_PATH fix (Block 4) |
| `torch... cuda: False` after conda install | Conda shuffled torch deps | `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `AttributeError: module 'torchvision.io' has no attribute 'VideoReader'` | LeRobot routing pyav → torchvision (wrong version) | Use submodule lerobot, not pip (Block 6) |
| `AttributeError: GPT2TokenizerFast has no attribute fake_image_token_id` | Pushed VLM missing tokenizer files | Sync `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `vocab.json`, `merges.txt`, `preprocessor_config.json`, `chat_template.json` from base SmolVLM2 |
| `OSError: [Errno 24] Too many open files` | ulimit too low | Block 10 (raise to 65535) |
| `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED` | batch_size too big | Drop to `--batch_size=16` |
| Training hangs at "Creating dataset" forever | `snapshot_download` HEAD checks stalled | Pre-download dataset with `hf download --max-workers 4` BEFORE training; remove `hf_xet`; set `HF_HUB_DOWNLOAD_TIMEOUT=30` |
| `IndexError: Invalid key: NNNN is out of bounds for size MMMM` | Dataset metadata mismatch | Wipe `~/.cache/huggingface/datasets/`; re-download dataset; try `--dataset.revision=v3.0` if tagged |
| `429 Too Many Requests` on HF | Free tier rate limit | Buy HF Pro ($9/mo) |
| `403 Forbidden: You don't have the rights to create a model under the namespace "X"` | HF token doesn't have write access to that namespace | Use a namespace your token CAN write to (check `hf auth whoami`) |

---

## When training finishes

LeRobot auto-pushes the final checkpoint to `--policy.repo_id` (e.g.
`HBOrtiz/smolvla_toy_v0`). Intermediate checkpoints are at
`outputs/<job_name>/checkpoints/<step>/`.

To deploy on the robot:
```bash
lerobot-record \
    --policy.path=HBOrtiz/smolvla_toy_v0 \
    ...
```

See `eval_3/HANDOVER_TO_DEPLOY.md` for the deploy-side recipe (rollout
script template, reference photo handling, etc.).

---

## Delete the H100 when you're done

```bash
brev delete <instance-name>
```

$2.28-$4.56/hr adds up. Don't leave idle instances running.
