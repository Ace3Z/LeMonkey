# Run the KLAL + LoRA co-training on a multi-GPU cluster

Step-by-step for running the 25k-step SmolVLA co-training (10:1 robot:VL,
KLAL attention supervision + LoRA) on a fresh cluster with several H200s.
Everything the run needs is in the repo - no extra data to fetch by hand.

The only thing you must supply is a **HuggingFace token with write access**
(checkpoints are pushed to HF every 5k steps).

---

## 1. Clone the repo + check out the branch

`third_party/lerobot` is a git submodule - clone **with submodules**:

```bash
git clone --recurse-submodules https://github.com/Ace3Z/LeMonkey.git
cd LeMonkey
git checkout dev/mahbod/kl-divergence
git submodule update --init --recursive   # safe to re-run; ensures lerobot is present
```

Branch: **`dev/mahbod/kl-divergence`**.

---

## 2. Set up the Python environment (one-time)

Python **3.12** (lerobot v0.5.1 requires `>=3.12`). Conda or venv - either works.

```bash
conda create -y -n cotrain python=3.12 && conda activate cotrain
# (or, with a system python3.12:  python3.12 -m venv .venv && source .venv/bin/activate)

# 1) PyTorch for your CUDA (H200 = Hopper; cu124 wheels are fine):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) lerobot (from the vendored copy) + co-train deps, and zstandard:
pip install -e "third_party/lerobot[smolvla,dataset,av-dep]" zstandard
```

Verify:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs')"
python -c "import lerobot.policies.smolvla.modeling_smolvla; print('SmolVLA OK')"
```

Both must print cleanly. (`eval_3/scripts/smolvla_cotrain/setup_env.sh` is a
conda-based bootstrap example if your cluster matches it - but the two `pip`
commands above are the portable path.)

---

## 3. HuggingFace token + push target

The script needs an HF token with **write** access (checkpoints are pushed).
Give it the token one of two ways - **never commit a token into the repo**:

**Option A - env var:**
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**Option B - token file (set once, gitignored):**
```bash
echo "hf_xxxxxxxxxxxxxxxxxxxxx" > eval_3/scripts/smolvla_cotrain/.hf_token
```
`.hf_token` is in `.gitignore`, so it stays local and is never pushed.

Either way, also set the push target:
```bash
export PUSH_REPO=youruser/smolvla_klal_lora_25k   # HF model repo for checkpoints
```

`PUSH_REPO` is created automatically if it doesn't exist. Checkpoints land at
`PUSH_REPO/step_005000`, `/step_010000`, …, `/final`.

The two training datasets (`HBOrtiz/so101_eval3_cotrain`,
`HBOrtiz/eval3_vl_pairs`) and the base model (`lerobot/smolvla_base`)
download automatically on the first run - the `HF_TOKEN` you set also
authorizes those downloads.

---

## 4. Run

```bash
bash eval_3/scripts/smolvla_cotrain/run_cluster.sh
```

That's it. The script:
- autodetects **every GPU** on the node and launches one process per GPU
  (`torchrun`, manual data-parallel gradient all-reduce);
- trains **50,000 steps**, 5:1 robot:VL, with KLAL + LoRA - KLAL's attention
  target is built from the VL dataset's `quad_corners_norm` column;
- saves a checkpoint and **pushes it to HF every 5,000 steps** (+ a final one).

**First run downloads ~15 GB of datasets** before training starts - this is
normal; let it run.

To run detached (recommended for a multi-hour job):

```bash
nohup bash eval_3/scripts/smolvla_cotrain/run_cluster.sh > cotrain.log 2>&1 &
tail -f cotrain.log
```

---

## 5. What you should see

Per-step log lines (rank 0 only):

```
step    123  flow_loss=1.2034  (last flow=0.21 vqa=10.4 klal=0.95)  grad=5.1  steps/s=...
```

- `flow_loss` - robot action loss; should trend down.
- `vqa_loss` - VL VQA loss (every 11th step); should trend down.
- `klal` - attention-supervision loss; non-zero and finite.

Every 5k steps: `[cotrain] checkpoint saved → …` then `[cotrain] pushed → https://huggingface.co/…`.

---

## Tunables (override by exporting before step 4)

| Env var | Default | Notes |
|---|---|---|
| `STEPS` | `25000` | total training steps |
| `SAVE_FREQ` | `5000` | checkpoint + push interval |
| `BATCH_SIZE` | `200` | robot batch **per GPU** - sized for 141 GB H200 cards |
| `VL_BATCH_SIZE` | `100` | VL batch per GPU - keep at `BATCH_SIZE/2` |
| `NUM_WORKERS` | `16` | dataloader workers per GPU process |
| `OUT_DIR` | `outputs/smolvla_klal_lora_25k` | local checkpoint dir |

`BATCH_SIZE=200 / VL_BATCH_SIZE=100` is sized from two clean VRAM measurements
- batch 56 → 33.5 GB, batch 96 → 56.1 GB - which give `VRAM ≈ 1.9 GB +
0.55 GB × BATCH_SIZE`. At 200 that predicts a **~115 GB peak on a 141 GB
card (~80%)**, with ~26 GB headroom for the extrapolation. With 8 GPUs the
effective batch is 8×200 = 1600 robot / 800 VL.

**If your cards are not 141 GB**, scale `BATCH_SIZE` to your VRAM:
`BATCH_SIZE ≈ (0.80 × VRAM_GB − 1.9) / 0.55` (e.g. 80 GB → ~110; 96 GB →
~135), and keep `VL_BATCH_SIZE = BATCH_SIZE/2`. Watch `nvidia-smi` on the
first run.

This script targets a **single node** with several GPUs. For a multi-node job,
replace `--standalone` in `run_cluster.sh` with your cluster's torchrun
rendezvous arguments.

## If something fails

- `third_party/lerobot` is empty → the submodule wasn't fetched. From the
  repo root: `git pull && git submodule sync && git submodule update --init
  --recursive`. (If you cloned before the submodule pin was fixed, `git pull`
  first so it points at the v0.5.1 tag.)
- `cannot import lerobot SmolVLA` → the env isn't active or step 2 didn't
  finish - re-do step 2 (needs `third_party/lerobot` populated first).
- `nvidia-smi not found` / `no GPUs` → run on a GPU node.
- HF push errors mid-run are logged as `[WARN]` and **do not stop training** -
  the local checkpoint under `OUT_DIR` is kept; check `HF_TOKEN` write access.
