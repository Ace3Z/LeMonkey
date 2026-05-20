# Run the KLAL + LoRA co-training on a multi-GPU cluster

Step-by-step for running the 25k-step SmolVLA co-training (10:1 robot:VL,
KLAL attention supervision + LoRA) on a fresh cluster with several H200s.
Everything the run needs is in the repo — no extra data to fetch by hand.

The only thing you must supply is a **HuggingFace token with write access**
(checkpoints are pushed to HF every 5k steps).

---

## 1. Clone the repo + check out the branch

`third_party/lerobot` is a git submodule — clone **with submodules**:

```bash
git clone --recurse-submodules https://github.com/Ace3Z/LeMonkey.git
cd LeMonkey
git checkout dev/mahbod/kl-divergence
git submodule update --init --recursive   # safe to re-run; ensures lerobot is present
```

Branch: **`dev/mahbod/kl-divergence`**.

---

## 2. Set up the Python environment (one-time)

Python 3.10–3.12. Conda or venv — either works.

```bash
conda create -y -n cotrain python=3.11 && conda activate cotrain
# (or: python -m venv .venv && source .venv/bin/activate)

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
conda-based bootstrap example if your cluster matches it — but the two `pip`
commands above are the portable path.)

---

## 3. Export your HuggingFace token + push target

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx          # must have WRITE access
export PUSH_REPO=youruser/smolvla_klal_lora_25k   # HF model repo for checkpoints
```

`PUSH_REPO` is created automatically if it doesn't exist. Checkpoints land at
`PUSH_REPO/step_005000`, `/step_010000`, …, `/final`.

The two training datasets (`HBOrtiz/so101_eval3_track3_v3_baseline`,
`HBOrtiz/eval3_objectvla_vl_pairs`) and the base model (`lerobot/smolvla_base`)
are public — they download automatically, no token needed for them.

---

## 4. Run

```bash
bash eval_3/scripts/smolvla_cotrain/run_cluster.sh
```

That's it. The script:
- autodetects **every GPU** on the node and launches one process per GPU
  (`torchrun`, manual data-parallel gradient all-reduce);
- extracts the bundled KLAL data (`m2_klal_data.tar.zst`, already in the repo);
- trains **25,000 steps**, 10:1 robot:VL, with KLAL + LoRA;
- saves a checkpoint and **pushes it to HF every 5,000 steps** (+ a final one).

**First run downloads ~15 GB of datasets** before training starts — this is
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

- `flow_loss` — robot action loss; should trend down.
- `vqa_loss` — VL VQA loss (every 11th step); should trend down.
- `klal` — attention-supervision loss; non-zero and finite.

Every 5k steps: `[cotrain] checkpoint saved → …` then `[cotrain] pushed → https://huggingface.co/…`.

---

## Tunables (override by exporting before step 4)

| Env var | Default | Notes |
|---|---|---|
| `STEPS` | `25000` | total training steps |
| `SAVE_FREQ` | `5000` | checkpoint + push interval |
| `BATCH_SIZE` | `48` | robot batch **per GPU** — measured: 65.5 GB peak on an 80 GB card |
| `VL_BATCH_SIZE` | `24` | VL batch per GPU |
| `NUM_WORKERS` | `8` | dataloader workers per GPU process |
| `OUT_DIR` | `outputs/smolvla_klal_lora_25k` | local checkpoint dir |

`BATCH_SIZE=48 / VL_BATCH_SIZE=24` was measured at **65.5 GB peak on an 80 GB
card (~80%, no OOM over 220 steps)** — a good fit with safe headroom. Going to
64/32 extrapolates past 80 GB and risks OOM; 48/24 is the recommended setting
for 80 GB cards. With 7 GPUs the effective batch is 7×48 = 336 robot / 168 VL.

This script targets a **single node** with several GPUs. For a multi-node job,
replace `--standalone` in `run_cluster.sh` with your cluster's torchrun
rendezvous arguments.

## If something fails

- `cannot import lerobot SmolVLA` → the env isn't active or step 2 didn't
  finish — re-do step 2.
- `nvidia-smi not found` / `no GPUs` → run on a GPU node.
- HF push errors mid-run are logged as `[WARN]` and **do not stop training** —
  the local checkpoint under `OUT_DIR` is kept; check `HF_TOKEN` write access.
