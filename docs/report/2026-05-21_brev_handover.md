# Brev 8×H100 — KLAL + LoRA co-train: handover & setup findings

**Date:** 2026-05-21 · **Branch:** `dev/mahbod/kl-divergence` · **Instance:** Brev `final`

This is the handover for running the **Track T2** co-training run (SmolVLA 10:1
co-train + KLAL attention supervision + LoRA) on the Brev 8×H100 instance. It
records the instance state, the setup gotchas hit along the way (so they are not
re-hit), and the exact commands to launch the run.

---

## TL;DR — current state

| Thing | State |
|---|---|
| Instance `final` | 8× H100 PCIe **80 GB**, 252 vCPU, 1.4 TB RAM, `/ephemeral` 6.3 TB |
| Conda env | `cotrain` (Python 3.12.13) at `/ephemeral/miniconda3` — torch 2.10.0+cu128, lerobot 0.5.1 ✅ verified (`cuda True 8 GPUs`) |
| Repo | `/ephemeral/LeMonkey` — git clone, branch `dev/mahbod/kl-divergence`, SSH remote |
| HF token | `eval_3/scripts/smolvla_cotrain/.hf_token` in place (gitignored) |
| KLAL data | pre-extracted to `eval_3/scripts/smolvla_cotrain/m2_klal_data/m2bundle` ✅ |
| Run | **not yet launched** — probe + full-run commands below |

Everything is staged. The run has **not** been started — that is the next step
(§ "How to run").

---

## What this run is

Track **T2** of [`EVAL_3_FINAL_PLAN.html`](EVAL_3_FINAL_PLAN.html): vanilla
SmolVLA 10:1 robot:VL co-training **+ KLAL attention supervision + LoRA**.

- **L_action** — SmolVLA flow-matching action loss (robot steps).
- **L_vl** — VQA cross-entropy on SmolVLM-2's LM head (every 11th step).
- **L_attn (KLAL)** — KL-divergence loss pushing the name-token's attention onto
  the prompted celeb's face bbox (robot steps).
- 25,000 steps, checkpoint + HF push every 5,000.
- **KLAL λ = 1.0** — chosen deliberately (the KLAL paper's value).
  `EVAL_3_FINAL_PLAN.html` §2 recommends 0.05–0.2 warmed-from-0; λ=1.0 makes
  KLAL ≈ 89 % of the robot-step loss. Watch the loss curves for instability.

---

## The instance

- **GPUs:** 8× H100 PCIe, 80 GB each (`81559 MiB`). PCIe, not SXM — inter-GPU is
  PCIe, but LoRA gradients are small so the all-reduce is cheap.
- **`/ephemeral` (6.3 TB)** is the working volume. The root disk is only 97 GB —
  **everything heavy lives on `/ephemeral`**: the conda env, the repo, the HF
  cache, checkpoints.
- **Conda env `cotrain`** — use it one of two ways:
  - PATH prefix (no shell init needed):
    `export PATH=/ephemeral/miniconda3/envs/cotrain/bin:$PATH`
  - or `source /ephemeral/miniconda3/etc/profile.d/conda.sh && conda activate cotrain`

---

## Setup findings / gotchas (already worked around — FYI)

1. **conda `defaults` channels need a ToS now.** A non-interactive
   `conda create` against `pkgs/main` / `pkgs/r` fails with
   `CondaToSNonInteractiveError`. Fixed by creating the env from conda-forge
   only: `conda create ... -c conda-forge --override-channels`.
2. **`conda activate` does not reliably switch `pip` in a non-interactive
   script**, and an env created without an explicit `pip` package has none — so
   a bare `pip` silently falls through to the system Python 3.10. **Always call
   the env's interpreter by absolute path:**
   `/ephemeral/miniconda3/envs/cotrain/bin/python -m pip ...`.
3. **No `unzstd` binary** on the instance (only `zstd`). `run_cluster.sh` calls
   `tar --use-compress-program=unzstd`. The KLAL bundle is **already
   extracted**, so `run_cluster.sh` skips it. To re-extract manually:
   `zstd -dc m2_klal_data.tar.zst | tar -xf - -C m2_klal_data`.
4. **Step-0 DataLoader hang** (seen on the friend's SLURM run) was the HF
   tokenizer fork deadlock — the VL collator runs a fast tokenizer inside forked
   workers. Fixed in commit `45b076c`: `TOKENIZERS_PARALLELISM=false` set in
   `cotrain.py` (module top) and exported in `run_cluster.sh`.
5. **LeRobot's dataset cache = `$HF_HOME/lerobot`** (`HF_LEROBOT_HOME` derives
   from `HF_HOME`). Export **`HF_HOME=/ephemeral/hf_cache`** so the ~15 GB of
   datasets land on `/ephemeral`, not the 97 GB root.
6. **GitHub auth on the instance is SSH** (`~/.ssh/id_ed25519`, registered to
   `Ace3Z`). HTTPS git fails — use `git@github.com:` URLs.

---

## Resource sizing — how the run is maximized

- **All 8 GPUs** — `NGPU=8`. Dedicated instance, no reason to idle one.
- **`BATCH_SIZE=110 / VL_BATCH_SIZE=55`** per GPU. From the measured VRAM curve
  `VRAM ≈ 1.9 + 0.565 × batch` (smoke points: batch 56 → 33.5 GB, 96 → 56.1 GB),
  batch 110 predicts a **~64 GB peak on an 80 GB card ≈ 80 %**.
  ⚠️ This is formula-extrapolated — **the probe (Step 1) confirms it on this
  instance**. If the probe's `PEAK` line is above ~74 GB, lower `BATCH_SIZE`.
- **`NUM_WORKERS=12`** — 8 ranks × 2 dataloaders × 12 = 192 workers, comfortably
  under 252 vCPU.
- Effective batch = 8 × 110 = **880 robot / 440 VL**.

---

## How to run

All commands run **on the instance**, from `/ephemeral/LeMonkey`.

```bash
cd /ephemeral/LeMonkey
export PATH=/ephemeral/miniconda3/envs/cotrain/bin:$PATH
export HF_HOME=/ephemeral/hf_cache
export HF_TOKEN=$(tr -d ' \t\r\n' < eval_3/scripts/smolvla_cotrain/.hf_token)
export TOKENIZERS_PARALLELISM=false
```

### Step 1 — VRAM probe (~15 min: confirms the batch fits + warms the dataset cache)

Direct `cotrain.py` call, 30 steps, **no HF push** (`--push_to_hub_repo` omitted):

```bash
M2B=eval_3/scripts/smolvla_cotrain/m2_klal_data/m2bundle
torchrun --standalone --nproc_per_node=8 eval_3/scripts/smolvla_cotrain/cotrain.py \
  --robot_dataset=HBOrtiz/so101_eval3_track3_v3_baseline \
  --vl_manifest=HBOrtiz/eval3_track3_vl_pairs \
  --pretrained_path=lerobot/smolvla_base \
  --steps=30 --save_freq=99999 \
  --batch_size=110 --vl_batch_size=55 --vl_ratio=10 \
  --lr=5e-5 --num_workers=12 --output_dir=/ephemeral/probe_out \
  --enable_lora --lora_r=16 --lora_alpha=32 \
  --enable_klal --klal_layers=10,12,14 --klal_lambda=1.0 --klal_sigma=1.0 \
  --face_labels_dir=$M2B/face_labels --celeb_manifest=$M2B/celeb_embeddings.json \
  --aug_root=$M2B/aug --episode_mapping=$M2B/episode_mapping.json 2>&1 | tee /ephemeral/probe.log
```

First run downloads ~15 GB of datasets — normal. Check the output for:
- `PEAK at batch 110/55: <N> MiB of 81559` — **N should be ≲ 74000**. If higher,
  drop `BATCH_SIZE` (and `VL_BATCH_SIZE = BATCH_SIZE/2`) and re-probe.
- `step ...` lines appearing — confirms the step-0 hang is gone.

### Step 2 — full 25k-step run (detached, ~12 h)

```bash
NGPU=8 BATCH_SIZE=110 VL_BATCH_SIZE=55 NUM_WORKERS=12 \
  PUSH_REPO=HBOrtiz/smolvla_klal_lora_25k \
  nohup bash eval_3/scripts/smolvla_cotrain/run_cluster.sh > /ephemeral/cotrain_run.log 2>&1 &
tail -f /ephemeral/cotrain_run.log
```

- `run_cluster.sh` reads `HF_TOKEN` from `.hf_token` automatically.
- `PUSH_REPO` — the HF repo for checkpoints (created if missing). Checkpoints
  land at `step_005000`, `step_010000`, …, `final`.
- Use `BATCH_SIZE` from whatever the probe confirmed.

### Monitoring

- Per-step log lines (rank 0): `step N flow_loss=... (last flow=.. vqa=.. klal=..) grad=.. steps/s=.. eta=..`.
- `nvidia-smi` — all 8 GPUs should show ~60–65 GB used and high utilisation.
- Every 5k steps: `[cotrain] checkpoint saved → …` then `[cotrain] pushed → https://huggingface.co/…`.
- ~12 h for 25k steps at ~0.6 steps/s.

---

## Open items / risks

- **KLAL λ = 1.0** — KLAL dominates the robot-step loss ~8:1. If `klal` loss
  diverges or `flow` stops decreasing, λ is the first knob to drop.
- **Batch 110 is formula-extrapolated** — Step 1's probe is the real check.
- **KLAL layers 10/12/14** are a "mid-late retained layers" heuristic, not the
  SmolVLA routing probe `EVAL_3_FINAL_PLAN.html` §7 calls for.
- The instance is a dedicated cloud box billed while up — stop it when the run
  and checkpoint pulls are done.
