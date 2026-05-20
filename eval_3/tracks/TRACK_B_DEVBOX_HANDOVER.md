# Track B — Dev-box handover (corrections from Brev 2026-05-19 run)

Owner: Roham · Goal: reproduce the Brev Track B Pi0.5 run on the dev box,
verify locally, then re-launch on Brev (or any future H100).

This is the **updated, post-debugging** handover. It supersedes the parts of
[`TRACK_B_BREV_HANDOVER.md`](TRACK_B_BREV_HANDOVER.md) that don't survive
contact with lerobot's actual CLI and the dataset's actual schema. The
canonical TRACK_B.md still describes the *why*; this doc captures the
*what-actually-works*.

The Brev run hit 7 distinct failures on the way to a stable training loop.
Every one of them is reflected as a concrete action here.

---

## 0 · Pre-flight (host system)

These are non-conda system requirements. Skip if already installed.

```bash
# torchcodec needs ffmpeg shared libs (libavutil.so.56-60)
sudo apt-get install -y ffmpeg
ldconfig -p | grep libavutil.so.56  # must print a path
```

If FFmpeg is missing, training crashes in the first batch with
`OSError: libavutil.so.60: cannot open shared object file`.

You also need:

- A GPU with **≥80 GB VRAM** for the recipe in this doc (batch=48 + compile +
  grad_ckpt). H100 80GB or RTX PRO 6000 Blackwell 96GB both work. On a 24 GB
  card you'd need batch=12 + grad_ckpt and probably drop `compile_model=True`.
- **≥128 GB host RAM** for the dataloader at `num_workers=2`. The Brev run
  OOM-killed at default `num_workers=4` (each worker held ~43 GB of mp4 frame
  buffers in steady state).
- Conda env `lemonkey` with editable lerobot install (`pip install -e
  third_party/lerobot`).

---

## 1 · Dataset materialisation (do this once, ~5 min)

The Pi0.5-corrected dataset lives at two places on HF:

- **`HBOrtiz/so101_eval3_track3_v3_baseline`** — full data (9 394 episodes,
  ~13 GB on disk after filtering to one camera). This is what SmolVLA reads.
- **`HBOrtiz/so101_eval3_track3_v3_pi05`** — only contains the corrected
  `meta/stats.json` (Pi0.5 quantiles), to avoid clobbering Hans's SmolVLA
  baseline.

**You must materialise a local copy that combines both**, then apply five
schema renames, before pointing lerobot-train at it. The reason: lerobot's
`LeRobotDatasetMetadata` calls HF every time it can't satisfy the request
fully from a local root, and the `_pi05` repo doesn't have the data — so
`--dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_pi05` alone hangs.

Use the prep script:

```bash
cd ~/ETH_Uni/LeMonkey
bash eval_3/scripts/prepare_dataset_track_B.sh
```

The script:

1. Downloads only `camera1 + meta + data` from
   `HBOrtiz/so101_eval3_track3_v3_baseline` via `snapshot_download` (~5 min,
   ~13 GB).  Skips `observation.images.reference` — that was added by Track 3
   augmentation v3 for the image-as-prompt era; the 2026-05-18 TA ruling
   forbids reference-image input so it's dead weight.
2. Pulls the corrected `meta/stats.json` from
   `HBOrtiz/so101_eval3_track3_v3_pi05` (LFS-aware via `hf_hub_download`).
3. Strips `observation.images.reference` from `info.json["features"]` and
   from `stats.json` keys.
4. Renames `observation.images.camera1` → `observation.images.right_wrist_0_rgb`
   in:
   - `meta/info.json["features"]`
   - `meta/stats.json` top-level
   - `meta/episodes/*.parquet` columns (all `videos/.../*` and `stats/.../*`
     columns containing `camera1`)
   - `videos/observation.images.camera1/` directory itself
5. Patches `info.json["total_frames"]` to the actual sum of parquet row
   counts (5 053 812 instead of the meta's nominal 5 053 972 — there's a 160-frame
   gap in the source data that causes IndexError ~1 400 steps into training
   when an oversampled index hits OOB).

The script is idempotent — safe to re-run.

**Why all this surgery?** Three independent reasons:

- **Pi0.5's pretrained base** (`lerobot/pi05_base`) has only 3 camera slots
  (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`). Renaming our
  wrist cam to `right_wrist_0_rgb` lands our camera on a *pretrained* slot
  rather than a fresh fourth slot. Verified against
  [`pi05_base/config.json` on HF](https://huggingface.co/lerobot/pi05_base/raw/main/config.json).
- **`--rename_map`** in lerobot is a half-solution: it renames the *batch*
  dict keys but NOT the policy's `input_features` registry. Renaming the
  dataset itself avoids that mismatch.
- **The `reference` feature** has 9 394 mp4 files we don't need and would
  bloat the local copy by ~12 GB.

After it finishes you should see at `$LOCAL_ROOT`:

```
~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_pi05/
├── data/chunk-000/file-{000..004}.parquet            (5 files, ~750 MB)
├── meta/
│   ├── info.json                       # right_wrist_0_rgb, total_frames=5053812
│   ├── stats.json                      # right_wrist_0_rgb, pi05 quantiles
│   ├── episodes/chunk-000/file-000.parquet  # renamed columns
│   └── tasks.parquet
└── videos/observation.images.right_wrist_0_rgb/
    └── chunk-{000..009}/file-*.mp4    # 9 394 mp4s, ~12 GB
```

If the dev box already has `~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_merged`
from earlier augmentation work, you may want to:

```bash
DATASET_ROOT=~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_pi05 \
    bash eval_3/scripts/prepare_dataset_track_B.sh
```

so the new pi05-ready copy lands beside the old merged one and doesn't
overwrite it.

---

## 2 · Launch-script differences vs the original `run_training_track_B.sh`

The script in `eval_3/scripts/brev/run_training_track_B.sh` has been updated
on Brev (and committed). The changes that *matter*:

| Flag | Was | Now | Why |
|---|---|---|---|
| Conda activation | implicit (assumed login shell) | explicit `source conda.sh && conda activate lemonkey` at top of script | `nohup`'d shells are non-login. Without this the script aborts with `lerobot-train: command not found`. |
| `--peft.lora_alpha` | `=32` | **removed** | Lerobot's `PeftConfig` exposes only `target_modules / method_type / r / init_type / full_training_modules`. `lora_alpha` and `lora_dropout` are PEFT defaults (alpha=r, dropout=0). |
| `--peft.lora_dropout` | `=0.05` | **removed** | Same — not exposed in lerobot's CLI. |
| `--rename_map` | `'{...camera1: right_wrist_0_rgb}'` | **removed** | Half-broken (renames batch dict but not policy `input_features`). Replaced by renaming the dataset directly in step 1. |
| `--dataset.root` | — | **added** (`$DATASET_ROOT`) | Forces lerobot to read fully from local, bypasses HF lookups + a separate lerobot bug where `RevisionNotFoundError.__init__` is missing the `response` kwarg. |
| `--num_workers` | (default 4) | **`=2`** | At default 4, each worker accumulates ~43 GB mp4 buffer RAM → kernel OOM kills a worker at ~2 h. |
| `BATCH_SIZE` | 24 | **48** | We have an 80 GB H100. batch=64 OOMs because torch.compile's CUDA-graph pool reserves ~48 GB on top of normal training memory. batch=48 + grad_ckpt + compile peaks at ~67 GB, leaves comfortable headroom. |
| `GRAD_CKPT` | True | **True** (kept, but env-driven) | Required at batch=48. Without it batch=48 also OOMs. |

The script is now env-var driven. Run it with the verified Brev settings:

```bash
HF_TOKEN="$(cat secrets/huggingface/token_hbortiz)" \
BATCH_SIZE=48 \
GRAD_CKPT=True \
NUM_WORKERS=2 \
DATASET_ROOT=~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_pi05 \
bash eval_3/scripts/brev/run_training_track_B.sh
```

The actual `lerobot-train` invocation it produces:

```bash
lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=True \
    --policy.train_expert_only=False \
    --policy.empty_cameras=2 \
    --policy.optimizer_lr=1e-5 \
    --policy.gradient_checkpointing=True \
    --policy.compile_model=True \
    --peft.method_type=LORA \
    --peft.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    --peft.r=32 \
    --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_pi05 \
    --dataset.root=$DATASET_ROOT \
    --batch_size=48 \
    --num_workers=2 \
    --steps=30000 \
    --output_dir=outputs/pi05_track_B \
    --policy.push_to_hub=True \
    --policy.repo_id=HBOrtiz/pi05_eval3_track_B
```

---

## 3 · Smoke test (must pass before the full 30 k-step run)

The Brev run survived 1 400 steps before the IndexError fired (~80 min wall),
and then survived another ~2 100 steps before the dataloader OOM-kill (~2 h).
On a fresh machine you'd like to catch both failures faster. A 100-step
smoke test catches:

- FFmpeg / torchcodec missing
- info.json / stats.json key mismatches
- IndexError (via random sampling — usually fires within 200-500 steps if
  total_frames is wrong)
- OOM-prone compile autotune

Smoke command:

```bash
HF_TOKEN="$(cat secrets/huggingface/token_hbortiz)" \
BATCH_SIZE=48 GRAD_CKPT=True NUM_WORKERS=2 STEPS=200 \
DATASET_ROOT=~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_pi05 \
OUT_DIR=outputs/pi05_track_B_smoke \
PUSH_REPO= \
bash eval_3/scripts/brev/run_training_track_B.sh 2>&1 | tee ~/smoke.log
```

(Setting `PUSH_REPO=` blanks it so the smoke run doesn't push. You may need
to add `if [ -n "$PUSH_REPO" ]; then` guards in the script — easier in
practice: just let it try to push the partial checkpoint at step 200 and
delete the repo afterwards, or kill manually before step 200 lands.)

What "passing" looks like:

- `Loaded state dict from model.safetensors` appears
- `num_learnable_params=59056128 (59M)` confirms LoRA wrap
- `Effective batch size: 48 x 1 = 48`
- Steps 1-50 take 3-5 min total (most of that is torch.compile autotune)
- After autotune, you should see ~3.5 s/step. Loss should start at ~3-5 and
  decrease.
- GPU util `nvidia-smi`: 95-100 %. Memory: 66-68 GB.
- Host RAM `free -h`: ~80-90 GB used, ~80 GB available.

If any of those don't match, stop and diagnose before launching the 30 k
run. See [§ 5 troubleshooting](#5--troubleshooting).

---

## 4 · Full training run

If the smoke test passed:

```bash
# foreground (you'll see the progress bar; ctrl-C kills it)
HF_TOKEN="$(cat secrets/huggingface/token_hbortiz)" \
BATCH_SIZE=48 GRAD_CKPT=True NUM_WORKERS=2 \
DATASET_ROOT=~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_pi05 \
bash eval_3/scripts/brev/run_training_track_B.sh

# OR background / persistent (24-30 h, survives SSH disconnect)
nohup env \
    HF_TOKEN="$(cat secrets/huggingface/token_hbortiz)" \
    BATCH_SIZE=48 GRAD_CKPT=True NUM_WORKERS=2 \
    DATASET_ROOT=~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_pi05 \
    bash eval_3/scripts/brev/run_training_track_B.sh \
    > ~/track_B.log 2>&1 &
echo $! > ~/track_B.pid
echo "Launched. PID=$(cat ~/track_B.pid). Log: ~/track_B.log"
```

Monitor:

```bash
tail -f ~/track_B.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
ps -p $(cat ~/track_B.pid)
```

Expected milestones (on H100 80GB):

| t | event |
|---|---|
| 0 | conda activate, lerobot-train fires |
| ~1 m | "Creating dataset", "Creating policy" |
| ~2 m | Pi0.5 weights loaded from cache, PEFT-wrapped, optimizer init |
| ~3-7 m | torch.compile autotune (lots of Triton kernel timing in log) |
| ~7-9 m | first training step lands |
| ~10 m | step ~50, loss ~3-4 |
| ~1 h | step ~1 000, loss ~2.0 |
| ~10 h | step ~10 000 |
| ~28-30 h | training finishes, policy pushed to `HBOrtiz/pi05_eval3_track_B` |

If you push to a fresh repo name (e.g. `HBOrtiz/pi05_eval3_track_B_v2` for
the dev-box run), set `PUSH_REPO=HBOrtiz/pi05_eval3_track_B_v2` in the env.
Don't overwrite the Brev run's output unless you mean to.

---

## 5 · Troubleshooting

Failures observed during the Brev run and their root cause:

| Symptom | Root cause | Fix |
|---|---|---|
| `lerobot-train: command not found` | nohup'd shell is non-login | the script now `source`s conda.sh + activates env at top |
| `lerobot-train: error: unrecognized arguments: --peft.lora_alpha=64` | flag doesn't exist in `lerobot/configs/default.py:PeftConfig` | removed (see step 2 table) |
| `FileNotFoundError: meta/info.json` then `TypeError: HfHubHTTPError.__init__() missing 'response'` | `_pi05` repo only contains stats.json, not the full data | use `--dataset.root=<local materialised copy>` (this doc step 1) |
| `OSError: libavutil.so.60: cannot open shared object file` | missing FFmpeg system libs | `sudo apt install -y ffmpeg` |
| `ValueError: All image features are missing from the batch` | `--rename_map` renames batch dict but not policy `input_features` | rename in dataset metadata directly (step 1.4 of prep script) |
| `KeyError: 'videos/observation.images.right_wrist_0_rgb/chunk_index'` | episodes parquet still has old `camera1` column name | step 1.4 renames the parquet columns too |
| `IndexError: Invalid key: 5053957 is out of bounds for size 5053812` | source data has 160-frame gap between meta and parquet rows | step 1.5 patches info.json `total_frames` to actual row count |
| `RuntimeError: DataLoader worker (pid X) is killed by signal: Killed.` | host RAM exhausted at default `num_workers=4` (each worker 43 GB RSS) | `--num_workers=2` (step 2 of launch flags) |
| `torch.OutOfMemoryError: Tried to allocate ... in private pools` at batch=64 | torch.compile CUDA-graph pool reserves ~48 GB extra | drop to `BATCH_SIZE=48` (max that fits with compile + grad_ckpt) |

Each one is documented in `docs/experiments/2026-05-19_track_b_brev_debugging.md`
if you need a more verbose post-mortem.

---

## 6 · Two-box plan

The Brev run is currently running (launched 2026-05-19 ~05:55 UTC,
ETA ~30 h, expected completion ~2026-05-20 ~11:00 UTC). Two completed runs
gives you a sanity comparator:

- **Brev run** pushes to `HBOrtiz/pi05_eval3_track_B`.
- **Dev-box run** push to a fresh name (`HBOrtiz/pi05_eval3_track_B_devbox`)
  so they don't collide. If their final-step losses are within 5-10 % of
  each other, the recipe is reproducible. If they diverge wildly, something
  on one of the machines is non-deterministic — most likely num_workers
  shuffling at different seeds.

Both pushed models are eligible for the Day-3 Strix rollout protocol per
[`TODO.md#strix-testing-protocol-darius`](../../TODO.md).

---

## 7 · References

- [`TRACK_B.md`](TRACK_B.md) — the *why* (LoRA recipe, validation findings,
  fallback decisions)
- [`TRACK_B_BREV_HANDOVER.md`](TRACK_B_BREV_HANDOVER.md) — the pre-debug
  Brev runbook (now partially outdated; this doc supersedes the
  recipe-mechanics parts)
- [`docs/experiments/2026-05-19_track_b_brev_debugging.md`](../../docs/experiments/2026-05-19_track_b_brev_debugging.md)
  — full post-mortem of each failure
- [`eval_3/scripts/prepare_dataset_track_B.sh`](../scripts/prepare_dataset_track_B.sh)
  — the idempotent dataset prep script described in § 1
- [`eval_3/scripts/brev/run_training_track_B.sh`](../scripts/brev/run_training_track_B.sh)
  — the updated launch script (now env-var driven)

---

*Last updated 2026-05-19. Status: handover ready; Brev run live at step ~1 000.*
