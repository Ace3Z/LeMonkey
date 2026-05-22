# Track B Brev debugging post-mortem (2026-05-19)

Owner: Claude / Roham · Run: Track B Pi0.5-LoRA on Brev H100 80GB.

This document captures the 7 failures hit between **2026-05-19 00:54 UTC**
(first launch attempt) and **2026-05-19 05:55 UTC** (stable training start)
when getting `lerobot-train` to produce its first loss line on Brev. The
goal is to make these failures cheap to diagnose on the dev box and on any
future H100.

Currently-stable launch as of 06:00 UTC: `BATCH_SIZE=48 GRAD_CKPT=True
NUM_WORKERS=2`, ~3.55 s/step on H100 80GB, ETA ~30 h.

---

## Failure 1 — `lerobot-train` not on PATH (00:55 UTC)

```
eval_3/scripts/brev/run_training_track_B.sh: line 49: lerobot-train: command not found
```

**Cause.** The launch script was being `nohup`'d from a parent shell that
had conda activated. The `nohup env ... bash script.sh` spawns a non-login,
non-interactive child shell that does NOT inherit conda hooks, so `conda
activate lemonkey` was never run and `lerobot-train` was not on PATH.

**Fix.** Bake conda activation into `run_training_track_B.sh` itself, with
a `[WARN]` if conda.sh is missing.

---

## Failure 2 — CLI rejected unsupported `--peft.lora_alpha` and friends (00:39 UTC)

```
unrecognized arguments: --peft.lora_alpha=64 --peft.lora_dropout=0.05
```

**Cause.** [TRACK_B.md](../../eval_3/tracks/TRACK_B.md) §4 documents the
ideal LoRA config including `lora_alpha=32` and `lora_dropout=0.05`. But
lerobot's `PeftConfig` at
[`third_party/lerobot/src/lerobot/configs/default.py:85-130`](../../third_party/lerobot/src/lerobot/configs/default.py)
only exposes 5 fields: `target_modules`, `full_training_modules`,
`method_type`, `init_type`, `r`. PEFT defaults take over for the rest:

- `lora_alpha` defaults to `r` (so r=32 → alpha=32 effective)
- `lora_dropout` defaults to 0

**Fix.** Removed the unsupported flags. Documented this in TRACK_B.md
inline note that PEFT defaults are used.

Same applies to `--dataset.rename_map=...` (top-level `--rename_map`, not
under `--dataset`), though we removed that flag entirely for a different
reason (Failure 5).

---

## Failure 3 — HF dataset `_pi05` repo has no data, only stats override (00:54 UTC)

```
FileNotFoundError: meta/info.json
... during fallback HF lookup ...
TypeError: HfHubHTTPError.__init__() missing 1 required keyword-only argument: 'response'
```

**Cause.** To avoid clobbering Hans's SmolVLA-baseline dataset, we'd
pushed only the Pi0.5-corrected `meta/stats.json` to a NEW repo
`HBOrtiz/so101_eval3_track3_v3_pi05`. The full data (parquets + mp4s) lives
in `HBOrtiz/so101_eval3_track3_v3_baseline`. When lerobot couldn't satisfy
its metadata request from the `_pi05` repo, it tried to fall back via
`get_safe_version` → which hit a separate lerobot bug (the
`RevisionNotFoundError` is raised with a missing `response` kwarg in
huggingface_hub).

**Fix.** Materialise a local copy that combines both: snapshot_download
baseline, then swap in the pi05 stats.json. Pass `--dataset.root=<local>`
to lerobot-train. The `LeRobotDatasetMetadata.__init__` short-circuits the
HF lookup if local metadata loads successfully.

This also requires filtering the snapshot_download to `camera1 + meta +
data` only — skipping the `reference` video stream (~9 394 mp4s, ~12 GB)
that was added by Track 3 augmentation v3 for the image-as-prompt
deprecated approach.

---

## Failure 4 — torchcodec failed to load FFmpeg shared libs (01:30 UTC)

```
OSError: libavutil.so.60: cannot open shared object file: No such file or directory
... (and so.59, so.58, so.57, so.56)
```

**Cause.** Lerobot uses
[`torchcodec`](https://github.com/pytorch/torchcodec) to decode mp4 frames
during training. torchcodec ships precompiled binaries against FFmpeg 4-7
shared libs (libavutil.so.56-60) but does NOT bundle FFmpeg itself. The
Brev VM didn't have FFmpeg installed.

**Fix.** `sudo apt-get install -y ffmpeg` (Ubuntu 22.04's repo has FFmpeg
4.4.2 → libavutil.so.56 → torchcodec_core4 loads).

---

## Failure 5 — `--rename_map` half-renames the dataset (01:44 UTC)

```
ValueError: All image features are missing from the batch.
  (batch: dict_keys([..., 'observation.images.right_wrist_0_rgb', ...]))
  (image_features: {'observation.images.camera1': ..., ...})
```

**Cause.** Lerobot's `--rename_map` flag renames the keys of the *batch
dict* being passed to the policy, but the policy's static `input_features`
registry is built from the dataset's info.json features at construction
time. The policy ended up expecting `camera1` while the batch had
`right_wrist_0_rgb`.

**Fix.** Don't use `--rename_map`. Instead, rename the dataset feature
directly in `meta/info.json["features"]`, `meta/stats.json` top-level, the
`meta/episodes/*.parquet` columns, and the `videos/` directory itself.
This is what the policy reads at startup.

The rename target (`right_wrist_0_rgb`) was chosen because
`lerobot/pi05_base` has 3 pretrained camera slots
(`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`) per its config.json;
landing our wrist cam on the pretrained `right_wrist_0_rgb` slot is better
than creating a 4th uninitialised slot.

---

## Failure 6 — episodes parquet still references old column name (01:47 UTC)

```
KeyError: 'videos/observation.images.right_wrist_0_rgb/chunk_index'
```

**Cause.** I renamed the feature key in info.json and stats.json but
forgot the episodes parquet, which uses dotted column names like
`videos/observation.images.camera1/chunk_index` to track file-shard
locations per episode.

**Fix.** Rename those columns too via `pyarrow.Table.rename_columns`,
matching every column whose name contains `observation.images.camera1`
(both `videos/.../*` and `stats/.../*` columns).

---

## Failure 7 — info.json total_frames overstates parquet rows by 160 (~02:00 UTC, fired ~03:30)

```
IndexError: Invalid key: 5053957 is out of bounds for size 5053812
```

**Cause.** `info.json["total_frames"]` reports 5 053 972, but the actual
parquet row count is 5 053 812 — a 160-row gap in the source data
(present in `HBOrtiz/so101_eval3_track3_v3_baseline` itself, not caused by
my renames). Lerobot's sampler uses meta's `total_frames` to compute the
valid index range; with random sampling at batch=48, hitting an OOB index
happens with roughly probability ~3.2 × 10⁻⁵ per sample → expected first
hit around step 600-1500. The Brev run made it to step 1404.

**Fix.** Patch info.json `total_frames` to the actual sum of parquet row
counts (5 053 812). Should also report this upstream to the dataset author.

---

## Failure 8 — DataLoader worker OOM-killed by kernel (05:48 UTC, after 2h13m)

```
RuntimeError: DataLoader worker (pid 61138) is killed by signal: Killed.
ConnectionResetError: [Errno 104] Connection reset by peer
```

`dmesg` confirmed kernel OOM-killer:

```
oom-kill:... task=pt_data_worker,pid=61138... uid=1001
Out of memory: Killed process 61138 (pt_data_worker)
  total-vm:79272196kB, anon-rss:43230644kB
```

**Cause.** Each DataLoader worker held ~43 GB of RAM in steady state —
decoded mp4 frame buffers accumulated by torchcodec. At lerobot's default
`num_workers=4`, total RAM from workers alone is ~172 GB. Brev has 177 GB.
We were sitting at ~99% RAM utilization for ~2 h before the kernel killed
one worker.

This isn't strictly a leak — torchcodec's buffer pool grows up to its
configured limit and stays there. But 43 GB per worker × 4 workers is
above the safe limit.

**Fix.** `--num_workers=2`. This halves both peak RAM (~86 GB workers
total) and per-step prefetch, but doesn't slow down training because GPU
is the bottleneck at 100% util.

A more principled fix would be to drop torchcodec's per-worker buffer
ceiling, but that's lerobot-internal and not exposed.

---

## Side notes worth remembering

### Memory budget at the stable settings

```
Brev H100 80GB, batch=48, num_workers=2, grad_ckpt=True, compile=True
  GPU:  66-68 GB / 80 GB  (CUDA graph pool reserves ~48 GB, weights+activations ~18-20 GB)
  Host: ~90 GB / 177 GB   (2 dataloader workers × ~43 GB + everything else)
```

### Things that DID work first try (don't break them)

- `--policy.dtype=bfloat16`
- `--policy.freeze_vision_encoder=True`
- `--policy.train_expert_only=False`
- `--policy.empty_cameras=2`
- `--policy.compile_model=True`
- `--peft.method_type=LORA --peft.r=32 --peft.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'`
- `--steps=30000`

### Batch-size ladder explored

| Batch | grad_ckpt | compile | Outcome |
|---|---|---|---|
| 24 | True | True | works (TRACK_B.md baseline) |
| 64 | False | True | OOM in autotune (79 GB used, hit 80 GB ceiling) |
| 64 | True | True | OOM in autotune (CUDA graph pool eats 48 GB) |
| 48 | True | True | **stable**, ~67 GB peak |

### Things deferred

- **VQA / web co-train (M5)** — blocked by lerobot's `MultiLeRobotDataset =
  NotImplementedError`. Would need 3-5 days of plumbing.
- **FAST CE on VLM LM head (M4)** — would need to integrate
  `physical-intelligence/fast` tokenizer + intercept actions pre-flow-
  matching + add LM-head CE. No lerobot integration exists.
- **Pi0.5-KI stop-gradient (M3)** — verified nobody has ported it; would
  need patches to `modeling_pi05.py`. Skipped per pivot decision since M4
  is also missing (M3 alone gives ~0% per Pi0.5-KI Fig 4a/6b).

The current run is therefore a "LoRA-only" Pi0.5 fine-tune. Whether that's
enough depends on whether PaliGemma's WebLI prior already contains the 9
celebrity identities — see [TRACK_B.md §3 Validation 1](../../eval_3/tracks/TRACK_B.md#validation-1)
for the literature review.

---

## Verification (smoke test) recommended on dev box

Before the full 30 k-step run:

```bash
HF_TOKEN="$(cat secrets/huggingface/token_hbortiz)" \
BATCH_SIZE=48 GRAD_CKPT=True NUM_WORKERS=2 STEPS=200 \
DATASET_ROOT=~/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_pi05 \
OUT_DIR=outputs/pi05_track_B_smoke \
bash eval_3/scripts/brev/run_training_track_B.sh 2>&1 | tee ~/smoke.log
```

If steps 1-200 land cleanly (loss decreases from ~3 to ~2.5), the recipe
is reproducible on the dev box.

---

*Last updated 2026-05-19 ~06:00 UTC. Training continues; will update with
final loss + push status when run completes.*
