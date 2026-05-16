# torchcodec OOM during SmolVLA training on LeRobot v3 dataset

**Date:** 2026-05-15 → 2026-05-16
**Reporter:** Roham (LeMonkey team, ETH RC FS26 Project 1)
**Status:** workaround in place (swap to `pyav` backend); root cause not yet confirmed in torchcodec; **bug report not yet filed**

## TL;DR

Training SmolVLA on a 4,195-episode LeRobot v3 dataset (8,390 unique mp4 files, 2.26 M frames total) caused the host RAM to climb monotonically until the OOM killer fired on a DataLoader worker. The leak was **per-worker** and grew proportional to the number of distinct mp4s opened over time. **Swapping `--dataset.video_backend=pyav` (libav-based Python bindings, the older default) for the lerobot 0.5.1 default `torchcodec` completely resolved the issue**: the 8.3 h, 30 k-step training then ran to completion with stable per-worker RSS.

We did not file the bug upstream yet — this report is the seed for that filing.

## Environment

| Field | Value |
|---|---|
| Host | Brev cloud VM `daddy-sejohn` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 GB VRAM, 177 GB host RAM) |
| OS | Ubuntu 22.04 (Linux 6.8) |
| Python | 3.12 (miniconda env `lemonkey`) |
| Driver | NVIDIA 580.126.09 |
| PyTorch | whatever `lerobot[smolvla]==0.5.1` pulled — likely 2.5.x |
| LeRobot | 0.5.1 (PyPI) |
| torchcodec | bundled with LeRobot 0.5.1 — exact version not captured (TODO before filing) |
| ffmpeg | system, 4.4.2-0ubuntu0.22.04.1 |

## What we were running

```
python -u lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.add_image_special_tokens=true \
  --policy.empty_cameras=1 \
  --policy.optimizer_lr=5e-5 \
  --policy.use_amp=true \
  --dataset.repo_id=local/so101_eval3_all \
  --dataset.root=/home/shadeform/LeMonkey/datasets/eval3_merged \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.reference": "observation.images.camera2"}' \
  --batch_size=64 \
  --steps=30000 \
  --num_workers=<4 or 8 — see below> \
  --wandb.enable=false
  # NOTE: torchcodec is the default — no --dataset.video_backend flag
```

Dataset characteristics that may be relevant:

- **8,390 unique mp4 files** (4,195 episodes × 2 cameras, one mp4 per camera per episode)
- Each mp4 is H.264, ~530 frames, 480×640 (camera1) or 480×480 (camera2/reference)
- Set deliberately to "one mp4 per episode per camera" via `video_files_size_in_mb=0.01` to avoid PyAV bitstream-concat issues at merge time (see `eval_3/aug/STRATEGY_v3.md`).
- Iteration order during training is shuffled across all 4,195 episodes — so every worker opens many distinct files quickly.

Of note: the reference camera mp4 is a **constant-frame** video (the same celebrity photo repeated 530× per episode). That's an unusual stream but should be a benign edge case for any decoder.

## Symptoms

- Training kicks off fine. GPU util 100 %, VRAM holds steady at ~83 GB / 97 GB.
- Host RAM starts climbing within the first ~10 minutes. Climb is monotonic and roughly linear.
- Around minute ~30, the kernel's OOM killer fires on a single `pt_data_worker` process.
- After the worker dies, the DataLoader either hangs or PyTorch raises `RuntimeError: DataLoader worker (pid <N>) is killed by signal: Killed.`
- No CUDA error, no `MemoryError` from Python — kill comes from the host kernel.

## Forensic evidence

From `dmesg` immediately before/after the kill (logged at the time, since lost to VM deletion):

| Run | `--num_workers` | Peak `anon-rss` on a single `pt_data_worker` | Time-to-OOM |
|---|---|---|---|
| 1 | 4 | **34.9 GB** | ~30 min |
| 2 | 8 | **17.9 GB** (per worker) — still climbing when killed | ~30 min |

Critical observation: **with 4 workers the per-worker RSS reached ~35 GB; with 8 workers it was ~18 GB at the same wall-clock point. The leak rate per worker scales inversely with worker count.** That is exactly what you'd expect if (a) the leak is per-decode (or per-file-open), and (b) total decode load is split across workers — so each worker individually leaks at half the rate when there are twice as many.

The aggregate "leak rate" (sum across workers) is comparable in both runs — the kernel just killed whichever single worker breached its slice of host RAM first.

## Root-cause hypothesis (unverified)

We hypothesize that `torchcodec.decoders.VideoDecoder` (or whichever lerobot-internal wrapper of it) does **not** release the underlying libav decoder context when a worker transitions from one mp4 to the next during iteration.

Specifically, our dataset has the property that successive `__getitem__` calls from the LeRobotDataset iterator tend to open *different* mp4 files (because we have 8,390 of them and the dataloader shuffles). If the previous file's decoder context isn't `avcodec_close`'d / freed before the next is opened, every iteration leaks one decoder context.

Concretely the chain is something like:

1. LeRobotDataset reads a row → resolves `videos/observation.images.camera1/chunk-NNN/file-MMM.mp4` and a `from_timestamp`.
2. Internally instantiates (or fetches from a per-worker cache) a `VideoDecoder` for that mp4.
3. Reads frames.
4. **Doesn't explicitly close** the decoder — relies on Python `__del__` / GC.
5. The decoder's underlying libav context (a C struct allocated via `avcodec_alloc_context3`) doesn't get freed promptly — possibly held by a Python-side cache, possibly retained because of FFI ref-counts.

The "doesn't get freed" step is what would need to be confirmed by reading torchcodec's source.

Why we suspect torchcodec and not LeRobot:

- The fix that worked is a *single flag* (`--dataset.video_backend=pyav`) that switches the same lerobot code path to use the older `pyav` (Python bindings for libav) decoder library. Same LeRobot iterator, same code paths, same dataset, same dataloader, same workers — only the decoder library changes. So whatever is leaking lives in the decoder library.
- `pyav` is a long-established, well-tested wrapper that has been used in dozens of video-ML projects without leaks of this magnitude.

## Workaround that fixed it

```diff
- # default: torchcodec
+ --dataset.video_backend=pyav
```

With pyav, the same 30 k-step run completed cleanly in ~8 h. Final loss 0.018, checkpoints at 5 k / 10 k / 15 k / 20 k / 25 k / 30 k pushed to `HBOrtiz/smolvla_eval3` on HF.

We also kept `--num_workers=8` (was dropped to 4 with torchcodec in a previous attempt to slow the bleed); with pyav, 8 workers was stable. We also added `--property=LimitNOFILE=524288` to the systemd-run launcher because pt_data_workers + mmapped parquet/video shards hit the 1024 default FD limit at step ~40 (separate issue, not torchcodec-related).

## Minimal reproducer (to build before filing)

This is the next step — none has been built yet. Sketch:

```python
# repro_torchcodec_leak.py
# Goal: reproduce the ~10 MB/iteration host-RAM growth without involving LeRobot.
import gc, os, resource, glob, time
import torchcodec
from torchcodec.decoders import VideoDecoder

mp4s = sorted(glob.glob('/path/to/many/different/mp4s/*.mp4'))
print(f'iterating over {len(mp4s)} mp4 files; pid={os.getpid()}')

for i, p in enumerate(mp4s * 10):     # multi-pass over the set
    dec = VideoDecoder(p)
    frame = dec.get_frame_at(0)        # touch at least one frame
    del dec
    gc.collect()
    if i % 100 == 0:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # KiB
        print(f'iter {i:6d}  rss={rss/1024:.1f} MB')
```

Expected (if bug confirmed):
- RSS climbs ~10 MB / 100 iterations, never stabilizes.
- After 10 k iterations: ~1 GB. Etc.

Run with:
- Many distinct files (≥ 1000) to defeat any per-path cache.
- `gc.collect()` after each `del` to rule out gc latency.
- Try with `tracemalloc` to confirm Python-side allocation is not the culprit.
- Then compare with the same loop using `av.open(p)` (pyav) — should be flat.

## To do before filing the upstream issue

1. **Capture the torchcodec version** that was bundled with lerobot 0.5.1. (We can pin it from the PyPI metadata even though the VM is gone.)
2. **Build the minimal reproducer above** and confirm the leak exists *without* LeRobot in the chain. (~1-2 h.)
3. **Profile with `tracemalloc` and `pympler`** to localize the leak (Python-side cache vs FFI / C-context).
4. **Read torchcodec's `VideoDecoder` cleanup path** to look for the missing `avcodec_close` or equivalent.
5. **File issue at https://github.com/pytorch/torchcodec/issues** with: env, reproducer, RSS-growth measurements, our dmesg/anon-rss numbers from this report, hypothesized cause.
6. If the maintainers concur, **submit a PR** to fix the cleanup. (Bigger time investment; gate on issue triage outcome.)

## Pointers

- The script that actually triggered this: [`eval_3/scripts/brev/run_training.sh`](../scripts/brev/run_training.sh) — see the long header comment for the live diagnostic notes captured during the incident.
- The dataset: [`HBOrtiz/so101_eval3_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_all) — pulling and iterating this exact dataset will reproduce the workload.
- The successful (post-fix) training log: [`smolvla_eval3.log`](smolvla_eval3.log) in this same dir — for "this is what a healthy 30 k-step run looks like" reference.
