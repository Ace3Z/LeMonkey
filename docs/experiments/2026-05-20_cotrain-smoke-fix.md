# 2026-05-20 — SmolVLA cotrain dry run: action-chunk fix

## What we ran

`STEPS=25 BATCH_SIZE=4 VL_BATCH_SIZE=2 LOG_EVERY=1 bash eval_3/scripts/smolvla_cotrain/launch.sh`
— the cotrain smoke test, on the AWS Blackwell node (lemonkey conda env).

## Symptom

The previous smoke run (`smoke.log`) crashed on the first **robot** step (step 1):

```
RuntimeError: The size of tensor a (291) must match the size of tensor b (245)
at non-singleton dimension 2
  in make_att_2d_masks  (modeling_smolvla.py:132)
```

Step 0 (VL/VQA) succeeded; the action path was the failure.

## Root cause

`load_robot_dataset()` constructed `LeRobotDataset(..., delta_timestamps=None)`.
With no `delta_timestamps`, each frame yields a **single** action `(action_dim,)`
instead of an action **chunk** `(chunk_size, action_dim)`.

Tracing through `VLAFlowMatching.forward`:

- `actions` arrives as `[B, action_dim]` (2-D, no time axis).
- `time_expanded` is `[B,1,1]`; `time_expanded * noise` broadcasts
  `[B,1,1] * [B,32]` → `[B, B, 32]` — a spurious axis of size `B`.
- `embed_suffix` then produced `embs/pad` of seq-len `B` (=4) but
  `att_masks` of seq-len `config.chunk_size` (=50), because
  `att_masks += [1] * chunk_size` is hard-coded while the pad mask follows
  the actual tensor shape.
- `prefix(241) + suffix_pad(4) = 245` vs `prefix(241) + suffix_att(50) = 291`
  → exactly the reported mismatch.

Probe confirmed: `embed_suffix OUT: embs=(4,4,720) pad=(4,4) att=(4,50)`.

## Fix

`load_robot_dataset()` now builds `delta_timestamps` from the policy config
via lerobot's own `resolve_delta_timestamps(cfg, ds_meta)` — the same path
`make_dataset()` uses. For SmolVLA this resolves to action → `chunk_size`
(=50) future steps, observations → `[0]`. The dataset now yields
`action [B,50,6]`, `action_is_pad [B,50]`, and time-axed observations.

Post-fix probe: `embed_suffix OUT: embs=(4,50,720) pad=(4,50) att=(4,50)`,
`forward OK, loss=0.897`.

## Result — smoke test passes

25 steps, both losses fire:

- VQA loss (steps 0/11/22): `15.86 → 11.69 → 14.41`
- Flow loss (robot steps): oscillating `~0.20–0.91`
- ~8–9 steps/s after warmup, no OOM (102 GB VRAM, used a fraction).
- Final checkpoint written to `outputs/smolvla_cotrain_10to1/final`.

## Second bug — partial-cache missing video file

The first 200-step smoke crashed at step ~100 (a robot step), in a DataLoader
worker:

```
FileNotFoundError: .../videos/observation.images.reference/chunk-000/file-934.mp4
```

Cause: the robot dataset is a **partial download** — 9394 episodes total, only
~938 episodes' video cached. The `reference` stream has a non-contiguous gap
(`file-934.mp4` absent; 933 and 935 present). `launch.sh` capped episodes by
*counting* reference files (938) and assumed indices `0..937` were contiguous —
but 934 falls inside that range with no file. Episode↔file is 1:1 here
(verified against `meta.episodes`: `videos/.../file_index == episode_index`).

Fix:
- `cotrain.py` `_episodes_with_complete_files()` — resolves every episode's
  data + video (chunk_index, file_index) against the on-disk path templates
  and passes only fully-covered episodes as an explicit `episodes=` list. A
  partial cache can no longer crash mid-run. Emits a `[WARN]` (CLAUDE.md §5).
- `launch.sh` — removed the brittle file-count cap; coverage is now decided
  authoritatively in `cotrain.py`.

## Result — 200-step smoke passes

`STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2`, `EXIT=0`, final checkpoint saved.

- `[WARN]` fired: 9394 episodes total, 938 fully-covered, 8456 files missing —
  trained on the 938-episode subset.
- VQA loss (19 VL steps): `15.86 → 11.67 → 10.59 → … → 3.80` — clear downward
  trend.
- Flow loss: `~0.06–1.09`, no non-finite values, grad norms 3–50 (clip=10).
- ~7–8 steps/s steady-state; no OOM.

## Reference video stream downloaded — full dataset unlocked

The 938-episode cap was a *cache* limitation. Per-stream coverage audit:

| stream    | files needed | cached before | cached after |
|-----------|--------------|---------------|--------------|
| data      | 5            | 5             | 5            |
| camera1   | 9394         | 9394          | 9394         |
| reference | 9394         | 938           | **9394**     |

`reference` was the sole bottleneck. Downloaded the missing 8456 files
(~2 GB) via `hf download --include "videos/observation.images.reference/*"`
into the snapshot dir. The download hung twice (all 8 worker threads stalled
on dead connections); a retry loop with `timeout -s KILL 150` per attempt +
`HF_HUB_DOWNLOAD_TIMEOUT=20` carried it through — hf download is resumable so
each attempt picked up from disk.

Post-download, `_episodes_with_complete_files()` reports **9394/9394**
episodes covered, contiguous `0..9393`.

## Result — 200-step smoke passes on the full dataset

`STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2`, final checkpoint saved.

- `[robot_dataset] 5053812 frames across 9394 episodes` — no `[WARN]`, no cap.
- VQA loss (19 VL steps): `15.86 → 11.68 → 10.59 → … → 3.87` — clear downward
  trend, action loss not collapsed.
- Flow loss: `~0.03–0.78`, no non-finite values, grad norms 3–43 (clip=10).
- No `FileNotFoundError` — partial-cache crash gone.

## Next steps

- Pipeline is verified end-to-end on the complete 9394-episode dataset.
  Ready for the full run (`STEPS=30000`, `BATCH_SIZE=32`).
- During the long run, watch that VQA loss keeps trending down without
  spiking the flow loss.
