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

## Next steps

- The full run still needs the standard 200-step / larger-batch smoke per
  `launch.sh` header before the 24h job (this run was a fast 25-step check).
- VQA loss is noisy across only 3 samples — not yet a trend; watch over a
  longer run.
