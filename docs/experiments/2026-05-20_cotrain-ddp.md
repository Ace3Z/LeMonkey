# 2026-05-20 — SmolVLA cotrain: multi-GPU (DDP)

## Goal

Use both GPUs on the AWS node (2× RTX PRO 6000) for a single cotrain run.
Previously `cotrain.py` was single-process — `cuda:0` only, second card idle.

## Why not torch DistributedDataParallel

The trainer has **two forward paths**: robot batches call `policy.forward()`
(flow-matching loss), VL batches call `policy.model.vlm_with_expert.vlm(...)`
directly (VQA CE loss). DDP syncs gradients via hooks tied to the wrapped
module's `forward()` — the VL path would bypass it entirely.

Instead: **manual DDP**. Plain modules; after `loss.backward()` the gradients
are averaged across ranks by hand with `dist.all_reduce`. Correct for both
paths, matches the script's hand-rolled loop. Launched via
`torchrun --nproc_per_node=N` (launch.sh auto-detects GPU count).

## Design

- `batch_size` / `vl_batch_size` are **global** — split evenly across ranks
  (`per_gpu_bs = batch_size // world_size`), so the optimization math matches
  the validated single-GPU runs. Divisibility is asserted.
- `DistributedSampler` on both loaders → disjoint shards; `set_epoch` on reset.
- Params + buffers broadcast from rank 0 before step 0 (insurance).
- Gradient all-reduce iterates **every** trainable param in fixed order,
  materialising a zero grad where backward produced none — NCCL matches
  collectives positionally, so a rank-dependent param set would deadlock.
  (The VQA path leaves the action expert grad-less; the robot path leaves the
  LM head grad-less — without the zero-fill the call sequence would differ.)
- Grad clip runs **after** the all-reduce (every rank clips identically).
- Rank 0 only: logging, checkpoints, HF push.
- Final barrier + `destroy_process_group` happen **before** the HF push so a
  slow upload can't trip the NCCL watchdog on waiting ranks.
- `init_process_group(timeout=2h)` so periodic-save barriers never time out.

## Review

Audited by a parallel review agent (CLAUDE.md §9). It flagged the
positional-collective hazard in the gradient all-reduce (originally iterated
only params with non-`None` grad) — fixed to the fixed-order zero-fill loop
above. It also flagged NCCL-timeout risk around the final HF push and periodic
saves — fixed via the pre-push teardown and the 2h group timeout. The
per-rank-mean grad averaging (`SUM / world_size`) is standard DDP semantics and
exact here because per-GPU batches are equal (`drop_last=True`) — kept as-is.

## Smoke tests

All on the 2× RTX PRO 6000 node, `ROBOT_MAX_EPISODES` capped for speed.

- **2-GPU DDP**, 40 steps: `EXIT=0`, world_size=2, both losses fire
  (VQA `15.43 → 9.45`, flow `~0.13–0.84`), periodic checkpoints saved at
  steps 15 & 30 (save barrier holds), final checkpoint saved. No deadlock,
  no NCCL error. ~38 steps/s.
- **1-GPU regression** (`NUM_GPUS=1`), 13 steps: `EXIT=0`, VQA loss at step 0
  = `15.8639` — bit-identical to the pre-DDP single-GPU runs, confirming the
  non-distributed path is unchanged.

## Usage

`launch.sh` auto-detects GPUs (via `torch.cuda.device_count()`, so it honours
`CUDA_VISIBLE_DEVICES`) and uses `torchrun` when >1. Override with `NUM_GPUS`.

## Next steps

- Ready for the full 2-GPU run (`STEPS=30000`, `BATCH_SIZE=32`).
- During the run, confirm both GPUs show ~equal utilisation in `nvidia-smi`.
