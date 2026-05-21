# KLAL moved to the VL forward pass — rewrite + validation

**Date:** 2026-05-21 · **Branch:** `dev/mahbod/kl-divergence` · **Instance:** Brev 8×H100

## What changed and why

KLAL (the `L_attn` attention-supervision loss for Track T2) was computed on the
**robot-action forward pass**, with its bbox target built from the **M2 bundle**
(per-frame face-detector boxes). That data path was built against an older,
wrong dataset version.

It now runs on the **VL forward pass**: the VL dataset `eval3_track3_vl_pairs`
feeds **both** `L_vl` (VQA) and `L_attn` (KLAL), with the attention target built
from its `quad_corners_norm` column (the printed-portrait quad). This matches
`EVAL_3_FINAL_PLAN.html` §6 and the ObjectVLA-style recipe. The Eval-3 rig uses
a static top-down camera, so a frame-0 portrait quad is valid for the whole
episode (see memory `eval3-static-topdown-camera`).

**New module `eval_3/aug/m2_klal_vl.py`** — `KLALHookSetSmolVLMVL` captures
q/k + `rotary_emb` on SmolVLM2's text model (a stock `LlamaModel`, verified
against installed transformers) and recomputes name-token→image-patch
attention with Llama RoPE + a **causal mask** (the VL forward is a causal LM,
unlike the bidirectional robot prefix). `quad_to_patch_mask` rasterises the
quad onto a clean 8×8 grid — the collator now passes `do_image_splitting=False`
so each image is exactly 64 contiguous `<image>` tokens.

`cotrain.py`: VL collator surfaces `quad_corners_norm` / `celeb_slug` /
`bbox_refit_ok`; the VL step computes `loss = vqa + λ·klal`; the robot-step
KLAL block and the M2-bundle args/builder are removed. `run_cluster.sh` /
`launch.sh` updated accordingly.

## Research grounding

- **SmolVLM2 text model** is a stock transformers `LlamaModel` — shared
  `rotary_emb`, `apply_rotary_pos_emb`, GQA 15/5 heads, causal. `output_attentions`
  is too memory-heavy (~58 GB); hook q/k + recompute instead (bit-exact verified).
- **ObjectVLA's 10:1** (arXiv 2502.19250 §3.3, §7.2) is a **data-size ratio**
  for *simple object detection* co-training; the paper does not specify the
  per-batch mechanism. For Eval-3's harder face-grounding task we use **5:1**
  (2× the VL of ObjectVLA, still 83% robot). We also found the implementation
  was accidentally ~20:1 (`VL_BATCH_SIZE = BATCH/2`) — fixed: `VL_BATCH = BATCH`.

## Bugs found during validation (2 review agents + 3 smokes)

1. **`quad_corners_norm` parse** (HIGH) — the parquet column is an object-array
   of four `(2,)` sub-arrays; `np.asarray(..., float32)` raises `ValueError`.
   Fixed with `np.stack(...)`.
2. **`celeb_slug` silent mismatch** (HIGH) — dataset slugs are long-form
   (`barack_obama`); `build_name_token_ids` keys on short form (`obama`). KLAL
   would have found zero name tokens and silently contributed 0. Fixed with a
   `_celeb_short` normaliser; verified name tokens found 8/8.
3. **Robot dataset metadata overcounts frames** (HIGH, pre-existing) —
   `so101_eval3_track3_v3_baseline` `meta/info.json` claims `total_frames =
   5,053,972` but the parquet has `5,053,812` rows (off by 160).
   `LeRobotDataset.__len__` trusts the metadata, so the sampler emitted
   out-of-range indices and crashed a DataLoader worker. Worked around by
   capping the dataloader to `len(robot_ds.hf_dataset)` with a `[WARN]`.
   *Upstream fix (re-merge / fix the dataset metadata) still pending.*

## Validation

- **Component smoke** (`/ephemeral/smoke_klal_vl.py`): 11/11 — collator → 64
  image tokens, name tokens found 8/8, KLAL finite & >0, gradient reaches the
  LoRA q/k adapters in layers 10/12/14, all grad params trainable (DDP-safe).
- **2 parallel §9 review agents** — attention recompute verified faithful to
  `LlamaAttention.forward`; both HIGH bugs above caught here.
- **8-GPU smoke, 30 steps** — cap `[WARN]` fired, no crash, 5:1 cadence,
  `flow_loss` 0.36→0.12, `vqa` 17.1→16.4 (decreasing), `klal` stable ~1.5–1.6.

## Run config (50k)

50,000 steps · 5:1 robot:VL · KLAL λ=1.0 on the VL step · KLAL layers 10,12,14 ·
LoRA r=16 α=32 on q/k/v/o · checkpoint + HF push every 5,000 steps.

## Open items

- **λ=1.0** — KLAL now sits next to the VQA loss (not the small flow loss);
  λ=1.0 is the KLAL-paper value and appropriate here, but watch the `klal`
  curve early — drop λ if it diverges or VQA stalls.
- The robot dataset's metadata frame-count bug should be fixed upstream.
