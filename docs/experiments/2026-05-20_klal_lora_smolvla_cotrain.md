# KLAL + LoRA on the SmolVLA 10:1 co-training baseline — 2026-05-20

Adds two opt-in enhancements to the `eval_3/scripts/smolvla_cotrain/` trainer
(merged in from `feat/cotrain-smolvla-darius`): **LoRA** parameter-efficient
VLM adaptation and the **KLAL** attention-supervision loss. Both are off by
default — with no new flags, `cotrain.py` is byte-for-byte the original
RT-2 §3.2 trainer.

## What was built

| File | Role |
|---|---|
| `eval_3/aug/m2_lora.py` | Hand-rolled LoRA (`LoRALinear`, `inject_lora`, merge helpers). |
| `eval_3/aug/m2_klal_smolvla.py` | `KLALHookSetSmolVLA` — attention recompute for SmolVLA's custom forward; name-token locator. |
| `eval_3/scripts/smolvla_cotrain/cotrain.py` | Wired LoRA + KLAL into the trainer (CLI flags, robot-step loss, merge-on-save). |
| `eval_3/scripts/smolvla_cotrain/launch.sh` | `ENABLE_LORA` / `ENABLE_KLAL` env vars. |

The `klal_loss` / `gaussian_target_from_mask` / `KLALConfig` from `m2_klal.py`
are model-agnostic and reused unchanged.

## Design decisions (and why)

- **LoRA replaces the VLM full-fine-tune.** The original cotrain runs
  `train_expert_only=False` (whole VLM body trainable). `--enable_lora` flips
  it to `train_expert_only=True` (VLM base + lm_head frozen) and injects
  low-rank adapters on the LM attention projections. Both the VQA-CE loss and
  the action-flow loss back-prop into the **same** adapters — RT-2 §3.2's
  anti-forgetting mechanism is preserved, and the frozen base is a stronger
  anti-forgetting guarantee than full-FT. Default r=16, α=32, dropout=0.0,
  all 16 LM layers, modules q/k/v/o.
- **Hand-rolled LoRA, not PEFT.** `peft` could not be import-verified in the
  dev env; pinning its API blind violates CLAUDE.md §8. The rank-decomposition
  update `W0 + (α/r)·BA` is unambiguous — a ~30-line module *is* the canonical
  method (Hu et al. 2021, arXiv:2106.09685).
- **Merge-on-save.** Every checkpoint is saved with the LoRA delta folded
  into a fresh plain `nn.Linear` (base weights never mutated — exact
  round-trip), so checkpoints load as a **vanilla `SmolVLAPolicy`**. Eval-day
  `from_pretrained` recipe is unchanged; intermediate checkpoints are
  probe-loadable.
- **KLAL supervises the robot batches.** Per the user's call: KLAL teaches the
  *deployed policy path* (`SmolVLMWithExpertModel` action forward) directly,
  rather than the VL VQA forward. The robot frames carry no bboxes, so the M2
  toolkit's `M2SupervisionBuilder` supplies per-frame face boxes — **bboxes
  only; the M2 ArcFace loss is NOT added** ("no enhancements" beyond KLAL+LoRA).
- **KLAL layers ⊆ LoRA layers.** KLAL can only move attention where q/k are
  trainable; the launcher asserts the subset relation. Default KLAL layers
  {10,12,14}; σ=1.0 patches (SmolVLA's 8×8 grid is coarser than Pi0.5's
  16×16, where the KLAL was σ=1.5 — empirical, flagged for tuning).

## Faithfulness of the KLAL attention recompute

`KLALHookSetSmolVLA` hooks `text_model.layers[n].self_attn.{q,k}_proj` (fire on
the VLM prefix stream only) and recomputes `softmax(QK^T·scale)`:

- **RoPE** is applied with SmolVLA's own `apply_rope` (not Gemma's) — the
  HIGH-severity no-RoPE-proxy bug from the Pi0.5 KLAL is avoided by
  construction.
- **position_ids** are captured live from `SmolVLMWithExpertModel.forward`
  (`cumsum(pad_masks)-1`, which repeats on padded language tokens — a plain
  arange would mis-RoPE).
- **scale** = `head_dim**-0.5`, matching `eager_attention_forward`.
- **prefix→prefix only**: SmolVLA's prefix is fully bidirectional and prefix
  rows attend only to prefix columns, so the recompute equals the policy's
  real prefix attention.
- **image-prefix length** (needed to map name-token → prefix row) is
  *measured at runtime* by counting connector calls — not guessed from
  `empty_cameras` config.

## Open risks / to verify on the GPU smoke test

1. **UNSMOKED.** cotrain.py was already "written but unsmoked"; these edits
   add to that. Run the 200-step smoke (`STEPS=200 ENABLE_LORA=1 ENABLE_KLAL=1
   ...`) and confirm: `klal=` prints non-zero, finite; `flow`/`vqa` still
   trend down; no OOM.
2. **Robot batch keys.** KLAL assumes the raw `LeRobotDataset` batch carries
   `episode_index` + `frame_index`. If it doesn't, the run raises loudly
   (no silent fallback) — fix by deriving `frame_index` from `index`.
3. **Name-token match.** If the SmolVLA task tokenizer disagrees with
   `vl_processor.tokenizer`, `extract_name_token_positions` returns None and
   KLAL logs `[WARN] no supervision` once — check the smoke log for it.
4. **σ = 1.0** on the 8×8 grid is an untuned default — revisit if the
   step-~10k attention probe shows the target too broad/narrow.
5. **Resume.** A mid-run resume re-injects LoRA fresh (B=0) on top of the
   merged checkpoint — continuous, but LoRA optimizer momentum is lost. Fine
   for single-shot 30k runs; noted.

## How to run

```bash
ENABLE_LORA=1 ENABLE_KLAL=1 \
  FACE_LABELS_DIR=eval_3/aug/stats/face_labels \
  CELEB_MANIFEST=eval_3/aug/stats/celeb_embeddings.json \
  AUG_ROOT=/data/eval3_track3_aug \
  EPISODE_MAPPING=eval_3/aug/stats/episode_mapping.json \
  STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2 LOG_EVERY=1 \
  bash eval_3/scripts/smolvla_cotrain/launch.sh
```
