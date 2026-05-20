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
- **position_ids** are captured by wrapping the module-level `apply_rope`
  (SmolVLA calls `vlm_with_expert.forward` directly, bypassing nn.Module
  `__call__`, so a forward-pre-hook never fires — this was caught by the
  smoke test). `cumsum(pad_masks)-1` repeats on padded tokens, so a plain
  arange would mis-RoPE.
- **scale** = `head_dim**-0.5`, matching `eager_attention_forward`.
- **prefix→prefix only**: SmolVLA's prefix is fully bidirectional and prefix
  rows attend only to prefix columns, so the recompute equals the policy's
  real prefix attention.
- **image-prefix length** (needed to map name-token → prefix row) is
  *measured at runtime* by counting connector calls — not guessed from
  `empty_cameras` config.

## Component smoke test — PASSED (33/33)

`eval_3/aug/tests/test_klal_lora_smoke.py` (run on CPU with the `lemonkey`
conda env) verifies the components with real torch:

- LoRA: no-op at init (B=0), fp32 merge round-trip exact (~1e-6), bf16 merge
  drift small (~1.6e-2), inject/swap, vanilla state-dict keys after merge.
- KLAL: Gaussian target sums to 1, loss ≈ 0 when attention matches the
  target and > 0 when uniform, name-token subsequence located.
- **Real `lerobot/smolvla_base`**: LoRA injected (64 modules, 1.64M adapter
  params), `q_proj.weight.dtype` access works (no crash), a real
  `VLAFlowMatching.forward` runs (`flow_loss=2.88`), KLAL hooks capture q/k,
  position_ids + image-prefix length measured, **KLAL loss finite (2.38)**,
  and **KLAL's gradient reaches the LoRA adapter** (`q_proj.lora_B`,
  |grad|≈7.9e-2).

The smoke caught one real bug — the original position_ids capture used a
forward-pre-hook on `vlm_with_expert`, but SmolVLA calls `.forward()`
directly so it never fired; fixed by wrapping `apply_rope` (above).

## Full cotrain smoke on a-toy-pi05 (H100) — PASSED

200-step run on the real datasets (`so101_eval3_track3_v3_baseline` robot +
`eval3_objectvla_vl_pairs` VL), `--enable_lora --enable_klal`, bs 4 / vl_bs 2:

- 200/200 steps, no NaN (`non-finite=0`), no OOM, final checkpoint saved.
- `flow_loss` (robot, pure) ↓ ~0.5–1.1 → ~0.10–0.26.
- `vqa_loss` ↓ ~15.9 → ~10–12, on real face images.
- `klal` active and finite throughout, ~1.3 → ~0.95–1.1.
- merge-on-save verified — the final `model.safetensors` has 0 LoRA/base
  keys: a vanilla loadable `SmolVLAPolicy`.

Reaching a green smoke required fixing **five pre-existing bugs in the
(never-smoke-tested) cotrain base** — none in the KLAL/LoRA code:
1. VL collator passed `images` as a flat list (SmolVLM wants list-of-lists).
2. VL collator truncated at 256 tokens, cutting the ~1088 image tokens.
3. `SmolVLMVisionEmbeddings` built `boundaries` on CPU (transformers 4.55).
4. robot dataloader used `delta_timestamps=None` — no action chunk.
5. `VLPairsDataset` only extracted `images.tar.zst`; the dataset ships
   `data.tar.zst`, so all 176k VL images had fallen back to gray.

## Multi-GPU cluster run

`run_cluster.sh` + `RUN_ON_CLUSTER.md` (`eval_3/scripts/smolvla_cotrain/`)
let another cluster run the 25k-step KLAL+LoRA cotrain, one command:

- `cotrain.py` gained torchrun **manual-DDP** data parallelism — explicit
  gradient all-reduce, not a DDP wrapper, because the VQA step calls
  `vlm(...)` directly and would bypass DDP's forward. DistributedSampler on
  both loaders, rank-0 broadcast sync after LoRA inject, distributed-safe
  non-finite skip (all-reduced finite flag), rank-0 save + barrier.
- Verified with a **2-rank `gloo` run on a-toy-pi05**: broadcast sync,
  all-reduce, save+barrier, 24/24 steps, flow/vqa/klal all fire,
  `step_000012` + `final` saved. NCCL on real multi-GPU uses the identical
  collective API (the test used gloo only because NCCL forbids 2 ranks on
  one GPU).
- `_save_and_push` uploads each checkpoint to `<repo>/step_NNNNNN` every 5k
  steps. `m2_klal_data.tar.zst` (9 MB, in-repo) bundles the KLAL bbox data
  so the cluster needs no extra access.
- **Not live-tested:** the HF upload itself (no write token on the test box)
  — wrapped in try/except, logs `[WARN]` and keeps the local checkpoint if it
  fails, never kills the run.

## Open risks / to verify on a longer run

1. **Smoke ≠ convergence.** 200 steps confirms the mechanics, not that KLAL
   actually binds names to faces — that is the step-~10k attention-probe
   gate, on a full 30k run.
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
