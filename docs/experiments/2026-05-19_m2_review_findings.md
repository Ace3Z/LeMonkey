# 2026-05-19 — M2 alignment module: 3-reviewer audit + reconciliation

**Branch:** [`dev/m2-arcface-toolkit`](https://github.com/Ace3Z/LeMonkey/tree/dev/m2-arcface-toolkit)

Per [CLAUDE.md §9](../../CLAUDE.md) ("two or three independent reviewers in
parallel and reconcile their findings"), three general-purpose agents
audited [`eval_3/aug/m2_alignment.py`](../../eval_3/aug/m2_alignment.py)
from independent angles:

- **Reviewer A — math + LeRobot-source correctness** (hostile witness).
- **Reviewer B — data join correctness on real frames** (ArcFace sanity sweep over a layout-balanced 5-source sample, then widened to all 151).
- **Reviewer C — forward-looking integration plan critique** (hook semantics, layer choice, DDP, torch.compile, bf16 precision).

## Reviewer A — math + LeRobot source

**1 BUG** (now fixed), 3 WARNINGs (all addressed).

| # | Finding | Action |
|---|---|---|
| 1 | `_resize_with_pad_box` used centered padding, but LeRobot's [`resize_with_pad`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L134) pads **LEFT+TOP only** (`F.pad(img, (pad_w, 0, pad_h, 0))`). Every bbox was off by ~64 px vertically (one full patch row). | **Fixed in commit on `dev/m2-arcface-toolkit`** — `_resize_with_pad_box` now uses `ratio = max(...)` and `pad_x` left / `pad_y` top, exactly mirroring [`modeling_smolvla.py:140-152`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L140-L152). The dbg overlay uses the same mapping. |
| 3 | Defensive: `valid=True` + `mask.sum()=0` silently produces zero-pool noise. | Added `[WARN]` log + automatic `valid := valid ∧ mask.any()` filter in `m2_align_loss`. |
| 4 | `slot_to_celeb('OXX')` raised raw `KeyError`. | Now raises explicit `ValueError` naming the bad letter. |
| 9 | bf16 over 512-D cosine accumulates ~1e-2 noise, blurring `cos=0.55` vs `cos=0.60` (the discriminative region). | Upcast `projected` and `target_centroids` to fp32 before `F.normalize` + dot; cast loss back to working dtype at return. `eps` bumped 1e-8 → 1e-6 (bf16-safe). |

All other findings (2, 5, 6, 7, 8, 10) were verified **OK** by Reviewer A:
- Einsum `"bsp,bph→bsh"` is the correct contraction.
- Loss sign correct (cos=+1 ⇒ loss=−1).
- `(cos.sum() * 0.0)` zero-loss branch preserves autograd graph.
- Manifest centroids confirmed L2-normalized at
  [`cache_arcface_embeddings.py:182-183`](../../eval_3/aug/cache_arcface_embeddings.py#L182-L183).
- Camera1-at-offset-0 layout correct **when** `add_image_special_tokens=False`
  (the default).

## Reviewer B — data join correctness

Audited the slot→celeb mapping against ArcFace's actual prediction on the
camera1 frame, across 5 sources covering layouts {LSO, SLO, SOL, OSL, OLS}.

**14 of 15** slot identifications matched. The 1 mismatch was upstream:

| Source | Slot | Expected | Actual ArcFace top-1 |
|---|---|---|---|
| `quick_lecun_SOL_ep01_212006` | R | yann_lecun | **taylor_swift** (cos=0.632) |

Widened sweep across all 151 sources found **4 sources (2.6%)** with the
same pattern: the augmentation pipeline's photo-replacement step failed on
the right slot for specific `orig_R=S, new_R=L` permutations.

- `quick_lecun_SLO_ep01_211540`
- `quick_lecun_SLO_ep04_211734`
- `quick_lecun_SOL_ep01_212006`
- `quick_obama_SLO_ep05_204851`

The **join logic itself is correct.** This is a data-curation issue in the
upstream Track 3 aug pipeline. Mitigation at training time: insert an
ArcFace consistency check that flips `bbox_valid[b,s]=False` when the
top-1 ArcFace identity over the bbox crop disagrees with the
augmentation.json claim. Bounded cost (≤1.5% of bboxes); zero risk of
training on miscoded labels.

**Action for Roham (when he's back):** rerun [`generate_aug_track3.py`](../../eval_3/aug/generate_aug_track3.py)
on these 4 sources, OR flag them and exclude their 4 × ~65 = ~260 variants
from training. Tracking in `docs/experiments/2026-05-19_m2_data_audit.md`
(written by Reviewer B).

## Reviewer C — integration plan critique

8 numbered concerns, classified:

| # | Concern | Verdict | Action |
|---|---|---|---|
| 1 | Layer choice: BlindVLA used layer 16/28 ≈ 57 % depth. Our `num_vlm_layers=16` truncation → matched depth is **layer 9** (57 % × 16 ≈ 9.1), not 12. | **PICK 9.** | Use `text_model.layers[10].input_layernorm` as the hook target. |
| 1b | Pre-hook semantics on the custom forward — need a probe to confirm the captured tensor really equals "post-layer-N output." | **NEEDS-CONFIRM.** | Write a 30-line probe script before integration. |
| 2 | `add_image_special_tokens=False` is the default but anyone could toggle it. | **WARNING.** | Assert at hook-attach time. |
| 3 | `set_requires_grad`'s `else` branch re-freezes layer 15 for DDP unused-params reasons. If we re-enable grad on layer 15 we hit DDP errors. | **OK for single-GPU**, BUG-if-DDP. | Either single-GPU Brev (matches Track A) or keep the last-layer re-freeze. Confirm Brev mode first. |
| 4 | torch.compile + Python-attribute writes from hooks cause graph breaks. Track A has `compile_model=False` so safe; Track B has `compile_model=True` so unsafe. | **OK for Track A.** | Assert `compile_model is False` at hook-attach. Don't enable M2 on Track B. |
| 5 | bf16 cosine precision. | Same as Reviewer A finding 9 — already fixed. | — |
| 6 | Dataloader: 151 JSONs, ~40 MB total. Per-`__getitem__` load = re-parse cost. | **OK with worker_init_fn preload.** | Implement preload in dataloader. |
| 7 | λ=0.2 balance vs action loss. | **NEEDS-CONFIRM via logging.** | Start at 0.2 (BlindVLA), monitor `per_slot_cos` mean; if > 0.3 at 1k steps drop to 0.1. |
| 8 | `set_requires_grad` runs once at init; checkpoint reloads may re-pin. | **OK with safeguard.** | Apply partial-freeze post-pass after every `policy.load_state_dict`. |

## Reconciliation: what changes now vs at integration time

**Changed in `m2_alignment.py` this turn** (commit pending):
- Geometry bug fix.
- fp32 upcast for cosine.
- `[WARN]` for valid+empty-mask.
- Stricter `slot_to_celeb` validation.

**Deferred to integration code** (next file: `eval_3/aug/m2_smolvla_hook.py`):
- Hook on `text_model.layers[10].input_layernorm` (layer 9 capture, depth-matched).
- Hook-probe verification script.
- Assert `add_image_special_tokens=False`, `compile_model=False`, `present_img_keys[0]=='observation.images.camera1'`.
- Partial-freeze post-pass on layers 0-7; keep upstream re-freeze of layer 15 (DDP-safe).
- Dataloader `worker_init_fn` to preload all 151 face_labels JSONs.
- Optional ArcFace consistency check at training time to mask the ~1.5 % miscoded slots.

## Bottom line

The alignment module is **ready to integrate**, conditional on:
1. **Pick layer 9** (depth-matched to BlindVLA) unless we have a reason to deviate.
2. **Confirm single-GPU Brev** (matches Hans's Track A defaults).
3. **Write the hook-probe** as the first integration step (catches the "captured wrong tensor" silent failure before we waste a 6 h Brev run).

Smoke test still passes 4/4 with the corrected geometry.
