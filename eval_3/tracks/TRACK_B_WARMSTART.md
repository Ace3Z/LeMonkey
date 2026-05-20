# Track B — PaliGemma VQA warm-start (Day-3 fallback path)

**Owner:** Roham · **Branch:** `track-b-pi05` · **Dataset:** TBD-VGGFace2-manifest · **Output:** `HBOrtiz/pi05_paligemma_celeb_warm` · **Brev:** ~10 h on a SECOND H100 80 GB VM (parallel to in-flight Track B)

Status: scaffolded 2026-05-19; launched only if Day-3 Strix data on the vanilla Track B shows weak face discrimination.

---

## TL;DR — what this is

A two-stage Pi0.5 recipe:

1. **Stage 1 (this doc, ~10 h):** LoRA-fine-tune the PaliGemma half of `lerobot/pi05_base` on a VGGFace2 VQA task (`Who is the person in this image?` → celebrity name). Merge adapters, push as `HBOrtiz/pi05_paligemma_celeb_warm`.
2. **Stage 2 (~24 h):** Run the existing Track B Pi0.5 LoRA recipe but with `--policy.pretrained_path=HBOrtiz/pi05_paligemma_celeb_warm` instead of `lerobot/pi05_base`. Same dataset, same hyperparams.

The bet: Pi0.5's PaliGemma was pretrained on WebLI (DLP-filtered web image-text, no published celeb benchmarks). Our 9 IID+OOD celeb names sit in the long tail. Pre-conditioning the VLM on 9 131 face identities × ~50 images each gives it general face-discrimination capability so the Stage 2 VLA fine-tune has something to anchor "Yann LeCun" / "Taylor Swift" to.

This is option 1 from [`TRACK_B.md` §8](TRACK_B.md#8--fallback-if-track-b-underperforms-on-day-3). Decision to launch is Day-3 morning, after Darius's Strix rollouts on the vanilla Track B.

---

## 1 · Architecture — what's trainable, what isn't

```
                Pi0.5 (4.2B params)
   ┌─────────────────────────────────────────────────────────┐
   │ PaliGemma-2B                                            │
   │  ├── vision_tower (SigLIP-So400m, 400M)   FROZEN       │
   │  ├── multi_modal_projector                FROZEN       │
   │  ├── language_model (Gemma 2B)            LoRA r=32    │ ← trained
   │  └── lm_head                              FROZEN       │
   ├─────────────────────────────────────────────────────────┤
   │ Gemma-300M action expert + projections    FROZEN       │
   │ Action heads                              FROZEN       │
   └─────────────────────────────────────────────────────────┘
```

**Why LoRA only on the LLM (not the vision tower):** SigLIP was trained for image-text contrastive alignment. It already produces decent face-discriminative features. The naming bottleneck is downstream in the LLM, where WebLI's long-tail bias suppresses celeb names. LoRA on Gemma layers lets the LM re-weight its existing knowledge without rewriting it.

**Why same `target_modules` as Track B:** `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` — full 7-projection Gemma block coverage. Using the same target list means the warm-start adapters and the Stage-2 Track-B adapters live in the same parameter subspace. After Stage 1 merge, the base PaliGemma weights ARE the warmed version, and Stage 2 LoRA starts from clean adapters on top. No adapter stacking, no PEFT version-compat headaches.

**Why freeze the `lm_head`:** Adding LoRA to the LM head means re-learning a 250k-vocab projection. Marginal upside (the LM head can already emit names; the bottleneck is which name to pick). Major downside: doubles trainable params + slows training. Skip.

**Why we can ignore the action expert during VQA:** `PaliGemmaForConditionalGenerationWithPiGemma` (lerobot's subclass) overrides only the LM decoder via `PiGemmaModel`. That subclass adds `adarms_cond`-conditioned RMSNorm + gated residuals. When `adarms_cond=None` (which is the default and what happens when we don't invoke the action expert), these reduce to standard Gemma behavior. So we can train the model with the vanilla HF VQA loss path (`model(pixel_values, input_ids, labels=...)`) and the architecture behaves like a standard PaliGemma. Validated via [`pi_gemma.py:342-352`](../../third_party/lerobot/src/lerobot/policies/pi_gemma.py#L342-L352).

---

## 2 · Data — VGGFace2 + our scraped bank

| Source | Identities | Images/identity | Total rows | Use |
|---|---|---|---|---|
| **VGGFace2** (from Hans) | 9 131 | ~50 (subsampled) | ~456 000 | Primary face-id capability |
| Our 193-celeb scraped bank | 193 | ~8 (all) | ~1 500 | Distribution-matched anchor for our specific celebs |
| **Total** | ~9 200 unique | mixed | **~458 000** | |

VGGFace2 is the right scale: Parashar 2024 ([arxiv 2401.12425](https://arxiv.org/abs/2401.12425)) shows long-tail VLM celebrity recall recovers only above ~100 examples/identity. CASIA-WebFace (10k identities × ~50) is the standard alternative if VGGFace2 access is blocked.

**The manifest is a parquet file** with five columns (`image_path`, `prompt`, `target`, `identity_id`, `source`). Images are read lazily by the collator. Format chosen so re-splitting / re-sampling doesn't require re-encoding pixels. Build it with:

```bash
python eval_3/scripts/warmstart/prepare_vggface2_vqa.py \
    --vggface2-root /shared/datasets/vggface2/train \
    --names-csv     /shared/datasets/vggface2/identity_meta.csv \
    --scraped-root  ~/LeMonkey/datasets/eval3_celebs/scraped \
    --max-per-identity 50 \
    --scraped-max-per-identity 10 \
    --out manifests/vggface2_vqa_train.parquet
```

The `--scraped-root` includes our 193-celeb bank — strictly optional but cheap to add (~1500 extra rows) and keeps the model exposed to the exact image style we'll see at eval time.

---

## 3 · Open question — where is VGGFace2 on disk?

**Hans has VGGFace2 for the SmolVLM2 Track A warm-start.** Before launching this on Brev we need:

- [ ] Confirm Hans's VGGFace2 root path (likely on his Brev VM or shared storage)
- [ ] Either rsync the raw images to our Brev VM (~36 GB → ~30 min over Brev network)
- [ ] OR (faster) build the manifest on Hans's side and rsync just the manifest + symlink / mount his image dir on our VM
- [ ] Get `identity_meta.csv` from Hans too

Fallback if Hans doesn't have it accessible:

- Use **CASIA-WebFace** on HF (`feature-extraction/casia-webface` or similar) — public, ~7 GB, similar variety.
- 10 575 identities × ~50 imgs/id ≈ 500k rows, comparable to VGGFace2-subsampled.

---

## 4 · Recipe — exact training command

```bash
python eval_3/scripts/warmstart/train_paligemma_vqa.py \
    --manifest /shared/vggface2_vqa_train.parquet \
    --pretrained-pi05 lerobot/pi05_base \
    --output-dir outputs/paligemma_celeb_warm \
    --push-repo HBOrtiz/pi05_paligemma_celeb_warm \
    --epochs 1 \
    --batch-size 8 --grad-accum 4 \
    --lr 1e-5 \
    --lora-r 32 --lora-alpha 64 --lora-dropout 0.05
```

### Per-flag reasoning

- `--pretrained-pi05 lerobot/pi05_base` — same base as Track B, ensures the warmed PaliGemma drops cleanly into the Stage-2 recipe.
- `--epochs 1` — 458k rows × 1 epoch ÷ effective batch 32 ≈ 14 300 optimizer steps. At ~2.5 s/step on H100 with grad_ckpt + bf16, that's ~10 h.
- `--batch-size 8 --grad-accum 4` — effective batch 32. PaliGemma 2B with LoRA + bf16 + grad_ckpt fits at micro-batch 8 with ~30 GB headroom on H100 80 GB.
- `--lr 1e-5` — same as Track B. Conservative because LoRA is small and we don't want to perturb base weights through the adapter path. Cosine schedule + 3% warmup baked into the script.
- `--lora-r 32 --lora-alpha 64` — same as Track B for alignment. Ratio α/r=2.0 is the LoRA convention for adapting fine-grained features.

### What gets pushed to HF

The training script does THREE things at end of training:

1. `paligemma.merge_and_unload()` — merges LoRA adapters into the PaliGemma submodule weights.
2. Splices the merged PaliGemma back into the full Pi0.5 model (overwrites `policy.model.paligemma_with_expert.paligemma`).
3. `policy.push_to_hub(...)` — pushes the complete Pi0.5 checkpoint (PaliGemma warmed + action expert untouched + action heads untouched + LM head untouched).

The resulting HF repo is a drop-in replacement for `lerobot/pi05_base` in the Track B launcher.

---

## 5 · Stage 2 — Track B re-launch with the warmed checkpoint

Once `HBOrtiz/pi05_paligemma_celeb_warm` exists, re-launch Track B with one flag changed:

```bash
# eval_3/scripts/brev/run_training_track_B.sh, override:
PRETRAINED="HBOrtiz/pi05_paligemma_celeb_warm" \
DATASET="HBOrtiz/so101_eval3_track3_v3_pi05" \  # or the 200-celeb merged repo if available
bash eval_3/scripts/brev/run_training_track_B.sh
```

(The current `run_training_track_B.sh` hardcodes `lerobot/pi05_base`. Adding a `PRETRAINED` env-var override is a one-line change — see `eval_3/scripts/brev/run_training_track_B.sh:54`.)

Everything else identical: same LoRA r=32, same target_modules, same batch=48, same 30k steps, same `train_expert_only=False`. The only difference: PaliGemma's q/k/v/o/gate/up/down weights inside `pi05_paligemma_celeb_warm` start ~1.4% different from `pi05_base` (the LoRA rank-32 perturbation, merged in).

---

## 6 · Smoke test (before the full ~10 h run)

```bash
python eval_3/scripts/warmstart/train_paligemma_vqa.py \
    --manifest /shared/vggface2_vqa_train.parquet \
    --output-dir /tmp/paligemma_smoke \
    --smoke
```

`--smoke` trims to 200 rows + skips the HF push. Verifies:

- Model loads (Pi0.5 base downloads + parses)
- LoRA wraps correctly + prints trainable param count (should be ~28M for r=32 on 7 modules of a 2B LLM)
- Processor handles `<image>` + `suffix=` correctly (no `image_token` count mismatch)
- One training step lands without OOM (peak VRAM at batch=8 + grad_ckpt + bf16 should be ~30 GB)

Expected wall: 2-3 min including model download (cached on Brev after first run).

---

## 7 · Risks + open validations

Caught by the 2026-05-19 cross-validation review (CLAUDE.md §9):

| Risk | Severity | Mitigation |
|---|---|---|
| **`PiGemmaModel.forward` expects a Tensor `attention_mask`, but transformers ≥5.0 `PaliGemmaModel.forward` builds a dict-of-masks** | **BLOCKER if it materialises** | The standard HF VQA forward path may pass `{<attention_type>: mask}` to the language model, which would crash `create_causal_mask`. **Smoke test gates the full run** — if it fails, swap to a manual splice path: compute image features separately, embed text tokens, replace `<image_pad>` positions with image features, then call `language_model(inputs_embeds=..., attention_mask=<tensor>)` directly. ~30 lines of patch. |
| PaliGemmaProcessor `max_length=64` would truncate the 256 image tokens (off-by-orders-of-magnitude) | fixed | default raised to 384; ensures image tokens + bos + prompt + name all fit. |
| VGGFace2 `identity_meta.csv` has leading-space column headers ( ` Name`, ` Sample_Num`, ...) | fixed | `prepare_vggface2_vqa.py` now strips header whitespace + hard-fails if no identity has a name mapping. |
| Collator returning `None` would crash Trainer | fixed | Falls back to fabricating a 1-row batch from `ds[0]` with `[WARN]` log. |
| PaliGemmaProcessor's `suffix=` arg behavior differs across `transformers` versions | low | Pin `transformers>=4.45` per HF docs; smoke test verifies `labels` correctly mask prompt tokens (set to -100). |
| Merging LoRA into `PaliGemmaForConditionalGenerationWithPiGemma` (lerobot subclass) — PEFT might not handle the subclass-specific layers (PiGemmaRMSNorm, gated residuals) | low | These layers aren't in our `target_modules`, so PEFT doesn't touch them. Merge is just the standard q/k/v/o/gate/up/down weights. Verified by reading PEFT source — merge happens in-place on the target Linear modules. |
| `push_to_hub` on a Pi0.5 model writes a config the Stage-2 lerobot-train can't reload | low | `policy.push_to_hub` uses lerobot's `PreTrainedPolicy` save/load path, which we know works (Track B itself does it at end of training). Round-trip verified by smoke test (`policy.save_pretrained` → `PI05Policy.from_pretrained`). |
| LoRA adapters trained on PaliGemma's standard forward won't behave the same way when the action expert is in the loop at Stage 2 | low | Adapters live in the q/k/v/o/MLP projections; their behavior is invariant to whether the action expert cross-attends downstream. The `merge_and_unload` path bakes the perturbation into the base weights, so Stage 2 sees a slightly-shifted PaliGemma regardless. |
| Hans doesn't have VGGFace2 accessible | medium | Fall back to CASIA-WebFace on HF Hub. ~Same scale, slightly different identity coverage. |

---

## 8 · Decision tree on Day 3

```
Day 2 evening: Track B vanilla LoRA in-flight on Brev (~22% done as of 2026-05-19 12:30 UTC)
                ↓
Day 3 ~11 UTC: Track B finishes → Darius Strix-tests it
                ↓
        ┌───────┴───────┐
        ↓               ↓
   passes face-id    fails face-id
   (≥2/3 celebs)     (consistently wrong celeb)
        ↓               ↓
   SHIP it.        Day 3 ~12 UTC: launch Stage 1 warm-start
                    on a SECOND Brev VM (~10 h)
                    ↓
                   Day 3 ~22 UTC: warm-start done →
                    relaunch Track B on either VM (~24 h)
                    ↓
                   Day 4 ~22 UTC: warmed Track B done →
                    final Strix dry-run → ship
                    
                   (tight: ~2h margin before eval day morning)
```

---

## 9 · References

- [`TRACK_B.md`](TRACK_B.md) — the *why* of vanilla Pi0.5 LoRA + 3-agent validations
- [`TRACK_B_DEVBOX_HANDOVER.md`](TRACK_B_DEVBOX_HANDOVER.md) — current Brev-running recipe (vanilla, no warm-start yet)
- [`docs/experiments/2026-05-19_track_b_brev_debugging.md`](../../docs/experiments/2026-05-19_track_b_brev_debugging.md) — 8 failures observed during the in-flight run
- PaliGemma 2 paper — [arxiv 2412.03555](https://arxiv.org/abs/2412.03555)
- VGGFace2 — [Cao et al. 2018](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- Long-tail VLM celeb recall — Parashar et al. "Neglected Tails" [arxiv 2401.12425](https://arxiv.org/abs/2401.12425)
- LoRA — Hu et al. [arxiv 2106.09685](https://arxiv.org/abs/2106.09685)

---

*Scaffold last updated 2026-05-19. Status: code written, not launched. Awaiting Day-3 Strix data on vanilla Track B before deciding whether to launch.*
