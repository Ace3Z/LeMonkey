# PaliGemma VQA warm-start — execution strategy + log

**Date:** 2026-05-19
**Branch:** `track-b-warmstart-vqa`
**Machine:** brev_instance2 (`brev-9yew2fxz2`, RTX PRO 6000 Blackwell 97 GB, 16 cores, 141 GB RAM)
**Owner:** Roham · **Pushed model (target):** `HBOrtiz/pi05_paligemma_celeb_warm`

This doc is the execution playbook for the run we're firing tonight. The *why* is documented in [`eval_3/tracks/TRACK_B_WARMSTART.md`](../../eval_3/tracks/TRACK_B_WARMSTART.md); this one captures the actual recipe + decision points + what we expect to see.

---

## 1 · What we're doing in one paragraph

We're LoRA-fine-tuning the PaliGemma 2B half of `lerobot/pi05_base` on a face-VQA task ("Who is the person in this image?" → `<celebrity name>`) using a pre-cropped VGGFace2 (9 131 identities × ~340 images each, pre-cropped to 224×224 = PaliGemma's exact input size). After ~10 h, we merge the LoRA adapters into the PaliGemma submodule, splice it back into the full Pi0.5 model, and push as `HBOrtiz/pi05_paligemma_celeb_warm`. That HF repo then becomes a drop-in `--policy.pretrained_path` for a Day-3 Track B re-launch IF the vanilla Track B (currently running on brev_instance1, 22% complete) fails face-discrimination on the Day-3 Strix rollouts.

This is the option-1 fallback from [`TRACK_B.md` §8](../../eval_3/tracks/TRACK_B.md#8--fallback-if-track-b-underperforms-on-day-3) — Hans's "LoRA-on-VGGFace2 + then VLA fine-tune" pattern, ported from SmolVLM2 to PaliGemma.

---

## 2 · Why now, on this machine

| Constraint | Resolution |
|---|---|
| brev_instance1 is busy with vanilla Track B (~28 h total, finishes Day-3 ~11 UTC) | brev_instance2 was provisioned specifically for this parallel work — different VM, different cost line |
| Day-3 morning decision needs the warmed checkpoint to be READY when the Strix call comes | Starting now (~Day-2 ~18 UTC) means the 10 h run finishes Day-3 ~04 UTC — plenty of slack before Strix data lands |
| If vanilla Track B passes, this is wasted spend | RTX PRO 6000 is ~$2/h, 10 h = ~$20. Cheap insurance for a project where one rollout = ~$5 of score |
| Risk of dict-attention-mask failure (validator's BLOCKER #10) | Smoke test gates the 10 h full run; ~5 min cost, catches the failure mode in 1 forward pass |

---

## 3 · Recipe — the exact run

### Data

- **Source:** [`chronopt-research/cropped-vggface2-224`](https://huggingface.co/datasets/chronopt-research/cropped-vggface2-224)
  - 40 parquet shards, ~19 GB total, on disk at `~/face_data/cropped-vggface2-224/data/`
  - Schema: `{image: HF Image (auto-decoded to PIL), label: ClassLabel (0-9130)}`
  - Pre-cropped to 224×224 = PaliGemma 2's exact input. No further preprocessing needed.
- **Identity → name map:** [`ProgramComputer/VGGFace2:meta/identity_meta.csv`](https://huggingface.co/datasets/ProgramComputer/VGGFace2/blob/main/meta/identity_meta.csv)
  - 9 131 entries; format `n000001 → 14th Dalai Lama` (after our parser strips the leading-space-headers bug + handles ragged rows)
  - Cached locally at `~/.cache/huggingface/hub/datasets--ProgramComputer--VGGFace2/.../meta/identity_meta.csv`

**VGGFace2 does NOT include our 9 eval celebs (Obama/Swift/LeCun/etc.) by design** — it was curated to exclude western A-list celebrities. That's fine: the warm-start teaches PaliGemma face-discrimination as a *general skill*; Stage 2 (Track B re-train on our augmented dataset) anchors that skill to our 9 specific names.

### Model

- **Base:** `lerobot/pi05_base` (4.2 B params total). Loaded as `PI05Policy` via lerobot's port.
- **Trainable:** PaliGemma's language_model only, via LoRA (r=32, α=64, dropout=0.05) on q/k/v/o + gate/up/down projections. **~28 M trainable params (<0.7% of base)**.
- **Frozen:** vision_tower (400 M, SigLIP), multi_modal_projector, lm_head, Gemma-300M action expert, action heads.
- **Mixed precision:** bf16 for compute, fp32 kept for vision_tower + projector + RMSNorms (per `to_bfloat16_for_selected_params`).

### Hyperparams

| Param | Value | Why |
|---|---|---|
| `epochs` | 1 | ~3.14 M rows × 1 epoch ÷ effective batch 32 ≈ 98 k steps. At ~2.5 s/step on RTX PRO 6000 with grad_ckpt, that's ~68 h. **Too long.** See "subsample" decision below. |
| **`max_train_samples`** | 460 000 | Subsample 50/identity (×9 131) ≈ 456 k rows to keep wall-time at ~10 h. Matches Hans's recipe scale + sufficient per Parashar 2024 (≥100 examples/id for long-tail entity recovery, halved for the LoRA-only context). |
| `batch_size` | 8 micro × 4 grad_accum = 32 effective | PaliGemma 2B + LoRA + bf16 + grad_ckpt fits at micro-batch 8 with ~30 GB peak on RTX PRO 6000 (97 GB) — wide headroom. |
| `lr` | 1e-5 | Same as Track B. Conservative; LoRA adapters are small. Cosine schedule, 3% warmup. |
| `lora_r / α / dropout` | 32 / 64 / 0.05 | Same target modules + same shape as Track B → adapter weights align with the architecture used at Stage-2 re-train. |
| `dataloader_num_workers` | 4 | PIL decode + processor tokenize is the main bottleneck. 4 workers keeps GPU fed. |
| `max_text_len` | 384 | PaliGemma 2 prepends 256 image tokens + BOS + prompt + name; ≥320 needed to avoid the validator's `image_token` mismatch failure. |

### What we push at end of run

After training, the script does:

1. `paligemma.merge_and_unload()` — bakes LoRA weights into the PaliGemma submodule
2. Splice merged submodule back into the full `PI05Policy`
3. `policy.save_pretrained(...)` locally → verify with reload
4. `policy.push_to_hub("HBOrtiz/pi05_paligemma_celeb_warm")` → ready for Stage 2

---

## 4 · Execution plan

### Step 1 — Smoke test (5-10 min)

Gates the dict-attention-mask risk + 5 other check items from the validator review.

```bash
cd ~/LeMonkey
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lemonkey
python eval_3/scripts/warmstart/train_paligemma_vqa.py \
    --local-parquet-dir ~/face_data/cropped-vggface2-224/data \
    --identity-meta-csv ~/.cache/huggingface/hub/datasets--ProgramComputer--VGGFace2/snapshots/ad5f6b5a5f560621fd7efb9b79c956d27d427a08/meta/identity_meta.csv \
    --output-dir ~/outputs/paligemma_smoke \
    --smoke
```

Pass criteria (all must hold):
- ✓ `Pi05Policy.from_pretrained("lerobot/pi05_base")` completes without OOM
- ✓ `print_trainable_parameters` reports ~28 M trainable (~0.7%)
- ✓ Processor accepts `<image>` token + `suffix=` without `image_token count mismatch`
- ✓ Trainer completes ≥1 optimizer step without crash (gates the dict-mask risk)
- ✓ `policy.save_pretrained(...)` writes a valid local checkpoint
- ✓ `PI05Policy.from_pretrained(local_dir)` round-trips cleanly

If smoke fails on the dict-mask path, the documented patch is in [`TRACK_B_WARMSTART.md` §7](../../eval_3/tracks/TRACK_B_WARMSTART.md): manually splice image features + call `language_model(inputs_embeds=..., attention_mask=<tensor>)` directly. ~30 lines of code.

### Step 2 — Full run (10-12 h)

```bash
# Subsample the dataset to ~50 rows per identity before training
# (the trainer will respect HF Dataset.select for this).
nohup python eval_3/scripts/warmstart/train_paligemma_vqa.py \
    --local-parquet-dir ~/face_data/cropped-vggface2-224/data \
    --identity-meta-csv ~/.cache/huggingface/hub/datasets--ProgramComputer--VGGFace2/snapshots/ad5f6b5a5f560621fd7efb9b79c956d27d427a08/meta/identity_meta.csv \
    --output-dir ~/outputs/paligemma_celeb_warm \
    --push-repo HBOrtiz/pi05_paligemma_celeb_warm \
    --batch-size 8 --grad-accum 4 \
    --lr 1e-5 --epochs 1 \
    --lora-r 32 --lora-alpha 64 \
    > ~/paligemma_warmstart.log 2>&1 &
echo $! > ~/paligemma_warmstart.pid
```

Monitor:

```bash
tail -f ~/paligemma_warmstart.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
free -h
```

### Step 3 — Verify the push (5 min after run completes)

```bash
hf download HBOrtiz/pi05_paligemma_celeb_warm --revision main --local-dir /tmp/verify_warm
python -c "
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
p = PI05Policy.from_pretrained('/tmp/verify_warm')
print('reload OK; trainable params:', sum(x.numel() for x in p.parameters() if x.requires_grad))
"
```

Expected: model loads, sum-of-trainable matches Track B's expected count (since the merged weights look identical to `pi05_base` to PEFT; trainable count is determined by the NEW LoRA adapters Stage 2 would apply).

### Step 4 — Stage 2 decision (Day-3 morning, ~11 UTC)

```
Track B (brev_instance1, vanilla LoRA) finishes ~11 UTC Day 3
                        ↓
        Darius runs the 3-rollout Strix protocol
                        ↓
              ┌───────────┴───────────┐
              ↓                       ↓
       passes face-id        fails face-id (wrong celeb consistently)
       (≥2 of 3 celebs)              ↓
              ↓                Re-launch Track B on brev_instance1 OR brev_instance2 with
       SHIP vanilla              --policy.pretrained_path=HBOrtiz/pi05_paligemma_celeb_warm
       Track B; warm-start       (same dataset, same recipe, 24 h)
       sits unused                       ↓
                              ~Day-3 evening: re-trained Track B done
                                          ↓
                              Day-4 dry-run → ship
```

---

## 5 · What success looks like at end of run

A good Stage-1 (warm-start) outcome:

- Training loss decreases steadily from ~3-4 at step 100 to ~0.5-1.5 by step ~50 k.
- No NaN or loss-spike events.
- At end, `merge_and_unload()` succeeds without "tensor shape mismatch" errors.
- `policy.save_pretrained(...)` + reload round-trip works.
- `hf download HBOrtiz/pi05_paligemma_celeb_warm` shows a valid Pi0.5 checkpoint (~6 GB safetensors).

A good Stage-2 outcome is measured separately on the Strix rollouts.

---

## 6 · Open risks

1. **Dict-attention-mask through PiGemmaModel** — see [`TRACK_B_WARMSTART.md` §7 row 1](../../eval_3/tracks/TRACK_B_WARMSTART.md#7--risks--open-validations). Gated by smoke; ~30-line manual-splice patch documented if it fires.
2. **VGGFace2 doesn't include our 9 eval celebs.** The warm-start adds a *general capability*, not specific memorization. Stage 2 grounds the capability onto our specific celebs through the VLA fine-tune. If the general capability doesn't transfer to OOD celebs at Strix, Stage 2's job is harder; we'd fall back to Track A/C (SmolVLA, Hans's warm-VLM).
3. **License (cc-by-nc-4.0).** VGGFace2 + chronopt parquet are research-use only. We're using it for the ETH RC course project (non-commercial). Pushed model `HBOrtiz/pi05_paligemma_celeb_warm` inherits this restriction — mark it private if needed.
4. **Disk space:** 646 GB free; warm-start outputs (checkpoints + logs) ~10 GB. No issue.
5. **VRAM:** RTX PRO 6000 has 97 GB; PaliGemma 2B + LoRA + bf16 + grad_ckpt at batch 8 = ~30-40 GB peak. Wide headroom.

---

## 7 · Communication template

Per the handover template — Slack-style updates:

- **Smoke launched / passed / failed** — one line each
- **Full run launched** — PID + ETA
- **Mid-run check (every ~2 h)** — step / loss / GPU util
- **End** — pushed repo URL + final loss + any anomalies

---

## 8 · Live log

- 2026-05-19 18:30 UTC — env verified, VGGFace2 + identity_meta.csv on disk, trainer updated for parquet input, branch `track-b-warmstart-vqa` created.
- 2026-05-19 ~19:00 UTC — smoke test passed (loss 12.37, no dict-mask crash, save/reload round-trip OK).
- 2026-05-19 ~19:10 UTC — full run launched (`--epochs 0.15`, ~470k rows, batch 32).
- 2026-05-20 ~00:30 UTC — full run finished. 15 507 steps, 5.6 h wall, final loss ~5.0 (mean-over-run 6.12). Merged + pushed `HBOrtiz/pi05_paligemma_celeb_warm`. Loads from HF clean (4.143 B params).

---

## 9 · Validation results (2026-05-20) — honest assessment

### Method note — why not generation

First attempt used `.generate()` on the PaliGemma submodule → **garbage tokens** from
*both* baseline and warmed models. Root cause: lerobot's Pi0.5 port
(`PaliGemmaForConditionalGenerationWithPiGemma` / `PiGemmaModel`) was built for
flow-matching *action* inference; its autoregressive *text* `.generate()` path is
untested and broken. Not a model defect — a harness defect.

Also discovered + ruled out: pi05 uses the **PaliGemma 1** tokenizer
(`google/paligemma-3b-pt-224`); the warm-start trained with the PaliGemma **2**
processor. Verified harmless — PG1 and PG2 tokenizers are byte-identical
(`GemmaTokenizer`, vocab 257152, same IDs for all test strings, same `<image>`
id 257152). pi05_base `lm_head` is healthy (norm 4259, tied to embeddings).

Switched to **teacher-forced N-way discrimination** — the working forward path
(same one training used). For each face, score candidate names by cross-entropy
on the name tokens; the model "picks" the lowest-CE candidate. This mirrors the
real eval task (closed-set celeb selection).

### Results

`eval_3/scripts/warmstart/eval_warmstart_vqa.py --n-way 5 --n-vggface2 60`

| Test | BASELINE (`pi05_base`) | WARMED (`pi05_paligemma_celeb_warm`) | Random |
|---|---|---|---|
| **A — VGGFace2, 5-way** (identities the warm-start trained on) | 20 % (12/60) | **37 % (22/60)** | 20 % |
| **B — our 8 eval celebs, 8-way** (Swift/Obama/LeCun/Federer/Bezos/Musk/Messi/Ronaldo — NOT in VGGFace2) | 0/8 | 1/8 | 12.5 % |

### Honest interpretation

- **Test A: real but modest gain.** The warm-start lifted in-distribution face
  discrimination from chance (20 %) to 37 % on a 5-way task. The LoRA adapters
  genuinely learned *some* face→name capability. But 37 % is weak — loss
  plateaued at ~5.0 (perplexity 148); the model never became a strong
  recognizer.
- **Test B: no usable transfer.** On our 8 actual eval celebs, both models are
  at random. WARMED *collapses to "Jeff Bezos" for 7 of 8 celebs* — it is not
  discriminating, just emitting one name. The "1/8 correct" is a lucky artifact
  of that collapse (Bezos is in the set), not real recognition.

### What this means

1. A light VGGFace2 LoRA warm-start (0.15 epoch, ~50 img/identity, r=32, frozen
   `lm_head`) is **too weak to transfer** to identities it never saw. The
   general-skill-then-transfer hypothesis did not hold at this training budget.
2. The warmed checkpoint is, at best, a **marginally better Track B init**
   (slightly better in-distribution face features). It is **not** a fix for our
   celeb-discrimination problem.
3. **Strategic consequence:** the planned Day-3 fallback ("if vanilla Track B
   fails Strix → re-train with warmed checkpoint") is **weaker than designed**.
   If vanilla Track B fails on face discrimination, the warmed re-train will
   most likely also fail. The real fallback is **Track A / Track C (SmolVLA)**.

### Candidate next steps (not yet decided)

- **C1 — warm-start on our scraped bank instead.** 193 celebs incl. our 8 eval
  ones, ~8-11 photos each + heavy augmentation. Directly targets our celebs.
  ~2-4 h. Cheapest way to learn whether *any* PaliGemma warm-start can move the
  8-way number.
- **C2 — stronger VGGFace2 run.** Full 1-2 epochs, higher LoRA rank, unfreeze
  `lm_head`. ~12-30 h. Higher cost, uncertain payoff.
- **C3 — ArcFace distillation (Track A's M2) ported to Pi0.5.** The "collapse to
  one name" is the signature of a vision encoder not extracting identity-
  discriminative features. ArcFace distillation explicitly fixes that. Code lift.
- **C4 — accept Pi0.5 VLM face-discrimination is weak; lean on Track A/C.**

The vanilla Track B run on brev_instance1 is still the primary; its Day-3 Strix
result is the load-bearing data point.
