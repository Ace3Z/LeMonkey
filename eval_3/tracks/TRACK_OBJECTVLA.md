# Track 2 — ObjectVLA co-train (Pi0.5)

**Owners:** Sejohn (lerobot-train patch + training) · Darius (VL-pairs data + Strix testing)
**Backbone:** `lerobot/pi05_base` · **Output:** `HBOrtiz/pi05_eval3_objectvla` · **VM:** brev_instance2 (RTX PRO 6000 96 GB)
**ETA:** ~2-4 h data prep + ~24 h training.

This is the **highest paper-backed** track. Full strategy context: [`docs/report/EVAL_3_FINAL_PLAN.html`](../../docs/report/EVAL_3_FINAL_PLAN.html) §3 Track 2, and the research synthesis in [`docs/experiments/2026-05-20_pivot_research.md`](../../docs/experiments/2026-05-20_pivot_research.md).

---

## 1 · Why this track

ObjectVLA ([arxiv 2502.19250](https://arxiv.org/abs/2502.19250)) is the only published method that directly tackles our exact failure mode — *"the robot sees a novel object/face but cannot bind it to a name → action."* Their recipe:

- **Co-train** the VLA on robot demonstrations + vision-language pairs in a **single training run** at a **10:1 robot:VL ratio**.
- Each VL pair carries **explicit bounding-box grounding** — e.g. `"the face of Yann LeCun is at [0.21, 0.32, 0.58, 0.74]"`.
- Quantitative result: **without the bounding boxes, OOD success drops from 64% → 19%.** The grounding is the load-bearing ingredient — not the text caption alone.

Independent confirmation: Pi0.5's own ablation ([arxiv 2504.16054](https://arxiv.org/abs/2504.16054) Fig. 11) shows removing web/VL co-training data drops OOD object success ~75% → ~45-50%.

This beats the deprecated two-stage VQA-warm-start (see [`TRACK_B_WARMSTART.md`](TRACK_B_WARMSTART.md)) because co-training never disconnects the VLM updates from the action-expert updates — so there's no catastrophic-forgetting / action-expert-miscalibration risk.

---

## 2 · The two work-streams

### Darius — VL-pairs data generation (~2-4 h, on edna)

Generate ~15k bbox-grounded face-VL pairs from the 193-celeb scraped bank.

edna already has the conda env `aug` with InsightFace + `buffalo_l` models installed (see [`HANDOVER_EDNA.md`](../HANDOVER_EDNA.md)). The bank is at `~/LeMonkey/datasets/eval3_celebs/scraped/` (193 celeb subdirs, 8-11 photos each).

For each photo:
1. Run InsightFace RetinaFace → face bounding box.
2. Normalize bbox to `[x1, y1, x2, y2]` in [0,1] image coords.
3. Emit ~10 caption variants (mix of forms):
   - **Location-explicit (ObjectVLA-style, ~50%):** `"The face of {Name} is at [{x1},{y1},{x2},{y2}]."`
   - **Q&A grounded (~30%):** prompt `"Who is the person at [{x1},{y1},{x2},{y2}]?"` → target `"{Name}"`
   - **Q&A open (~10%):** prompt `"Who is the person in this image?"` → target `"{Name}"`
   - **Caption form, no bbox (~10%):** `"{Name} is in this image."` (variety; keeps some examples bbox-free)

Use the celeb's **most-common name variant** for `{Name}` (REAL / Parashar finding — most-googled name beats formal name). For our 193 celebs the slug-derived Title Case is usually fine; spot-check the AI researchers (e.g. `yann_lecun` → "Yann LeCun" ✓).

**Deliverable:** parquet manifest pushed to `HBOrtiz/eval3_objectvla_vl_pairs`. Columns: `image_path, prompt, target, bbox, celeb, caption_type`.

A starting point script: clone `eval_3/scripts/warmstart/prepare_vggface2_vqa.py` (it already does the scraped-bank walk + InsightFace) and add the bbox extraction + caption synthesis. ~1 h of edits.

**Optional enrichment:** also sample 5 wrist-cam frames per episode from `HBOrtiz/so101_eval3_aug_v3_200celebs` and emit the same caption forms — this adds the real print/lighting distribution. +~50k pairs. Do this only if there's time.

### Sejohn — the lerobot-train mixed-batch patch + training

The core change: lerobot-train must interleave robot batches and VL batches in the same optimizer loop at a 10:1 ratio.

**Patch design (~150 lines):**

1. Add a `--vl_dataset.manifest` CLI arg + a `--vl_ratio` (default 10, meaning 1 VL batch per 10 robot batches).
2. Build a second dataloader over the VL parquet manifest. Collator: `PaliGemmaProcessor` with `suffix=` (reuse the collator from `eval_3/scripts/warmstart/train_paligemma_vqa.py` — it already handles the `<image>` + suffix masking).
3. In the training loop:
   ```python
   if step % (vl_ratio + 1) == 0:
       batch = vl_loader.next()
       loss = pi05_vqa_loss(model, batch)     # PaliGemma + lm_head CE, no action expert
   else:
       batch = robot_loader.next()
       loss = pi05_flow_loss(model, batch)    # standard Pi0.5 flow-matching path
   loss.backward(); optimizer.step()
   ```
4. The VQA-mode forward goes through `PaliGemmaWithExpertModel.forward()` with `inputs_embeds=[image_text_embeds, None]` — the prefix-only branch (verified at `modeling_pi05.py:462-473`). The action expert is not invoked for VL batches.

**Heads-up — the dict-attention-mask risk:** the warm-start scaffolding flagged that transformers ≥5.0's `PaliGemmaModel.forward` may pass a dict-of-masks to the language model, which `PiGemmaModel.forward` (expects a Tensor) can't handle. **Smoke-test the VQA forward path on brev_instance2 before the full run.** If it fails, the fix is to call the LM directly with `inputs_embeds` + a tensor `attention_mask` rather than going through PaliGemma's top-level forward. Full detail in [`TRACK_B_WARMSTART.md`](TRACK_B_WARMSTART.md) §6.

**Training recipe** (start from the working Track 1 recipe in `eval_3/scripts/brev/run_training_track_B.sh`):

```bash
--policy.type=pi05
--policy.pretrained_path=lerobot/pi05_base
--policy.freeze_vision_encoder=True
--policy.train_expert_only=False
--peft.method_type=LORA --peft.r=32
--peft.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
--dataset.repo_id=HBOrtiz/so101_eval3_aug_v3_200celebs
--vl_dataset.manifest=<local copy of eval3_objectvla_vl_pairs>
--vl_ratio=10
--batch_size=48 --steps=30000 --optimizer_lr=1e-5
--output_dir=outputs/pi05_objectvla
--policy.push_to_hub=True --policy.repo_id=HBOrtiz/pi05_eval3_objectvla
```

Run on brev_instance2 — env is already bootstrapped (`lemonkey` conda env, cu128 PyTorch, lerobot, peft). See [`HANDOVER_BREV_INSTANCE2.md`](../HANDOVER_BREV_INSTANCE2.md).

---

## 3 · Step-by-step

| # | Step | Owner | ETA |
|---|---|---|---|
| 1 | Generate VL-pairs manifest from scraped bank, push to HF | Darius | 2-4 h |
| 2 | Write the lerobot-train mixed-batch patch (`--vl_dataset`, `--vl_ratio`) | Sejohn | 4-6 h |
| 3 | Smoke test: 200-step run, verify both loss paths fire + no dict-mask crash | Sejohn | 30 min |
| 4 | Full run on brev_instance2 (~24 h) | Sejohn | 24 h |
| 5 | Strix test the result — 3-rollout protocol | Darius | 2 h |

---

## 4 · Smoke test (must pass before the 24 h run)

```bash
# On brev_instance2, after the patch is in place:
python <lerobot-train entrypoint> \
    ...same flags... \
    --steps=200 --vl_ratio=10 \
    --output_dir=/tmp/objectvla_smoke
```

Verify:
- Robot batches AND VL batches both run (you'll see ~18 robot + ~2 VL in 200 steps).
- VQA forward path does NOT crash on the dict-attention-mask issue.
- Both losses decrease (flow-matching loss ~3→2, VQA CE loss ~5→3).
- GPU memory peaks < 90 GB (RTX PRO 6000 has 96 GB).

---

## 5 · Risks

| Risk | Mitigation |
|---|---|
| Dict-attention-mask crash in VQA forward | Smoke-test gates it. Fallback: call `language_model` directly with tensor mask. |
| VL batches dominate gradient if ratio is wrong | Stick to 10:1 — ObjectVLA's published ratio. Don't improvise. |
| bbox coords in the wrong convention (xyxy vs xywh, pixel vs normalized) | Standardize on normalized xyxy. Verify against a rendered debug overlay before generating all 15k. |
| The 200-celeb dataset's quantile stats are wrong for Pi0.5 | Run `eval_3/scripts/fast_recompute_quantiles.py` on it first — same fix as Track 1's `_pi05` dataset. |

---

## 6 · Sources

- ObjectVLA — [arxiv 2502.19250](https://arxiv.org/abs/2502.19250)
- Pi0.5 (web co-train ablation Fig. 11) — [arxiv 2504.16054](https://arxiv.org/abs/2504.16054)
- Pi0.5-KI — [arxiv 2505.23705](https://arxiv.org/abs/2505.23705)
- Research synthesis — [`docs/experiments/2026-05-20_pivot_research.md`](../../docs/experiments/2026-05-20_pivot_research.md)

---

*Scaffolded 2026-05-20. Owners: Sejohn + Darius. Status: assigned, not started.*
