# Track 2 — ObjectVLA co-train, Sejohn's enhanced spec

**Sibling of:** [`TRACK_OBJECTVLA.md`](TRACK_OBJECTVLA.md) (canonical, owned by Roham's commit `5dbae0a`).
**Owners:** Sejohn (lerobot-train patch + training + enhancements) · Darius (VL pairs data + Strix testing).
**Branch:** `dev/SjohnU/track_2_objectvla`.
**Status:** scaffolded 2026-05-20.

This doc PRESERVES the canonical Track 2 ObjectVLA recipe and stacks five
training-time-only enhancements on top, all data-side, single-flag, or config-only, none
deviating from the +45 pp OOD published mechanism (ObjectVLA arxiv 2502.19250).

If you're reading this to ship, the diff vs canonical is in §C "Combined launch
command". §F lists explicit non-deviations from canonical.

---

## 0 · Why we enhance the canonical spec

The canonical Track 2 spec already cites a +45 pp OOD lift (ObjectVLA Fig. 8:
19% → 64% on novel-object placement). Why not just ship it vanilla? Six
reasons, each targeting a specific concern:

**0.1 — The +45 pp number is on a different task.** ObjectVLA's eval is novel
**object** placement; ours is novel **celeb-face** matching. The mechanism
family (bbox-grounded VL co-training of the VLM text head) transfers, but the
specific number does not. Inpainted celeb faces at wrist-cam angle are harder
than clean tabletop objects — we should expect lower lift unless we squeeze
every free signal available.

**0.2 — Cold `pi05_base` was ObjectVLA's baseline because they didn't have a
warm-PG. We do.** Roham's `HBOrtiz/pi05_paligemma_celeb_warm` is verified ready
on HF (4 B params, LoRA-merged from 9 131-identity VGGFace2 VQA fine-tuning).
Starting cold ignores this. **Enhancement B-1** flips one flag to stack
offline + online face knowledge — the kind of cost-free signal you take.

**0.3 — Our dataset has noise the published recipe didn't have.** ObjectVLA
used clean ground-truth bboxes on real objects. Ours come from
ArcFace + RetinaFace applied to inpainted celeb faces from
`generate_aug_v3.py`. Mahbod's M2 data audit
([`docs/experiments/2026-05-19_m2_data_audit.md`](../../docs/experiments/2026-05-19_m2_data_audit.md))
already flagged that 14/15 audited frames matched the expected celeb — meaning
1/15 didn't. Project-wide that's a 5–10% label-noise floor. **Enhancement
B-2** removes the worst noise; **B-3 + B-5** make sure the model sees hard
discriminations early enough to develop fine-grained features.

**0.4 — 24 h budget forces gradient efficiency.** ObjectVLA trained 50 k steps
on 8× A100s. We have 30 k steps on a single H100. Every gradient step matters
~50% more. **Enhancement B-4** (layer-wise LoRA rank) concentrates trainable
capacity in the layers known to host face-binding (BlindVLA Table 12: layer 10
of Gemma-2B). **Enhancement B-7** (EMA) reduces gradient-noise variance.

**0.5 — Team-level portfolio coverage.** Three Pi0.5 bets:

| Track | Mechanism | What it addresses |
|---|---|---|
| 1 (Roham) | vanilla LoRA + warm-PG | safety floor — proven recipe with celeb-aware base |
| **2 (Sejohn — this spec)** | **ObjectVLA bbox grounding via VQA CE on LM head** | **face-name-position binding via text head** |
| 3 (Mahbod) | M2 ArcFace cosine at mid-layer + KLAL attention bias | mid-VLM features become face-discriminative |

The three target different points in the VLM stack with different loss forms.
If they fail in correlated ways, ONE of them needs to be the strongest
possible individual run. Enhancing Track 2 (highest published precedent)
maximizes the team's ceiling without changing anyone else's plan.

**0.6 — Each enhancement is a targeted fix for a specific failure mode:**

| Enhancement | Failure mode it addresses |
|---|---|
| B-1 warm-PG start | Mahbod's probe found PaliGemma fails zero-shot celeb naming on teleop frames |
| B-2 ArcFace filter | Inpainted-face noise in 200-celeb dataset contaminates training signal |
| B-3 hard-neg oversampling | Easy "Obama vs Swift" pairs dominate gradient; confusable pairs underlearned |
| B-4 layer-wise LoRA rank | Uniform r=32 wastes capacity on layers that don't host face-binding |
| B-5 curriculum learning | Early noise prevents face features from forming before action loss settles |
| B-7 EMA of weights | Gradient-noise oscillation late in training reduces final checkpoint quality |

None of these add a new mechanism. Each one is a published technique applied
to a specific known concern. **The spine of the recipe is sacred; only the
periphery is enhanced.** See §F for the full non-deviation list.

---

## A · Canonical Track 2 (locked, do not modify)

From [`TRACK_OBJECTVLA.md`](TRACK_OBJECTVLA.md):

- **Backbone**: `lerobot/pi05_base` cold → overridden by warm-PG (see §B-1)
- **Co-training** at **10:1 robot:VL ratio**: 10 robot batches (flow-matching loss)
  for every 1 VL batch (LM-head CE loss).
- **VL pairs** = bbox-grounded face captions from Darius's 193-celeb scraped bank
  (~15 k pairs, 4 caption forms 50/30/10/10).
- **Action expert NOT invoked on VL batches** —
  `PaliGemmaWithExpertModel.forward(inputs_embeds=[image_text_embeds, None])`
  prefix-only branch (`modeling_pi05.py:462-473`).
- `train_expert_only=False` (VQA CE needs gradients in PaliGemma body).
- LoRA r=32 on `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`
  → overridden by layer-wise rank (see §B-4).
- `batch_size=48`, `steps=30000`, `optimizer_lr=1e-5`, `compile_model=True`,
  `gradient_checkpointing=True`, `empty_cameras=3`.
- **Output**: `HBOrtiz/pi05_eval3_objectvla`.
- **VM**: brev_instance2 (RTX PRO 6000 96 GB).

**Critical risk (from canonical §2)**: transformers ≥5.0 dict-attention-mask issue
in PaliGemma's top-level forward. Smoke test gates this. Fallback: call
`language_model` directly with tensor mask.

---

## B · Enhancements on top of canonical (Sejohn's additions)

All six are training-time only, TA-compliant, and compose with the canonical
spec without changing the 10:1 ratio, the VL/robot batch separation, or the
LoRA target list.

### B-1 · Warm-PG starting point (1 flag, ~0 h)

Swap cold `lerobot/pi05_base` → warm `HBOrtiz/pi05_paligemma_celeb_warm`
(Roham's warm-PG, verified ready on HF: 4 B params, safetensors, F32/BF16).
Stacks offline VGGFace2 face-knowledge with online ObjectVLA bbox grounding.

```diff
- --policy.pretrained_path=lerobot/pi05_base
+ --policy.pretrained_path=HBOrtiz/pi05_paligemma_celeb_warm
```

**Why it works**: warm-PG already LoRA-fine-tuned PaliGemma's language body on
9 131-identity VGGFace2 VQA. Starting from it means PaliGemma's LM head input
distribution already encodes celeb knowledge; Track 2's bbox VQA CE then
refines + applies on the wrist-cam distribution. Pi0.5 blog Fig. 11 confirms:
removing web/VQA co-train drops OOD 75% → ~47%.

**Risk**: catastrophic forgetting if Track 2's LoRA fine-tune drifts the warm
weights. LoRA's small parameter budget bounds the drift; VQA CE re-anchors
celeb knowledge from step 0.

### B-2 · ArcFace data quality filter (~1 h, after Darius's bboxes)

Uses Mahbod's `celeb_embeddings.json` (192 centroids on
`HBOrtiz/eval3_m2_arcface_toolkit`). For each variant in the 200-celeb dataset,
per-frame target-face cosine to centroid. Drop episodes with mean target_cos
< 0.50.

```diff
+ --dataset.episodes_file=eval_3/scripts/track_2/keep_episodes.txt
```

**Threshold rationale (CLAUDE.md §7 triple-source)**:
- ArcFace LFW verification FAR=1e-3 → 0.36 (InsightFace docs)
- Mahbod's M2 data audit on inpainted faces: same-celeb cos 0.5–0.8, cross 0.0–0.2
  ([`docs/experiments/2026-05-19_m2_data_audit.md`](../../docs/experiments/2026-05-19_m2_data_audit.md))
- 0.50 chosen as "well above noise floor, below clean-face mean" — empirically
  verify via histogram on a 5-variant sample before committing the full threshold.

Expected retention: 85–95% of variants.

Script: [`eval_3/scripts/track_2/arcface_audit_200celeb.py`](../../eval_3/scripts/track_2/arcface_audit_200celeb.py).

### B-3 · Hard-negative oversampling (~2 h)

Per-frame `hardneg_gap = target_cos − max_distractor_cos`. Frames with gap
< 0.10 (confusable distractor present) get 2× sample weight on robot batches.
Forces fine-grained discrimination during the action-loss path.

```diff
+ --dataset.sample_weights=eval_3/scripts/track_2/hardneg_weights.npy
```

**Composes** with B-2 (filter first, then weight the survivors). **Composes**
with B-5 (curriculum reads same `hardneg_gap` column).

### B-4 · Layer-wise LoRA rank concentration (~30 min)

Uniform r=32 across 18 Gemma-2B layers wastes capacity on layers that don't
host face-binding. Concentrate at face-discriminative zones per BlindVLA
Table 12 (~57% LM depth) + Voita 2019 head-specialization.

Per-layer rank config ([`layer_rank_track2.json`](../../eval_3/scripts/track_2/layer_rank_track2.json)):

| Layers | Rank | Purpose |
|---|---|---|
| 0–4 | 16 | Preserve warm-PG WebLI prior |
| 5–7 | 32 | Lower-mid scaffold |
| 8–12 | **64** | Mid-LM face-discrim zone (BlindVLA layer 10 ± 2) |
| 13–14 | 32 | Upper scaffold |
| 15–17 | 48 | Top-LM name-token alignment (LM head reads from these) |

Total trainable params ≈ uniform r=32, but capacity is in face-relevant layers.
If PEFT version doesn't accept per-layer rank, fall back to uniform r=32
(no critical-path impact).

### B-5 · Curriculum learning by ArcFace difficulty (~30 min)

First 5 k steps: sample only variants with `hardneg_gap ≥ 0.10` (easy, clear
identity). Step ≥ 5 k: full distribution. Standard curriculum (Bengio 2009).

```python
# In sampler:
if step < curriculum_switch_step:  # default 5000
    weights = hardneg_weights * (hardneg_gap >= 0.10).astype(float)
else:
    weights = hardneg_weights
```

```diff
+ --dataset.curriculum_switch_step=5000
```

### B-7 · EMA of weights (optional stretch, ~5 min)

α = 0.999 shadow copy. At inference, use EMA weights. Standard SGD stability.

```diff
+ --train.use_ema=True --train.ema_alpha=0.999
```

Low risk, modest expected lift (~1–3%). Skip if time-constrained.

---

## C · Combined launch command (diff vs canonical)

```diff
  python eval_3/scripts/lerobot_train_with_vl_cotrain.py \
    --policy.type=pi05 \
-   --policy.pretrained_path=lerobot/pi05_base \
+   --policy.pretrained_path=HBOrtiz/pi05_paligemma_celeb_warm \
    --policy.freeze_vision_encoder=True \
    --policy.train_expert_only=False \
    --policy.empty_cameras=3 \
    --policy.optimizer_lr=1e-5 \
    --policy.compile_model=True \
    --policy.gradient_checkpointing=True \
    --peft.method_type=LORA \
-   --peft.r=32 \
+   --peft.layer_rank_config=eval_3/scripts/track_2/layer_rank_track2.json \
    --peft.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
    --dataset.repo_id=HBOrtiz/so101_eval3_aug_v3_200celebs \
+   --dataset.episodes_file=eval_3/scripts/track_2/keep_episodes.txt \
+   --dataset.sample_weights=eval_3/scripts/track_2/hardneg_weights.npy \
+   --dataset.curriculum_switch_step=5000 \
    --vl_dataset.manifest=HBOrtiz/eval3_objectvla_vl_pairs \
    --vl_ratio=10 \
+   --train.use_ema=True --train.ema_alpha=0.999 \
    --batch_size=48 --steps=30000 \
    --output_dir=outputs/pi05_objectvla \
    --policy.push_to_hub=True \
    --policy.repo_id=HBOrtiz/pi05_eval3_objectvla
```

---

## D · Gantt — 30 h to demo lock

| Hour | Activity | Critical | Owner |
|---|---|---|---|
| 0–0.5 | Pull Mahbod's `celeb_embeddings.json`, pull 200-celeb meta/ | YES | Sejohn |
| 0–0.5 | Slack Darius — confirm VL-pairs gen started on edna | YES | Sejohn |
| 0–1 | **Strix VRAM + latency probe** on cold `pi05_base` | GATE | Darius |
| 0–4 | Darius generates VL-pairs manifest (parallel) | parallel | Darius |
| 0–6 | Sejohn writes mixed-batch patch + reads `train_paligemma_vqa.py` | YES | Sejohn |
| 0–5 | Sejohn ports CLI runner to Pi05Policy (Task #5) | parallel | Sejohn |
| ~4 | Darius's VL manifest available + robot-frame bboxes | gate | both |
| 4–5 | Run ArcFace audit on robot frames (~1 h locally) | YES | Sejohn |
| 5–5.5 | Generate `keep_episodes.txt` + `hardneg_weights.npy` | YES | Sejohn |
| 5.5–6 | Wire layer-wise LoRA rank config | YES | Sejohn |
| 6–6.5 | **200-step smoke test on brev_instance2** | GATE | Sejohn |
| 6.5–7 | Launch full 30 k step run | — | Sejohn |
| 7–31 | Training | — | brev |
| 31–33 | Push to HF, Strix download, latency probe | YES | Darius |
| 33–35 | Strix 3-rollout protocol | YES | Darius |
| 35–36 | Final lock for demo | margin | — |

Margin: ~2–3 h. Three abort gates protect against burning Brev compute.

---

## E · Risk + mitigation

| Risk | Probability | Mitigation |
|---|---|---|
| Dict-attention-mask crash in VQA forward | medium | Smoke test gates. Fallback: call `language_model` directly with tensor mask. (Canonical §2.) |
| Warm-PG drift / catastrophic forgetting | low–medium | LoRA r=32 bounds drift; VQA CE re-anchors celeb knowledge from step 0. |
| ArcFace filter drops too much data (>20%) | low | Histogram cos distribution before filtering; adjust threshold to keep ≥85%. |
| Layer-wise LoRA rank incompatible with PEFT version | low–medium | Fall back to uniform r=32. No critical-path impact. |
| Darius's bboxes arrive late (>hour 4) | low–medium | Track 2 still launches without B-2/B-3/B-5; B-1, B-4, B-7 still apply. |
| Multi-source VL data (stretch) distribution mismatch | n/a | Skipping for first pass; deferred to Day-3 retry. |

---

## F · Explicit non-deviations from canonical

This spec does NOT change:

- The 10:1 robot:VL ratio
- The VL/robot batch separation (no robot-batch bbox CE)
- The VQA loss form (no focal, no token-level reweighting)
- The action loss form (standard flow-matching)
- The partial-freeze pattern (`train_expert_only=False`, no extra freeze)
- The LoRA target_modules list (still the canonical 7 Gemma projections)
- The output repo name pattern

This spec does NOT add:

- ArcFace cosine distillation (Mahbod's Track 3 lane)
- Robot-batch bbox CE (deviates from ObjectVLA's published 10:1 separation)
- LoRA on `lm_head` (Roham rejected in [`TRACK_B_WARMSTART.md`](TRACK_B_WARMSTART.md))
- Mixed-prompt training (shortcut risk)
- State channel ArcFace (TA grey-zone)
- Variant B aux head on action expert (no precedent + Pi0.5-KI 0%-frozen warning)
- Spatial prompt enrichment (shortcut risk)

---

## G · Cross-validation history

Per CLAUDE.md §9, the enhancement stack was validated against the following:

- **Round 1**: M2 port feasibility — agent verdict 6–8 h, conditional gates ([`docs/experiments/2026-05-19_m2_review_findings.md`](../../docs/experiments/2026-05-19_m2_review_findings.md))
- **Round 2**: 200-celeb dataset TA-compliance — agent verified prompt mix (75/15/10) requires filtering OR canonical Track 2 uses HF episodes parameter
- **Round 3**: Pi0.5 wrapper LM head feasibility — confirmed Pi0.5 supports the VQA forward path used in canonical
- **Round 4**: ObjectVLA bbox-CE engineering verdict — not feasible in 24 h via bbox tokens, BUT this spec uses the canonical CE on text tokens (not bbox tokens), avoiding the agent's flagged blockers
- **Round 5**: focal loss / gradient scaling / head-selective training — all rejected for VLA face-rec, fallback to data-side which is what this spec implements
- **Round 6**: layer-wise LoRA rank — published precedent (BlindVLA Table 12, Voita 2019); included in enhanced spec

Conservative-additive design: every enhancement is published or trivial; no
novel-substitutive mechanisms layered onto Track 2's published spine.

---

*Scaffolded 2026-05-20. Owner: Sejohn. Status: enhancement spec, awaiting Darius's bbox manifest.*
