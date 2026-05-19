# Track B — Pi0.5 (PaliGemma-2B + Gemma-300M expert)

**Owner:** Roham · **Branch:** `track-b-pi05` · **Dataset:** [`HBOrtiz/so101_eval3_track3_v3_pi05`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_pi05) · **Output:** `HBOrtiz/pi05_eval3_track_B` · **Bonus:** +16 · **Brev:** ~24 h on RTX PRO 6000 Blackwell

This document is the **load-bearing source of truth for Track B**. It supersedes the
generic Track B description in [`docs/report/EVAL_3_FINAL_PLAN.html`](../../docs/report/EVAL_3_FINAL_PLAN.html#tracks)
on three points where deep validation (2026-05-19, 3 parallel agents) changed the recipe.

---

## TL;DR — what changed vs the original plan

| Original plan | Validated correction | Reason |
|---|---|---|
| `--policy.train_expert_only=True` (freeze entire VLM) | **`train_expert_only=False` + LoRA on PaliGemma LLM via `--peft`** | Pi0.5-KI paper [arxiv 2505.23705 §4 + Fig 4a/8](https://arxiv.org/html/2505.23705v1) explicitly: *"fully freezing the VLM yields ~0% task performance"*. Full SFT risks catastrophic forgetting (Hans observed on SmolVLM). LoRA splits the difference. |
| `--policy.empty_cameras=3` | **`empty_cameras=2` + `rename_map` camera1 → right_wrist_0_rgb** | `lerobot/pi05_base` was pretrained with **3 camera slots** (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`), not 4. Source: [HF config.json](https://huggingface.co/lerobot/pi05_base/raw/main/config.json). |
| (implicit) trust merged `meta/stats.json` | **must re-run `augment_dataset_quantile_stats.py`** | Our merger's "min-of-q01, max-of-q99, median-of-q50" aggregation across episodes is **wrong** for Pi0.5's QUANTILE normalization. Pi0.5 reads quantiles from `meta/stats.json` and uses them for action/state clipping. Bad quantiles → training instability. |

All three corrections are in the launch script `eval_3/scripts/brev/run_training_track_B.sh`.

---

## 1 · Why Track B exists

The team's prior at 2026-05-18 was: **"Pi0.5 is the safer bet because of PaliGemma's WebLI prior"**. That instinct was based on:

- PaliGemma-2B was pretrained on **WebLI** (10B image-text pairs, web-scraped) — Google's frontier image-text corpus, likely contains celebrity coverage.
- Pi0.5 has **6.6× more parameters** than SmolVLA-450M (2.6B vs 450M). Capacity is generally helpful for fine-grained reasoning.
- Pi0.5-KI introduces explicit protection mechanisms (stop-gradient, FAST CE, web/VQA co-train) for fine-tuning a VLM without losing its priors.

Track B tests this prior. The +20 bonus for the smaller SmolVLA model is forfeited
(-4 points = ~0.7 rollouts of slack) in exchange for hopefully-much-better face
recognition from a bigger backbone with a better prior.

Mahbod summarised the choice (Slack 2026-05-18 17:14): *"I rather we lose the bonus points and our model works properly, so I think ditching SmolVLA is better."*

---

## 2 · Architecture summary

```
            ┌──────────────────────────────────────────────────────┐
            │  PaliGemma-2B (Google, pretrained on WebLI)          │
            │  ┌──────────────────────┐   ┌────────────────────┐   │
            │  │   SigLIP-So400m       │   │  Gemma-2B  (LLM)   │   │
            │  │   vision tower        │──▶│  18 transformer    │   │
            │  │   (frozen)            │   │  layers            │   │
            │  └──────────────────────┘   └────────────────────┘   │
            └─────────────────────────────────┬────────────────────┘
                                              │  hidden states (K, V)
                                              ▼
            ┌──────────────────────────────────────────────────────┐
            │  Gemma-300M action expert                            │
            │  - cross-attends to PaliGemma K/V (with sg(·))       │
            │  - emits flow-matching velocity field over actions   │
            │  - chunk_size=50 actions per forward pass            │
            │  - num_inference_steps=10 denoising steps            │
            └──────────────────────────────────────────────────────┘
```

**Param accounting** (verified via [`modeling_pi05.py:558-580`](../../third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L558)):

| Component | Params | Status in Track B (LoRA recipe) |
|---|---|---|
| SigLIP vision tower (PaliGemma) | ~400 M | **Frozen** (`freeze_vision_encoder=True`) |
| Gemma-2B LLM (PaliGemma) | ~2.0 B | **LoRA-wrapped** (rank 16, ~14 M trainable) |
| Gemma-300M action expert | ~300 M | **Fully trainable** |
| Projections (`action_in_proj`, `action_out_proj`, `time_mlp`) | ~2 M | **Fully trainable** |
| **Total trainable** | ~316 M | ~12 % of total params |

---

## 3 · Validations (2026-05-19) — full text

Three parallel agents, see `docs/experiments/2026-05-19_track_b_validations.md` for the raw transcripts. Headline conclusions below.

### Validation 1 — PaliGemma celebrity prior + frozen-VLM viability

**Verdict: original plan's `train_expert_only=True` is high-risk.**

- Neither PaliGemma 1 nor 2 publishes person-identity/celebrity benchmarks. WebLI was DLP-filtered for sensitive identifiers (no claim that named-celebrity coverage is preserved or curated). [PaliGemma 2 paper arxiv 2412.03555](https://arxiv.org/abs/2412.03555).
- VLM literature on long-tail entities (Parashar et al. "Neglected Tails," [arxiv 2401.12425](https://arxiv.org/pdf/2401.12425); Bravo et al. [arxiv 2306.16048](https://arxiv.org/html/2306.16048v3)) shows web-scale VLMs systematically fail on fine-grained named entities — the regime "Yann LeCun"-as-printed-photo lives in.
- Our 2026-05-09 0/14 zero-shot PaliGemma celebrity probe is consistent with this literature, not an outlier.
- Pi0.5-KI paper [arxiv 2505.23705 §4 + Fig 4a/8](https://arxiv.org/html/2505.23705v1): *"fully freezing the VLM yields ~0% task performance."* The Pi0.5-KI recipe is **not** "freeze VLM" — it's "update VLM via discrete FAST action tokens **and** stop-gradients from the continuous action expert." Web/VQA co-train protects OOD generalization (Fig 8 caption).

**Implication:** `train_expert_only=True` is contradicted by the Pi0.5 paper itself.
Our fix: LoRA-wrap the PaliGemma LLM so it can update during training (via low-rank
adapters) without catastrophically forgetting the pretrained prior.

### Validation 2 — Camera slot count

**Verdict: `lerobot/pi05_base` was pretrained with 3 camera slots, not 4.**

From [HF config.json](https://huggingface.co/lerobot/pi05_base/raw/main/config.json) — the three `VISUAL` features are:
- `observation.images.base_0_rgb` (top / static external view)
- `observation.images.left_wrist_0_rgb` (left wrist cam)
- `observation.images.right_wrist_0_rgb` (right wrist cam)

We have one wrist camera (`observation.images.camera1`). The fix is **two** steps:

1. **`--policy.empty_cameras=2`** — adds 2 zero-pad-mask placeholder slots to bring our 1 camera up to Pi0.5's 3-slot expectation. Verified at [`configuration_pi05.py:125-133`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L125) (validate_features) + [`modeling_pi05.py:1199-1204`](../../third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L1199) (missing-key handling — fills with `-1` tensor + zero attention mask).
2. **`--dataset.rename_map='{"observation.images.camera1": "observation.images.right_wrist_0_rgb"}'`** — maps our wrist cam onto the *trained* slot so PaliGemma's vision-tower positional/key embeddings line up with how it was pretrained. Without the rename, our camera would land on a fresh fourth slot and the pretrained vision weights wouldn't apply correctly.

Right wrist chosen because (a) it's a wrist cam (matches mounting), and (b) SO-101 is a right-arm robot in most operator-perspective recordings.

### Validation 3 — Quantile stats + VRAM

**Verdict on quantile stats: MUST re-run `augment_dataset_quantile_stats.py`.**

- Our [merger](../scripts/merge_track3_custom.py#L162-L175) aggregates per-ep quantiles with min/max/median (e.g., `q99 = max_i (per_ep_q99_i)`). This is **wrong** when quantile distributions vary across episodes.
- Example failure: 50 eps with q99=0.5 + 1 ep with q99=10.0 → our merger reports global q99=10.0; true global q99 over the joined frames is ~0.52. Pi0.5's quantile normalization (`STATE: QUANTILES, ACTION: QUANTILES` at [`configuration_pi05.py:73-79`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L73)) clips by these bounds → bad clip with our wrong stats.
- Fix is one command (already running in background as of this writing):
  ```bash
  python third_party/lerobot/src/lerobot/scripts/augment_dataset_quantile_stats.py \
    --repo-id local/eval3_track3_v3 \
    --root /home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_merged \
    --overwrite
  ```
- After completion, re-push the dataset to HF to update `meta/stats.json` on the hub.

**Verdict on VRAM: both training and inference fit.**

| Phase | Hardware | Budget | Estimate |
|---|---|---|---|
| Train | Brev RTX PRO 6000 Blackwell, 96 GB | 96 GB | ~7 GB (weights 4.6 GB + LoRA grads 0.05 GB + AdamW state 0.1 GB + activations w/ grad_ckpt 0.5 GB + batch=24 forward 1.5 GB) |
| Infer | Strix, 16 GB | 16 GB | ~4.8 GB (weights 4.6 GB + KV cache 0.12 GB + activations 0.1 GB) |

Both have comfortable headroom. The bf16 dtype is set via `--policy.dtype=bfloat16`.

---

## 4 · Exact training recipe (`run_training_track_B.sh`)

```bash
lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=True \
    --policy.train_expert_only=False \
    --policy.empty_cameras=2 \
    --policy.optimizer_lr=1e-5 \
    --policy.gradient_checkpointing=True \
    --policy.compile_model=True \
    \
    --peft.method_type=LORA \
    --peft.target_modules='["q_proj","k_proj","v_proj","o_proj"]' \
    --peft.r=16 \
    --peft.lora_alpha=32 \
    --peft.lora_dropout=0.05 \
    \
    --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_pi05 \
    --dataset.rename_map='{"observation.images.camera1":"observation.images.right_wrist_0_rgb"}' \
    \
    --batch_size=24 \
    --steps=30000 \
    --output_dir=outputs/pi05_track_B \
    --policy.push_to_hub=True \
    --policy.repo_id=HBOrtiz/pi05_eval3_track_B
```

### Per-flag reasoning

- `dtype=bfloat16` — saves VRAM, standard for VLA fine-tunes. Pi0.5's mixed-precision path keeps vision tower + multi_modal_projector in fp32 ([`modeling_pi05.py:417-418`](../../third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L417)).
- `freeze_vision_encoder=True` — SigLIP stays frozen. We don't need it to learn new face features; the LLM downstream is where naming bias lives.
- `train_expert_only=False` — VLM is NOT entirely frozen (counterintuitive but research-validated; see Validation 1). The LoRA flag below controls *how* the VLM updates.
- `empty_cameras=2` — fills the 2 missing camera slots Pi0.5 was trained with.
- `optimizer_lr=1e-5` — half of Pi0.5's default 2.5e-5. Conservative because LoRA updates effective only at low LR; also reduces risk of perturbing base weights through the LLM's adapter path.
- `gradient_checkpointing=True` — drops peak activation memory by recomputing forward during backward. Free with `compile_model=True`.
- `compile_model=True` — torch.compile speeds Pi0.5 by ~25-40 % in published benchmarks.
- `peft.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` — LoRA-wrap **all 7 linear projections** in every Gemma-2B transformer block (4 attention + 3 gated-MLP). Targeting MLP layers in addition to attention is the LLaMA-Adapter / QLoRA convention and gives the adapters genuinely-meaningful capacity to absorb celeb-discriminative features. Standard attention-only LoRA would under-perform here because most of Gemma's representational capacity lives in the gated MLP, not attention.
- `peft.r=32, alpha=64` — rank 32, scaling alpha 64 → effective scaling ratio 2.0. Bumped from the initial r=16 because r=16 only allocates ~14M trainable params and the model needs to learn 9 distinct celeb identities through a small projection — that's plausibly under-capacity. r=32 gives ~28M trainable adapter params, which is still <1.4 % of Gemma-2B's frozen weights so forgetting risk remains low. (For reference: LoRA paper Hu 2021 §4 found rank ≥ 8 typically sufficient on language tasks, but face-recognition is fine-grained classification with many identities — closer to DreamBooth-style personalization where 16-32 is the recommended floor.)
- `peft.lora_dropout=0.05` — small dropout on adapters; standard.
- `rename_map` — re-routes our 1 camera onto the pretrained `right_wrist_0_rgb` slot.
- `batch_size=24` — bf16 + grad_ckpt + LoRA easily fits at 24 on RTX PRO 6000.
- `steps=30000` — same as Pi0.5's published `scheduler_decay_steps` default. Matches our other tracks.

### Variable flags you may want to tune

| Flag | Default | Alternative | Notes |
|---|---|---|---|
| `peft.r` | 32 | 16 or 64 | r=16 if Day-1 smoke run loss is unstable; r=64 if Day-2 Strix test still shows weak face discrimination. r=32 is the research-grounded middle. |
| `optimizer_lr` | 1e-5 | 2.5e-5 (Pi0.5 default) | If training loss plateaus too fast, increase. If unstable, decrease. |
| `batch_size` | 24 | 32 | If VRAM allows; gives slightly cleaner gradients. |

---

## 5 · Pre-flight checklist (must be true before launching)

- [x] **Quantile stats recomputed** — done via `eval_3/scripts/fast_recompute_quantiles.py` (27 s for 5M frames; the upstream `augment_dataset_quantile_stats.py` was loading mp4 frames sequentially with an ETA of ~12 days, so we wrote a focused action+state-only recompute since Pi0.5 only quantile-normalises those features).
- [x] **Dataset pushed to a Pi0.5-specific HF repo** — pushed to `HBOrtiz/so101_eval3_track3_v3_pi05` (separate from the SmolVLA baseline repo so we don't risk overwriting stats Hans / Sejohn are reading). Verified live at [huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_pi05](https://huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_pi05).
- [ ] **Brev VM provisioned** with RTX PRO 6000 Blackwell (96 GB), conda `lemonkey` env, lerobot in editable mode, this repo synced via `eval_3/scripts/brev/sync_to_brev.sh`.
- [ ] **HF token** available on the Brev VM at `~/LeMonkey/secrets/huggingface/token_hbortiz` so the policy push at end of training works.
- [ ] **(optional)** WandB or tensorboard logging configured.

See [`eval_3/tracks/TRACK_B_BREV_HANDOVER.md`](TRACK_B_BREV_HANDOVER.md) for step-by-step Brev setup.

---

## 6 · Run + monitor

```bash
# Local pre-flight (re-run if dataset changed since the last quantile-stats run)
bash eval_3/scripts/brev/sync_to_brev.sh

# On Brev VM
bash eval_3/scripts/brev/run_training_track_B.sh

# Monitor (from dev box)
bash eval_3/scripts/brev/follow_training.sh
bash eval_3/scripts/brev/training_status.sh
```

Expected milestones:
- t=0 to ~30 min — dataset download + tokenizer init + first batch
- t=~30 min — first training step logged; loss should start at ~3-5, decrease
- t=24 h — finished. Push to `HBOrtiz/pi05_eval3_track_B`.

If loss is NaN within the first 100 steps: **kill, reduce LR to 5e-6, restart.** Most
likely cause is bad quantile clipping. Double-check stats.json was regenerated.

---

## 7 · Strix deployment (Day 3, owner: Darius)

After Track B finishes:

```python
from lerobot.policies.pi05 import Pi05Policy
policy = Pi05Policy.from_pretrained("HBOrtiz/pi05_eval3_track_B")
# At inference:
input = {
    "observation.images.right_wrist_0_rgb": wrist_video_frame,  # NOTE: renamed key
    "observation.state": robot_state,                            # 6-d padded to 32
    "task": f"Place the coke on {target_celeb_name}.",
}
```

Three-rollout protocol per [`TODO.md`](../../TODO.md#strix-testing-protocol-darius).
Pay particular attention to OOD celebs — that's where the PaliGemma WebLI prior is
most stressed.

---

## 8 · Fallback if Track B underperforms on Day 3

If Day-2 Strix shows weak face discrimination (Pi0.5 picks wrong celeb most of the
time, like the v1 SmolVLA failure mode), the prior is weaker than hoped. Options:

1. **PaliGemma warm-start (Hans's recipe mirrored)** — LoRA fine-tune of PaliGemma on
   VGGFace2 VQA *before* the robot training. ~10 h Brev for the warm-start +
   re-launch Track B (~24 h). Total: ~34 h. Tight on Day 3 → Day 4. Need Hans's
   exact LoRA training script.
2. **Bump LoRA rank to 32 and increase LR** — gives the adapters more capacity to
   adapt PaliGemma during the 24h training run. ~24 h Brev again. Lower risk than
   option 1 but uncertain whether the extra adapter capacity helps.
3. **Drop Track B, ship Track A or C** — Tracks A and C also yield +20 bonus
   (smallest-model) and Hans's SmolVLM warm-VLM has empirical face-recognition
   verification. We forfeit the capacity hedge but keep the bonus.

Default fallback if face discrimination fails: **Option 3** (ship SmolVLA), since
Track A has empirical evidence of the warm-VLM working. Track B is the hedge, not
the primary.

---

## 9 · Sources cited

### Primary

- **Pi0.5-KI** — [arxiv 2505.23705](https://arxiv.org/abs/2505.23705) ([HTML v1](https://arxiv.org/html/2505.23705v1)) + [pi.website/research/knowledge_insulation](https://www.pi.website/research/knowledge_insulation). Eqs. 5-6 stop-gradient; §4 + Fig 4a/8 frozen-VLM ablation.
- **PaliGemma 2** — [arxiv 2412.03555](https://arxiv.org/abs/2412.03555). Backbone for Pi0.5.
- **WebLI dataset** — [Chen et al. arxiv 2209.06794](https://arxiv.org/abs/2209.06794) (PaLI). 10B image-text web-scrape, DLP-filtered.
- **Long-tail entities in VLMs** — [Parashar et al. "Neglected Tails", arxiv 2401.12425](https://arxiv.org/abs/2401.12425). Establishes that web-scale VLMs fail on long-tail named entities.
- **LoRA** — [Hu et al. arxiv 2106.09685](https://arxiv.org/abs/2106.09685). Original LoRA paper; standard target_modules choice.
- **OpenVLA-OFT** — [arxiv 2502.19645](https://arxiv.org/abs/2502.19645). Industry precedent for LoRA in VLA fine-tuning.

### Internal

- [`docs/report/EVAL_3_FINAL_PLAN.html`](../../docs/report/EVAL_3_FINAL_PLAN.html) — the canonical 4-day plan
- [`docs/EVAL_3_DATASETS.md`](../../docs/EVAL_3_DATASETS.md) — HF artifact inventory
- [`eval_3/STRATEGY.md`](../STRATEGY.md) §7d — locked plan summary
- [`eval_3/scripts/merge_track3_custom.py`](../scripts/merge_track3_custom.py) — the custom fast merger; has the buggy quantile aggregation flagged here

### LeRobot code (verified file:line)

- [`configuration_pi05.py:69`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L69) — `empty_cameras` default 0
- [`configuration_pi05.py:73-79`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L73) — QUANTILES normalization
- [`configuration_pi05.py:88-89`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L88) — `freeze_vision_encoder` + `train_expert_only` defaults
- [`configuration_pi05.py:125-133`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py#L125) — validate_features adds empty_camera_{i} keys
- [`modeling_pi05.py:421-428`](../../third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L421) — train_expert_only freezes the entire PaliGemma
- [`modeling_pi05.py:1142-1206`](../../third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L1142) — prepare_images, missing-camera handling
- [`configs/default.py:85 class PeftConfig`](../../third_party/lerobot/src/lerobot/configs/default.py#L85) — lerobot's PEFT/LoRA CLI integration
- [`lerobot_train.py:246`](../../third_party/lerobot/src/lerobot/scripts/lerobot_train.py#L246) — `policy.wrap_with_peft(...)` entry point

---

*Track B owner: Roham. Last updated 2026-05-19. Status: branch `track-b-pi05` created off main; pre-flight in progress.*
