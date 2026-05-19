# TODO.md — active work list for LeMonkey Eval 3

**Last updated:** 2026-05-18 23:10 CEST
**Status:** Post-TA-ruling pivot. 4-day sprint, 5-person team. Tracks A / B / C locked.

> **Read this file first** in every new session. It is the operational source of truth
> for what's being worked on. The deeper "why" lives in:
>
> - [`docs/report/EVAL_3_FINAL_PLAN.html`](docs/report/EVAL_3_FINAL_PLAN.html) — the canonical 4-day plan (text-only, 5-person)
> - [`docs/EVAL_3_DATASETS.md`](docs/EVAL_3_DATASETS.md) — what's on HF and what to use
> - [`eval_3/STRATEGY.md`](eval_3/STRATEGY.md) — full strategy history
> - [`docs/report/EVAL_3_RESEARCH_REPORT.md`](docs/report/EVAL_3_RESEARCH_REPORT.md) — M1–M8 mechanism enumeration + audits

---

## The plan in one table

| Track | Owner | Backbone | Mechanism | Bonus | Brev | Risk |
|---|---|---|---|---|---|---|
| **A** | **Hans** | SmolVLA-450M | Hans's warm-VLM frozen; action expert trains | **+20** | ~6 h | low — Hans verified warm-VLM works |
| **B** | **Roham** | Pi0.5-3B | VLM frozen (`train_expert_only=True`); preserves PaliGemma WebLI prior | +16 | ~24 h | medium — PaliGemma celeb prior untested |
| **C** | **Sejohn** | SmolVLA-450M | Vanilla `lerobot/smolvla_base`, defaults, no warm-VLM | **+20** | ~6 h | low — proven recipe |
| **D** (opt) | **Mahbod** | tooling | M2 ArcFace cosine distillation toolkit on SmolVLA, drop-in | — | — | n/a — on standby |
| — | **Darius** | Strix testing | deploys each checkpoint, runs 3-rollout protocol | — | — | — |

All three training tracks load the same input: `HBOrtiz/so101_eval3_track3_v3_baseline`
(178 base + 9,216 aug = 9,394 episodes, 14.3 GB, text-only prompts). The reference
camera channel is zero-padded via `--policy.empty_cameras=N`.

---

## Tonight (Day 1, May 18) — what's already done

- [x] Track 3 augmentation: 9,216 / 9,216 variants written (`datasets/eval3_track3_aug/`)
- [x] Prompt re-label: all 9,216 variants converted to default-bucket prompts
- [x] Custom fast merger written + ran (85 s vs estimated 2 h for upstream `aggregate_datasets()`)
- [x] Merged dataset pushed to **`HBOrtiz/so101_eval3_track3_v3_baseline`** (6 min upload)
- [x] Drive backup: `datasets/eval3_track3_aug.tar.zst` (13.2 GB, 38 s wall)
- [x] Code committed + pushed to `origin/main` (commits `3836e5f .. d1e8fca`)
- [x] EVAL_3_FINAL_PLAN.html distributed to team

## Tonight — still to do

- [ ] **Hans:** push warm-VLM to **`HansOrtiz/smolvlm2_celeb_warm`** (LoRA merged into base SmolVLM2-500M). Sanity-check it names Swift/Obama/LeCun correctly. Slack confirm.
- [ ] **Hans:** share LoRA training script + VGGFace2 VQA dataset format with Roham (for the Pi0.5 fallback path in Day 3).
- [ ] **Sejohn:** dev-box sanity loader on the merged HF dataset — 5 random episodes, verify 538 frames + default-bucket prompt + 6-d state + 6-d action. Slack confirm.
- [ ] **Roham:** compute Pi0.5 quantile state/action stats on the merged dataset:
  ```
  python eval_3/aug/compute_quantile_stats.py \
      --dataset HBOrtiz/so101_eval3_track3_v3_baseline \
      --output  eval_3/aug/stats/track3_v3_quantile.json
  ```
- [ ] **Roham:** write `eval_3/scripts/brev/run_training_track_{A,B,C}.sh` launch scripts.

---

## Day 2 (May 19) — 3 training launches + M2 toolkit prep

### Track A (Hans) — SmolVLA + Hans's warm-VLM

```
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.vlm_model_name=HansOrtiz/smolvlm2_celeb_warm \
  --policy.freeze_vision_encoder=True \
  --policy.train_expert_only=True \
  --policy.empty_cameras=1 \
  --policy.optimizer_lr=5e-5 \
  --policy.compile_model=False \
  --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \
  --batch_size=64 --steps=30000 \
  --output_dir=outputs/smolvla_track_A \
  --policy.push_to_hub=True \
  --policy.repo_id=HBOrtiz/smolvla_eval3_track_A
```

- Launch by 07:00. Finishes ~13:00. Push to `HBOrtiz/smolvla_eval3_track_A`.
- Hand off to **Darius** for Strix deployment.

### Track B (Roham) — Pi0.5

```
lerobot-train \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.freeze_vision_encoder=True \
  --policy.train_expert_only=True \
  --policy.empty_cameras=3 \
  --policy.optimizer_lr=1e-5 \
  --policy.compile_model=True \
  --policy.gradient_checkpointing=True \
  --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \
  --batch_size=24 --steps=30000 \
  --output_dir=outputs/pi05_track_B \
  --policy.push_to_hub=True \
  --policy.repo_id=HBOrtiz/pi05_eval3_track_B
```

- Launch by 06:00. Finishes ~Day 3 06:00 (~24 h). Push to `HBOrtiz/pi05_eval3_track_B`.
- `train_expert_only=True` is CRITICAL — preserves PaliGemma's WebLI celeb prior.

### Track C (Sejohn) — SmolVLA safety floor

```
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.freeze_vision_encoder=True \
  --policy.train_expert_only=True \
  --policy.empty_cameras=1 \
  --policy.optimizer_lr=5e-5 \
  --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \
  --batch_size=64 --steps=30000 \
  --output_dir=outputs/smolvla_track_C \
  --policy.push_to_hub=True \
  --policy.repo_id=HBOrtiz/smolvla_eval3_track_C_baseline
```

- Same as Track A but without Hans's warm-VLM (uses `lerobot/smolvla_base` vanilla).
- Launch by 07:00. Finishes ~13:00.

### Track D toolkit (Mahbod) — M2 ArcFace cosine distillation, drop-in patch

- [x] `eval_3/aug/m2_alignment.py` — frozen projector (LN → 960→2048 → SiLU → Drop(0.1) → 2048→2048 → SiLU → Drop(0.1) → 2048→512) + `m2_align_loss` (BlindVLA Eq. 9). Commit `b66091e`.
- [x] `eval_3/aug/m2_dataloader.py` — `M2SupervisionBuilder` reads `face_labels/` + `celeb_embeddings.json` + per-variant `augmentation.json`, emits `bbox_masks/bbox_valid/target_centroids`. Commit `b66091e`.
- [x] `eval_3/aug/m2_policy_wrapper.py` + `eval_3/scripts/lerobot_train_with_m2.py` — `M2WrappedPolicy` is a drop-in wrapper; launcher monkey-patches `make_policy` so upstream lerobot stays untouched. Includes a launch-time eligibility sanity check (commit `b66091e`, plus fix commits `e32e069`/`27626e6`/`fd7ab38`/`07d4f49`/`9c76ced`/`b8009b1`/`ea1501b`).
- [x] **Track D run launched 2026-05-19 06:36 CEST on Brev `time2sleep`** (A100 80GB). Step ~1.6 k / 30 k at last check, `mean_cos = +0.74`, step time ~2 s, ETA ~17-20 h. Output: `HBOrtiz/smolvla_eval3_track_D_m2_mahbod`. Detailed log: [`docs/experiments/2026-05-19_track_D_m2_brev_launch.md`](docs/experiments/2026-05-19_track_D_m2_brev_launch.md).
- [ ] **Strix-side pre-cache** (handoff to Darius): before eval-day load, pull the architecture+tokenizer for `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` onto the eval box. The training config has `load_vlm_weights=False` (VLM weights come from our safetensors), but `SmolVLAPolicy.from_pretrained` still resolves `vlm_model_name` to instantiate the graph. One-liner: `huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.

**Held for Day 3 decision applied early** because Hans's warm-VLM
(`HansOrtiz/smolvlm2_celeb_warm`) wasn't on HF yet at launch time; we used
vanilla `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, so this run is
effectively "M2 on top of Track C" rather than "M2 on top of Track A". If
Hans publishes the warm-VLM before Day 3, swap `--policy.vlm_model_name`
and re-run.

### Strix testing protocol (Darius)

For every checkpoint that finishes training:

1. `huggingface-cli download HBOrtiz/<track> --local-dir /tmp/<track>`
2. Spin up Strix + SO-101. Verify camera1 stream is live.
3. Place 3 printouts in a fixed layout (don't change between rollouts within a track).
4. 3 rollouts per checkpoint:
   - (a) TOY: 3 IID celebs in 3 positions, name them by prompt
   - (b) Held-out IID: same celebs, different photos (heldout_03/04/05)
   - (c) OOD: rotate one printout to a non-IID celeb (e.g. Federer, Bezos)
5. Log `{track, rollout_type, celeb, layout, success(0/1), notes}` to shared sheet.
6. Capture video of any failure (positional shortcut, wrong celeb, timeout).

---

## Day 3 (May 20) — iterate

- 06:00 — Track B finishes. **Darius** deploys to Strix.
- 10:00 — team checkpoint call. Compare Tracks A / B / C. Pick the front-runner.
- If any track has weak face discrimination on OOD: **Mahbod's M2 toolkit** gets applied + the affected track re-trains (~6 h Brev).
- If Pi0.5 vanilla underperforms broadly: **mirror Hans's LoRA-on-VGGFace2 recipe on PaliGemma** (~10 h Brev) + re-train Track B (~24 h Brev). Tight but doable.

---

## Day 4 (May 21) — dry-run + ship

- 08:00–12:00 — **Darius** full eval-day dry-run on top 1–2 candidates: 9 rollouts each (3 TOY / 3 held-out IID / 3 OOD).
- 12:00 — team picks final checkpoint. Lock inference recipe:
  ```python
  policy = SmolVLAPolicy.from_pretrained("HBOrtiz/<chosen-track>")
  # OR
  policy = Pi05Policy.from_pretrained("HBOrtiz/<chosen-track>")
  # required at inference:
  input = {
      "observation.images.camera1": wrist_video[-1],
      "observation.state": robot_state,
      "task": f"Place the coke on {target_celeb_name}.",
  }
  ```
- 12:00–17:00 — final checks: Strix VRAM fits, latency < 20 s, no `observation.images.reference` accessed.
- Evening — sleep before eval day.

---

## Eval-day reminders (must be true at submission)

- [ ] Policy input is `camera1 + state + text prompt` only (no `observation.images.reference`, no asset-table lookup at inference)
- [ ] `--policy.empty_cameras=N` set correctly so the unused camera slot is zero-padded
- [ ] M6/M7 stay parked (TA-disallowed); M1+M2+M3+M4-lite+M5 are training-only and OK
- [ ] Inference latency < 20 s per rollout (Pi0.5 must be VRAM-checked on Strix)

---

## How to mark items done

Change `- [ ]` to `- [x]` and add a short note with the commit hash:
```
- [x] eval_3/aug/cache_arcface_embeddings.py  (commit a1b2c3d)
```
