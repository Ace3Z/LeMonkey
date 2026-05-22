# TODO.md - active work list for LeMonkey Eval 3

**Last updated:** 2026-05-18 23:10 CEST
**Status:** Post-TA-ruling pivot. 4-day sprint, 5-person team. Tracks A / B / C locked.

> **Read this file first** in every new session. It is the operational source of truth
> for what's being worked on.

---

## The plan in one table

| Track | Owner | Backbone | Mechanism | Bonus | Brev | Risk |
|---|---|---|---|---|---|---|
| **A** | **Hans** | SmolVLA-450M | Hans's warm-VLM frozen; action expert trains | **+20** | ~6 h | low - Hans verified warm-VLM works |
| **B** | **Roham** | Pi0.5-3B | VLM frozen (`train_expert_only=True`); preserves PaliGemma WebLI prior | +16 | ~24 h | medium - PaliGemma celeb prior untested |
| **C** | **Sejohn** | SmolVLA-450M | Vanilla `lerobot/smolvla_base`, defaults, no warm-VLM | **+20** | ~6 h | low - proven recipe |
| **D** (opt) | **Mahbod** | tooling | M2 ArcFace cosine distillation toolkit on SmolVLA, drop-in | - | - | n/a - on standby |
| - | **Darius** | Strix testing | deploys each checkpoint, runs 3-rollout protocol | - | - | - |

All three training tracks load the same input: `HBOrtiz/so101_eval3_track3_v3_baseline`
(178 base + 9,216 aug = 9,394 episodes, 14.3 GB, text-only prompts). The reference
camera channel is zero-padded via `--policy.empty_cameras=N`.

---

## Tonight (Day 1, May 18) - what's already done

- [x] Track 3 augmentation: 9,216 / 9,216 variants written (`datasets/eval3_track3_aug/`)
- [x] Prompt re-label: all 9,216 variants converted to default-bucket prompts
- [x] Custom fast merger written + ran (85 s vs estimated 2 h for upstream `aggregate_datasets()`)
- [x] Merged dataset pushed to **`HBOrtiz/so101_eval3_track3_v3_baseline`** (6 min upload)
- [x] Drive backup: `datasets/eval3_track3_aug.tar.zst` (13.2 GB, 38 s wall)
- [x] Code committed + pushed to `origin/main` (commits `3836e5f .. d1e8fca`)
- [x] EVAL_3_FINAL_PLAN.html distributed to team

## Tonight - still to do

- [ ] **Hans:** push warm-VLM to **`HansOrtiz/smolvlm2_celeb_warm`** (LoRA merged into base SmolVLM2-500M). Sanity-check it names Swift/Obama/LeCun correctly. Slack confirm.
- [ ] **Hans:** share LoRA training script + VGGFace2 VQA dataset format with Roham (for the Pi0.5 fallback path in Day 3).
- [ ] **Sejohn:** dev-box sanity loader on the merged HF dataset - 5 random episodes, verify 538 frames + default-bucket prompt + 6-d state + 6-d action. Slack confirm.
- [ ] **Roham:** compute Pi0.5 quantile state/action stats on the merged dataset:
  ```
  python eval_3/aug/compute_quantile_stats.py \
      --dataset HBOrtiz/so101_eval3_track3_v3_baseline \
      --output  eval_3/aug/stats/track3_v3_quantile.json
  ```
- [ ] **Roham:** write `eval_3/scripts/brev/run_training_track_{A,B,C}.sh` launch scripts.

---

## Day 2 (May 19) - 3 training launches + M2 toolkit prep

### Track A (Hans) - SmolVLA + Hans's warm-VLM

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

### Track B (Roham) - Pi0.5

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
- `train_expert_only=True` is CRITICAL - preserves PaliGemma's WebLI celeb prior.

### Track C (Sejohn) - SmolVLA safety floor

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

### Track D toolkit (Mahbod) - M2 ArcFace cosine distillation, drop-in patch

- [ ] `eval_3/aug/cache_arcface_embeddings.py` - walk every variant, compute `buffalo_l` ArcFace embedding of the target face from `augmentation.json["workspace_photos"][target_pid]`, save as `aug_cache_target_arcface.npy` in each variant dir. ~2 h dev box.
- [ ] `eval_3/aug/face_align_projector.py` - frozen 3-layer MLP module (LN → Linear(hidden, 2048) → SiLU → Dropout(0.1) → Linear(2048, 2048) → SiLU → Dropout(0.1) → Linear(2048, 512)). All params `requires_grad=False` after init.
- [ ] Patch `third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` at Backbone2Enc - expose hidden state at SmolLM2 layer 8 (of 16), compute `0.2 · L_align` (BlindVLA Eq. 9), add to loss in `forward()`.

**Hold for Day 3 decision.** Apply only if Track A or C shows face-discrimination
weakness on the Day-2 Strix test.

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

## Day 3 (May 20) - iterate

- 06:00 - Track B finishes. **Darius** deploys to Strix.
- 10:00 - team checkpoint call. Compare Tracks A / B / C. Pick the front-runner.
- If any track has weak face discrimination on OOD: **Mahbod's M2 toolkit** gets applied + the affected track re-trains (~6 h Brev).
- If Pi0.5 vanilla underperforms broadly: **mirror Hans's LoRA-on-VGGFace2 recipe on PaliGemma** (~10 h Brev) + re-train Track B (~24 h Brev). Tight but doable.

---

## Day 4 (May 21) - dry-run + ship

- 08:00–12:00 - **Darius** full eval-day dry-run on top 1–2 candidates: 9 rollouts each (3 TOY / 3 held-out IID / 3 OOD).
- 12:00 - team picks final checkpoint. Lock inference recipe:
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
- 12:00–17:00 - final checks: Strix VRAM fits, latency < 20 s, no `observation.images.reference` accessed.
- Evening - sleep before eval day.

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
