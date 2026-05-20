# Eval 3 — Consolidated Experiment Record

**Status:** authoritative record of every important Eval 3 training run, probe,
result, and number. **Last updated:** 2026-05-20.
**Validation:** every figure below was cross-checked against source files on the
feature branches by **three independent read-only validation passes** plus a
four-branch documentation audit. Discrepancies found by the validators have been
reconciled and are reflected here; see `docs/experiments/2026-05-20_branch_audit.md`.

---

## 0 · How to read this

- **Provenance tags.** Each load-bearing figure is tagged:
  - `[committed]` — verified in a committed file on a feature branch.
  - `[checkpoint]` — verified by direct inspection of the pushed adapter
    checkpoint (`adapter_config.json` / `adapter_model.safetensors`).
  - `[run report]` — from a Brev training-session completion report relayed
    2026-05-20; accurate but not yet written into a committed log file.
- **The four feature branches:** `track-b-pi05`, `track-b-warmstart-vqa`,
  `main` (= `dev/SjohnU/track_2_objectvla`, identical commit `9ee14f8`), and
  `dev/m2-arcface-toolkit`.
- **Read §6 (validity ledger) before quoting any M2 number** — one bug
  invalidated a class of results.

| Canonical | Aliases | Backbone |
|---|---|---|
| S1 | image-as-prompt v1 | SmolVLA |
| S2 | vanilla SmolVLA | SmolVLA |
| S3 | SmolVLA + warm-VLM | SmolVLA |
| S4 | "Track D", M2-on-SmolVLA | SmolVLA |
| P1 | "Track B", vanilla Pi0.5 LoRA | Pi0.5 |
| P2 | warm-VLM v1 | Pi0.5 / PaliGemma |
| P3 | warm-VLM v2 (`pi05_paligemma_celeb_warm_v2`) | Pi0.5 / PaliGemma |
| TE | "Track E", Pi0.5 + M2 + KLAL | Pi0.5 |
| T2 | "Track 2", ObjectVLA co-train | Pi0.5 |

---

## 1 · The task and the consistent failure

The SO-101 arm must place a coke can on the printed portrait of a celebrity
named in a text prompt; three printed portraits are on the table; input is
wrist camera + text + proprioception only.

**Consistent across every run:** the motor behaviour works (the arm picks up
and places the can smoothly); the policy does **not** reliably place it on the
*correct* celebrity. Placement is near-random with respect to the prompt name.

**Data:** **178 base teleop episodes** were recorded by hand, then augmented
(digital portrait replacement, all permutations) into a 9,394-episode set
(178 base + 9,216 augmented variants) — the 3-celebrity "TOY" set (Swift /
Obama / LeCun) — and a separate ~10k-episode 200-celebrity set. The M2
face-labelling pipeline processed **151** of the base episodes (the subset
with usable segmentation caches). `[committed: dev/m2-arcface-toolkit]`

---

## 2 · SmolVLA experiments

*SmolVLA = SmolVLM2-500M vision-language model + action expert. S1–S3
reconstructed from team notes (approximate); S4 verified.*

### S1 — image-as-prompt
Reference photo of the target fed as a 2nd camera; ~4,200 episodes. Strix
rollout placed the can on the wrong celebrity (asked Swift, placed on Obama).
Abandoned after the input-modality ruling (no reference image at inference).

### S2 — vanilla SmolVLA
Stock SmolVLA, ~10k augmented episodes, action loss only. Places the can,
wrong celebrity.

### S3 — SmolVLA + warm-VLM
SmolVLA's VLM fine-tuned on ~200 celebrity images first, then the action head
trained on ~10k episodes (sequential). Wrong celebrity.

### S4 — SmolVLA + M2 ArcFace distillation ("Track D") `[committed: dev/m2-arcface-toolkit]`
- M2 = an ArcFace cosine-distillation auxiliary loss, intended to shape the
  vision features toward face-identity geometry.
- A **30,000-step** run (`steps=30,000`, `save_freq=5,000`, λ=0.2, lr=5e-5,
  batch 64), on **vanilla SmolVLM2-500M** — not Hans's warm VLM
  (`HansOrtiz/smolvlm2_celeb_warm` was 404 at launch). Checkpoints were probed
  at **step 10,000 and step 25,000**.
- The SmolVLA M2 capture hook was correct (`holder.captured = inputs[0]`, live
  tensor) — so S4's representation numbers are valid (contrast the Pi0.5 M2
  detach bug, §6).
- **Representation result:** M2 shaped the face crops well — mean cosine
  **≈0.88** (the figure cited across the M2 verification docs and commit
  `6807c1e`; an attention-probe README informally restates it as ≈0.85).
- **Behaviour result:** still places on the wrong celebrity. The step-10,000
  Strix rollout failed (arm did not move to the named celeb).
- **Attention:** step-10,000 and step-25,000 attention probes are identical and
  random — see §4.2.
- **False-confidence caveat:** the step-10,000 sanity checks reported 4/4 PASS
  (action vectors differ across prompts, |Δ| = 0.06 / 0.09 / 0.07) and claimed
  the "language pathway off-axis" concern was "empirically refuted." The later
  attention probe refuted *that* claim — prompt-dependent action variation is
  necessary but not sufficient for visual grounding.
- **Predicted in advance:** the 2026-05-18 M2 validation report
  (`docs/report/2026-05-18_m2_arcface_validation.md`) predicted M2 would fail —
  4 blockers, citing BlindVLA §7.6 (the method does not help fine-grained
  under-represented concepts). The step-10,000 probe + failed rollout
  vindicated that prediction.

---

## 3 · Pi0.5 experiments

*Pi0.5 = PaliGemma 1 (SigLIP-So400m vision tower + Gemma-2B language model) +
Gemma-300M action expert; `lerobot/pi05_base`. Pi0.5 uses **PaliGemma 1**
(`paligemma_variant=gemma_2b`), not PaliGemma 2.*

### P1 — vanilla Pi0.5 LoRA ("Track B")
- **Data:** `HBOrtiz/so101_eval3_track3_v3_pi05` — 9,394 episodes, 3-celebrity
  (Swift / Obama / LeCun) permutations. (The full video data lives in the
  `..._baseline` repo; the `_pi05` repo carries the Pi0.5 exact-quantile
  `stats.json`.) `[committed: track-b-pi05, run_training_track_B.sh]`
- **Training signal:** the Pi0.5 flow-matching **action loss only**. No face
  loss, no auxiliary loss, single dataset.
- **What trained — LoRA adapters only:** rank 32, `lora_dropout=0`. **Effective
  `lora_alpha = 8`** → scaling α/r = **0.25** `[checkpoint]`. (lerobot's
  `PeftConfig` does not expose `lora_alpha`, so PEFT's library default of 8
  applies — the intended ~64 was never set; the run adapts ~8× more gently than
  intended. Note: `2026-05-19_track_b_brev_debugging.md` states PEFT "defaults
  `lora_alpha` to `r`" — that is incorrect; PEFT's `LoraConfig` default is a
  fixed 8, confirmed by `adapter_config.json`.)
- **666 adapter tensors `[checkpoint]`** = 333 LoRA modules × 2 (`lora_A`,
  `lora_B`), on three sub-models: PaliGemma Gemma-2B LM (q/k/v/o/gate/up/down,
  all 18 layers → 126 modules); SigLIP vision tower (q/k/v only, all 27 layers
  → 81 modules); Gemma-300M action expert (q/k/v/o/gate/up/down, all 18 layers
  → 126 modules). 126+81+126 = 333. **59,056,128 trainable params**
  (~1.4% of the 4.2B model) `[committed: TRACK_B_DEVBOX_HANDOVER.md
  "num_learnable_params=59056128 (59M)"]`.
  - *Why SigLIP got adapters despite `freeze_vision_encoder=True`:* that flag
    freezes SigLIP's **base** weights only; PEFT still injects trainable
    adapters wherever `target_modules` name-matches — and q/k/v_proj matches
    SigLIP's attention layers. The base tower is frozen; the adapters on it are
    not.
- **Frozen:** all base weights; `action_in_proj`, `action_out_proj`,
  `time_mlp` (action I/O heads); the multimodal projector; SigLIP `out_proj` +
  MLP (`fc1`/`fc2`, name-mismatch, never wrapped).
- **Config `[committed: brev_debugging.md]`:** 30,000 steps, batch 48,
  `num_workers=2`, lr 1e-5 (cosine), bf16, ~3.5 s/step on H100 80GB.
- **Completion `[run report]`:** step 30,000/30,000 reached 2026-05-20
  11:34 UTC; wall time 29 h 37 m; epoch 0.28; final loss **0.019**
  (0.018 at the last logged step); grad-norm 0.21; lr at the cosine floor
  2.5e-6. Pushed to `HBOrtiz/pi05_eval3_track_B`. An intermediate step-20,000
  checkpoint is at `HBOrtiz/pi05_eval3_track_B_ckpt20k` (private — picked wrong
  celebs). *(These completion metrics are from the Brev session's own training
  log + completion message; they are not yet in a committed file on
  `track-b-pi05`.)*
- **Result:** placement smoother than any SmolVLA run, but still does not
  select the correct celebrity.
- **Recipe weaknesses found afterward:** (a) `lora_alpha` effectively 8, not
  the intended ~64; (b) the action expert is LoRA-bottlenecked at r=32 and the
  action I/O heads are frozen — under-powered for learning SO-101 control.

### P2 — PaliGemma VQA warm-start v1 `[committed: track-b-warmstart-vqa]`
- **Goal:** teach the VLM face→name before any robot training.
- **Data:** `chronopt-research/cropped-vggface2-224` — VGGFace2, pre-cropped
  224×224, **9,131 identities, ~3.14M images** (the cropped redistribution;
  canonical VGGFace2 is ~3.31M); integer labels → names via
  `ProgramComputer/VGGFace2:identity_meta.csv`.
- **Task:** plain VQA — `<image>Who is the person in this image?` → name;
  teacher-forced cross-entropy on the name tokens only.
- **What trained:** LoRA on the **Gemma LM only** (r=32, α=64); the SigLIP
  vision tower **frozen**.
- **Result:** **37%** (22/60) on a 5-way VGGFace2 identification test
  (random = 20%). v1 final loss ≈ 5.0.
- **Probe that explained the ceiling:** with SigLIP frozen, its features carry
  near-zero identity information — a same-person/different-person separation of
  only **0.048** (same-person cosine ≈0.867 vs different-person ≈0.819, per the
  warm-start probe). The LM cannot name from identity-blind features.

### P3 — PaliGemma VQA warm-start v2 (`pi05_paligemma_celeb_warm_v2`) `[committed: track-b-warmstart-vqa §10]`
- Same VGGFace2 VQA task as P2.
- **What trained:** LoRA r=**64**, dropout 0.05 — on the **full SigLIP vision
  tower** (q/k/v/out_proj/fc1/fc2, 27 layers) **and** the full Gemma-2B LM
  (q/k/v/o/gate/up/down, 18 layers). (This is the v1→v2 change: v1 was LM-only
  with the vision tower frozen.) `lora_alpha = 128` `[run report]` — note the
  `train_paligemma_vqa.py` default is 64, so the launch overrode it; confirm
  against `adapter_config.json` if exact. Trainable params **113.2M (3.18%)**
  `[run report]`.
- **Config:** bf16, batch 24, grad-accum 1, gradient checkpointing OFF, lr 1e-5
  cosine, 3% warmup, epochs 0.12 (~377k of 3.14M images), **16,541 steps**
  `[run report]`, 3.0 h wall on the RTX PRO 6000. On-the-fly aug:
  RandomResizedCrop(224, 0.75–1.0), HFlip, Rotation(±12°), ColorJitter.
  Final loss ≈ 4.4 (v1 ended ≈ 5.0). Grad-norm elevated 55–75 during the run
  (pre-clip).
- **Post-training:** `merge_and_unload()` bakes adapters into PaliGemma,
  spliced back into full Pi0.5, pushed to `HBOrtiz/pi05_paligemma_celeb_warm_v2`.
- **Result A — VGGFace2 5-way, in-distribution:** **48%** (baseline 20%, v1
  37%). Monotonic 20 → 37 → 48 — unfreezing the vision tower works.
- **Result B — our 8 eval celebs, 8-way:** **2/8**, of which **1 genuine**
  (Taylor Swift, confident margin 0.32). v2 still collapses to "Cristiano
  Ronaldo" for 6/8. Weak because the 8 eval celebs are not in VGGFace2.
- **Status:** a sequential warm-start. Not yet used for a robot fine-tune.

### TE — Pi0.5 + M2 + KLAL co-train ("Track E") `[committed: dev/m2-arcface-toolkit]`
- Introduced commit `dd1d981` (2026-05-20 00:20). Combines, in one co-trained
  run: the Pi0.5 action loss + M2 (ArcFace cosine distillation) + **KLAL**
  (KL-divergence attention supervision). Loss: `total = action_loss + λ_m2·m2 +
  klal`, one backward pass.
- **KLAL** (`eval_3/aug/m2_klal.py`, **288 lines**, + `m2_pi05_policy_wrapper.py`):
  hooks `q_proj`/`k_proj` on PaliGemma LM layers, recomputes
  `softmax(QK^T·scale)` (GQA-aware) **applying the model's own RoPE** (an
  earlier no-RoPE proxy was a bug, fixed in commit `1c66387`), slices
  name-token rows × 256 image-patch columns, KL-divergence against a Gaussian
  (`target_sigma_patches = 1.5`) centred on the face-bbox. Bbox source = the M2
  toolkit's `face_labels/`.
- **M2 detach bug** (commit `5c65ce2`, 2026-05-20 12:12): before this fix the
  Pi0.5 M2 capture pre-hook stored `h.detach().clone()`, severing autograd —
  M2 loss was logged but trained **zero parameters**. KLAL was unaffected
  (separate hookset). See §6.
- **Config changes after the fix** (commit `6807c1e`, 2026-05-20 14:49):
  `M2_LAMBDA` 0.2 → **1.0** (Pi0.5 trains at lr 1e-5, 5× below SmolVLA's 5e-5,
  so M2's effective step was 5× too small; BlindVLA's verbatim paper default is
  λ=0.2, so 1.0 is a deliberate lr-compensating deviation); `KLAL_LAYERS`
  6,10,14,17 → **10,14,17** (layer 6 is frozen by the partial-freeze, trained
  0 params, diluted the layer-averaged loss).
- **Status:** post-fix smoke test PASSES — M2 `mean_cos` climbs +0.011 →
  +0.562 over 800 steps; M2 and KLAL both train. Retrain config finalized;
  **mid-flight on Brev**. The decisive step-~10k face-binding probe has not yet
  returned.

### T2 — ObjectVLA co-train ("Track 2") `[committed: main]`
- Planned: co-train Pi0.5 on robot episodes + bbox-grounded face VL pairs at a
  **10:1 robot:VL ratio** (ObjectVLA recipe — published mechanism: without the
  bbox grounding, OOD success drops 64% → 19%, i.e. a +45 pp swing).
- VL-pairs manifest in use: `HBOrtiz/eval3_objectvla_vl_pairs` (built from
  wrist-cam teleop frames — ~176k rows, 192 celebs).
- **Status — not runnable yet.** The mixed-batch co-train loop is unimplemented
  scaffold: `lerobot_train_with_vl_cotrain.py` `main()` prints an integration
  checklist and `return 0` (no training). ~4–5 h of Brev integration work
  remains. `TRACK_OBJECTVLA_ENHANCED.md` (Sejohn's B-1..B-7 enhanced spec)
  supersedes the baseline `TRACK_OBJECTVLA.md` runbook.

---

## 4 · Diagnostic probes

### 4.1 Frozen-SigLIP identity-separation probe `[committed: track-b-warmstart-vqa]`
With the SigLIP vision tower frozen, matching vs non-matching face pairs are
nearly indistinguishable — an identity separation of only **0.048**
(same-person cosine ≈0.867 vs different-person ≈0.819). This is the basis for
the P2→P3 decision to unfreeze the vision tower.

### 4.2 Attention probe — the root-cause finding `[committed: dev/m2-arcface-toolkit]`
Controlled probe: fix the scene, vary only the celebrity name in the prompt
(Swift → Obama → LeCun), inspect the name-token → image cross-attention.

- **Result (S4, SmolVLA):** the name-token attention does **not change** with
  the name. Its argmax sits on the same fixed background patch — 8×8-grid cell
  **(1,7)**, a top-right corner — for *every* prompt at *every* probed layer
  (9, 11, 13, 15). Attention weight is **below uniform** (uniform = 1/64 =
  0.0156; observed max 0.0052–0.0240 at layers 9–13). Identical at step 10,000
  and step 25,000. `[2026-05-19_attention_probe_step10000/README.md]`
- **VLM face-detection probes:** the PaliGemma warm-start **v1 and v2 are both
  attention-sink-locked** — name-token argmax constant across all three celeb
  prompts (v1 at 3 of 4 probed layers; v2 likewise prompt-invariant /
  below-uniform). The frozen-VLM path is dead.
  `[2026-05-20_vlm_face_detection_probes/README.md]`
- **Diagnosis:** the failure is **attention routing** — the name-token →
  face-patch binding — not representation quality. M2 produced good
  representations (§2 S4, mean cos ≈0.88) and the routing still did not move.
  KLAL (§3 TE) supervises this routing directly.

### 4.3 Pi0.5 gating probes G1 / G2 `[committed: dev/m2-arcface-toolkit: 2026-05-20_pi05_gating/README.md]`
Run before committing to Track E:
- **G1 — vanilla-PaliGemma attention probe: PASS.** Unlike the warm-VLMs and
  SmolVLA (fully sink-locked), vanilla PaliGemma's name-token argmax **does
  shift** across prompts at layers 6/10/14 (it sinks only at layer 17). So the
  base PaliGemma is not hopelessly sink-locked — there is routing capacity to
  build on.
- **G2 — zero-shot celebrity VQA: MARGINAL.** Obama recognized; Swift and LeCun
  not. 0/3 open-ended, 1/3 verification-style. The base VLM has thin celebrity
  knowledge — consistent with the 0/14 zero-shot probe from 2026-05-09.

### 4.4 Celebrity-separability / confusion matrix `[committed: main]`
Over 192 celebrities (960 task strings → 192 centroids via
`build_task_to_centroid.py`): the celebs are well-separated. The single most
confusable centroid pair is `kate_beckinsale` vs `mira_murati` at cosine
**0.316** (`confusable_topk.json`), with the p99 of pairwise cosines ≈0.163.
(The per-run mean pairwise cosine is computed by `build_confusion_matrix.py`
but not recorded in a committed results file.)

### 4.5 Leave-one-out identity check `[committed: dev/m2-arcface-toolkit]`
Scraped-bank LOO top-1 = **1438/1445 = 99.5%**.

---

## 5 · Data audits and pipeline bugs

### 5.1 Miscoded augmentation supervision `[committed: 2026-05-19_m2_data_audit.md]`
**4 of 151** source episodes (2.6%) have miscoded slot-R supervision — the
augmentation pipeline failed to replace a Swift portrait with LeCun
(`orig_R=S, new_R=L` pattern):
- `quick_lecun_SLO_ep01_20260511_211540`
- `quick_lecun_SLO_ep04_20260511_211734`
- `quick_lecun_SOL_ep01_20260511_212006`
- `quick_obama_SLO_ep05_20260511_204851`

For `quick_lecun_SOL_ep01_20260511_212006`, **17/17** LeCun-at-R variants render
Swift. Upstream bug in `generate_aug_track3.py`; not yet swept across the full
9,216-variant set.

### 5.2 Undersized face detection `[committed: 2026-05-19_m2_data_foundation.md]`
The first foundation run used InsightFace `det_size=320` (default) — faces are
only ~10% of a 640×480 frame, so only **50.4%** of frames had all 3 faces
detected (5 source recordings at 0%). A re-run at `det_size=640` is queued;
its re-validation result is not yet documented.

### 5.3 Broken scraped-bank identity `[committed: 2026-05-19_m2_data_foundation.md]`
`oier_mees` is structurally broken — 6/8 photos misclassify under leave-one-out
(intra-celeb cosine 0.05–0.13, noise-level). Remove/re-scrape before use.

---

## 6 · Validity ledger — what to trust, what not to

| Item | Status | Reason |
|---|---|---|
| S4 (SmolVLA) M2 mean-cosine ≈0.88 | **VALID** | SmolVLA M2 hook used a live tensor — no detach bug |
| Any Pi0.5 M2 `mean_cos` before commit `5c65ce2` (2026-05-20 12:12) | **INVALID** | The Pi0.5 M2 capture pre-hook detached the hidden state — M2 loss logged but trained zero parameters. Includes the step-1000 Track E probe's M2 numbers. |
| KLAL numbers (any date) | **VALID** | KLAL used a separate hookset, no detach |
| P3 (warm-VLM v2) 48% / 2-of-8 | **VALID** | VQA warm-start, separate pipeline, unaffected by the M2 bug |
| `lora_alpha` in any lerobot-train Pi0.5 run | **= 8 (not the intended 64)** | lerobot `PeftConfig` does not expose `lora_alpha`; PEFT's library default (8) applies. `brev_debugging.md`'s claim that it "defaults to r" is incorrect. |
| SmolVLA M2 bbox alignment, pre-fix | **WAS MISALIGNED** | `_resize_with_pad_box` in `modeling_smolvla.py` used centered padding, but LeRobot pads left+top only — every bbox was off by ~64 px (one patch row). Fixed before the S4 run (`2026-05-19_m2_review_findings.md`, Reviewer A). |
| KLAL "no-RoPE" recompute | **SUPERSEDED** | An early KLAL version recomputed attention without RoPE (a proxy); commit `1c66387` replaced it with the model's real RoPE. Current code applies RoPE. |

### Other gotchas (carry to any new run)
- **transformers 4.55 breaks Pi0.5** without `pi05_inference_patch.apply()` —
  patches `embed_image`/`get_image_features` (no `pooler_output`) and
  `create_causal_mask` (kwarg rename `inputs_embeds`→`input_embeds`). Apply
  before any `PI05Policy` construction. `[committed: pi05_inference_patch.py]`
- **Dataset 160-frame gap:** `HBOrtiz/so101_eval3_track3_v3_baseline`
  `info.json.total_frames` = 5,053,972 but actual parquet rows = 5,053,812 →
  `IndexError` ~1,400 steps into training. Patch `total_frames` to the real
  count. `[committed: brev_debugging.md Failure 7]`
- **torchcodec dataloader OOM:** each worker holds ~43 GB of mp4 buffers
  (`anon-rss:43230644kB`); at the default `num_workers=4` that is ~172 GB →
  kernel OOM-kill after ~2 h. Use `num_workers=2`. `[committed: brev_debugging.md
  Failure 8]`
- **PEFT silent fallback:** `PI05Policy.from_pretrained` on a PEFT-adapter
  directory silently loads **random weights** if `peft` is not installed —
  observed live as ~100-scale random actions. Always verify `peft` is present.
- **`compile_model=True` must be OFF for rollout** — torch.compile autotune
  runs inside the episode loop on a laptop GPU and starves the policy of time.

---

## 7 · Open / unmeasured — the gating risks

1. **Strix VRAM + latency probe has never been run.** `probe_pi05_strix.py`
   exists on `main`; no results. Thresholds: < 14 GB VRAM (16 GB card), < 20 s
   p95 latency (the TA rule). A failure on either disqualifies or forces a
   pivot off Pi0.5. **This is the largest unmeasured risk.** Run it.
2. **Track 2 (T2) co-train training loop is unimplemented.**
   `lerobot_train_with_vl_cotrain.py` `main()` prints an integration checklist
   and returns 0 — the robot/VL alternation loop does not exist. ~4–5 h of work
   remains.
3. **Track E (TE) step-~10k face-binding probe** has not returned. That probe
   is the decisive "does KLAL bind names to faces" test.
4. **`det_size=640` face-detection re-validation** result not yet documented.
5. **The 4 miscoded episodes** (§5.1) have not been swept across all 9,216
   variants, and the bad variants have not been removed or re-rendered.

---

## 8 · Source branches

| Branch | Holds |
|---|---|
| `track-b-pi05` | Pi0.5 strategy docs, P1 (Track B) recipe + debugging, pivot research |
| `track-b-warmstart-vqa` | P2 / P3 PaliGemma VQA warm-start (v1, v2) + their evals |
| `main` (= `dev/SjohnU/track_2_objectvla`) | Track 2 ObjectVLA co-train scaffold, VL-pairs tooling, Strix probe |
| `dev/m2-arcface-toolkit` | S4 (Track D), TE (Track E), M2 + KLAL toolkit, all attention/face probes, data audits |

---

*Compiled 2026-05-20 from the four-branch documentation audit
(`docs/experiments/2026-05-20_branch_audit.md`) and the source experiment logs
on each branch. Every figure was cross-checked by three independent read-only
validation passes; discrepancies were reconciled. Figures tagged `[run report]`
come from Brev training-session reports relayed 2026-05-20 and should be
written into a committed training log when convenient.*
