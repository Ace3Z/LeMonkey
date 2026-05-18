# M2 ArcFace cosine distillation patch — research validation

**Date:** 2026-05-18
**Author:** Claude session, cross-validating Mahbod's Track D specification.
**Status:** **Recommend re-spec or reallocate Mahbod's Day-2.** The patch as
currently described in `TODO.md` (lines 122-130) and `docs/report/EVAL_3_FINAL_PLAN.html`
(Track D block) will not deliver its intended effect under the post-2026-05-18
text-only constraint.

## Executive verdict

Three independent agents (paper deep-read, precedent survey, hostile-witness
code review) converged on **four blocker findings** that make the patch a
likely no-op or anti-discriminative under the current Track A/C launch
configuration:

1. **`train_expert_only=True` (the default in both Track A and Track C launch scripts)
   freezes the entire VLM.** The M2 loss has no gradient path to the parameters
   it is supposed to shape. Without flipping this flag, the patch is a literal
   no-op.
2. **One cached ArcFace embedding (target only) + camera1 frame showing 3
   faces = anti-discriminative loss.** The current spec ("mask-gate to face
   patches") pulls distractor-face patches toward the target identity. To be
   discriminative, the loss must route per-face embeddings to per-face patch
   regions (or be reformulated as contrastive target-vs-distractor).
3. **The published BlindVLA recipe is OpenVLA-7B + general teachers
   (C-RADIOv3, DINOv2, Theia, SigLIP), not 450M + ArcFace.** Three deviations
   simultaneously (smaller student, narrow-domain teacher, mask-gating —
   none of which the paper validates) put the patch outside its evidence
   base. The paper's §7.6 explicitly warns the method does **not** help on
   fine-grained, under-represented visual concepts — i.e. exactly our
   celebrity-identity task.
4. **The patch is not "drop-in."** Realistic diff is **120-180 LoC across
   three files** (`modeling_smolvla.py`, `smolvlm_with_expert.py`, new
   projector module + dataloader changes for per-pid face-region routing).
   Plus a custom cache step that must rectify printed faces via
   `portrait_corners.json` before running ArcFace, not the magazine
   references.

The proposed combination is also **first-mover** — Agent B found no
published paper distilling a face-recognition encoder into a VLM/VLA.
Closest precedent is BlindVLA itself with a general teacher.

## Method

Three parallel general-purpose agents, briefed independently:

- **Agent A:** verify BlindVLA paper (arxiv 2510.25616) + repo
  (`CognitiveAISystems/BlindVLA`) against our notes on Eq. 9, λ=0.2, frozen
  MLP architecture, Backbone2Enc layer choice, and tested
  students/teachers.
- **Agent B:** survey published precedent for face-encoder→VLM/VLA
  distillation; identify alternatives that work under text-only inference;
  compare ArcFace vs SigLIP/CLIP on face verification.
- **Agent C:** read `modeling_smolvla.py`, `smolvlm_with_expert.py`,
  `5_verify_identity.py`, and the Track A/C launch scripts; assess patch
  feasibility, silent-failure modes, and whether the loss actually moves
  the action expert off its positional shortcut.

## Findings

### F1 — Paper recipe is verified verbatim (Agent A)

| Claim | Status | Source |
|---|---|---|
| Loss `−(1/k)·Σ cos(F.normalize(u_j), F.normalize(z_j))` | ✓ verbatim | `finetune_align.py:423-427` |
| λ = 0.2 constant from step 0 | ✓ verbatim | `finetune_align.py:121`; paper §8 |
| Frozen 3-layer MLP projector (LN → 2048 → 2048 → out) | ✓ verbatim | `finetune_align.py:138-152` |
| **Output dim = teacher patch-token dim, not 512** | ✓ verbatim — **our spec is wrong if we want a direct port** | `finetune_align.py:268-273` — DINOv2-L=1024, C-RADIOv3-L=1280, Theia=768. 512 matches **only** buffalo_l ArcFace. |
| Injection at "Backbone2Enc," layer 16 (not 8) | ✓ verbatim | `finetune_align.py:122` default; paper Table 12 |
| Loss applied to **all** visual patches, not mask-gated | ✓ verbatim — mask-gating is **our novel deviation** | `finetune_align.py:391-395` |
| Headline gains | ≈right deltas (+12pp semantic OOD, +9pp vision OOD) but **on SimplerEnv + VL-Think, not LIBERO** as our docs claim | paper Table 1 |
| Student tested | OpenVLA-7B only; pi0.5 alignment in repo TODO, unchecked | repo README |
| Teachers tested | DINOv2-L/G, C-RADIOv3-L/H, Theia (SigLIP referenced in paper but not in released code) | `finetune_align.py:165-178`; paper Table 4 |
| ArcFace / faces / fine-grained identity | Never mentioned | — |
| **§7.6 — where the method does NOT help** | "improvements on abstract domains … remain mostly unchanged"; "VL-Think domain forgetting persists" for under-represented concepts | paper §7.6 |

### F2 — No ArcFace→VLM precedent; ArcFace strictly dominates SigLIP/CLIP on faces (Agent B)

- **No published paper** distils ArcFace (or any face-recognition encoder)
  into a VLM/VLA hidden state. Closest: BlindVLA (general teachers, not
  face). VisPer-LM (NeurIPS 2025, [arxiv 2412.09585](https://arxiv.org/abs/2412.09585))
  distils segmentation/depth/generation experts — dense spatial signals,
  not a global identity vector. ArcFace's 512-D identity vector is a
  different geometric regime than what VLM patch tokens were shaped for.
- **No VLA explicitly trained for face/identity grounding.** ObjectVLA
  ([arxiv 2502.19250](https://arxiv.org/abs/2502.19250)) does
  object-name bbox co-training, but requires visual exemplars at fine-tune
  time and provides no published face transfer number.
- **ArcFace vs CLIP/SigLIP on face verification at FMR=10⁻⁴** ([arxiv 2507.03541](https://arxiv.org/html/2507.03541v2)):
  | Encoder | LFW | WebFace42M | IJB-C |
  |---|---|---|---|
  | ArcFace `buffalo_l` | **98.86 %** | **98.67 %** | **95.06 %** |
  | CLIP-L-14-336 | 63.49 % | 30.86 % | 87.33 % |
  | OpenCLIP-H-14 | 64.97 % | 32.58 % | — |

  The discriminative-headroom argument is real. **If** the distillation
  transfers (open question), the upside is significant. **If** it doesn't,
  we are spending Day-2 engineering on a research bet.

### F3 — Code-grounded blockers (Agent C)

1. **`train_expert_only=True` blocker.**
   `third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:144-146`
   sets every VLM parameter `requires_grad=False`. Both Track A
   (`TODO.md:67`) and Track C (`TODO.md:110`) launch with
   `train_expert_only=True`. Without flipping this flag, the M2 loss
   produces a non-zero scalar but has zero gradient to any weight it is
   supposed to shape. **The patch silently no-ops.** Flipping the flag
   destroys Hans's warm-VLM prior (Track A) or the SmolVLA pretrain
   (Track C) — both rely on `train_expert_only=True` to avoid the
   catastrophic forgetting documented in Pi0.5-KI Fig. 6b (94 → 74 %
   OOD).

2. **Single-target embedding + 3-faces-in-camera1 = anti-discriminative.**
   `aug_cache_target_arcface.npy` is one embedding (the target's). Camera1
   shows all 3 printed portraits every frame. A mask that covers "face
   patches" without per-face routing pulls distractor patches toward the
   target identity — exactly the opposite of what we want. To be sound,
   the loss needs per-pid face-region masks routed through the batch via
   `portrait_corners.json` + `augmentation.json["workspace_photos"]`, with
   distractors either excluded or used as negatives. The Track D spec does
   not describe this routing.

3. **Realistic diff size.** Capturing layer-N hidden state requires
   modifying `SmolVLMWithExpertModel.forward`
   (`smolvlm_with_expert.py:403-498`) to expose intermediates, then
   plumbing the captured tensor through `VLAFlowMatching.forward`
   (`modeling_smolvla.py:763-799`) and `SmolVLAPolicy.forward`
   (`modeling_smolvla.py:379-402`) to aggregate into the returned loss.
   The prefix layout (image patches at front + optional special tokens
   + language + state) depends on `add_image_special_tokens`
   (`configuration_smolvla.py:90`) and `empty_cameras` — a "mask the wrong
   tokens" bug is the highest-probability silent failure.

4. **Cached-embedding domain gap.** `5_verify_identity.py:79-108` runs
   `buffalo_l` only after rectifying the printed quad to 224×320 via
   `crop_portrait_from_corners` (using SAM-2 corners). If the cache is
   computed from magazine reference photos (high quality), it distils the
   model toward `clean-magazine-face` features, not the
   `wrist-cam-printed-face` distribution the model actually sees at
   inference. BlindVLA never addresses cross-domain teacher-student
   distillation.

5. **The shortcut is at the action-head level, not perception.** v1's
   documented failure (`docs/EVAL_3_OPTIONS.md` §1.2 hypothesis 3,
   confirmed empirically) is that the action expert learned
   `(state, generic-visual) → motion` and ignored language. Better VLM
   patch features may not change this because the action expert is
   updating in parallel and can keep its shortcut if the language pathway
   doesn't reinforce identity selection. The patch addresses (G2)
   representation gap; the shortcut is partly a (G1)+(G3 action-head
   prior) failure.

## What would a defensible Track D look like?

If the team insists on activating M2, the only sound form is a **re-spec**:

1. `train_expert_only=False` with VLM layers 0–7 frozen, 8–15 trainable
   (preserves most of Hans's warm prior, gives the loss something to
   move).
2. **Per-pid face-region masks** routed through the batch via the
   existing `portrait_corners.json` + `augmentation.json` artifacts.
3. **Cache ArcFace embeddings from the printed-face crops**, not the
   magazine references — use the rectification path
   `5_verify_identity.py:crop_portrait_from_corners`.
4. **Target-patches-only** loss (distractors excluded); or, strictly
   stronger, a **contrastive variant** (cosine to target + margin from
   distractors).
5. **Projector output dim = 512** (matches ArcFace) **only if** loss is
   reformulated for one-embedding-per-face. The BlindVLA k-patch sum
   does not directly apply — instead pool target-region VLA patches to
   one vector, project to 512-D, cosine against the ArcFace embedding.
6. **Layer choice** — BlindVLA defaults layer 16 of OpenVLA's LLM. For
   SmolLM2 truncated to 16 layers, the analogous mid-to-late position is
   ~layer 12, not layer 8. Layer 8 is from `EVAL_3_FINAL_PLAN.html` and
   appears to have no source — verify before patching.

That is **1.5–2 days** of engineering, **plus** small-scale validation
against a BlindVLA-C-RADIOv3 baseline at matched compute, **plus** the
~6 h Brev re-train if the team activates it on Day 3.

## Cheaper alternatives that dominate on cost/risk

Agent C surfaced four alternatives that get most of the signal for
≤4 hours of work and zero surgery on `third_party/lerobot/`:

- **A. Prompt-suffix identity injection at training time.** Append
  `"[target=Taylor_Swift; distractors=Obama,LeCun]"` to the task string.
  The LM gets the identity binding via cross-attention. Eval-day prompts
  drop the suffix. ~2 h. (Note: this is a soft variant of M5/M8 that
  doesn't require an asset table at inference.)
- **B. Patch-level CE classification head on face patches.** Train a
  tiny linear head SmolVLM-patches → 192-way celeb-id, masked to face
  regions via `portrait_corners.json`. Same supervision signal as M2
  with cleaner gradients, no frozen-projector cargo cult, no Eq. 9
  ambiguity. Bypasses `modeling_smolvla.py` if wired as an external
  hook. ~6 h.
- **C. Layout-hint prompt re-label.** "Place the coke on Swift — the
  leftmost portrait" turns the IID rollouts into a task whose answer is
  in the prompt. Eval-day prompt is the canonical phrasing. Trains the
  action expert to ground language to position. ~3 h.
- **D. Hans's warm-VLM is already M5.** If Track A on the merged dataset
  shows even 5/9 success, Track D's marginal value is small because the
  face-binding is already in the frozen VLM. Track A's Day-2 Strix test
  is the decision gate.

## Recommendation for Mahbod's Day-2

1. **Do not build the patch as currently specified.** It is a research
   bet outside the source paper's evidence, and the most common
   activation path (`train_expert_only=True`) makes it a silent no-op.
2. **Defer the decision to Day-3.** Day-2 Strix tests of Tracks A and C
   tell us whether face discrimination is actually the bottleneck.
3. **Reallocate Mahbod's Day-2** to one of:
   - **Alternative A or C** above (prompt-side fixes that can be applied
     before any Brev re-train and don't require code surgery), OR
   - **Build a small evaluation harness** that, given a Strix rollout
     video + the celeb prompt, classifies the failure mode (positional
     shortcut vs face-confused vs correct). Useful regardless of which
     mechanism we apply.
4. **If Day-3 decision is "M2 anyway"**, re-spec per the 6 points above
   before any code is written. Quote this doc as the spec.

## Sources

### Primary
- BlindVLA paper — [arxiv 2510.25616](https://arxiv.org/abs/2510.25616)
- BlindVLA code — [github.com/CognitiveAISystems/BlindVLA](https://github.com/CognitiveAISystems/BlindVLA),
  especially `openvla/vla-scripts/finetune_align.py`
- ArcFace — [arxiv 1801.07698](https://arxiv.org/abs/1801.07698)
- InsightFace `buffalo_l` — [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- SmolVLA — [arxiv 2506.01844](https://arxiv.org/abs/2506.01844)
- Pi0.5-KI — [arxiv 2505.23705](https://arxiv.org/abs/2505.23705)

### Survey
- Foundation vs Domain-Specific FR — [arxiv 2507.03541](https://arxiv.org/html/2507.03541v2)
- VisPer-LM — [arxiv 2412.09585](https://arxiv.org/abs/2412.09585)
- MoVE-KD — [arxiv 2501.01709](https://arxiv.org/abs/2501.01709)
- Theia (CoRL 2024) — [arxiv 2407.20179](https://arxiv.org/abs/2407.20179)
- ObjectVLA — [arxiv 2502.19250](https://arxiv.org/abs/2502.19250)

### Code (this repo)
- `third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`
- `third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py` (esp. lines 144-146, 403-498)
- `third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`
- `eval_3/aug/5_verify_identity.py` (rectified-face ArcFace pattern)
- `eval_3/aug/3_extract_corners.py` (per-pid quad coordinates)
- `TODO.md` (Track D spec, lines 122-130; Track A launch lines 60-77; Track C launch lines 105-118)

### Project docs
- `docs/report/EVAL_3_FINAL_PLAN.html` (Track D block)
- `docs/report/EVAL_3_RESEARCH_REPORT.md` §3.3 (current M2 description, camera2-based)
- `docs/report/EVAL_3_OPTIONS_BRIEFING.md` §2 M1+M2 (the +12pp claim, BlindVLA Table 1)

---

**For the next reader:** the three agent transcripts are durable only in
this session. The headlines above survive. If you need to re-run the
audit, the prompts are reproducible from the three Agent-tool invocations
in the 2026-05-18 session log.
