# 2026-05-20 — Pivot to Pi0.5-only strategy, 4-agent research synthesis

## Context

The team (Roham + Hans + Sejohn + Mahbod + Darius) verified via attention-map analysis on the trained SmolVLA checkpoints that the VLM head puts **essentially zero attention weight on the correct celebrity face** when prompted with names like "Taylor Swift" or "Yann LeCun". The action expert defaults to a positional shortcut from the training distribution. All SmolVLA variants tried (vanilla, Hans's warm-VLM, Mahbod's M2 ArcFace toolkit) exhibit the same failure mode.

**Decision:** abandon SmolVLA entirely. The whole team converges on Pi0.5-3B (PaliGemma vision-language + Gemma-300M flow-matching action expert) as the only viable backbone, primarily because of PaliGemma's larger language model and richer pretraining corpus.

Roham listed 4 candidate next steps:
1. Fine-tune Pi0.5 on the 10k TOY (3-celeb) dataset — already in flight on brev_instance1
2. VQA pretrain → VLA fine-tune (PaliGemma warmstart on faces, then VLA)
3. ArcFace ↔ SigLIP latent alignment (MLP projector + cosine aux loss, or attention bridge)
4. Fine-tune Pi0.5 on the 10k 200-celeb dataset (just-pushed `HBOrtiz/so101_eval3_aug_v3_200celebs`)

Per CLAUDE.md §9 (cross-validate non-trivial decisions with parallel agents), I spawned **4 independent research agents** to validate each idea against published literature before committing to a strategy. This document is the durable record of their findings.

## Method

4 agents launched in parallel via the Agent tool. Each was given:
- Project context (SmolVLA failure mode, Pi0.5 architecture, 2-day timeline)
- A focused research question with 5-10 sub-questions
- A standard for sourcing (triple-source numerical claims, verify via WebFetch/WebSearch)
- An output spec ("structured report under N words, source + claim + project-implication per finding")

The 4 questions:

1. **Agent 1 — VLA + named-entity recognition SOTA.** What VLA papers in 2024-2026 address fine-grained classification or named-entity-conditioned manipulation? What does the long-tail VLM literature say?
2. **Agent 2 — ArcFace ↔ VLM alignment.** Find published methods for aligning a frozen face encoder with a VLM's vision tower. MLP projector vs attention bridge. Dimensional considerations.
3. **Agent 3 — VQA-pretrain → VLA fine-tune precedents.** Two-stage vs co-training. Pi0.5-KI joint recipe details. Catastrophic forgetting risk.
4. **Agent 4 — Pi0.5 ecosystem + alternatives.** What variants exist? PaliGemma 1 or 2? KI implementation status. Alternative VLAs.

All 4 agents completed within ~5 minutes of each other (10-15 min each).

## Findings (synthesis)

### Architectural corrections

- **Pi0.5 uses PaliGemma 1, not PaliGemma 2.** Verified by Agent 4 via `lerobot/pi05_base/config.json`: `paligemma_variant=gemma_2b`. PaliGemma 2's entity-tuned WebLI splits (Entity-WebLI, OVEN benchmark) **do not apply to our backbone**. This had been an unstated assumption in our prior planning docs; correction propagates to the strategy.
- **Pi0.5 was originally trained two-stage** (Agent 3) — stage 1 = FAST-only pretrain (α=0, 280k steps, web/VQA + action tokens), stage 2 = flow-matching + action expert (α=10, 80k steps). So the "two-stage" instinct isn't crazy, but the field has moved on (see below).
- **Pi0.5-KI is not implemented in lerobot** (Agent 4, via openpi issue #649). Only flow-matching head supported. KI re-implementation = ~3-5 days of port work.

### Recipe ranking by paper backing

**Strongest paper backing — ObjectVLA-style co-training** (Agent 1):
- ObjectVLA ([arxiv 2502.19250](https://arxiv.org/abs/2502.19250)) directly tackles "robot sees novel object but can't bind name → action" — verbatim our failure mode.
- Recipe: mix 10:1 robot:VL with **bounding-box-grounded** VL captions (`"the face of <NAME> is at [x1,y1,x2,y2]"`) in a single training run.
- Quantitative ablation: **without bboxes, OOD success drops 64% → 19%**. Grounding is the load-bearing intervention.
- Independent confirmation: Pi0.5 Fig. 11 shows removing web co-train drops OOD ~75% → ~45-50% (~25-30pp).

**Paper-validated mechanism, novel teacher — BlindVLA ArcFace alignment** (Agent 2):
- BlindVLA ([arxiv 2510.25616](https://arxiv.org/abs/2510.25616)) validates the *mechanism*: frozen MLP projector + per-token cosine alignment loss at λ=0.2, injected mid-layer.
- Their teacher was C-RADIOv3 (general ViT); **no published paper uses ArcFace specifically as the teacher**.
- All face-MLLM papers to date (Face-LLaVA, FaceInsight, FaceLLM, Face-MLLM) use either landmark detectors or face-pretrained CLIP variants — not ArcFace embedding injection.
- MLP projector > cross-attention at our data scale (LLaVA-1.5 ablation; Q-Former needs 10²-10³× more data).
- **If we run it, we'd be the first.**

**Suboptimal vs co-training — Two-stage VQA warmstart** (Agent 3):
- VLM2VLA ([arxiv 2509.22195](https://arxiv.org/abs/2509.22195)) explicitly argues sequential VQA-pretrain **causes** catastrophic forgetting.
- Pi0.5-KI's stop-gradient design exists *because* updating PaliGemma while the action expert sees its features causes miscalibration. A separate pre-stage where we shift PaliGemma independently makes this *worse*, not better — and is not paper-validated.
- LangGap (2603.00592) supports VQA pretrain in principle but says fine-grained discrimination needs *thousands* of task-specific examples. Our 50 demos/celeb is borderline.
- **Verdict: co-training beats sequential.**

**Unvalidated extrapolation — Vanilla LoRA on Pi0.5** (Agents 1+4):
- Pi0.5-KI did not test LoRA. Their recipe is full backbone updates + action-expert stop-gradient + FAST CE on lm_head.
- LoRA-only on a Pi0.5 backbone is paper-extrapolation, not paper-validated.
- But: it's cheap and gives us a baseline. In flight on brev_instance1.

### Other findings

- **Pi 0.6 / RECAP** (Agent 4) is irrelevant — offline RL for execution recovery, not language grounding.
- **Pi0-FAST autoregressive variant** is NOT a better pick than Pi0.5 for our failure mode (Agent 4). Discrete action gradients don't help face→name binding; gradient signal is in wrong channel.
- **DROID-fine-tuned Pi0.5 variants** add negative value — DROID is Franka 7-DoF, our SO-101 is 6-DoF; action prior pollution.
- **No alt VLA** (TinyVLA, X-VLA, FlowerVLA, OpenVLA, RoboFlamingo, RT-2) has demonstrated face-binding behavior.
- **Strix bf16 inference fits Pi0.5 at batch=1** with `num_inference_steps=10`. ~8 GB weights + 2-3 GB activations + 4-5 GB margin on 16 GB card. No quantization needed.

## Strategic verdict

The 4 user-listed options compress to 3 tracks for execution:

| Track | Mechanism | Owner | Paper backing |
|---|---|---|---|
| **1. Vanilla LoRA** (= user's Option 1+4) | Pi0.5 LoRA on robot data alone | Roham | Weak / extrapolation |
| **2. ObjectVLA co-train** (NOT on user's list) | 10:1 robot:VL with bbox-grounded face captions | unassigned | **Strongest** |
| **3. ArcFace alignment** (= user's Option 3) | Frozen MLP projector + cosine aux loss | unassigned | Mechanism validated, teacher novel |

User's Option 2 (VQA warmstart → VLA finetune) is deprioritized — Track 2 (co-training) is the same idea done correctly.

## Files updated

- `docs/report/EVAL_3_FINAL_PLAN.html` — full strategy rewrite reflecting the 3-track plan
- `eval_3/tracks/TRACK_B_WARMSTART.md` — earlier two-stage warm-start plan; remains as scaffolded code (Option 2) but is no longer the primary direction
- (TODO): write Track 2 (ObjectVLA co-train) and Track 3 (ArcFace alignment) design docs as `eval_3/tracks/TRACK_OBJECTVLA.md` and `eval_3/tracks/TRACK_ARCFACE.md` when owners are assigned.

## Pending decisions

1. Who owns Track 2 (ObjectVLA co-train)?
2. Who owns Track 3 (ArcFace alignment)?
3. Do we provision a 3rd Brev VM so Tracks 2 and 3 can run in parallel?
4. Should brev_instance2's existing setup (cu128 PyTorch, lerobot, etc.) be repurposed for Track 2 or Track 3?

---

*Cross-validated by 4 parallel research agents on 2026-05-20.
Agent run details persist in `/tmp/claude-1000/.../tasks/{agentId}.output` for the duration of the Claude session.*
