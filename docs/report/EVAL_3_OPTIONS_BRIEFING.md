# Eval 3 — Options briefing for team decision

**Status:** active. Self-contained briefing document for the LeMonkey team to decide which training tracks to run for Eval 3.

**Background:** SmolVLA-450M was trained for the Eval 3 face-matching task (place coke on celebrity portrait specified by name + reference photo) and **fails empirically** — picks wrong celebrity, falls back to positional shortcuts. This document presents:

- **§1.** Why v1 failed (failure mode diagnosis)
- **§2.** The 8 protective mechanisms (M1-M8) from the source papers we cited but skipped
- **§3.** All 17 training options we could run, each explained
- **§4.** Decision matrix + budget
- **§5.** My recommendation, and three "what if budget is X" scenarios

All numerical claims and mechanisms below have been triple-source-validated across two waves of 8 parallel research agents (4 deep-readers + 4 skeptical fact-checkers). Audit results are in [`docs/report/EVAL_3_RESEARCH_REPORT.md` §P2.7b](EVAL_3_RESEARCH_REPORT.md).

---

## §1. Why v1 failed (the diagnosis)

Empirical observations on Strix (2026-05-17 smoke-test):

- Pipeline runs end-to-end (no crashes, no OOM, no timing issues)
- Camera2 reference photo verified to reach the model (instrumented the tensor at `prepare_images()`)
- Wrist-cam aim verified identical to training distribution
- **Failure mode: positional shortcut.** Rotating Swift's portrait through left/middle/right positions: the can lands at the same physical spot regardless. The model is not conditioning actions on identity.

Two compounding gaps explain this:

- **(G1) Domain gap.** Training-time reference photos are magazine/web style. Eval-day workspace is paper printouts photographed through a wrist cam. Published face-recognition literature ([arxiv 2404.06559 §4](https://arxiv.org/html/2404.06559v2)) measures this transformation at +5.6-16.0% FMR shift on ArcFace verification. Our model never saw the print domain at training.
- **(G2) Representation gap.** SmolVLA's vision tower compresses each image into **64 tokens via 2×2 pixel-shuffle** ([SmolVLM, arxiv 2504.05299 §3.1](https://arxiv.org/html/2504.05299v1)). For a face occupying ~30% of the frame, that's ~10-20 identity-bearing tokens. SigLIP was pretrained for image-text contrastive loss, not face-discriminative geometry.

Plus we discovered 6 additional mechanisms we under-applied from the very papers we cited (see §2).

---

## §2. The 8 protective mechanisms

Each mechanism is a research-published intervention that we **could add** to a training recipe. Each addresses a specific failure mode. The 4 architectures we considered (SmolVLA, Pi0.5, OpenVLA, X-VLA) can be combined with subsets of these mechanisms.

### M1. Frozen 3-layer MLP projector + Backbone2Enc injection

**Source:** "Don't Blind Your VLA" ([arxiv 2510.25616](https://arxiv.org/html/2510.25616v1)), Tables 5 and 6.

**Plain-language description:** We need to map the VLA's internal features into the same space as a teacher's features so we can compare them with a cosine loss. The mapping module is a small 3-layer MLP. The paper finds:
1. **The MLP must be FROZEN** (initialized once, then `requires_grad=False`). Otherwise the model "cheats" by adjusting the MLP instead of fixing the VLA's hidden state. Table 6: frozen 0.61 vs trainable 0.54 on semantic OOD.
2. **The loss must be injected at a mid-LLM layer, not at the vision encoder output.** Table 5: Backbone2Enc 0.61 vs Enc2Enc 0.55 on semantic OOD (note: gap is significant only on semantic axis; vision axis 0.66/0.66 and execution 0.38/0.38 are p=0.04/0.64 respectively).

**Exact architecture (code-verified at [`finetune_align.py:326-338`](https://github.com/CognitiveAISystems/BlindVLA/blob/main/openvla/vla-scripts/finetune_align.py)):**
```python
nn.Sequential(
    nn.LayerNorm(hidden_size),
    nn.Linear(hidden_size, projector_dim),  # projector_dim = 2048 default
    nn.SiLU(),
    nn.Dropout(0.1),
    nn.Linear(projector_dim, projector_dim),
    nn.SiLU(),
    nn.Dropout(0.1),
    nn.Linear(projector_dim, z_dim),         # z_dim = teacher's feature dim
)
```

**Cost to implement:** ~1 day engineering (modify SmolVLA's `embed_image` to expose mid-LLM hidden states; add the projector module; wire it through `forward`).

**Validation status:** ✓ confirmed verbatim by V2 fact-checker. The frozen-vs-trainable result is in Table 6; the Backbone2Enc result in Table 5.

---

### M2. Cosine alignment loss against a frozen vision teacher

**Source:** "Don't Blind Your VLA" Equation 9.

**Plain-language description:** Train the VLA's vision features to match a frozen "teacher" encoder's features. The teacher knows things the VLA doesn't (e.g., ArcFace knows face identity). This is the actual loss term that makes M1 work.

**Exact equation:**
```
L_align = − (1/k) · Σ_{j=1}^{k} cos(F.normalize(u_j), F.normalize(z_j))
L_total = L_action + λ · L_align
λ = 0.2
```
where `u_j` are projected student patches, `z_j` are teacher features, `k` is the number of aligned patches.

**Key details:**
- Cosine, not L2 or InfoNCE (paper Table 8: cosine 0.61/0.72/0.39 > L2 0.54/0.63/0.34; p<0.01 on semantic and vision axes, p=0.05 borderline on execution)
- `F.normalize` is explicit in the code, not `F.cosine_similarity` (verified `finetune_align.py:420-428`)
- λ=0.2 from step 0, constant (verified `finetune_align.py:311`)
- Patch-level, not CLS-level

**Headline empirical impact (Table 1):** Default SFT (no alignment) gets semantic OOD 0.49 ± SD 0.02 / vision 0.74 / execution 0.28. Align (their method) gets 0.61 / 0.83 / 0.35. So **+12pp on semantic OOD, +9pp on vision OOD**.

**Note on teacher choice:** Their best teacher is C-RADIOv3 (general vision foundation). For our face-matching task we'd use **ArcFace** instead — task-specific. This is a first-mover combo (no published paper does ArcFace→SigLIP distillation), but the recipe transfers in principle.

**Cost:** Combined with M1 — ~1 day eng + ~1.5-3h Brev fine-tune.

**Validation status:** ✓ Equation 9 confirmed verbatim by V2. Code lines verified. Table numbers verified.

---

### M3. Stop-gradient between action expert and VLM

**Source:** Pi0.5 Knowledge Insulation ([arxiv 2505.23705](https://arxiv.org/html/2505.23705v1)), Equations 5-6.

**Plain-language description:** When the action expert reads from the VLM (via cross-attention), the gradients flowing back from the action loss should NOT update the VLM. The action expert can *read* VLM features but cannot *perturb* them. This prevents the VLM's pretrained knowledge from drifting during robot fine-tune.

**Exact mechanism (Eq. 5 verbatim, K is wrapped):**
```
softmax( Q_a · sg(K_b)^T   Q_a · K_a^T )
```
And Eq. 6 (V is also wrapped):
```
P_ab · sg(V_b) + P_aa · V_a
```

The `sg(·)` is PyTorch's `detach()`. Subscript `b` = backbone (VLM), `a` = action expert.

**Empirical impact:** KI's paper §4 shows:
- Frozen VLM = 0% on items-in-drawer task ([verbatim §4 L284: "0% performance"](https://arxiv.org/html/2505.23705v1))
- Joint training (no stop-gradient, our v1 setup) **performs significantly worse** than KI on action success AND collapses language following (body text confirmed; exact bar heights are visual reads from Fig. 4a/4b)
- KI is **7.5× faster training** to reach same performance (verbatim Fig. 6 caption)

**Cost to implement:** Multi-day engineering. Requires modifying SmolVLA's `embed_prefix` and `forward` to insert `detach()` calls on the cross-attention K and V from VLM to action expert. Not currently exposed in lerobot 0.5.1.

**Validation status:** ✓ Eqs. 5-6 verbatim. Frozen-VLM = 0% verbatim. 7.5× speedup verbatim. Bar-height percentages flagged as visual reads.

---

### M4. FAST-token cross-entropy loss through the VLM

**Source:** Pi0.5-KI §5.1.

**Plain-language description:** Even though the action expert outputs continuous actions via flow-matching, the VLM is also trained to predict a *discretized* version of the action sequence as a sequence of language tokens. This gives the VLM a supervised signal that ties its representations to robot actions, without polluting the action expert.

**Exact mechanism:** Eq. 4 contains an autoregressive token-prediction term `−Σ Mj^ℓ log p_θ(ℓ̂_{j+1} | x_{1:j})` where `ℓ̂` includes both natural-language tokens AND FAST-tokenized action tokens. `Mj^ℓ` is a per-sample mask selecting which tokens count.

**Empirical impact:** KI §5.1 verbatim: *"The autoregressive objective is only used at training time as a representation learning objective, which enables the model to train much faster."* This is what produces the 7.5× speedup.

**Cost to implement:** Multi-day engineering. SmolVLA's training forward computes only flow-matching MSE — there's no LM head exposed for next-token CE loss. Adding one requires exposing `vlm_with_expert`'s prefix output through `lm_head`, plus the FAST tokenizer.

**Validation status:** ✓ §5.1 mechanism confirmed verbatim by V1.

---

### M5. Web/VQA co-training

**Source:** Pi0.5 paper ([arxiv 2504.16054](https://arxiv.org/abs/2504.16054)) + Pi0.5 blog ([pi.website/blog/pi05](https://www.pi.website/blog/pi05)).

**Plain-language description:** Mix non-robot data into the robot fine-tune. Pi0.5's pretraining mixture is **97.6% non-mobile-manipulator data** (verbatim Pi0.5 §I L110-113) including image-text pairs from the web. This preserves the VLM's pretrained world knowledge during robot fine-tune.

**Empirical impact:** Pi0.5 blog verbatim quote: removing web data drops OOD object recognition from 94% to 74% on the object-into-drawer eval. **−20pp on OOD object recognition** — the closest published analog to our face-name binding task.

**Note:** This number is in the blog text. In the arXiv paper, the same finding is described qualitatively (Pi0.5 §V-C / Fig. 11) but the literal 94/74 numbers are only in the figure chart.

**Cost to implement:** Two ways:
1. **Full co-training** (the proper way): mixed-batch training of robot data + VQA data with per-sample loss masking. Blocked by `MultiLeRobotDataset = NotImplementedError` in lerobot 0.5.1. 3-5 days eng to build a custom dataloader and multi-loss forward.
2. **Sequential VLM warm-start** (cheap proxy): pretrain just the VLM (SmolVLM2-500M or PaliGemma) on VQA pairs in plain HuggingFace `transformers`, then use the warm VLM as the SmolVLA/Pi0.5 init. Sidesteps the LeRobot blocker. ~3-6h Brev for the VLM pretrain + 1 day eng.

**Validation status:** ✓ 97.6% verbatim from Pi0.5 §I. ✓ 94→74% verbatim from blog (not paper text proper).

---

### M6. Reference image inlined IN the language token stream

**Source:** Interleave-VLA ([arxiv 2505.02152](https://arxiv.org/abs/2505.02152)), §3.2.

**Plain-language description:** Instead of passing the reference photo as a separate camera observation, the reference image is inserted INSIDE the language prompt — between text tokens that reference it. This lets the language model directly ground a noun phrase ("Yann LeCun") to the visual content of the reference image via positional adjacency.

**Exact mechanism (paper §3.2 L184-186):**
- Sequence is `I = (l₁, I₁, l₂, I₂, ...)` — alternating text segments and image tensors
- Example prompt format: `"Place [image1] into [image2]."` where `[image]` is replaced by image tokens at the right position in the text stream
- **Note:** The paper does NOT define specific token strings like `<BOI>`/`<EOI>`. Concrete tokens come from the underlying VLM (e.g., PaliGemma's `<image>` soft tokens)

**What SmolVLA currently does (code-verified at [`modeling_smolvla.py:626-705`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py)):**
The prefix order is `[img1_tokens, img2_tokens, ..., language_tokens, state_tokens]` — language is appended as a single block AFTER all images. SmolVLA's `add_image_special_tokens` flag only wraps each individual image with `<global-img>` / `<end_of_utterance>` tokens. **It cannot interleave images BETWEEN text segments.**

So our v1 architecture cannot do what Interleave-VLA does — the LM has no syntactic mechanism to know that camera2 "is" the celebrity named in the prompt.

**Empirical impact (Interleave-VLA Table 1, validated by V3):**
- Semantic L1 (novel object, known category): π₀ 26.7 → Interleave-VLA 63.7 (≈2.4×)
- Semantic L2 (novel category): 21.0 → 53.0 (≈2.5×)
- Real-FANUC: π₀ w/PT mean ≈28% vs Interleave-VLA w/PT mean ≈58% across 5 OOD object axes (≈2×; per-axis ratios 0.7×-9×)

**Caveat (V3 audit):** The "second-camera approach is INVALID compared to inlined" is **our hypothesis**, not paper-tested. Interleave-VLA never tested the 2nd-camera variant nor explicitly argued against it. Our diagnosis is based on the paper's grounding mechanism, not on a head-to-head experiment.

**Cost to implement:** Multi-day engineering. Requires processor changes in `processor_smolvla.py` (insert image tokens INTO the text token stream) plus `embed_prefix` modifications. Backbone tested in paper only at ≥3.3B params (π₀-PaliGemma-3B, OpenVLA + InternVL2.5-8B) — **no sub-1B evidence**.

**Validation status:** ✓ Mechanism confirmed. ✗ "BOI/EOI" token names were fabricated (paper doesn't define them). ✗ Original "13→71" Real-FANUC claim was wrong number — corrected to ~28% → ~58%. ⚠ The "2nd-camera is invalid" framing softened to "our hypothesis."

---

### M7. 3-5 reference photos per object, sampled per step

**Source:** Interleave-VLA Table 4.

**Plain-language description:** Use multiple reference photos per celebrity, sampled randomly at each training step. Variation should come from different angles, lighting, etc.

**Exact recipe (paper §3.3 verbatim):** "trajectory frames" (plural — exact frame-selection rule not specified) plus internet-source images of the same noun. Cropped via OWLv2 then verified via Qwen2.5-VL + SAM.

**Empirical impact (Table 4 verbatim):**
- Internet-only: 59.2 / 69.1
- Task-specific-only: 67.5 / 67.1
- **Mixed: 71.0 / 71.7**

Paper §3.3 explicitly: cropped-only "lacks diversity," internet-only "lacks task relevance" — combination is required.

**Cost to implement:** ~3-4h engineering. Dataset-side change to expand the 192-celeb bank from 1 photo/celeb to 3-5 photos/celeb (we have ≥5 photos for most celebs already) and modify the dataloader to sample one per training step.

**Validation status:** ✓ Table 4 numbers and §3.3 mechanism confirmed verbatim.

---

### M8. Bbox-grounding co-fine-tune at 10:1 robot:VL

**Source:** ObjectVLA ([arxiv 2502.19250](https://arxiv.org/html/2502.19250v1)), §3.2-3.3, §4.1.2.

**Plain-language description:** Mix a vision-language task into the robot fine-tune: given an image of an object, predict the bounding box of the object. The "language" output is the bbox coordinates as text. The same bbox prediction is injected as a reasoning prefix in the robot trajectories.

**Exact recipe (verbatim):**
- VL pair format: `(image, "Detecting the bounding box of <object>.", "(x1,y1),(x2,y2)")`
- Robot:VL ratio: **10:1** (§3.3 verbatim)
- Dataset size: 100 objects × 20 images = 2000 VL pairs (§3.2 verbatim)
- Grounding token injection in robot trajectory prompts: `<object_ref_start>name<object_ref_end><box_start>(x1,y1),(x2,y2)<box_end>` then the action chunk
- Training: 8× A800, **Adam** (not AdamW), LR 2e-5 constant, batch 128, 50k steps (§7.2 verbatim)

**Empirical impact (verbatim §4.1.2):**
- ObjectVLA full recipe: 100% ID → **64% OOD**
- Without bbox-grounding co-train: 100% ID → **19% OOD**
- Without any VL co-train (DiVLA baseline): 100% ID → **8% (random)**

The bbox-grounding co-train carries **45 percentage points** of OOD performance. **This is the largest single-mechanism gain in the literature we surveyed.**

**Cost to implement:** Two ways:
1. **Full co-training** (the proper way): requires multi-dataset training infra. Same LeRobot blocker as M5. 3-5 days eng.
2. **Prompt-relabel proxy** (cheap): precompute the face bbox per reference photo offline, inject it as text in the existing prompts. Example: `"<ref> shows Yann LeCun in bbox (12,15)-(245,230). Set the coke down on his picture."` ~6h eng. Weakened version of the mechanism — may or may not transfer.

**Validation status:** ✓ All numbers verbatim per V4. ⚠ Original "AdamW" was wrong — corrected to Adam.

---

### Summary of mechanisms

| M | What | Source | Headline gain | Cost (eng + Brev) |
|---|---|---|---|---|
| M1 | Frozen 3-layer MLP at Backbone2Enc | Blind-VLA T5+T6 | -7pp if trainable; -6pp if Enc2Enc on semantic | 1 day + 0 |
| M2 | Cosine alignment loss vs frozen teacher | Blind-VLA Eq 9 | +12pp semantic OOD (vs default SFT) | 1 day + 2h |
| M3 | Stop-gradient action↔VLM | KI Eqs 5-6 | KI ≫ joint > frozen (=0%); 7.5× speedup | multi-day + 0 |
| M4 | FAST-token CE loss on VLM | KI §5.1 | Enables M3's effectiveness | multi-day + 0 |
| M5 | Web/VQA co-train | Pi0.5 blog | +20pp OOD object recognition | 3-5d full / 1d proxy |
| M6 | Reference image inlined in language stream | Interleave-VLA §3.2 | +1.8-2.5× OOD (only ≥3.3B tested) | 1-2 days + 0 |
| M7 | 3-5 ref photos/object, sampled per step | Interleave-VLA T4 | Mixed 71 > Task-only 67 > Internet-only 59 | 3-4h + 0 |
| M8 | Bbox-grounding co-train at 10:1 | ObjectVLA §4.1.2 | +45pp OOD (100→19% without; 100→64% with) | 3-5d full / 6h proxy |

---

## §3. All 17 training options

Each option combines an architecture (SmolVLA, Pi0.5) × a protocol (image-as-prompt, name-only) × a subset of M1-M8.

### Option 0 — Ship as-is

**Description.** Use `HBOrtiz/smolvla_eval3` exactly as-is. No retraining.
**Mechanisms.** None added.
**Why it might work.** Some IID rollouts might still succeed by luck.
**Why it will fail.** Empirically demonstrated to fail (positional shortcut, smoke-test 2026-05-17).
**Cost.** $0.
**Bonus.** +20.
**Verdict.** Catastrophic floor (~31 pts). Comparison baseline only.

### Option 1 — SmolVLA + ref recuration only (M7)

**Description.** Filter the 192-celeb bank for enrollment-quality photos (NIST FRVT / ISO 19794-5 standards: pose ≤±15°, inter-eye ≥60px). Pre-crop offline. Ship as static asset table.
**Mechanisms.** M7 (partial — only 1 photo per celeb, the highest-quality one).
**Why it might work.** Pure inference-time fix. Removes magazine-shoot noise.
**Why it might fail.** Doesn't address representation gap or KI gaps.
**Cost.** 0 Brev, ~4h eng.
**Bonus.** +20.
**Verdict.** Cheap add-on. Likely +0-2 rollouts alone.

### Option 2 — SmolVLA + print-domain aug only

**Description.** Augraphy-inspired print emulation (Lab gamut → Floyd-Steinberg dither → Perlin grain → blur → JPEG) on camera2 at p=0.7. Re-fine-tune from 30k for 5-10k steps.
**Mechanisms.** None of M1-M8 — this is a domain-coverage fix orthogonal to the 8 mechanisms.
**Why it might work.** RoboEngine paper (intro verbatim with "even"): *"Methods even directly modify the scene using random images or texture…fail to respect physical constraints, leading to degenerated real-world performance due to distribution shifts."* — describes our current alpha-feather paste-on baseline. Print augmentation closes the magazine→print gap (arxiv 2404.06559 §4: +5.6-16.0% FMR shift).
**Why it might fail.** First-mover on this specific recipe; calibration against our printer is mandatory. Doesn't address representation gap.
**Cost.** ~3-4h Brev, ~6h eng.
**Bonus.** +20.
**Verdict.** Direct domain fix. Bundle with Options 1 + 3.

### Option 3 — SmolVLA + ArcFace cosine distillation only (M1+M2)

**Description.** Add Blind-VLA's auxiliary cosine loss at λ=0.2. Frozen 3-layer MLP projector. Mid-LLM injection (Backbone2Enc, layer 7-8 of SmolLM2's 16-layer truncation). Mask-gated to face patches via RetinaFace. ArcFace `buffalo_l` teacher.
**Mechanisms.** M1 + M2.
**Why it might work.** Blind-VLA Eq. 9 + Table 1: Default SFT semantic 0.49 → Align 0.61 (+12pp). Directly fixes representation gap.
**Why it might fail.** First-mover combo on ArcFace→SigLIP. Pooled-student adaptation for single-embedding teacher. Not validated at 450M scale.
**Cost.** ~1.5-3h Brev, ~1 day eng.
**Bonus.** +20.
**Verdict.** Direct representation-gap fix. Bundle with Options 1 + 2.

### Option 4 — VLM-only VQA warm-start (cheap M5)

**Description.** Pretrain SmolVLM2-500M on face-VQA pairs `(face_crop, "Who is this?", "Yann LeCun")` from the 192-celeb bank using plain HuggingFace `transformers.Trainer` — bypasses LeRobot's blockers. Save warm VLM, use it as init for the SmolVLA fine-tune.
**Mechanisms.** M5 (sequential proxy).
**Why it might work.** Pi0.5 blog 94→74% confirms web data has +20pp on OOD. Even a sequential warm-start should partially preserve the binding.
**Why it might fail.** Sequential ≠ KI's joint co-train; action fine-tune may drift the warm VLM if `train_expert_only=false`. Mitigation: freeze VLM during action fine-tune.
**Cost.** ~3-6h Brev (VLM pretrain), ~1 day eng.
**Bonus.** +20.
**Verdict.** Cheap, works regardless of protocol. Useful add-on to almost any option.

### Option 5 — Caption + bbox prompt-relabel (cheap M8)

**Description.** Rewrite training prompts to include the celebrity's face bbox in camera2. Precompute bboxes once with RetinaFace. Inject as text: `"<ref> shows Yann LeCun in bbox (12,15)-(245,230). Set the coke down on his picture."` No code changes — pure dataset relabel.
**Mechanisms.** M8 (text-only proxy).
**Why it might work.** ObjectVLA verbatim 45pp gain from bbox grounding. Proxy might capture some of it.
**Why it might fail.** Proxy is weaker than ObjectVLA's actual bbox-prediction co-training. May not transfer the full gain.
**Cost.** ~3-5h Brev (re-fine-tune), ~4h eng.
**Bonus.** +20.
**Verdict.** Cheapest path to the largest published single-mechanism gain.

### Option 6 — SmolVLA-boost-v2 (M1+M2+M5-proxy+M7+M8-proxy + low-LR + tight-jitter)

**Description.** Stack Options 1+2+3+5 plus lower LR to 2.5e-5 plus tighten hue jitter to ±0.02. Resume from 30k for 10-15k more steps.
**Mechanisms.** M1, M2, M5-proxy (caption emphasis), M7 (partial), M8-proxy.
**Why it might work.** Addresses both domain gap (G1) AND representation gap (G2). Keeps the +20 bonus. Lowest cost of the 4 originally-committed tracks.
**Why it might fail.** Doesn't include true M6 (in-stream interleaving) or true M3-M4 (KI mechanism). If failure is dominantly those, won't fix.
**Cost.** ~5-6h Brev, ~1.5 days eng.
**Bonus.** +20.
**Verdict.** **Primary bonus-preserving recommendation.** Currently committed as Track A.

### Option 7 — Option 6 + true Interleave-VLA inlining (adds M6) [Track A-2]

**Description.** Option 6 plus modify `processor_smolvla.py` + `embed_prefix` in `modeling_smolvla.py:626-705` so prefix is `[language_with_inlined_image_tokens, state]` instead of `[images, language, state]`.
**Mechanisms.** M1, M2, M5-proxy, M6, M7, M8-proxy.
**Why it might work.** Fixes the architectural deviation that V3 audit identified as the most likely root cause of v1's failure. LM can directly ground noun-phrase to camera2 image position.
**Why it might fail.** V3 caveat: the "2nd-camera fails, inlined works" comparison is **our hypothesis** — Interleave-VLA never tested it head-to-head. Also: Interleave-VLA only validated at ≥3.3B params; no sub-1B evidence.
**Cost.** ~5h Brev, **1-2 days eng** (substantial code change to processor + embed_prefix).
**Bonus.** +20.
**Verdict.** Activate ONLY if Option 6 (Track A) underperforms. Currently deferred as "Track A-2 follow-up."

### Option 8 — Option 6 + VLM warm-start (adds Option 4)

**Description.** Run VLM-only VQA pretrain first (Option 4), then Option 6 fine-tune from the warm VLM checkpoint.
**Mechanisms.** Option 6 + M5 (sequential variant).
**Why it might work.** Strongest combination of bonus-preserving mechanisms (5 of 8 mechanisms with proxies or full versions).
**Why it might fail.** Risk that action fine-tune drifts the VLM. Mitigation: freeze VLM during second stage.
**Cost.** ~7-9h Brev total, ~1.5 days eng.
**Bonus.** +20.
**Verdict.** Max-effort SmolVLA option. Worth doing if budget and time allow.

### Option 9 — Pi0.5 + image-as-prompt (vanilla) [Track C]

**Description.** Train `lerobot/pi05_base` on our existing image-as-prompt dataset. Just swap architectures.
**Mechanisms.** None of M1-M8 — pure capacity bet.
**Why it might work.** PaliGemma-2B vision tower (SigLIP-So400m at 400M) is bigger than SmolVLM's vision component. Pi0.5 §I L110-113: 97.6% of phase-1 data is non-mobile — includes WebLI with celebrity coverage.
**Why it might fail.** Same architectural issue as Track A v1: Pi0.5's prefix also concatenates images-then-language; LM doesn't ground celeb names to camera2 in-stream. Also: Pi0.5 lacks `add_image_special_tokens` (regression vs SmolVLA for image-as-prompt).
**Cost.** ~27-33h Brev (bs=16-24 with grad-checkpoint, bf16, compile_model on RTX PRO 6000), ~6h eng (quantile-stats preprocessing).
**Bonus.** +16 (loses 4pt vs SmolVLA tracks).
**Verdict.** Currently committed as Track C, but **strictly weaker than Option 10** for similar cost. Consider dropping.

### Option 10 — Pi0.5 + ArcFace distillation (M1+M2) [Track B]

**Description.** Option 9 plus port the Blind-VLA alignment loss recipe to Pi0.5's PaliGemma vision tower.
**Mechanisms.** M1 + M2.
**Why it might work.** Capacity bet (Pi0.5) AND representation fix (ArcFace distill). Strictly stronger than Option 9.
**Why it might fail.** Multi-day eng to port the patch from SmolVLA's SmolLM2 (16 layers) to Pi0.5's Gemma-2B (18 layers). Mid-network injection layer needs to be re-picked for Gemma.
**Cost.** ~30-35h Brev, ~2 days eng.
**Bonus.** +16.
**Verdict.** Best Pi0.5 option for the budget. **Currently committed as Track B.**

### Option 11 — Pi0.5 + true KI recipe (M3+M4+M5)

**Description.** Implement KI's full mechanism: `sg(K_b)`, `sg(V_b)` in cross-attention (Eqs. 5-6), FAST-token CE loss (§5.1), web/VQA co-train at Pi0.5's ratio.
**Mechanisms.** M3, M4, M5.
**Why it might work.** The published Pi0.5 recipe end-to-end. If Pi0.5 generalizes to face matching, this is the most likely configuration.
**Why it will fail.** **3-5 days engineering across multiple code paths** (forward-pass `sg(·)` wrapping, LM head exposure, FAST tokenizer integration, multi-dataset co-training). Not feasible in 24-48h timeline.
**Cost.** ~30-35h Brev + 3-5 days eng.
**Bonus.** +16.
**Verdict.** The "right" Pi0.5 setup, but engineering-blocked.

### Option 12 — Pi0.5 + name-only (pre-pivot plan)

**Description.** Train Pi0.5 with text-only prompts ("Place the coke on Yann LeCun"). No camera2 reference stream. Banks on PaliGemma's WebLI prior.
**Mechanisms.** None added.
**Why it might work.** The original team plan before the 2026-05-09 pivot. PaliGemma has seen billions of image-text pairs including many tagged celebrities.
**Why it might fail.** No closed-set fine-tune evidence at our scale. The 2026-05-09 zero-shot probe (frozen PaliGemma, no fine-tune) failed naming 14 TOY images — but that test had no fine-tuning OR web co-training, so it's unclear if a real Pi0.5 fine-tune would succeed.
**Cost.** ~25-30h Brev, ~6h eng (dataset relabel — drop reference stream).
**Bonus.** +16.
**Verdict.** Epistemically valuable (directly tests whether the pivot was right). Lower priority than Option 10 unless TAs reject image-as-prompt.

### Option 13 — Pi0.5 + name-only + VLM warm-start

**Description.** Option 12 plus VLM warm-start (Option 4) applied to PaliGemma-3B for celeb-name binding.
**Mechanisms.** M5 (sequential proxy).
**Why it might work.** Pi0.5 capacity + explicit name-binding refresh via VQA pretrain. Highest-capacity text-only option.
**Why it might fail.** PaliGemma fine-tune for VQA is non-trivial (~10-15h Brev for the pretrain alone). Most expensive of text-only options.
**Cost.** ~30-35h Brev total, ~1.5 days eng.
**Bonus.** +16.
**Verdict.** Best text-only option if TAs require text-only.

### Option 14 — SmolVLA + name-only + VLM warm-start

**Description.** Text-only variant of Track A. Drop reference stream entirely, name-only prompts, with VQA pretrain on SmolVLM2-500M.
**Mechanisms.** M5 (sequential proxy).
**Why it might work.** If image-as-prompt isn't allowed, this is the bonus-preserving text-only option.
**Why it might fail.** 500M VLM has limited capacity for binding 192 celebrity names to faces.
**Cost.** ~8-10h Brev, ~1.5 days eng.
**Bonus.** +20.
**Verdict.** Text-only insurance. Run if TAs reject image-as-prompt.

### Option 15 — SmolVLA 3-celeb baseline (name-only on 178 base teleops only) [Track D]

**Description.** Filter merged dataset to just Swift/Obama/LeCun episodes (178 base teleops). Name-only prompts. Train SmolVLA from `lerobot/smolvla_base`. `--policy.empty_cameras=2`.
**Mechanisms.** None — minimal training.
**Why it might work.** Reduces task to closed-set 3-way classification. Each celeb has ~60 demos (above SmolVLA's recommended ≥50/task per their doc). Visually distinct between 3 celebs is easy at any scale.
**Why it might fail.** Zero OOD generalization by design — concedes the 3 OOD eval runs (max 16.7 pts loss).
**Cost.** ~5-7h Brev, ~6h eng.
**Bonus.** +20.
**Verdict.** Safety net. Maximum reliability on 6 IID runs. **Currently committed as Track D.**

### Option 16 — Pi0.5 + everything (M1-M8 all-in)

**Description.** Track A v2's full mechanism stack ported to Pi0.5 + true KI recipe + true Interleave-VLA inlining + true bbox-grounding co-train.
**Mechanisms.** M1-M8 (all).
**Why it might work.** The "everything we can do" option.
**Why it will fail.** **3-5 days engineering across 4 different code paths** + ~$160 Brev. Not feasible in deadline.
**Cost.** ~35h Brev + 3-5 days eng.
**Bonus.** +16.
**Verdict.** Skip — engineering timeline incompatible.

---

## §4. Decision matrix

| Option | Mechanisms | Eng time | Brev cost | Bonus | Verdict for today |
|---|---|---|---|---|---|
| 0 | — | 0 | $0 | +20 | Baseline only |
| 1 | M7 partial | 4h | $0 | +20 | Add-on |
| 2 | — (domain) | 6h | $15 | +20 | Add-on |
| 3 | M1+M2 | 1d | $15 | +20 | Add-on |
| 4 | M5 proxy | 1d | $25 | +20 | Helps everything |
| 5 | M8 proxy | 4h | $25 | +20 | Largest single signal |
| **6** | **M1+M2+M5p+M7+M8p** | **1.5d** | **$30** | **+20** | **Primary** |
| 7 | + M6 | 2-3d | $30 | +20 | Follow-up |
| 8 | + M5 warm-start | 2d | $45 | +20 | Max-effort SmolVLA |
| 9 | — (capacity) | 6h | $150 | +16 | Skip (10 is stronger) |
| **10** | **M1+M2** | **2d** | **$160** | **+16** | **Capacity hedge** |
| 11 | M3+M4+M5 full | 3-5d | $160 | +16 | Skip (eng-blocked) |
| 12 | — | 6h | $140 | +16 | Pivot test |
| 13 | M5 warm-start | 1.5d | $165 | +16 | Best text-only |
| 14 | M5 warm-start | 1.5d | $50 | +20 | Text-only insurance |
| **15** | **— (3-celeb)** | **6h** | **$35** | **+20** | **Safety net** |
| 16 | M1-M8 | 3-5d | $170 | +16 | Skip (eng-blocked) |

**Bold rows** are the currently-committed 4 tracks. Cost in $ assumes ~$5/Brev-hour on RTX PRO 6000.

---

## §5. Recommendations (3 budget scenarios)

### Scenario A: Original 4 tracks ($370 over $130 remaining)

Committed: Track A (Option 6) + Track B (Option 10) + Track C (Option 9) + Track D (Option 15). Total ~$370. Over budget.

**Action:** Either request budget extension OR drop one track.

### Scenario B: Drop Track C, keep 3 tracks (~$225)

Committed: Track A (Option 6) + Track D (Option 15) + Track B (Option 10). Drop Option 9.
- Rationale: Option 10 (Pi0.5+ArcFace) is strictly stronger than Option 9 (Pi0.5 vanilla) for similar Brev cost. We lose the "pure capacity test" but gain a track that includes a surgical fix.
- Still ~$95 over remaining budget. Need ~$95 extension OR partial scope reduction.

### Scenario C: Minimum viable plan, 2 tracks (~$65, fits remaining budget)

Committed: Track A (Option 6) + Track D (Option 15). Both bonus-preserving.
- Pros: Fits budget. Both have +20 bonus. Track A has 5 mechanisms (proxies for M5, M8 + full M1, M2, M7), Track D is reliable IID safety.
- Cons: No Pi0.5 capacity test. If A doesn't lift face-matching, no fallback that's stronger.
- **Mitigation:** Add Option 4 (VLM warm-start) at ~$25 — boosts both A and D, total $90, still under budget.

### Scenario D: Text-only branch (if TAs reject image-as-prompt)

Committed: Option 14 (SmolVLA + name-only + VLM warm-start) + Option 15 (Track D) + optionally Option 13 (Pi0.5 + name-only + VLM warm-start).
- Total: $50 + $35 + $165 = $250 if all three. $85 if just first two.
- Need TA answer in Slack before committing.

---

## §6. Open question for the team

**The image-as-prompt permission question is still unanswered.** The official spec says the prompt is text ("Place the coke on [name]"). Our image-as-prompt protocol requires us to look up a reference photo internally from a name→photo asset table. The TAs may or may not consider this protocol-compliant.

If allowed: scenarios A/B/C apply.
If disallowed: pivot to scenario D (text-only).

**Action:** someone Slack TAs with: *"For Eval 3, may our policy take a reference image of the named celebrity as a 2nd camera input (looked up internally from a pre-built name→photo asset table that ships with the policy)? Or must the policy take only the text prompt with no auxiliary lookups?"*

---

## §7. Citations (primary sources only, audit-verified)

### VLA papers
- **SmolVLA** — [arxiv 2506.01844](https://arxiv.org/abs/2506.01844)
- **Pi0.5** — [arxiv 2504.16054](https://arxiv.org/abs/2504.16054) + [pi.website/blog/pi05](https://www.pi.website/blog/pi05)
- **Pi0.5-KI** — [arxiv 2505.23705](https://arxiv.org/html/2505.23705v1)
- **Interleave-VLA** — [arxiv 2505.02152](https://arxiv.org/abs/2505.02152)
- **SmolVLM** — [arxiv 2504.05299](https://arxiv.org/html/2504.05299v1)

### Training-technique papers
- **"Don't Blind Your VLA"** — [arxiv 2510.25616](https://arxiv.org/html/2510.25616v1)
- **BlindVLA reference code** — [github.com/CognitiveAISystems/BlindVLA/blob/main/openvla/vla-scripts/finetune_align.py](https://github.com/CognitiveAISystems/BlindVLA/blob/main/openvla/vla-scripts/finetune_align.py)
- **ObjectVLA** — [arxiv 2502.19250](https://arxiv.org/html/2502.19250v1)
- **ArcFace** — [arxiv 1801.07698](https://arxiv.org/abs/1801.07698)

### Augmentation papers
- **Print-and-Scan morph attacks** — [arxiv 2404.06559](https://arxiv.org/html/2404.06559v2)
- **Augraphy paper-emulation** — [arxiv 2208.14558](https://arxiv.org/abs/2208.14558)
- **RoboEngine** — [arxiv 2503.18738](https://arxiv.org/abs/2503.18738)
- **LIBERO-Plus** — [arxiv 2510.13626](https://arxiv.org/abs/2510.13626)
- **GenAug** — [arxiv 2302.06671](https://arxiv.org/abs/2302.06671)
- **ROSIE** — [arxiv 2302.11550](https://arxiv.org/abs/2302.11550)

### Biometric standards
- **NIST FRVT Quality** — [pages.nist.gov/frvt/html/frvt_quality](https://pages.nist.gov/frvt/html/frvt_quality.html)
- **ISO/IEC 19794-5:2011** — [iso.org/standard/50867](https://www.iso.org/standard/50867.html)
- **MagFace** — [arxiv 2103.06627](https://arxiv.org/abs/2103.06627)
- **InsightFace** — [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

### Audit trail
- Two waves of parallel research agents (4 deep-readers + 4 skeptical fact-checkers) verified every numerical claim, with corrections applied. Full per-agent audit results in [`docs/report/EVAL_3_RESEARCH_REPORT.md` §P2.7b](EVAL_3_RESEARCH_REPORT.md).

### Project docs
- [`docs/PROJECT.md`](../PROJECT.md) — eval rubric
- [`docs/VLA_ARCHITECTURES.md`](../VLA_ARCHITECTURES.md) — architecture inventory
- [`docs/EVAL_3_OPTIONS.md`](../EVAL_3_OPTIONS.md) — earlier 15-option enumeration
- [`docs/report/EVAL_3_RESEARCH_REPORT.md`](EVAL_3_RESEARCH_REPORT.md) — full research synthesis with validation audit
- [`/eval_3/STRATEGY.md`](../../eval_3/STRATEGY.md) — chosen strategy
- [`/TODO.md`](../../TODO.md) — operational checklist
