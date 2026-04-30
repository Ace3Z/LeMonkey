# VLA Architectures, VLM Backbones, and Tunable Knobs — LeMonkey

Companion doc to [`PROJECT.md`](PROJECT.md) (eval definitions) and [`RELATED_WORK.md`](RELATED_WORK.md) (prior work). This file answers two practical questions:

1. **For each candidate VLA we'd realistically run on the SO-101**, what is the model under the hood — which **VLM backbone**, how many parameters, what action head, what training data, and which of our three evals it fits.
2. **What knobs we can actually tune per eval** — which parameters get fine-tuned, what the data → train → deploy pipeline looks like end to end, and the trade-off space when a stronger VLM (e.g. PaliGemma-3B in Pi0.5) is on the table.

Eval definitions are sourced from [`PROJECT.md §2`](PROJECT.md#2-evaluation--150-pts-main--50-pts-bonus) (the consolidated brief built from the course [Google Doc](https://docs.google.com/document/d/1YsQ_Qe4vEwDp1dJdqn3l9vSt7oJBkc6JazjbmWLxAXg/edit?tab=t.0) and the Brev PDF). The 20 s rollout limit, single-camera rule, $200 GPU budget, and smallest-model bonus all flow from there.

> **VLA-only constraint (load-bearing).** [`PROJECT.md §3`](PROJECT.md#3-architecture--procedure-constraints) forbids any other foundation / pretrained model (LLMs, orchestration VLMs, YOLO, face-recognition models, etc.) anywhere in the inference pipeline. **The policy must consist of the VLA alone.** This kills two-stage recipes (face-ID → policy, VLM-grounder → policy, OCR → policy) — every reasoning step must happen inside the VLA itself.
>
> **Per-eval flexibility.** [`PROJECT.md §4`](PROJECT.md#cameras-updated-project-rule) now allows a different camera mount per eval and different model checkpoints per eval. We exploit both below.

---

## 1. TL;DR — VLA × VLM × per-eval fit

| VLA | VLM backbone | Total params | Action head | Open weights | Best fit |
|---|---|---|---|---|---|
| **SmolVLA** | `SmolVLM2-500M-Video-Instruct` | ~450 M | Flow-matching action expert (75 % VLM width, 16 VLM layers) | Yes — `lerobot/smolvla_base` ([HF](https://huggingface.co/lerobot/smolvla_base)) | Eval 1 (primary), Eval 2 (with `smolvla_vlabench` start) |
| **Pi0** | PaliGemma-3B (`gemma_2b` variant — Gemma-2B language + SigLIP-So400m vision ≈ 3 B) | ~3.3 B (≈3 B PaliGemma + ~300 M action expert) | Flow-matching | Yes — `lerobot/pi0` ([HF](https://huggingface.co/lerobot/pi0)) | Reasoning-heavy generalist baseline |
| **Pi0.5** | PaliGemma-3B (`gemma_2b` variant) | ~3.3 B (≈3 B PaliGemma + ~300 M action expert) | Flow-matching, **quantile** state/action norm, longer tokenizer (200) | Yes — `lerobot/pi05_base` ([HF](https://huggingface.co/lerobot/pi05_base)) | Eval 2 (fallback), Eval 3 (primary VLA) |
| **FLOWER-VLA** | Pruned LLM (~50 % removed) + fusion head | ~950 M | Diffusion w/ Global-AdaLN ([repo](https://github.com/intuitive-robots/flower_vla_pret)) | Pretraining-only repo; no fine-tune scripts shipped | Reference architecture; not plug-and-play |
| **TinyVLA** | LLaVA-Pythia (400 M / 700 M / 1.3 B) ([repo](https://github.com/liyaxuanliyaxuan/TinyVLA)) | 400 M – 1.3 B | Diffusion policy head | Yes — `lesjie/Llava-Pythia-{400M,700M,1.3B}` on HF | Smallest-model bonus candidate |
| **OpenVLA** | Llama-2-7B + DINOv2 + SigLIP ([HF](https://huggingface.co/openvla/openvla-7b)) | 7 B | Autoregressive action tokens | Yes — open weights (MIT code, Llama-2 license on backbone) | Reference; too large for our $200 budget |
| **GR00T-N1.5** | NVIDIA Eagle-2 VLM + diffusion transformer | ~3 B (per NVIDIA model card) | Diffusion / flow-matching | Yes — NVIDIA license ([blog](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)) | Reference; kills size bonus |
| **ObjectVLA** | Base VLA + VLM co-fine-tuning ([page](https://objectvla.github.io/)) | not disclosed | Continuous action | Weights not released as of writing | Pattern reference for OOD-target generalisation |
| **Interleave-VLA** | π0-based, image-of-image prompting ([paper](https://arxiv.org/abs/2505.02152) · [code](https://github.com/Interleave-VLA/Interleave-VLA)) | not disclosed | π0 inheritance | Code released | Eval 3 architecture inspiration |

**Per-eval headline (full justification in §3):**

| Eval | Primary VLA | VLM under it | Fallback VLA |
|---|---|---|---|
| Eval 1 — colour-named pick | SmolVLA fine-tuned from `smolvla_base` | `SmolVLM2-500M-Video-Instruct` | Pi0 (PaliGemma-3B) only if SmolVLA fails |
| Eval 2 — compositional | SmolVLA fine-tuned from `lerobot/smolvla_vlabench` | `SmolVLM2-500M-Video-Instruct` | Pi0.5 (PaliGemma-3B) |
| Eval 3 — celebrity image | **Pi0.5** end-to-end on celebrity-prompt demos (VLA-only) | **PaliGemma-3B** (web-pretrained world knowledge of public figures) | SmolVLA fine-tuned with image-as-prompt format (Interleave-VLA-style, but a single VLA — no extra models) |

---

## 2. Lit review

Every model spec below is sourced. Where we ship the policy locally in [`third_party/lerobot/src/lerobot/policies/`](third_party/lerobot/src/lerobot/policies/), the corresponding config file is cited. External claims point to the paper, model card, or repo README.

### 2.1 SmolVLA

- **Paper:** [arXiv 2506.01844](https://arxiv.org/abs/2506.01844) (Shukor et al., 2025) · **Model:** [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) · **Docs:** [`smolvla.mdx`](third_party/lerobot/docs/source/smolvla.mdx).
- **VLM backbone:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` ([HF card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)) — set as the default in [`configuration_smolvla.py:87`](third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py).
- **Total params:** ~450 M — stated explicitly in the LeRobot doc page (line 49 of `smolvla.mdx`: *"`smolvla_base`, our pretrained 450M model"*).
- **Action head:** flow-matching action expert, runs alongside the VLM with cross-attention. Default action chunk = 50 steps ([`configuration_smolvla.py:32`](third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py)).
- **Training data:** community LeRobot datasets; the paper reports community pretraining lifts SO-100 success 51.7 → 78.3 %.
- **Best at:** small-data fine-tuning on SO-100 / SO-101, consumer-GPU inference. The official guide recommends ~50 demos and ~4 h on a single A100 for 20 k steps (`smolvla.mdx` line 50).
- **Per-eval fit:** Eval 1 primary; Eval 2 primary (start from [`lerobot/smolvla_vlabench`](https://huggingface.co/lerobot/smolvla_vlabench) instead of `smolvla_base` — it's already fine-tuned on reasoning prompts).

### 2.2 Pi0 (Physical Intelligence π₀)

- **Paper:** [arXiv 2410.24164](https://arxiv.org/abs/2410.24164). **Reference repo:** [`Physical-Intelligence/openpi`](https://github.com/Physical-Intelligence/openpi). **LeRobot port:** [`lerobot/pi0`](https://huggingface.co/lerobot/pi0).
- **VLM backbone:** PaliGemma `gemma_2b` variant ([`configuration_pi0.py:32`](third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py)) — i.e. [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224). PaliGemma-3B = Gemma-2B language model (~2 B) + SigLIP-So400m vision encoder (~400 M) + projection ≈ 3 B total. The "gemma_2b" string in the config refers to the language-model variant inside PaliGemma, not the VLM total.
- **Action expert:** `gemma_300m` ([`configuration_pi0.py:33`](third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py)) — separate ~300 M Gemma-style transformer that produces actions via flow-matching.
- **Total params:** ~3.3 B (≈3 B PaliGemma + ~300 M action expert).
- **Action head:** flow-matching denoiser, default `num_inference_steps=10` ([`configuration_pi0.py:45`](third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py)).
- **Image resolution:** 224 × 224 ([`DEFAULT_IMAGE_SIZE`](third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py)).
- **Normalisation:** `MEAN_STD` for state and action ([`configuration_pi0.py:72-77`](third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py)).
- **Fine-tuning defaults:** `freeze_vision_encoder=False`, `train_expert_only=False` (lines 87-88) — i.e. by default the entire VLM also updates.
- **Best at:** zero-shot generalisation, dexterous tasks, cross-embodiment. Worse fit for the smallest-model bonus.

### 2.3 Pi0.5 (π₀.₅)

- **Paper:** [arXiv 2504.16054](https://arxiv.org/abs/2504.16054). **Model:** [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base).
- **VLM backbone:** identical to Pi0 — PaliGemma `gemma_2b` ([`configuration_pi05.py:32`](third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py)).
- **Differences from Pi0** (verified by diffing the configs):
  - **Normalisation:** `STATE` and `ACTION` use `QUANTILES`, not `MEAN_STD` ([`configuration_pi05.py:76-77`](third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py)). More robust to outliers in joint/action distributions.
  - **Tokenizer length:** `tokenizer_max_length=200` (vs 48 in Pi0) — supports longer prompts and subtask annotations.
- **Best at:** long-horizon home tasks, semantic subtask reasoning. The combination of stronger VLM (PaliGemma-3B) plus a longer tokenizer is what makes it the strongest reasoning candidate available locally.

### 2.4 FLOWER-VLA

- **Paper:** [arXiv 2509.04996](https://arxiv.org/abs/2509.04996). **Repo:** [`intuitive-robots/flower_vla_pret`](https://github.com/intuitive-robots/flower_vla_pret).
- **VLM backbone:** the README does not name a single VLM; the model uses a pruned LLM (~50 % removed) plus an intermediate-modality fusion head.
- **Total params:** ~950 M (paper).
- **Action head:** diffusion with Global-AdaLN.
- **Caveat:** repo ships **pretraining code only** — no fine-tune scripts and no released weights. ~200 H100-hours to retrain. Not plug-and-play for our $200 budget.
- **Use as:** architecture reference only.

### 2.5 TinyVLA

- **Paper:** [arXiv 2409.12514](https://arxiv.org/abs/2409.12514). **Repo:** [`liyaxuanliyaxuan/TinyVLA`](https://github.com/liyaxuanliyaxuan/TinyVLA).
- **VLM backbone:** LLaVA-Pythia in three sizes (400 M / 700 M / 1.3 B); checkpoints `lesjie/Llava-Pythia-{400M,700M,1.3B}` on HF.
- **Action head:** diffusion policy head.
- **Best at:** few-shot fine-tuning, fast inference. Strong **smallest-model bonus** candidate at the 400 M tier.
- **Caveat:** original eval was on Franka / ALOHA, not SO-101. Output dir name must contain `"llava_pythia"`; data must be HDF5 in a specific layout; eval requires the `process_ckpts` post-processing pass (per repo README).

### 2.6 OpenVLA

- **Paper:** [arXiv 2406.09246](https://arxiv.org/abs/2406.09246). **Model:** [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b).
- **VLM backbone:** Llama-2-7B + DINOv2 + SigLIP visual encoder.
- **Total params:** 7 B.
- **Action head:** autoregressive — discretised actions emitted as language tokens (no separate action expert).
- **Caveat:** kills the smallest-model bonus; LoRA fine-tuning is supported but VRAM cost is still high.
- **Use as:** reference / sanity baseline only.

### 2.7 GR00T-N1.5 (NVIDIA)

- **Blog:** [SO-101 fine-tune guide](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning).
- Larger generalist VLA shipped with an SO-101 fine-tuning recipe; locally available under [`third_party/lerobot/src/lerobot/policies/groot/`](third_party/lerobot/src/lerobot/policies/groot/).
- **Use as:** reference. Loses the size bonus.

### 2.8 ObjectVLA

- **Page:** <https://objectvla.github.io/> · **Paper:** <https://arxiv.org/html/2502.19250v1>.
- Co-fine-tunes a base VLA on vision-language pairs + teleop; generalises to ~100 novel objects with no per-target demos.
- **Pattern reference for Eval 3:** demonstrates that **inside a single VLA** you can pair vision-language data + teleop to teach OOD-target generalisation without per-target demos. Directly relevant to the ~100-celebrity OOD list (see [`PROJECT.md §2 Eval 3`](PROJECT.md#eval-3--coke-can-on-celebrity-image-50-pts)).

### 2.9 Interleave-VLA

- **Paper:** [arXiv 2505.02152](https://arxiv.org/abs/2505.02152) · **Code:** [`Interleave-VLA/Interleave-VLA`](https://github.com/Interleave-VLA/Interleave-VLA) · **Page:** <https://interleave-vla.github.io/Interleave-VLA-Anonymous/>.
- π0-based VLA that takes interleaved text+image prompts; closest public precedent for *image-of-image* reasoning **inside a single VLA**.
- **Use as:** Eval 3 architecture inspiration. We could feed the celebrity prompt + a reference photo of that celebrity (sourced from the ~100-name list TAs will publish) as an interleaved prompt — the VLA does the matching internally. **This stays VLA-only**: no separate face-ID model, no orchestration LLM. The reference photos are training-data assets, not a runtime model.

---

## 3. Per-eval recommendation (VLA + VLM)

Each eval below quotes the constraint that drives the choice, the recommended VLA, the VLM the recommendation rests on, and a one-fallback option in case the primary fails on real-robot eval day.

### Eval 1 — "Put the banana in the [red / green / blue] bowl"

> *"Three bowls (blue, red, green) in a semicircle. Banana in smiley orientation in front of the arm. **The banana never rotates** — always smiley, ±5 cm from your teleop position. Prompt names the colour. 9 rollouts / team, 20 s each."* — [`PROJECT.md §2 Eval 1`](PROJECT.md#eval-1--direct-color-conditioned-pick-and-place-50-pts).

- **Why colour-name lookup is easy on a small VLM:** the prompt names a colour that is directly observable in pixels. SmolVLM-style VLMs handle simple referential expressions well; we don't need world knowledge.
- **Banana is a fixed-pose object** (never rotated). Treat it as a near-constant in image space — the variation that matters is *bowl colour position*, not banana pose. Demos should over-sample bowl-colour permutations rather than banana orientations.
- **Primary VLA:** **SmolVLA** (`lerobot/smolvla_base`), 50–75 demos with intentional bowl colour-position variation.
- **Primary VLM:** `SmolVLM2-500M-Video-Instruct` (default in [`configuration_smolvla.py:87`](third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py)). 500 M backbone is sufficient. Maximises the smallest-model bonus.
- **Camera (per [`PROJECT.md §4`](PROJECT.md#cameras-updated-project-rule)):** wrist mount is fine — three bowls fit in the wrist field of view at the start pose.
- **Fallback VLA:** Pi0 only if SmolVLA fails on the real robot. PaliGemma-3B inside Pi0 is overkill for direct colour lookup and costs more compute on the $200 Brev budget.

### Eval 2 — Compositional reasoning ("2nd bowl from the left", "right of the red bowl", "not green and not blue")

> *"Banana at the **exact same position as Eval 1** (smiley, fixed). The **three colored bowls may switch places** (same colour set as Eval 1, just reshuffled). Prompts require reasoning beyond direct colour lookup."* — [`PROJECT.md §2 Eval 2`](PROJECT.md#eval-2--compositional-instruction-following-50-pts).

Actual example prompts (per [`PROJECT.md`](PROJECT.md)):
- *"Put the banana into the 2nd bowl from the left."* (ordinal-position reasoning)
- *"Put the banana into the bowl on the right of the red bowl."* (relative-position reasoning, requires colour grounding too)
- *"Put the banana into the bowl that is not green and not blue."* (negation → red bowl)

- **Why compositional prompts stress the VLM:** the policy must combine **colour grounding** (which bowl is red?) with **spatial reasoning** (left of / right of / 2nd from). These are language tasks the VLM has to handle before the action expert ever fires.
- **Banana position is fixed** (same as Eval 1) — the model only has to vary its target *bowl*, not its grasp. This makes Eval 2 a great candidate for **fine-tuning from the Eval 1 checkpoint** rather than starting from scratch.
- **Primary VLA:** **SmolVLA fine-tuned from [`lerobot/smolvla_vlabench`](https://huggingface.co/lerobot/smolvla_vlabench)** (not `smolvla_base`). Same 500 M backbone, but the starting checkpoint has already seen 3.11 M frames of VLABench reasoning prompts.
- **Primary VLM:** `SmolVLM2-500M-Video-Instruct`. Despite the reasoning aspect, the prompt vocabulary is small and the eval is closed-set; we expect SmolVLM2-500M to be sufficient when warm-started from a reasoning-tuned checkpoint.
- **Demo strategy:** sample all 3! = 6 bowl-position permutations roughly evenly. Within each permutation, hit the three reasoning prompt patterns (ordinal / relative / negation). ≥ 50 demos total per [`smolvla.mdx`](third_party/lerobot/docs/source/smolvla.mdx).
- **Camera:** same as Eval 1 (wrist or shoulder, but be **consistent** with the Eval 1 setting if you fine-tune from that checkpoint — see [issue #1763](https://github.com/huggingface/lerobot/issues/1763) on camera-order matching).
- **Fallback VLA:** **Pi0.5** (`lerobot/pi05_base`) with PaliGemma-3B — picked over Pi0 because Pi0.5's `tokenizer_max_length=200` ([`configuration_pi05.py:71`](third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py)) tolerates longer reasoning prompts and quantile normalisation is more robust on small fine-tune sets. Costs the size bonus.
- **Fallback VLM:** PaliGemma-3B (`google/paligemma-3b-pt-224`).

### Eval 3 — "Place the coke on [Taylor Swift / Obama / LeCun / Federer / Merkel]"

> *"DIN A5 **portrait** colour prints of celebrities in a semicircle; a **normal 330 ml** coke can (no Coke Zero) in the middle, may be crumbled but must stand. Bring your exact can to demo day. In-distribution: Obama / Swift / LeCun. OOD: drawn from a list of ~100 candidates that TAs publish on the weekend of 2026-05-02 / 03."* — [`PROJECT.md §2 Eval 3`](PROJECT.md#eval-3--coke-can-on-celebrity-image-50-pts).

> ⚠️ **VLA-only:** [`PROJECT.md §3`](PROJECT.md#3-architecture--procedure-constraints) bans face-recognition models, OCR models, orchestration LLMs/VLMs, and YOLO. The earlier "ArcFace frontend → SmolVLA" recipe is **not allowed**. Everything below is a single end-to-end VLA.

#### Why this eval is hard with VLA-only

The policy must, inside one VLA pass:
1. **Recognise the named celebrity** (Obama / Swift / LeCun in-distribution; Federer / Merkel and others from the ~100-name pool out-of-distribution).
2. **Match the name to one of the A5 prints on the table** — visual identity grounding from a photo it has likely never seen at that exact angle.
3. **Place the coke can** on top of that print without knocking over the other prints.

The bottleneck is **(1) + (2)** — celebrity world-knowledge and visual face-grounding. SmolVLM-500M's pretraining is too weak for arbitrary celebrities. We need a stronger VLM backbone or we need to feed the celebrity's identity into the prompt as visual context.

#### Primary architecture: Pi0.5 end-to-end on celebrity-prompt demos

- **Primary VLA:** **Pi0.5** ([`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)).
- **Primary VLM:** **PaliGemma-3B** ([`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224)). Pretrained on WebLI (filtered web image-text) — has materially more world knowledge of public figures than SmolVLM-500M. Pi0.5's `tokenizer_max_length=200` also fits the prompt template *"Place the coke on Taylor Swift"* + scene description comfortably.
- **Demo collection (in-distribution).** ≥ 50 demos for the in-distribution set (Obama / Swift / LeCun): cycle through different print images per celebrity (different angles / outfits — buy stock photos or print 3–5 different photos per celebrity), randomise print positions in the semicircle, randomise the can's starting offset (±5 cm), randomise lighting. Each demo's task string is *"Place the coke on `<name>`"*.
- **OOD strategy.** Once TAs publish the ~100-candidate list this weekend, **add reference prints + demos for as many OOD candidates as compute allows**. The eval will sample some subset — we can't cover all 100 with full demos, so prioritise the most-likely-tested half plus diverse training signal so PaliGemma's pretrained face-association generalises to the rest.
- **Camera (per-eval choice):** **shoulder mount** is strongly preferred for Eval 3. The wrist camera tilts with the gripper and may lose sight of the prints on the far ends of the semicircle; a fixed shoulder/agent view sees all prints throughout the rollout. [`PROJECT.md §4 Cameras`](PROJECT.md#cameras-updated-project-rule) explicitly allows a different camera setting per eval, so this is fine even if Eval 1 / 2 use wrist.
- **Why Pi0.5 over Pi0.** Pi0.5's quantile state/action normalisation is more robust on small fine-tune sets, and the longer tokenizer handles the celebrity-name vocabulary without truncation.

#### Fallback architecture: image-as-prompt, single VLA

- **Fallback VLA:** **SmolVLA** trained Interleave-VLA-style — accept the celebrity prompt **plus a reference photo of that celebrity** as part of the input (image-of-image prompting, all inside one VLA forward pass).
- The reference photos come from the ~100-name candidate list; at runtime we look up the name → reference photo and feed both to the VLA. The VLA does the matching internally — no separate face-ID model.
- **VLA-only?** Yes. The reference photos are training-data assets / a runtime lookup table, not a separate model. No face-recognition model runs at inference.
- **Why fallback, not primary:** Interleave-VLA's open-source training pipeline is less mature than SmolVLA / Pi0.5 in LeRobot. We'd be doing more plumbing work for a smaller VLM (SmolVLM-500M) — only worth it if PaliGemma-3B turns out to be too slow on the 20 s rollout cap or fails on OOD celebrities.

#### Smallest-model-bonus accounting

[`PROJECT.md §2 Bonus`](PROJECT.md#bonus-up-to-50-pts--smallest-model) ranks by total active inference params. With VLA-only:
- Pi0.5 primary: **~3.3 B** total (3 B PaliGemma + 0.3 B action expert) — costs the size bonus, but it's the only path with strong celeb recognition.
- SmolVLA fallback (image-as-prompt): **~450 M** — far better for the bonus, but uncertain on celeb recognition.

If we can make SmolVLA work via image-as-prompt, that's the size-bonus play; if not, eat the bonus loss for Pi0.5.

#### Baselines to run before eval day

1. **Zero-shot PaliGemma-3B probe.** Before fine-tuning, ask `google/paligemma-3b-pt-224` to identify each in-distribution celebrity from the print images and a sample of OOD celebrities. If it scores < 80 % on in-distribution, Pi0.5 alone won't be enough — pivot to image-as-prompt.
2. **Pi0.5 fine-tuned without explicit name-prints supervision.** Train on demos with name-only prompts. If OOD success collapses, the name → image association isn't transferring — switch to image-as-prompt.
3. **Compute budget check.** Pi0.5 fine-tuning on Brev: 3 B-param model means smaller `batch_size` and more steps than SmolVLA. Plan ~10–15 h of A100 time vs ~4 h for SmolVLA.

---

## 4. Architecture deep-dive

### 4.1 Generic VLA stack

```
[wrist or shoulder camera] → vision encoder ──┐
                                              ├─→ VLM (frozen or partially fine-tuned)
[natural-language prompt]    → tokenizer  ────┘            │
                                                           ▼
                                            [contextual features]
                                                           │
                                                           ▼
                                          action head (flow-matching / diffusion / autoregressive)
                                                           │
                                                           ▼
                                       chunk of N action vectors → SO-101 motors
```

For SmolVLA, Pi0, Pi0.5, FLOWER-VLA, TinyVLA the action head is **decoupled** from the VLM — it's a separate transformer ("action expert") trained jointly. For OpenVLA the action head **is** the VLM (actions emitted as discrete tokens).

### 4.2 SmolVLA internals

Sourced from [`configuration_smolvla.py`](third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py):

| Knob | Default | Meaning |
|---|---|---|
| `vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | The VLM backbone. **Swappable in one flag**, but a swap forces full retraining (see §7). |
| `num_vlm_layers` | 16 | Number of VLM transformer layers actually used (truncates the backbone). |
| `expert_width_multiplier` | 0.75 | Action expert hidden size = 75 % of VLM hidden size. |
| `num_expert_layers` | -1 (= same as VLM) | Action expert depth. |
| `chunk_size` / `n_action_steps` | 50 / 50 | Predict 50 actions per VLM forward pass; execute all 50. |
| `freeze_vision_encoder` | True | Vision encoder frozen by default. |
| `train_expert_only` | True | Default fine-tune mode trains only the action expert + projections (cheap). |
| `tokenizer_max_length` | 48 | Short prompts only — relevant for Eval 2 phrasing. |
| `num_steps` | 10 | Flow-matching denoising steps at inference. |
| `resize_imgs_with_padding` | (512, 512) | Input image size. |
| `optimizer_lr` / scheduler | 1e-4 cosine, 1k warm-up, 30k decay | Default training schedule. |

### 4.3 Pi0 / Pi0.5 internals

Sourced from [`configuration_pi0.py`](third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py) and [`configuration_pi05.py`](third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py):

| Knob | Pi0 default | Pi0.5 default | Notes |
|---|---|---|---|
| `paligemma_variant` | `gemma_2b` | `gemma_2b` | Both wrap PaliGemma-3B. |
| `action_expert_variant` | `gemma_300m` | `gemma_300m` | Separate ~300 M action expert. |
| `num_inference_steps` | 10 | 10 | Flow-matching denoising. |
| `image_resolution` | 224 × 224 | 224 × 224 | Smaller inputs than SmolVLA. |
| `tokenizer_max_length` | 48 | **200** | Pi0.5 supports much longer prompts. |
| State / action norm | `MEAN_STD` | **`QUANTILES`** | Quantile norm is more outlier-robust. |
| `freeze_vision_encoder` | False | False | Default trains the VLM too. |
| `train_expert_only` | False | False | Default trains the full stack. |
| `optimizer_lr` | 2.5e-5 | 2.5e-5 | Lower than SmolVLA's 1e-4 because PaliGemma-3B is updating. |

### 4.4 What changes between Pi0 and Pi0.5

Per the configs, two things differ at the framework level: tokenizer length (48 → 200) and state/action normalisation (mean-std → quantiles). Architectural changes (longer-horizon hierarchical reasoning, web co-training) live in the model weights and the openpi training pipeline rather than the LeRobot port.

> Footnote: `tokenizer_max_length=200` is set twice in [`configuration_pi05.py`](third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py) (lines 71 and 105) — an upstream duplicate. Both set the same value so behaviour is unaffected, but a future reader changing one and not the other would silently break.

---

## 5. Knobs we can tune per eval

The space splits into **data**, **training**, and **inference** knobs. The same VLA exposes the same knobs across evals; what changes is how we *use* them.

### 5.1 Data knobs

| Knob | Eval 1 | Eval 2 | Eval 3 |
|---|---|---|---|
| Number of demos | ≥ 50 (per [`smolvla.mdx:32`](third_party/lerobot/docs/source/smolvla.mdx)) | ≥ 50, balanced over **3! = 6** bowl-colour permutations × 3 reasoning prompt types | ≥ 50 in-distribution (Obama / Swift / LeCun) + add reference demos for as many of the ~100 OOD candidates as compute allows |
| Language strings | 3 fixed colours | Ordinal ("2nd from left"), relative ("right of red bowl"), negation ("not green and not blue"). **No** colour-mixing prompts (not in the actual eval) | `"Place the coke on <celebrity name>"` for every in-distribution + OOD-candidate name in demos |
| Augmentation | colour jitter, light, table | colour jitter; randomise bowl colour positions; **banana stays fixed** | randomise print positions in the semicircle, vary lighting, vary the print images per celebrity (3–5 different photos each) |
| Camera placement (different per eval allowed, see [PROJECT.md §4](PROJECT.md#cameras-updated-project-rule)) | wrist (close-up of bowl rim helps placement) | same as Eval 1 (consistency for fine-tune-from-Eval-1 path) | **shoulder mount** — fixed view sees all prints on the semicircle, doesn't lose sight when the gripper tilts |

### 5.2 Training knobs

| Knob | Where | Effect |
|---|---|---|
| `freeze_vision_encoder` | SmolVLA / Pi0 config | True = much cheaper, only action expert + projections train |
| `train_expert_only` | SmolVLA / Pi0 config | True = freezes the VLM entirely (SmolVLA default). Set False on Pi0.5 if we want PaliGemma to adapt to bowl colours |
| LoRA on VLM | not in default LeRobot configs | Compromise between full FT and frozen VLM; not exposed by the configs we ship — would need a custom training script |
| `batch_size` | CLI flag | 64 from `smolvla.mdx`; lower for larger backbones |
| `steps` | CLI flag | 20k for SmolVLA per official guide |
| `optimizer_lr` | config | 1e-4 (SmolVLA) / 2.5e-5 (Pi0/0.5). Lower the LR if we unfreeze the VLM |

### 5.3 Inference knobs

| Knob | Effect |
|---|---|
| `num_inference_steps` (Pi0/0.5) / `num_steps` (SmolVLA) | 10 default. Lower = faster but lower quality. **Relevant: 20 s rollout limit** ([`PROJECT.md §2`](PROJECT.md#2-evaluation--150-pts-main--50-pts-bonus)). |
| `chunk_size` / `n_action_steps` | 50 default. Smaller chunk = faster reaction, more VLM forwards per second |
| Camera order | Must match between train and deploy (see [`huggingface/lerobot#1763`](https://github.com/huggingface/lerobot/issues/1763)). Hard-code one source of truth |

### 5.4 Eval 3-specific architecture knob

VLA-only constraint forces an end-to-end VLA. The remaining choice is **how the celebrity identity reaches the policy**:

- **Name-only prompt (default Pi0.5 path).** Prompt = `"Place the coke on Taylor Swift"`. The VLM (PaliGemma-3B) must contain the name → face association internally. Best on celebrities the VLM saw a lot of in pretraining (very famous public figures); weaker on OOD niche names.
- **Image-as-prompt (Interleave-VLA-style fallback path).** Prompt = `"Place the coke on this person"` + a reference photo of the named celebrity, looked up at runtime from the published ~100-name candidate list. The VLA does the photo-to-print matching internally — still a single VLA, no extra models. Better on OOD names but more plumbing to build.

---

## 6. End-to-end workflow (LeRobot v3 → SO-101)

```
1. Calibrate SO-101 + teleop                       (lerobot-calibrate, lerobot-teleoperate)
2. Record 50+ episodes with single camera          (lerobot-record  → LeRobot v3 dataset)
3. Replay episodes on real robot (sanity)          (lerobot-replay)
4. Push dataset to HF                              (huggingface-cli upload …)
5. Fine-tune VLA on Brev GPU                       (lerobot-train --policy.path=…)
6. Pull checkpoint, deploy on SO-101                (lerobot-record --policy.path=local …)
```

**Canonical fine-tune command** ([source](third_party/lerobot/docs/source/smolvla.mdx) lines 56–66):

```bash
cd lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

For Eval 2, swap `--policy.path=lerobot/smolvla_vlabench`. For the Pi0.5 fallback, swap to `--policy.path=lerobot/pi05_base` and lower `--batch_size` (PaliGemma-3B is ~6 × the SmolVLA VRAM footprint).

**Deploy command** (from `smolvla.mdx` lines 99–117):

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_blue_follower_arm \
  --robot.cameras="{ front: { type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30 } }" \
  --dataset.single_task="Put the banana in the red bowl." \
  --dataset.repo_id=${HF_USER}/eval_test \
  --dataset.episode_time_s=20 \
  --dataset.num_episodes=9 \
  --policy.path=${HF_USER}/my_smolvla_finetune
```

Set `episode_time_s=20` to match the eval-day rollout limit.

---

## 7. Backbone-swap guidance — `huggingface/lerobot#2104`

Source: [Issue #2104 thread](https://github.com/huggingface/lerobot/issues/2104). Maintainer: `jadechoghari` (HuggingFace).

**Verbatim guidance, paraphrased:**

> SmolVLA defaults to `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. Larger alternatives include `google/paligemma-3b-pt-224` (which Physical Intelligence uses for Pi0 and Pi0.5). Both Pi0 and Pi0.5 already use PaliGemma-3B. Larger PaliGemma variants (10 B, 28 B) are available via the [PaliGemma-2-Mix collection](https://huggingface.co/collections/google/paligemma-2-mix-67ac6a251aaf3ee73679dcc4).
>
> **Swapping the backbone forces from-scratch training.** You cannot take an existing SmolVLA / Pi0 checkpoint and swap in a different VLM — the encoder dimensions and pretraining co-adapt with the action expert. Set `--policy.vlm_model_name=<new-vlm>` and the framework re-initialises and re-loads via `transformers`. No code surgery is needed, but compute cost is high.

**Implications for our evals:**

- **Eval 1.** Stay with SmolVLM2-500M. The reasoning bar is low and the size bonus rewards staying small.
- **Eval 2.** First try `lerobot/smolvla_vlabench` (still SmolVLM2-500M, no backbone swap). If reasoning fails, switch to **Pi0.5 (PaliGemma-3B)** rather than custom-swapping the SmolVLA backbone — Pi0.5 is already trained, so we avoid the from-scratch cost.
- **Eval 3.** **Pi0.5 with PaliGemma-3B is the primary play** — it's the only plug-and-play VLA in LeRobot whose VLM has serious world knowledge of public figures. Don't custom-swap a 10 B / 28 B PaliGemma in: maintainer guidance below says it's not real-time-practical. If Pi0.5 falls short on OOD celebrities, fall back to image-as-prompt with SmolVLA (still a single VLA — see §3 Eval 3 fallback).

**Caveats from the thread:**
- Switching backbones = no checkpoint reuse.
- Larger PaliGemma variants trade inference latency for capacity. Maintainer guidance (jadechoghari, 2025-10-11): "*not as practical yet for real-time control*" — directly relevant given our 20 s rollout cap.
- The default 500 M backbone is already lightweight; jumping to 3 B is a ~6 × VLM-capacity increase.

---

## 8. Sources

### Local source files (LeRobot submodule)
- [`third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py`](third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py)
- [`third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`](third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [`third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py`](third_party/lerobot/src/lerobot/policies/pi0/configuration_pi0.py)
- [`third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py`](third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py)
- [`third_party/lerobot/docs/source/smolvla.mdx`](third_party/lerobot/docs/source/smolvla.mdx)

### Papers
- SmolVLA — <https://arxiv.org/abs/2506.01844>
- Pi0 — <https://arxiv.org/abs/2410.24164>
- Pi0.5 — <https://arxiv.org/abs/2504.16054>
- FLOWER-VLA — <https://arxiv.org/abs/2509.04996>
- TinyVLA — <https://arxiv.org/abs/2409.12514>
- OpenVLA — <https://arxiv.org/abs/2406.09246>
- ObjectVLA — <https://arxiv.org/html/2502.19250v1>
- Interleave-VLA — <https://arxiv.org/abs/2505.02152>

### Model cards
- <https://huggingface.co/lerobot/smolvla_base>
- <https://huggingface.co/lerobot/smolvla_vlabench>
- <https://huggingface.co/lerobot/pi0>
- <https://huggingface.co/lerobot/pi05_base>
- <https://huggingface.co/google/paligemma-3b-pt-224>
- <https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct>
- <https://huggingface.co/openvla/openvla-7b>

### Repos
- <https://github.com/Physical-Intelligence/openpi>
- <https://github.com/intuitive-robots/flower_vla_pret>
- <https://github.com/liyaxuanliyaxuan/TinyVLA>
- <https://github.com/Interleave-VLA/Interleave-VLA>

### Discussions & issues
- VLM backbone swap — <https://github.com/huggingface/lerobot/issues/2104>
- Camera config gotcha — <https://github.com/huggingface/lerobot/issues/1763>
- NVIDIA GR00T-N1.5 SO-101 — <https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning>
