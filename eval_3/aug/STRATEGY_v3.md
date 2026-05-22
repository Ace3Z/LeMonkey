# Eval 3 - Augmentation Strategy v3 (Path A, 200-celeb bank)

> **Status:** Locked 2026-05-14 after 4-agent literature synthesis.
> **Supersedes:** `STRATEGY.md` (May 10 - pre-Path-A pipeline).
> **Companion:** `RESEARCH_v2.md` (May 10 - older research review of inpaint
> primitives; algorithmic choices there are still current, but the slot
> strategy and prompt mixture below override anything inconsistent).
> **Quality bar:** CLAUDE.md §7 - every numerical default is triple-sourced;
> §11 "Cross-check" appendix shows the verification.

---

## TL;DR

For each of the 180 base teleop episodes:

1. Generate **M = 25 augmented variants**.
2. Per variant: replace **ALL 3** visible portraits with celebs from a
   ~203-celeb bank (200 scraped + Swift/Obama/LeCun from `web/`).
3. The target celeb's reference photo (`observation.images.reference`) is
   a **different photo of the same celeb** than the one painted on the
   table.
4. Prompt is drawn from a **75 / 15 / 10 mixture** (default name + action,
   reference-only, counterfactual wrong-name).
5. SmolVLA's VLM backbone stays **frozen** (default); only the action
   expert trains.

Total training corpus: 180 × 25 = **4 500 augmented videos**, each celeb
appears as target **~22 times** and as distractor **~45 times**.

---

## §1 · Problem statement

The Eval 3 task is: given the wrist-camera view of a table with 3 printed
A5 celebrity portraits + a Coke can, plus a text prompt naming the target
celeb, plus a reference photo of the target celeb, place the can on the
matching printed paper.

The TA's evaluation sample is **9 rollouts** drawn from:

- 3 IID-known (printed photos seen at training) - 3 celebs from
  `eval3_celebs/heldout/`, namely Swift/Obama/LeCun;
- 3 IID-held-out (different photos of same 3 celebs);
- 3 OOD (top-50 famous people, never trained on).

The architecture decision **Path A - image as prompt** was locked
2026-05-09 after a [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224)
probe scored **0/14** on TOY names and **0/6** on OOD names - VLMs do not
reliably know celebrity faces from name alone. Path A side-steps this
by passing a clean reference photo of the target as a second image input
(`observation.images.reference`), turning the task into **face
verification** - a well-validated transfer problem ([Schroff et al.,
FaceNet, CVPR 2015](https://arxiv.org/abs/1503.03832); [Deng et al.,
ArcFace, CVPR 2019](https://arxiv.org/abs/1801.07698); [Cao et al.,
VGGFace2, 2017](https://arxiv.org/abs/1710.08092)).

The model therefore needs to learn a single skill: **match the
reference embedding to one of the 3 visible portrait embeddings, then
direct the can to that paper.** It does NOT need to memorise 200 celebs
by name. This is the single most important framing of the project, and
it's what makes M = 25 sufficient - see §5.

---

## §2 · Slot strategy - ALL 3 portraits replaced

For each augmented variant, all 3 visible portraits are inpainted with
new celebs drawn from the bank. **No celeb is held fixed across
variants.**

### §2.1 Why not "only the target"

If only the target portrait is replaced and Swift/Obama/LeCun remain as
the 2 distractors in every variant, the model has at least **three
shortcuts** that beat the intended face-matching objective on training
loss:

1. **Odd-one-out by identity.** The model learns
   `argmax_i 1[face_i ∉ {Swift, Obama, LeCun}]`. Training loss → 0; at
   evaluation, ALL 3 portraits are unknown celebs, the indicator fires
   on all 3 (or none), accuracy collapses to **chance (33 %)**. This is
   the documented LIBERO-Plus pathology
   ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626)).
2. **Photo-style shortcut.** The 2 unchanged Swift/Obama/LeCun papers
   were physically printed and re-photographed; the 1 inpainted target
   has different JPEG noise, color, blur. The model picks "the AI-edited
   one." Same eval collapse. Established in
   [Geirhos et al. 2020, "Shortcut Learning in DNNs"
   (arXiv:2004.07780)](https://arxiv.org/abs/2004.07780)
   and [Shortcut Learning in Generalist Robot Policies, Aug 2025
   (arXiv:2508.06426)](https://arxiv.org/abs/2508.06426).
3. **Distribution mismatch at eval (open-set).** Training distractors
   are drawn from `{Swift, Obama, LeCun}` (closed set, 3 identities);
   eval distractors are drawn from top-50 famous people. This is the
   classical closed-set→open-set generalisation failure
   ([Günther et al., "Toward Open-Set Face Recognition", arXiv:1705.01567](https://arxiv.org/abs/1705.01567);
   [Vareto et al., 2023, arXiv:2311.00400](https://arxiv.org/abs/2311.00400)).

### §2.2 Why "all 3 replaced" is correct

Under Path A (image-as-prompt), the action depends only on the
**match relation** `argmax_i sim(reference_embed, portrait_i_embed)`.
Identity per se is causally irrelevant - only the match matters. Under
RoCoDA's causal decomposition ([Doshi et al., Nov 2024,
arXiv:2411.16959](https://arxiv.org/abs/2411.16959)) all 3 portraits and
both distractor identities are causally-irrelevant scene state `s_I`.
The correct augmentation policy is to **resample `s_I` uniformly while
preserving the causal relation** - which means swapping all 3 slots
per variant, with the reference image updated in lock-step with whatever
slot is designated target.

This is the structural insight that **no published VLA inpainting paper
exploits** because they bind the target via *language* (ROSIE / CACTI /
NICE / GenAug all keep the target fixed because changing it would
violate the language label -
[ROSIE, arXiv:2302.11550](https://arxiv.org/abs/2302.11550);
[GenAug, arXiv:2302.06671](https://arxiv.org/abs/2302.06671);
[CACTI, arXiv:2212.05711](https://arxiv.org/abs/2212.05711);
[NICE, arXiv:2511.22777](https://arxiv.org/abs/2511.22777)). Path A
re-binds the target via the *reference image*, so swapping is
label-preserving.

### §2.3 Sampling within a variant

```
Sample c_T, c_D1, c_D2  uniformly without replacement from the 203-celeb bank
Sample 2 distinct photos of c_T  → (p_T_print, p_T_ref)
Sample 1 photo each of c_D1, c_D2 → (p_D1, p_D2)
```

`c_T` goes to whichever physical slot the original episode's target
occupied (preserves the recorded action trajectory). `c_D1`, `c_D2` fill
the other 2 slots in random order.

### §2.4 Stratification across variants

- Each celeb appears as target with equal marginal frequency
  (~22 / 4 500 across the full corpus).
- Each celeb appears as distractor with equal marginal frequency
  (~45 / 4 500).
- Each celeb appears in each slot position (left / middle / right) with
  uniform marginal - mitigates the positional bias documented in
  [LIBERO-Plus, arXiv:2510.13626](https://arxiv.org/abs/2510.13626).
- Within the per-celeb photo pool (5–10 photos each from the multi-
  source scrape, see `scrape_headshots.py`), photos are rotated across
  variants so no single photo is over-used.

---

## §3 · Reference photo strategy

### §3.1 Different photo, same celeb (split-pair)

For each variant the reference photo (`observation.images.reference`)
is a **different photo of the target celeb** than the one painted onto
the wrist view. Never the same file.

**Why:** if reference = the same image as the printed portrait, the
matching task degenerates to pixel similarity. At eval the TA hands a
different photo for both the printed and reference channels, so a
pixel-identity-trained model fails by construction.

Sources:
- [Schroff et al., FaceNet, CVPR 2015](https://arxiv.org/abs/1503.03832) -
  triplet anchor and positive must be DIFFERENT images of the same
  identity; this is the canonical face-verification training pattern.
- [Cao et al., VGGFace2, 2017 (arXiv:1710.08092)](https://arxiv.org/abs/1710.08092) -
  intra-class variance (many different photos per ID, 80–843) is *more*
  important than per-image quality for OOD face recognition.
- [Ruiz et al., DreamBooth, CVPR 2023 (arXiv:2208.12242)](https://arxiv.org/abs/2208.12242) -
  3–5 different photos per subject is the empirical minimum for identity
  learning; 10–20 for faces specifically.

### §3.2 Per-celeb photo pool

Target **≥ 5 distinct photos per celeb** (DreamBooth minimum extended
for faces); **6–10 ideal**. Our scraper (`eval_3/scripts/scrape_headshots.py`)
already targets 10/celeb and we currently have:

- 166 celebs with full 10/10 bank;
- 39 celebs with 1–9 photos (the long tail);
- 12 celebs with < 5 photos - flagged by §7 audit, supplemented or dropped.

For the 3 IID celebs (Swift, Obama, LeCun), the bank includes the
photos previously in `eval3_celebs/web/<short>/` folded into
`eval3_celebs/scraped/<full_name>/` (see §6.1).

### §3.3 The reference is `s_I` invariance

Under the same RoCoDA framing as §2.2, the reference photo is `s_I` from
the model's view: only its *identity* matters, the specific pixels
don't. Rotating through multiple photos per celeb across variants
forces identity-invariant features rather than photo-specific
memorisation - verified in [Kortylewski et al., 2018
"Empirically Analyzing the Effect of Dataset Biases on DCNNs"](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w41/Kortylewski_Empirically_Analyzing_the_CVPR_2018_paper.pdf)
which shows DCNNs fail to generalise pose when training pose is
restricted.

---

## §4 · Reference photo as a second image stream (the LeRobot wiring)

LeRobot's data schema is "any number of `observation.images.<name>` keys."
Eval 1 and 2 use one camera (`observation.images.camera1`). Eval 3 adds
a second key for the reference channel.

Two representations are viable; we choose **option A (constant-frame
video)** for v1 because it requires no LeRobot-config tweak:

| Option | Format | Pros | Cons |
|---|---|---|---|
| A | Encode reference as a 1280×720 (or similar) MP4 of length `n_frames`, every frame identical, written under `videos/observation.images.reference/chunk-000/file-000.mp4`. | Drop-in compatible with LeRobot's `from_pretrained`; no schema change. | Wastes disk (~5–10 KB/frame for h264 with no motion). |
| B | Store the reference image once in `meta/info.json` as a non-video feature, declare it as `IMAGE` in `input_features`. | Storage-optimal. | Requires LeRobot config tweak + custom loader. |

The training-side policy config registers both streams:

```yaml
policy:
  type: smolvla
  input_features:
    observation.images.camera1:
      type: VIDEO
      shape: [3, 480, 640]
    observation.images.reference:
      type: VIDEO
      shape: [3, 480, 640]
    observation.state:
      type: VECTOR
      shape: [6]
```

The SmolVLM-2 backbone (SmolVLA's VLM, [Shukor et al., 2025,
arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) is natively
multi-image - its [SigLIP](https://arxiv.org/abs/2303.15343) vision
encoder produces image tokens for each frame independently, and the
SmolLM2 decoder interleaves them with text tokens. The reference image
is just additional image tokens in the input sequence - no architecture
change.

This pattern is precedented in
[VIMA (Jiang et al., ICML 2023, arXiv:2210.03094)](https://arxiv.org/abs/2210.03094)
and [Interleave-VLA, May 2025 (arXiv:2505.02152)](https://arxiv.org/abs/2505.02152),
both of which interleave multiple image inputs as part of the prompt
sequence; the latter reports **2× improvement in OOD generalisation to
unseen objects** specifically from mixing internet-sourced reference
photos with robot observations.

---

## §5 · Per-celeb data math (M = 25)

With 180 base episodes × M variants × 3 portrait slots = `540 M`
slot-occurrences, distributed over a 203-celeb bank:

| M | Variants total | Target / celeb | Distractor / celeb | Status |
|---|---|---|---|---|
| 5  | 900     | 4   | 9   | **Below floor** - DreamBooth minimum is 3–5, 10+ for faces; risky for tail celebs |
| 10 | 1 800   | 9   | 18  | Marginal - at the floor; tail celebs (< 5 photos) under-cover |
| 20 | 3 600   | 18  | 36  | Solid - above DreamBooth ideal (10–20 for faces) |
| **25** | **4 500** | **22** | **45** | **Sweet spot** - comfortably above DreamBooth face minimum, well below Lin et al. plateau |
| 30 | 5 400   | 27  | 54  | OK - approaching Lin et al. K = 50 plateau region |
| 50 | 9 000   | 45  | 90  | Wasteful - beyond [Lin et al. ICLR 2025 (arXiv:2410.18647)](https://arxiv.org/abs/2410.18647) K = 50 plateau for per-pair imitation-learning data |
| 100 | 18 000 | 90  | 180 | Strongly diminishing returns; compute is better spent on more layouts |

### §5.1 Why face-matching plateaus near M = 25 for us

Three independent saturation signals converge:

1. **Personalisation literature** - DreamBooth recommends 3–5 photos for
   general objects, 10–20 for faces ([Ruiz et al., 2023,
   arXiv:2208.12242](https://arxiv.org/abs/2208.12242);
   [Kumari et al., Custom Diffusion, CVPR 2023](https://www.cs.cmu.edu/~custom-diffusion/)
   recommends 15–20 for faces specifically;
   [Gal et al., Textual Inversion, ICLR 2023, arXiv:2208.01618](https://arxiv.org/abs/2208.01618)
   uses 3–5). 22 target appearances × multiple photos per appearance is
   comfortably above all three thresholds.
2. **Self-supervised retrieval saturation** - DINOv2 k-NN classification
   reaches usable accuracy with as few as 5–32 examples per class
   ([Oquab et al., 2023, arXiv:2304.07193](https://arxiv.org/abs/2304.07193)).
   Our 22 target + 45 distractor appearances per celeb sit firmly in this
   band.
3. **Imitation-learning scaling** - [Lin et al., ICLR 2025
   (arXiv:2410.18647)](https://arxiv.org/abs/2410.18647) shows
   per-(environment × object) imitation data plateaus around K = 50
   demonstrations, with the steep part of the power-law curve in K =
   10–30. M = 25 sits on the steep part.

### §5.2 Image-prompt efficiency multiplier

Path A (image-as-prompt) framing reduces per-celeb data requirements vs
text-prompt by roughly an order of magnitude
([VIMA, arXiv:2210.03094](https://arxiv.org/abs/2210.03094) reports
2.7 × better with 10 × less data on hardest generalisation;
[OWLv2 / OWL-ViT](https://owlvit.com/) zero-shot image-conditioned
detection works from a single visual exemplar). The model doesn't have
to encode 200 names; it has to learn a shared face-matching skill.
This is why 22 target appearances per celeb is sufficient where
text-prompt 200-class fine-grained classification would need 100+ per
celeb to even reach 70 % accuracy
([CascadeVLM, EMNLP 2024 fine-grained-classification benchmark](https://aclanthology.org/2024.findings-emnlp.102.pdf)).

---

## §6 · Photo bank composition

### §6.1 Consolidating `web/` into `scraped/`

Current state under `datasets/eval3_celebs/`:

```
heldout/                  ← 4–5 photos per IID celeb that were PRINTED at recording
  swift/   {5 jpgs}
  obama/   {4 jpgs}
  lecun/   {5 jpgs}
web/                      ← reference photos used by record_eval3*.py at teleop time
  swift/   {5 jpgs}
  obama/   {12 jpgs}
  lecun/   {20 jpgs}
scraped/                  ← the 200-celeb OOD bank (from scrape_headshots.py)
  <full_name>/  {0–10 jpgs each}
```

The 3 IID celebs already have folders in `scraped/` from the bank build
(`taylor_swift`, `barack_obama`, `yann_lecun`). To centralise:

- Move `web/swift/*` → `scraped/taylor_swift/web_*.jpg` (prefix avoids
  collision with the scraped photos already there).
- Same for obama → `barack_obama/`, lecun → `yann_lecun/`.
- `heldout/` stays separate - it is the "training-set physical photos"
  reference and must NOT be used for augmentation (eval hygiene; see
  §6.2).

### §6.2 Eval hygiene - what we exclude

`heldout/` contains the exact photos that were physically printed and
photographed during the 180 teleop recordings. If we re-used those
images during augmentation, the model would see them with two distinct
photo styles (printed-and-photographed vs. raw JPEG) and could learn a
style discriminator. We therefore **draw augmentation photos
exclusively from the `scraped/` (now-merged) pool**, never from
`heldout/`.

Mention in the implementation: `4_inpaint_video.py` accepts
`--photo-bank` - point it at `eval3_celebs/scraped/` for all
production augmentation runs.

---

## §7 · Pre-training probes (run BEFORE generating augmentations)

Two probes catch silent data failures. Both are < 30 min runs and
required by CLAUDE.md §7.

### §7.1 ArcFace inter-celeb similarity probe

Compute the 203 × 203 ArcFace cosine matrix on the bank using the
buffalo_l model. Flag any celeb pair with cosine > 0.4 (the InsightFace
"same-identity" boundary, [Sefik 2020](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/);
[Deng et al. 2019, ArcFace, arXiv:1801.07698](https://arxiv.org/abs/1801.07698)).
For any such pair, manually:

- Swap one out of the bank (if the celeb is non-essential), OR
- Add a sampling constraint guaranteeing they never co-appear in the
  same training triple (target + distractors).

Implementation: `eval_3/scripts/audit_bank_arcface.py` (new). Output:
`docs/report/eval3_arcface_audit.json` with the matrix + the flagged
pairs.

### §7.2 Per-celeb photo count audit

Count `*.jpg` files per `scraped/<celeb>/` folder. For every celeb with
< 5 photos:

- Try one final scrape pass with `scrape_headshots.py --names "<celeb>"
  --force --target-n 10`.
- If still < 5: either drop the celeb from the bank (preferred), or
  flag for manual photo curation.

This protects against the DreamBooth-floor risk identified in §3.2 and
§5. The 12 celebs currently below 5 photos (Tim Cook, John Jumper,
Karol Hausman, etc.) get this treatment.

Implementation: `eval_3/scripts/audit_bank_counts.py` (new). Or just a
small inline Python.

---

## §8 · Prompt mixture (75 / 15 / 10)

Every augmented variant draws its language prompt from one of three
buckets:

| Bucket | Fraction | Template family | Ground-truth action |
|---|---|---|---|
| Default name + action (paraphrased) | **75 %** | 5 paraphrases per name: <br>"Place the coke on {name}.", "Put the coke on {name}.", "Place the can on the photo of {name}.", "Set the coke down on {name}'s picture.", "Put the can on {name}'s photo." | Place on the target celeb |
| Reference-only (no name) | **15 %** | "Place the coke on the person in the reference image." / "Put the coke on whoever is in the reference photo." / "Place the can on the celebrity shown in the reference image." | Place on the target celeb |
| Counterfactual wrong-name | **10 %** | "Place the coke on {WRONG_name}." (where `WRONG_name` is a randomly chosen non-target celeb) | Place on the target celeb (the one in the **reference**, ignoring the wrong name in the prompt) |

### §8.1 Why this exact mixture

- **75 % default** preserves the standard supervised channel - the
  vast majority of training looks like canonical teleop.
- **15 % reference-only** forces the policy to actually attend to the
  reference image instead of taking the easier name-lookup path. This is
  necessary because, by [LIBERO-Plus and shortcut-learning analyses
  (arXiv:2510.13626, 2508.06426)](https://arxiv.org/abs/2510.13626),
  VLAs default to the easier modality when both are redundant. At eval,
  the OOD celebs have names the model has never seen - so the
  *only* reliable signal is the reference image. We must train for that
  case explicitly.
- **10 % counterfactual** is the [CAST recipe
  (arXiv:2508.13446)](https://arxiv.org/abs/2508.13446) which reports
  **+27 % instruction-following improvement** by injecting prompt /
  visual mismatches. We adapt it inverted: instead of teaching the model
  to follow the language *over* visual priors, we teach it to follow
  the *reference image* over a mis-named prompt. Same mechanism, opposite
  modality preference, because the reference is the reliable channel
  for our task at OOD eval.

The 5 % "wrong reference" bucket the prompt-strategy agent suggested
(prompt correct, reference unrelated) is **omitted in v1** - it asks
the model to fall back to a name → face lookup that we don't have
training signal for (because we never trained name→face except as a
redundant signal). Adding it would inject noise. Revisit only if the
matching probe (§9.5) plateaus.

### §8.2 Optional VQA co-training

The forgetting agent recommended skipping VQA co-training since
SmolVLA's VLM is frozen - VLM features can't be "blinded" if they're
not training. We follow that recommendation. Revisit only if we observe
unexpected matching plateaus.

---

## §9 · Production parameters (locked)

| Parameter | Value | Source |
|---|---|---|
| `M` (variants per base episode) | **25** | §5 |
| Base episodes | 180 | `record_eval3_guided.sh` |
| Total augmented videos | 4 500 | M × base |
| Bank size | ~203 celebs | §6 |
| Per celeb as target | ~22 | §5 |
| Per celeb as distractor | ~45 | §5 |
| Slot replacement | All 3 portraits | §2 |
| Reference photo rule | ≠ printed photo of same celeb | §3 |
| Prompt mix | 75 / 15 / 10 | §8 |
| SmolVLA VLM | Frozen | SmolVLA paper §3.2 |
| Pre-flight probes | ArcFace audit, photo-count audit | §7 |
| Training steps | 200 k @ batch 32 | SmolVLA paper recipe |
| Reference channel format | Constant-frame MP4 (option A from §4) | §4 |

### §9.1 First-run plan: 20 examples in debug mode

Before launching the full 4 500-variant run we generate **20 sample
variants** with the full debug bundle (compare frames, stage-2 panels,
refit traces) so the visual quality + slot assignment + prompt mix can
be eyeballed end-to-end. Stratified across the 9 layout cells
(3 celebs × 3 layouts) - 2 from each cell + 2 extras = 20.

For the production run after the user signs off on the 20-example
output, **debug mode is OFF** to save ~70 % of pipeline time and
~95 % of output-disk usage. Only the LeRobot data (`videos/`, `data/`,
`meta/`) gets written per variant.

---

## §10 · Cross-check appendix (CLAUDE.md §7)

Each load-bearing decision verified against ≥ 3 independent sources.

### §10.1 "All 3 portraits replaced, not target only"

| Source | Evidence |
|---|---|
| [LIBERO-Plus, arXiv:2510.13626](https://arxiv.org/abs/2510.13626) | "Current VLAs exhibit positional bias rather than genuine semantic understanding of objects" - the exact failure mode predicted by target-only |
| [Shortcut Learning in Generalist Robot Policies, arXiv:2508.06426](https://arxiv.org/abs/2508.06426) | "Increased diversity moves models from zero success to shortcut-free performance" |
| [RoCoDA, arXiv:2411.16959](https://arxiv.org/abs/2411.16959) | Causal-invariance framework - under Path A the action depends only on the match relation; all 3 slots are causally-irrelevant `s_I` |
| [COLOSSEUM, arXiv:2402.08191](https://arxiv.org/abs/2402.08191) | Distractor identity is a 30–50 % vulnerability axis for VLAs |
| Convergent - all four point the same way | |

### §10.2 "Reference photo ≠ printed photo"

| Source | Evidence |
|---|---|
| [FaceNet, arXiv:1503.03832](https://arxiv.org/abs/1503.03832) | Triplet anchor and positive must be DIFFERENT images of the same ID |
| [VGGFace2, arXiv:1710.08092](https://arxiv.org/abs/1710.08092) | High intra-class variance > per-image quality for OOD generalisation |
| [DreamBooth, arXiv:2208.12242](https://arxiv.org/abs/2208.12242) | 3–5 different photos per subject is the minimum for identity learning |
| [Open-Set Face Recognition, arXiv:1705.01567](https://arxiv.org/abs/1705.01567) | OOD identity protocol requires non-trivial pos-pair structure at train time |
| Convergent - all four point the same way | |

### §10.3 "M = 25"

| Source | Plateau / floor for our regime |
|---|---|
| [DreamBooth, arXiv:2208.12242](https://arxiv.org/abs/2208.12242) | Floor 3–5 for objects, 10–20 ideal for faces |
| [DINOv2, arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | k-NN saturation at 5–32 examples/class |
| [Lin et al. ICLR 2025, arXiv:2410.18647](https://arxiv.org/abs/2410.18647) | Per-pair plateau K = 50 ⇒ M = 25 sits on steep part of curve |
| [SmolVLA paper, arXiv:2506.01844](https://arxiv.org/abs/2506.01844) | "Comparable performance with 31 demonstrations" on new SO-101 task |
| Convergent - M = 25 in the steep-gain region for all four | |

### §10.4 "Prompt mix 75/15/10"

| Source | Evidence |
|---|---|
| [CAST, arXiv:2508.13446](https://arxiv.org/abs/2508.13446) | Counterfactual labels at 10 % +27 % instruction following |
| [π0.5 KI, arXiv:2505.23705](https://arxiv.org/html/2505.23705v1) | Modality-mismatch co-training prevents VLM collapse to dominant modality |
| [Mees et al., CALVIN paraphrase, arXiv:2204.06252](https://arxiv.org/pdf/2204.06252) | Paraphrase-augmented language is consistently beneficial |
| Linguistic-blindness (multiple, [arXiv:2602.17659](https://arxiv.org/html/2602.17659), [2603.06001](https://arxiv.org/html/2603.06001v1)) | VLAs default to dominant modality unless explicitly forced; we need the 15 % no-name stress to ground in reference |
| Convergent - 75/15/10 sits at the lower bound of all four sources' "needed counterfactual" range, conservatively | |

### §10.5 Convergent rejection of "VQA co-training"

| Source | Position |
|---|---|
| [SmolVLA paper](https://arxiv.org/abs/2506.01844) | VLM frozen by default; no mechanism for VLM forgetting |
| [VLM2VLA, arXiv:2509.22195](https://arxiv.org/html/2509.22195) | LoRA + frozen-backbone preserves ≥ 85 % of base VQA |
| [Princeton "Without Forgetting" blog, Apr 2026](https://blog.ai.princeton.edu/2026/04/23/from-vision-language-models-to-robot-control-without-forgetting/) | LoRA / frozen is preferred over VQA-cofine-tune when compute is tight |
| Convergent - VQA co-train unnecessary for frozen-VLM Eval 3 | |

---

## §11 · References (consolidated)

### Slot strategy / inpaint augmentation
- [Yu et al., ROSIE, RSS 2023, arXiv:2302.11550](https://arxiv.org/abs/2302.11550)
- [Chen et al., GenAug, RSS 2023, arXiv:2302.06671](https://arxiv.org/abs/2302.06671)
- [Mandi et al., CACTI, 2022, arXiv:2212.05711](https://arxiv.org/abs/2212.05711)
- [Sylvest et al., LIBERO-Plus, Oct 2025, arXiv:2510.13626](https://arxiv.org/abs/2510.13626)
- [NICE, Nov 2025, arXiv:2511.22777](https://arxiv.org/abs/2511.22777)
- [Doshi et al., RoCoDA, Nov 2024, arXiv:2411.16959](https://arxiv.org/abs/2411.16959)
- [Pumacay et al., COLOSSEUM, RSS 2024, arXiv:2402.08191](https://arxiv.org/abs/2402.08191)
- [Shortcut Learning in Generalist Robot Policies, Aug 2025, arXiv:2508.06426](https://arxiv.org/abs/2508.06426)
- [Geirhos et al., "Shortcut Learning in DNNs", 2020, arXiv:2004.07780](https://arxiv.org/abs/2004.07780)
- [Chen et al., Semantically Controllable Augmentations, IJRR 2025, arXiv:2409.00951](https://arxiv.org/abs/2409.00951)

### VLA architectures
- [Shukor et al., SmolVLA, Jun 2025, arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- [Driess et al., π0.5 + KI, May 2025, arXiv:2505.23705](https://arxiv.org/html/2505.23705v1)
- [Kim et al., OpenVLA, 2024, arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- [Kim et al., OpenVLA-OFT, 2025, arXiv:2502.19645](https://arxiv.org/abs/2502.19645)
- [Brohan et al., RT-2, 2023, arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- [Jiang et al., VIMA, ICML 2023, arXiv:2210.03094](https://arxiv.org/abs/2210.03094)
- [Interleave-VLA, May 2025, arXiv:2505.02152](https://arxiv.org/abs/2505.02152)
- [ObjectVLA, Feb 2025, arXiv:2502.19250](https://arxiv.org/html/2502.19250v1)
- [VLM2VLA, Sep 2025, arXiv:2509.22195](https://arxiv.org/html/2509.22195)

### Face recognition / identity learning
- [Schroff et al., FaceNet, CVPR 2015, arXiv:1503.03832](https://arxiv.org/abs/1503.03832)
- [Deng et al., ArcFace, CVPR 2019, arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
- [Cao et al., VGGFace2, 2017, arXiv:1710.08092](https://arxiv.org/abs/1710.08092)
- [Günther et al., Toward Open-Set Face Recognition, 2017, arXiv:1705.01567](https://arxiv.org/abs/1705.01567)
- [Vareto et al., Open-Set FR with Maximal Entropy, 2023, arXiv:2311.00400](https://arxiv.org/abs/2311.00400)
- [LFW protocol](https://vis-www.cs.umass.edu/lfw/results.html)
- [IJB-C protocol, Maze et al., ICB 2018](http://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf)

### Personalisation / few-shot identity
- [Ruiz et al., DreamBooth, CVPR 2023, arXiv:2208.12242](https://arxiv.org/abs/2208.12242)
- [Gal et al., Textual Inversion, ICLR 2023, arXiv:2208.01618](https://arxiv.org/abs/2208.01618)
- [Kumari et al., Custom Diffusion, CVPR 2023](https://www.cs.cmu.edu/~custom-diffusion/)

### Self-supervised vision foundations
- [Oquab et al., DINOv2, 2023, arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- [Zhai et al., SigLIP, 2023, arXiv:2303.15343](https://arxiv.org/abs/2303.15343)

### Data scaling laws / imitation learning
- [Lin et al., Data Scaling Laws in Imitation Learning, ICLR 2025 Oral, arXiv:2410.18647](https://arxiv.org/abs/2410.18647)
- [Mandlekar et al., What Matters in Learning from Offline Human Demonstrations, 2021, arXiv:2108.03298](https://arxiv.org/abs/2108.03298)
- [Jang et al., BC-Z, 2022, arXiv:2202.02005](https://arxiv.org/abs/2202.02005)

### Prompt / language design for VLA
- [CALVIN, Mees et al., 2021, arXiv:2112.03227](https://arxiv.org/abs/2112.03227)
- [What Matters in Language Conditioned RIL, Mees et al., 2022, arXiv:2204.06252](https://arxiv.org/pdf/2204.06252)
- [CAST, Counterfactual labels for VLAs, Aug 2025, arXiv:2508.13446](https://arxiv.org/abs/2508.13446)
- [Linguistic-blindness papers, 2025-26: arXiv:2602.17659, 2603.06001, 2604.05595](https://arxiv.org/html/2602.17659)

### Project-internal
- `eval_3/aug/STRATEGY.md` - superseded by this doc
- `eval_3/aug/RESEARCH_v2.md` - older research notes; inpaint primitives still current
- `eval_3/aug/VALIDATION.md` - triple-source audit of v1 numerical defaults; still applies to MTF blur σ, Reinhard, etc.
- `eval_3/scripts/scrape_headshots.py` - multi-source bank builder
- `eval_3/scripts/record_eval3_guided.sh` - 180-ep recording structure
- `docs/PROJECT.md` - Eval 3 task brief

---

*Doc locked 2026-05-14. Any change requires re-validating the four
convergent-source tables in §10.*
