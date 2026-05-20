# Track E method validation — will M2 + KLAL make Pi0.5 bind celeb names to faces? — 2026-05-20

A deep cross-check of the Track E method (Pi0.5 + M2 ArcFace distillation +
KLAL attention supervision) before committing a multi-day GPU run. Combines
an online literature review, three independent code/data audits, and a fix
for the one real bug found.

## TL;DR — the method is sound; one HIGH-severity bug was found and fixed

| Question | Verdict |
|---|---|
| Is the KLAL paper real and is our loss faithful to it? | **Yes** — arXiv:2511.12738, WACV 2026. KL direction `KL(P_target‖Q_model)` matches. Two documented deviations (target shape, layers supervised). |
| Can attention supervision teach *identity*-conditioned attention, not a positional shortcut? | **Yes here** — the dataset kills the positional shortcut by construction (all 6 layouts balanced 1:1). This is the single most important finding. |
| Do the training prompts contain the celeb names KLAL needs? | **Yes** — all 9,394 episodes, exact casing. |
| Is the attention KLAL trains the attention the policy actually uses? | **It was NOT** — the loss recomputed attention *without RoPE*. **HIGH-severity bug. Fixed this session.** |
| Are the prefix offsets / camera / name-token indexing correct? | **Yes** — all verified against `modeling_pi05.py`. |
| Will 25k steps actually produce name→face attention? | **Unknowable in advance.** The losses are correct; the *outcome* is empirical — gated by a step-~10k checkpoint probe. |

**Bottom line:** Track E is theoretically sound, the dataset is close to a
best-case setup for it, and the implementation is now correct. The run is
worth launching — but it must be gated at step ~10k by an attention probe
(now also RoPE-fixed). Had the RoPE bug not been caught, the run would have
trained and "passed its own probe" on a proxy decoupled from the deployed
policy.

## What Track E is

Goal: a Pi0.5 policy that places a coke can on the printout of a **named**
celebrity, discriminating by face — not by position or color.

Two stacked auxiliary losses on the PaliGemma VLM inside Pi0.5:

- **M2** (ArcFace cosine distillation, BlindVLA Eq. 9): mean-pools the VLM
  hidden states under each face bbox, projects them through a frozen MLP,
  and pulls them toward that celeb's ArcFace identity centroid. This makes
  the *face-patch hidden states identity-distinct*.
- **KLAL** (KL Attention Loss, WACV 2026): supervises the attention from the
  language **name-token** to the **image patches** against a target peaked
  on the prompted celeb's face bbox. This trains the *name-token query to
  select the identity-matching patch*.

The two are co-dependent and that is the design: M2 puts identity *into* the
patch keys; KLAL trains the name query to *route to* it. Neither alone
suffices (M2-only was confirmed dead — see
[`2026-05-20_vlm_face_detection_probes`](2026-05-20_vlm_face_detection_probes/README.md)).

## 1. Literature validation — the KLAL paper is real, our loss is faithful

Verified by fetching the source:

- **arXiv:2511.12738** — *"Direct Visual Grounding by Directing Attention of
  Visual Tokens"*, Esmaeilkhani & Latecki (Temple University). Accepted at
  **WACV 2026** (confirmed via the CVF Open Access proceedings, not just the
  arXiv page). "KLAL" / "KL attention loss" is the paper's own term.
- **Loss form** — `L_KLAL = (1/L) Σ_l D_KL(P(S) ‖ Q^(l)(S))`: **forward KL**,
  `P` = bbox target, `Q` = model attention, head-averaged, summed over
  layers. Our `klal_loss` computes `Σ P·(logP − logQ)` = `KL(P‖Q)` —
  **direction matches the paper.** ✓

Two **deviations from the paper**, both now documented in `m2_klal.py`:

1. **Target shape.** The paper builds `P_target` from the bbox's *center
   line* of patches (tuned for elongated RefCOCO objects). We use a 2-D
   isotropic Gaussian on the bbox centroid (`gaussian_target_from_mask`,
   σ = 1.5 patches). For a *compact face* bbox this is appropriate — arguably
   better than a line. The paper publishes no σ; 1.5 patches is an empirical
   default (a face spans ~2–4 patches on the 16×16 grid).
2. **Layers supervised.** The paper averages KL over **all** LM layers; we
   supervise **`{10, 14, 17}`** only. This is forced by the partial-freeze
   (layers 0–9 are frozen for anti-forgetting; supervising a frozen layer
   trains nothing). A real reduction in supervision strength — accepted as
   the cost of protecting the PaliGemma prior.

**Calibration from the paper's own numbers:** attention supervision gives
*large* gains on pointing-type tasks (Grid-Patch pointing 28.6 % → 44.9 %)
but only *modest* gains on real referring-expression grounding (RefCOCO
≈ +0.7 pp). Our task — "which of 3 printouts" — is pointing-type, the
favorable regime. The paper also warns that **too-large λ degrades next-token
fluency**; we run λ_KLAL = 1.0, which the smoke test (below) showed does not
destabilize training.

## 2. The positional-shortcut risk — and why our dataset eliminates it

This is the deepest concern, and the literature is blunt about it: **no
attention-supervision paper, KLAL included, demonstrates that the technique
teaches a model to attend to the right object *by identity*** (object A vs
object B at a different location). A bbox-derived target is *structurally a
positional signal* — the loss `KL(P‖Q)` can be driven to zero by a model
that simply learns "name-token attention → screen coordinate (x, y)", with
no obligation to bind to *which* face lives there. If that happens, the
policy fails the moment a celeb appears in a new position at eval.

**The positional shortcut is broken if and only if a given celeb appears in
varied positions across training, so "name → fixed position" is not a
consistent mapping.** We audited the dataset
(`HBOrtiz/so101_eval3_track3_v3_baseline`) directly — tallied all 9,216
`augmentation.json` files:

```
Layout permutation count:  OLS OSL LOS LSO SOL SLO  = 1536 each (perfectly uniform)
Target-celeb slot:  obama L/M/R = 1024/1024/1024
                    lecun L/M/R = 1024/1024/1024
                    swift L/M/R = 1024/1024/1024
```

Every celeb is the prompted target at the left, middle, and right slot in
**exactly equal proportion**. The "name → fixed position" mapping does not
exist in this dataset. To minimize KLAL across the whole training set the
model is *forced* toward identity-conditioned attention — there is no
positional shortcut to take. Combined with the discrete 3-slot structure
(the model selects one of 3 slots, not a free coordinate) and M2 injecting
identity into the patch keys, this is close to a best-case setup for making
attention supervision learn identity.

This does **not** make success certain — it removes the *one failure mode the
literature most strongly warns about*. The remaining question (does training
actually converge to it) is empirical.

## 3. The RoPE bug — found, and fixed this session

**The bug.** `m2_klal.py` recomputed attention from hooked `q_proj`/`k_proj`
outputs as `softmax(QK^T·scale)` with **no RoPE applied**. The real Pi0.5
forward (`modeling_pi05.py:compute_layer_complete`, lines 257-260) applies
RoPE to query/key states *before* attention, on every layer. RoPE inserts a
per-(query, key) relative-position rotation `R(p_k − p_q)`; for a name-token
at prefix position ~1290 attending image patches at positions 0–255 the
relative offsets span ~ −1290 … −1035 — a large, patch-dependent rotation
that **re-ranks which patch the attention favors**.

Consequence: KLAL was shaping `q_proj`/`k_proj` to make the *no-RoPE* (content
-only) attention peak on the face, while the policy — and the action expert
reading the VLM — sees the *RoPE'd* attention. **We were optimizing a proxy
decoupled from the quantity that matters.** The code even carried a comment
claiming "the KLAL paper does the same" — the paper says no such thing (it
supervises the model's real attention, which for any RoPE model is RoPE'd).
The comment was a fabricated justification; it has been removed.

The probe (`attention_map_probe_pi05.py`) had the **same** no-RoPE recompute.
So every prior Pi0.5 attention verdict ("warm VLM sink-locked", etc.) measured
*proxy* attention. The qualitative *prompt-invariance* conclusions still hold
(RoPE depends only on position, not on the celeb name, so a prompt-invariant
no-RoPE map implies a prompt-invariant RoPE'd map) — but the exact argmax
coordinates in those reports are proxy artifacts, and a future "the attention
localizes the face" verdict on the *un*fixed probe would not have proven the
real attention localizes.

**The fix (this session).** Both the loss and the probe now capture the
model's *own* `(cos, sin)` via a forward hook on `text_model.rotary_emb` —
the exact rotary embedding the model applies — and call the model's own
`apply_rotary_pos_emb` in the recompute, before the GQA key expansion,
matching `compute_layer_complete` exactly. The softmax scale is now read from
`self_attn.scaling` rather than hard-coded. No position-id reconstruction is
needed: the captured `cos/sin` already encode the true positions, so the
recompute is faithful by construction (the Pi0.5 prefix is fully
bidirectional, so for prefix→prefix attention there is no mask term either —
the recompute now equals the real prefix attention up to numerics).

Files changed: `eval_3/aug/m2_klal.py`, `eval_3/scripts/attention_map_probe_pi05.py`.
Two independent review agents (CLAUDE.md §9) signed off: RoPE order, cos/sin
source, softmax scaling, GQA expansion, and gradient flow to `q_proj.weight`
all match the real forward. Two non-blocking hardening items from the review
were applied — an explicit `assert` on the prefix-first cos/sin slice, and a
comment recording that the recompute omits the (padding-only) attention mask,
a second-order deviation that re-normalising over the unpadded 256 image
columns divides out.

## 4. Implementation audit — everything else checks out

Three independent audits (data, RoPE, offsets) plus the prior M2-fix audit:

| Checked | Verdict |
|---|---|
| Prompts contain literal celeb names ("Taylor Swift" etc., exact casing) | ✓ all 9,394 episodes; 15 templates |
| `lang_offset` (image tokens before language tokens) | ✓ = 5 streams × 256 = **1280** (camera1 + reference + 3 empty_cameras). Computed dynamically from `config.image_features` — correct; only a stale comment said "4×256", now fixed |
| camera1 is image stream 0 | ✓ — M2 and KLAL both supervise the real wrist camera, not an empty pad |
| `q_proj`/`k_proj` hook captures prefix-only rows | ✓ — PaliGemma LM runs the prefix; the action expert is a separate module |
| name-token subsequence match (leading-space BPE variant) | ✓ — `[WARN] no name_token_positions` never fired in the step-1000 run |
| KLAL KL direction `KL(P‖Q)` | ✓ matches WACV 2026 |
| M2 gradient flow (the `.detach()` bug) | ✓ fixed + verified earlier — see [`2026-05-20_m2_fix_track_E_verification.md`](2026-05-20_m2_fix_track_E_verification.md) |
| KLAL supervised a frozen layer (old layer 6) | ✓ already dropped → `KLAL_LAYERS=10,14,17` |

## 5. Remaining risks (none fatal, all monitorable)

- **Attention sink.** SigLIP produces high-norm artifact patches; Gemma has a
  `<bos>` sink. Sinks are documented to survive ordinary fine-tuning. KLAL's
  `KL(P‖Q)` *does* penalize attention mass off-target (including on the sink)
  so KLAL is the right tool — but it has to fight a robust prior. Monitor:
  the step-10k probe should show mass *leaving* the sink patches.
- **3 of 18 layers supervised** (vs the paper's all-layers). Weaker signal;
  accepted cost of the anti-forgetting freeze.
- **M2 strength.** The smoke test reached `mean_cos` 0.56 at step 800 and was
  still rising; SmolVLA's M2 reached ~0.88. If M2 plateaus low, the patch
  keys are less identity-distinct and KLAL has less to route to. Monitor
  `mean_cos`.
- **Proxy ≠ task success.** KLAL optimizes attention, a *proxy* for "go to the
  right printout". Good attention is necessary, not provably sufficient — the
  action expert must use it. The ultimate test is the Strix rollout, not the
  probe.

## 6. Verdict and the decisive test

**The method is sound and the implementation is now correct.** The literature
backs the technique, the KL math is faithful, and — critically — the dataset
removes the positional-shortcut failure mode that the literature most warns
about. The one real bug (no-RoPE proxy) is fixed. Launching the run is
justified.

**What cannot be promised:** that 25k steps *converges* to face-localizing
attention. That is the empirical outcome of the run; no static analysis
settles it. The honest framing: the losses are correct and the setup is
favorable — the rest is training.

**The decisive go/no-go test:** probe a Track E checkpoint at **step ~10k**
with the now-RoPE-fixed `attention_map_probe_pi05.py`, on one scene with the
3 celeb prompts. Pass = the name-token attention argmax lands on the prompted
celeb's face **and moves to the correct face when the celeb name changes**
(that *is* the identity-vs-position test). Fail = still prompt-invariant or
sink-locked → kill the run. This was always the plan; the only change is that
the probe now measures the policy's real attention.
