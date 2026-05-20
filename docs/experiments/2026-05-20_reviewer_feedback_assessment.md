# Reviewer feedback assessment — does "co-train + prompt design" solve the celeb-routing failure? — 2026-05-20

A reviewer commented on the team's Change 1 / Change 2 (from
[`EVAL_3_FOR_TAS.html`](../report/EVAL_3_FOR_TAS.html) §6) and suggested
co-training "should mostly be the solution," with prompt design as the
remaining lever. This assesses that claim. Backed by two independent
research passes plus the team's own diagnosis.

## TL;DR

| Claim under review | Verdict |
|---|---|
| **Change 1** — co-train action loss + face-identity loss — "should mostly be the solution" | **Overstated.** Co-training is the right *anti-drift* tool, but the diagnosed failure is attention *routing*, not representation drift. Necessary, **not sufficient**. |
| **Prompt design** — inject a steering word ("celebrity") to elicit the behavior | **Cannot fix routing.** A constant word added to every prompt carries zero celeb-discriminative information. May marginally aid knowledge elicitation; not a solution. |
| **Change 2** — attention-supervision (KLAL) | **This is the load-bearing fix** — it directly supervises the diagnosed broken link. It *is* Track E, currently training. |

**The reviewer's emphasis is inverted.** Change 1 + prompt design is the
easy part and not the crux. Change 2 (KLAL routing supervision) is the
crux — and it is exactly what Track E runs. Following the reviewer's
suggestion as stated (lean on Change 1 + prompts) would likely reproduce
the S4 / Track D failure.

## The reviewer's feedback (verbatim)

> **Change 1 — Co-train, don't do it in stages.** […] The face loss
> continuously holds the identity ability in place while the policy learns
> to act — it cannot drift away, because it is still being graded. […]
> Should mostly be the solution to the problem. It then becomes a question
> on how to design the prompts for your VLM to elicit the behavior.
> Injecting the word *celebrity* might help. […] figure out […] which words
> in the prompt steer it towards recognizing the celebrities. You could then
> at eval time always inject this additional word into the prompt.
>
> **Change 2 — Supervise the attention routing directly.** […] we add an
> attention-supervision loss […] penalize the model when its name-token
> attention diverges from that target.

## 1. Change 1 — co-training a face-identity loss

**The reviewer's mechanism is correct for what it covers — and mis-scoped
for the actual failure.** Co-trained auxiliary losses are a well-evidenced
anti-forgetting tool: keeping a source objective active does prevent the
capability decay that a *staged* warm-start suffers. The team agrees — it
is exactly Refinement 2 of the attention-routing diagnosis.

But the diagnosed failure was **never representation drift.** It is that
the **name-token does not *attend to* the named face's image patch**.
Co-training a face-identity loss fixes a problem the team does not have.

This is not speculation — **the team already ran this experiment.** S4 /
Track D was M2 (ArcFace cosine distillation = a face-identity loss),
co-trained with the action loss. Result: it shaped the face
representations well — **mean cosine ≈ 0.88** — and the name→face attention
routing **still did not move** (step-10k and step-25k probes identical and
random). The team's own diagnosis states it plainly: *"representation-only
interventions like M2 didn't close the gap — they were never training the
thing that's broken."*

Recognition and grounding are **distinct capabilities**, and the former
does not confer the latter: an ArcFace loss supervises *"is this embedding
the right identity"* — a global, instance-level recognition objective on a
face crop. It carries **no spatial-selection signal** — nothing that says
*which of the three patches* the name-token should route to. Training
"recognize face X in isolation" does not train "find face X among three
faces"; the second is a multi-distractor referring task.

The decisive external number is **ObjectVLA's own ablation**
([arXiv:2502.19250](https://arxiv.org/abs/2502.19250); the team's
`EVAL_3_RESULTS.md` already cites it):

- robot data only: **8%** novel-object success
- vision-language co-training **without** bounding boxes: **19%**
- vision-language co-training **with** bbox grounding supervision: **64%**

The 19%→64% jump comes *entirely* from adding explicit spatial grounding to
the *same* co-training data. A co-trained recognition-style signal alone
moves the needle to the ~19% tier; explicit grounding supervision is what
works.

## 2. The steering-word / prompt-engineering idea

**Cannot fix the routing failure.** The argument is mechanical and airtight:

A steering word ("celebrity") added to *every* prompt is, by construction,
a **constant** — the same tokens, the same key/value vectors, in every
forward pass. Self-attention computes per-instance scores from query–key
dot products; a constant term shifts every score by the **same bias** and
therefore **cannot create a contrast between "Swift" and "Obama."** A
constant carries **zero bits** about which celebrity is named, so it cannot
route to a celebrity-specific face. The discriminative signal must come
from the input that *varies* — the name token — and the probe shows
precisely that token is sink-locked and prompt-invariant.

A steering word can change *how attention behaves in general* (e.g. nudge
it off the background sink toward faces-as-a-class); it **cannot make
attention depend on the name**. That is the whole task.

Two further problems:

- **PaliGemma's prefixes select a task *mode*, not a per-instance target.**
  PaliGemma was trained with strict task prefixes (`caption`, `detect`,
  `answer`, `segment`; [arXiv:2407.07726](https://arxiv.org/abs/2407.07726)).
  Prompt phrasing picks caption-vs-detect-vs-VQA; it does not pick *which
  object*. And in Pi0.5 the VLM is consumed as a feature extractor, further
  muting prompt-format effects.
- **A base-model "magic word" will not survive fine-tuning.** Determining a
  prompt trick by probing the *base* VLM, then injecting it at eval on the
  *fine-tuned* policy, is unsound — fine-tuning changes the weights, so it
  changes what any word does. And an eval-only word absent from training is
  textbook train/eval skew. If a cue word is used at all, it must be in
  **every training prompt**, not bolted on at eval.

**What is worth doing:** the base VLM's celebrity knowledge is thin (gating
probe G2: Obama recognized; Swift/LeCun not). A cue word cannot add
knowledge the WebLI prior never encoded. Still — one cheap, logged probe
varying prompt phrasing (does adding "celebrity" / "the person" shift the
name-token attention map at all?) is informative and low-cost. Treat it as
a diagnostic that could inform *training*-prompt design — not as a fix.

## 3. Change 2 — KLAL is the actual fix

The diagnosed failure is attention routing; the intervention that targets
it head-on is **explicit routing/grounding supervision** — i.e. Change 2,
KLAL. It directly penalizes the name-token→image attention for diverging
from a bbox-derived target on the named face. This is supported by
ObjectVLA's 19%→64% grounding jump and is the only mechanism on the table
that supervises the broken link rather than an adjacent quantity.

This is **not a new thing to build** — it is **Track E** (Pi0.5 + M2 +
KLAL), co-trained in one run, currently training on Brev. The decisive
test is the **step-~10k attention probe** (now RoPE-fixed, commit
`1c66387`). See [`2026-05-20_track_E_method_validation.md`](2026-05-20_track_E_method_validation.md).

Track E is not guaranteed to succeed — KLAL's gains on real grounding
benchmarks are modest, and the positional-shortcut risk is real (mitigated
here by the perfectly balanced layout dataset). But it is the *correct*
intervention for the diagnosis, and "would a co-trained identity loss +
prompt tweaks solve it instead" is answered by the evidence: no.

## 4. Recommendations

1. **Do not pivot to "Change 1 + prompt design."** It is the S4 / Track D
   recipe (which failed on routing) plus a prompt change that cannot
   discriminate instances. Keep the full stack — co-train **+ KLAL** —
   which is Track E, already running.
2. **Co-train: yes, keep it.** The reviewer is right that co-training beats
   a staged warm-start. Track E already co-trains action + M2 + KLAL in one
   pass.
3. **Steering word: one cheap probe, then decide.** Vary prompt phrasing in
   the RoPE-fixed attention probe; if a cue word measurably moves attention
   off the sink, fold it into the *training* prompts (never eval-only).
   Budget it as a 1-hour diagnostic, not a deliverable.
4. **The go/no-go remains the step-~10k Track E probe.** That is what tells
   us whether routing supervision actually binds names to faces.

## Sources

- Verified / corroborated: PaliGemma ([arXiv:2407.07726](https://arxiv.org/abs/2407.07726));
  ObjectVLA 8/19/64 ablation ([arXiv:2502.19250](https://arxiv.org/abs/2502.19250),
  also cited in the team's `EVAL_3_RESULTS.md`); KLAL ([arXiv:2511.12738](https://arxiv.org/abs/2511.12738)).
- The two research passes additionally cited recent VLA-grounding work
  (referring-grounding vs recognition, attention-supervised VLAs); those are
  consistent with the conclusion but were not independently re-verified
  here — the verdict rests on the mechanical argument (§2), the ObjectVLA
  ablation, and the team's own S4 / Track D probe evidence.

*Assessment: 2 parallel research agents + reconciliation against the team's
own attention-routing diagnosis, 2026-05-20.*
