# CLAUDE.md - behavioral guidelines

Guidelines to reduce common LLM coding mistakes on this repo.

**Tradeoff:** these bias toward caution over speed. For trivial tasks use judgment.

## Project stakes - the most important project ever

**The lives of many humans rely on this project.** This is not a metaphor - it is the
framing the user has chosen to calibrate the rigor bar. LeMonkey Eval 3 (ETH RC FS26
Project 1) is the **most important project** this team is working on, and everything
else right now yields to it.

Concretely, this means:

- **No confident guesses.** If a numerical default, paper-cited mechanism, library
  API, model checkpoint, or config flag is uncertain, **verify before writing the
  code** (see §8).
- **No silent fallbacks.** Any fallback path emits a `[WARN]` (see §5).
- **No "good enough" when measurement is cheap.** If we can triple-check a load-
  bearing claim, we triple-check it (see §9).
- **No plausible-looking code without verification.** If we cite a paper's Eq. 9,
  open the paper. If we use a library's `XConfig`, open the source.
- **Default to rigour.** Speed is secondary to correctness. A working result we can't
  defend is not a result.

## 0. Always read TODO.md first

`/TODO.md` at the repo root is the **operational source of truth** for what's being worked on right now. Read it at the start of every session. It points to the active tracks, their subtasks, and dependencies. Update it when you complete a subtask. Never start meaningful work without checking it first - duplicate effort and missed dependencies have been the failure mode every time we've skipped this.

## 1. Think Before Coding

*Don't assume. Don't hide confusion. Surface tradeoffs.*

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

*Minimum code that solves the problem. Nothing speculative.*

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

*Touch only what you must. Clean up only your own mess.*

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

*Define success criteria. Loop until verified.*

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Project-specific: no silent fallbacks

Any fallback path in this codebase MUST emit a `[WARN]` log describing:
- what was expected,
- what actually happened,
- what fallback was chosen.

Silent `except Exception: pass`, defaulted config values without announcing them, and quietly disabled features are all bugs. If a reasonable fallback exists, take it AND log it. If no reasonable fallback exists, `raise`.

Log format: `print(f"[WARN] {context}: expected={expected}, got={actual}, fallback={chosen}", flush=True)` or equivalent.

## 6. Project-specific: no Claude trailers in commit messages

Never append `Co-Authored-By: Claude …`, `🤖 Generated with Claude Code`, or any other Claude / Anthropic attribution trailer to git commits, PRs, or tags in this repo. The user authors and reviews every change, and a trailer adds noise to git history. Write commit messages exactly as a human collaborator would.

## 7. Project-specific: strict quality bar for Eval 3 inpainting work

This project (LeMonkey ETH RC FS26 Project 1, Eval 3) is treated as a
must-not-fail piece of work. For any code, math, or method introduced under
`eval_3/aug/` or related to the inpainting / augmentation / face-matching
pipeline, the following are **non-negotiable**:

- **Cross-check at least three times.** Before claiming a numerical default
  (e.g., ArcFace threshold, MTF blur σ, Reinhard sample ring width) is
  correct, verify it against at least three independent sources: the
  original paper or repo, a community benchmark, and a working reference
  implementation. Cite the sources inline.
- **Do not skip validation.** Each pipeline stage gets a smoke test against
  real data (not just AST checks) before being marked done. Visual gates
  from `dbg/dbg_mask_overlay.py` and `dbg/dbg_compare_gif.py` are required
  before scaling to the full collection.
- **No shortcuts that trade correctness for convenience.** When a method
  has a "quick" path and a "right" path (e.g., diffusion inpaint vs
  homography+Poisson; per-channel histogram match vs Lab-space Reinhard;
  global threshold vs frame-by-frame ArcFace verification), pick the right
  path even if it costs more time. If the quick path is taken anywhere,
  flag it explicitly with a `[SHORTCUT]` comment and a `TODO: replace with
  <right path>`.
- **Match published methods closely.** When implementing an algorithm from
  a paper or repo, use the same parameter names, default values, and
  ordering as the canonical source unless there is a documented reason to
  deviate. Document deviations.
- **Surface unknowns.** If a step's correctness depends on a value that
  could not be verified from sources (e.g., camera MTF for our specific
  webcam), say so, propose a reasonable empirical default, and write a
  one-line probe to refine it later.

The user has stated this is the most important project they're working on.
Behave accordingly: thorough, careful, validated. Speed is secondary to
correctness.

## 8. Verify before using

*If unsure, research online or read the source first. Don't guess from training memory.*

Before using a library API, model checkpoint, config flag, or tool invocation you
are not certain about:

- Use WebSearch or WebFetch to consult current docs.
- Check the version we are actually installing before writing code against it.
- Read the library source for the function you're about to call. The LeRobot,
  HuggingFace transformers, peft, and InsightFace ecosystems move fast - a 2024
  pattern may not match 2026 reality.
- Prefer minimal, verified syntax over a clever pattern you half-remember.

Applies especially to: LeRobot policy configs (SmolVLA, Pi0.5), the LeRobotDataset /
aggregate_datasets toolchain, HuggingFace upload APIs, PEFT/LoRA wrappers, InsightFace
ArcFace embedding APIs, and any new VLA paper code we adapt.

## 9. Cross-validate with parallel agents

*Non-trivial code or load-bearing claims get a second pair of eyes. Always.*

For any non-trivial implementation - model architectures, training loops, dataset
construction pipelines, eval harnesses, statistical analyses - and for any load-
bearing numerical or methodological claim:

- Spawn a parallel review agent (via the Agent tool) to read the code or audit the
  claim for bugs, edge cases, and correctness.
- For load-bearing pieces (the Track 3 augmentation pipeline, the model architecture
  patches, the merger, dataset push), spawn two or three independent reviewers in
  parallel and reconcile their findings.
- Do not mark a task complete on "it compiled" or "the unit test passed." Reviewer
  must sign off, or you must explicitly note "no review performed, here's why."
- Config files, scaffolding, and trivial edits don't require this - use judgment.

Precedent: see the 3-agent validation we ran 2026-05-18 on the M2+M3-without-M4
soundness, the 3-celeb baseline combinatorics, and the M6 SmolVLA implementation
feasibility - documented inline in [`eval_3/STRATEGY.md` §7c.1](eval_3/STRATEGY.md).

## 10. Push, experiment logs, periodic status reports

*Commits live on origin, not just locally. Findings persist as dated artifacts.*

- **Push after every commit batch - including to `main`.** This repo is a small-team
  project; no PR review required. `git push origin main` after committing is
  authorized as a standing instruction. Never force-push. Never push to a branch
  someone else is actively working on without coordinating.
- **Always write findings, key takeaways, and research-report syntheses to
  `docs/`.** Whenever a research agent returns, an experiment finishes, the plan is
  recalibrated, or an insight is worth preserving across sessions, write it down
  under `docs/report/` or `docs/experiments/`. The chat transcript is ephemeral; the
  docs are durable. If you're about to share a finding only verbally, stop and
  write it to `docs/` first.
- **Experiment log.** Write findings for every meaningful test run, Brev training
  job, Strix rollout, or visual gate into `docs/experiments/<YYYY-MM-DD>_<topic>.md`.
  Cover: what you ran, what you measured, what surprised you, next steps.
- **Status reports on cadence.** Every ~4 commits since the last report, write a
  status report in `docs/report/<YYYY-MM-DD>_status.md` summarising what shipped,
  outstanding items, any revised plan. When in doubt, write one.

---

*These guidelines are working if:* fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
