# 2026-05-21 — SmolVLA co-training import + 6-agent review

## What was done

Imported Darius's SmolVLA co-training implementation from
`feat/cotrain-smolvla-darius` onto `main` (commit `4699d47`) as a selective
byte-identical copy — `eval_3/scripts/smolvla_cotrain/` (5 files) +
`docs/experiments/2026-05-20_cotrain-{ddp,smoke-fix}.md`. Then reviewed with
6 parallel/sequential agents (2 independent `cotrain.py` reviewers, 1 shell-script
reviewer, 1 git-audit, 1 mahbod-branch investigator, 1 round-3 verification).

## Import audit — CLEAN (3/3 checks pass)

- **Completeness**: the 7 imported files are Darius's complete additive
  contribution across all 18 commits. Nothing missed.
- **No shared-code changes**: purely additive — no commit modifies any file
  that exists on `main`. (The branch's `init` commit deletes old docs
  README/TODO/STRATEGY/tracks; those deletions were intentionally not imported.)
- **Byte-identity**: all 7 files byte-identical to the branch.

## cotrain.py review — 2 confirmed blockers

Verified by runtime execution against the repo's pinned `lerobot 0.5.1` /
`transformers 5.3.0`.

| # | finding | verdict | severity |
|---|---|---|---|
| C1 | `set_requires_grad()` freezes `lm_head` → VQA objective crippled | **REFUTED** | none — freeze substrings are stale for transformers 5.3.0 so nothing freezes; `lm_head` is tied to trainable `embed_tokens` and trains fine. All 16 layers train. |
| C2 | VQA CE loss runs through a 16-of-32-layer truncated VLM trunk | **known limitation** | not a blocker — it's SmolVLA's own base config; loss runs and decreases. Document it. |
| C3 | `_episodes_with_complete_files` uses `meta.info.data_path` | **CONFIRMED — BLOCKER** | `meta.info` is a plain dict; `.data_path` is `AttributeError`. Crashes during dataset setup, **before step 0**. |
| C4 | VL collator `truncation=True, max_length=1024` cuts image tokens | **CONFIRMED — BLOCKER** | square / 224px images expand to 1088 `<image>` tokens > 1024 → processor raises `ValueError` in the collator on most VL batches. |

### The recorded "200-step smoke pass" is not valid evidence

The commit that introduced the C3 bug (`ec6bf96`) itself claims a passing
smoke; the "full-dataset 200-step pass" was recorded in a **docs-only commit**
(`10a33b5`, no code change). Against the pinned lerobot 0.5.1, the C3 line is an
unconditional `AttributeError`. The smoke was run against a different library
state (the cotrain docstring says "lerobot v0.5.2"; the repo pins/installs
0.5.1) or pre-bug code. **Re-run the smoke on the pinned env after fixing.**

### Fixes (small, runtime-verified)

- **C3** — `cotrain.py:448,452`: `meta.info.data_path` → `meta.data_path`,
  `meta.info.video_path` → `meta.video_path`.
- **C4** — `make_vl_collator` (`cotrain.py` ~311-330): `truncation=False` in
  both `processor(...)` calls (the collator already pads `padding="longest"`;
  image tokens must never be truncated).

## Shell scripts (launch.sh / predl_vl.sh / setup_env.sh) — no blockers

All 18 `launch.sh`→`cotrain.py` flags match. Issues found: env-activation
inconsistency (`launch.sh` doesn't source conda while the other two do; a
wrong-env run silently degrades to 1 GPU), stale README file-layout, a
`set -e`/`ls` pipeline edge case. Worth fixing before a 24h run; none block a
smoke.

## Bottom line

The import is clean and on `main`. `cotrain.py` **cannot reach step 0** until
C3 + C4 are fixed (both 1-line fixes, runtime-verified). C1 over-stated by the
first-round reviewers; C2 is a documented limitation.
