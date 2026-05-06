# eval_2 — SO-101 SmolVLA, Compositional Instruction Following

Runtime artifacts and scripts for **Eval 2** (50 pts): the policy must
decide which bowl is the target by *reasoning over a compositional prompt*,
not by mapping color words to fixed positions.

Per `docs/PROJECT.md` §2:
- The banana stays at the **same position as Eval 1** between Eval 1 and Eval 2.
- The 3 colored bowls get reshuffled across the 3 positions.
- Prompts are compositional. Examples from the brief:
  - `"Put the banana into the 2nd bowl from the left."`
  - `"Put the banana into the bowl on the right of the red bowl."`
  - `"Put the banana into the bowl that is not green and not blue."`
- 20 s per rollout.

## What's on Hugging Face Hub

| Repo | Type | Contents |
|---|---|---|
| [`HBOrtiz/smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2) | model | **Deployed Eval 2 policy** — 450M params, 25k steps from-scratch from `lerobot/smolvla_base`, image augmentation enabled. Final 25k checkpoint at the repo root for `from_pretrained()`; intermediates under `checkpoints/{005000,010000,015000,020000,025000}/`. |
| [`HBOrtiz/so101_eval2_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval2_all) | dataset | 180 teleop episodes, 107,820 frames, 123 distinct compositional prompts, balanced over 6 bowl arrangements × 6 prompt families |

The Eval 2 model is kept in its own repo (separate from `HBOrtiz/smolvla_eval1*`)
so the two tasks' artifacts never mix.

## Layout

```
eval_2/
├── README.md                ← this file
├── scripts/
│   ├── record_eval2.py        teleop recorder driven by a balanced 180-ep plan
│   ├── merge_eval2_episodes.py merges 180 ep dirs → one LeRobot v3 dataset
│   └── brev/                  scripts to scp to the Brev VM for training
│       ├── run_training.sh
│       ├── start_training.sh
│       ├── follow_training.sh
│       └── training_status.sh
├── state/                   plan.json — persistent recording state (gitignored)
├── train/                   model checkpoints (gitignored)
├── rollouts/                per-rollout dataset dumps (gitignored)
└── evals/                   per-session eval CSVs (gitignored)
```

Datasets live under `~/LeMonkey/datasets/eval2/` (one dir per episode).

## The 180-episode balanced plan

`record_eval2.py` generates and persists a **fixed plan** of exactly 180
episodes (`eval_2/state/plan.json`). The plan is **balanced** along three
independent axes:

| Axis | Levels | Per-level count |
|---|---|---|
| Arrangement | `BRG / BGR / RBG / RGB / GBR / GRB` | exactly 30 each |
| Family | `direct / spatial_absolute / spatial_ordinal / relational_lr / relational_between / negation` | exactly 30 each |
| Per (arr × family) cell | every cell has 5 episodes | 6×6×5 = 180 |

Within each arrangement batch the 30 episodes are family-shuffled, and the
6 batches themselves are presented in a randomized order — so **arrangement
and family-shuffling are decoupled** and neither is biased.

Target colors fall out at roughly 60 ± 10 each (verified empirically — the
slight green-skew comes from `relational_between` always targeting the middle
bowl, which is acceptable; PROJECT.md doesn't require perfect color balance).

The plan minimizes physical reshuffling: **only 5 reshuffles across the entire
180-episode collection**. You record ~30 episodes, then the script tells you
to set the bowls to a new arrangement.

## Why this design

The probe scripts (`eval_1/scripts/probe_*.py`) showed v2/25k:
- conditions on language strongly (`wrong_color ≈ 57`)
- but is overfit to the 13 verbatim Eval 1 phrasings (`paraphrase ≈ 61`)
- has **no compositional signal** (`spatial_*`, `relational`, `negation` all
  5–11 pairwise distance — 5–10× weaker than the wrong_color baseline)

So Eval 2 cannot reuse v2 as-is. Strategy: **fine-tune from v2/25k on this
new 180-episode compositional dataset** with broad phrasing diversity per
concept (~120+ distinct prompt strings).

## Prompt families and phrasing pools

| Family | # phrasings | Example |
|---|---|---|
| `direct` (Eval 1 carryover) | 8 × 3 colors | "Drop the banana in the red bowl" |
| `spatial_absolute` | 8 × 3 positions | "Put the banana in the bowl furthest to the left" |
| `spatial_ordinal` | 5 × 3 ordinals × 2 ref-sides × 2 ord-words | "Put the banana into the third bowl from the right." |
| `relational_lr` | 6 × 2 sides × 3 ref-colors | "Put the banana in the bowl directly to the right of the red bowl." |
| `relational_between` | 5 (only fires when target = middle bowl) | "Put the banana in the bowl that sits between the blue and green bowls." |
| `negation` | 7 × 2 (color1, color2) orderings | "Put the banana in the bowl that isn't blue or green." |

## Recording flow

```bash
./eval_2/scripts/record_eval2.py
```

The script reads the plan from `eval_2/state/plan.json` (or creates one on
first run). Each step:

1. If the next pending episode has a different arrangement than the last
   one, you'll see a big banner: `ARRANGEMENT CHANGE → please set bowls to: GRB`.
   Reshuffle the bowls to match the requested arrangement.
2. The script shows the prompt + which bowl is the target color/position.
3. On ENTER, runs `lerobot-record` (teleop mode, 20 s) labelled with the
   prompt as `--dataset.single_task`.
4. State auto-saves to `plan.json` so progress survives restarts.

### Controls

| Key | Action |
|---|---|
| **ENTER** | record this episode |
| **`d`** | delete the last recorded episode AND re-queue its prompt for the next ENTER (use this when teleop went badly and you want to redo) |
| **`p`** | print progress (recorded vs pending, broken down by arrangement / family / target color) |
| **`q`** | quit (state saved automatically) |

### Resuming, restarting, or deleting the plan

- Default: re-running `record_eval2.py` resumes where you left off.
- `--regenerate-plan` discards the existing plan and starts over (asks for
  confirmation; recorded data on disk is not deleted, only progress state).
- `--seed N` controls the plan generator (only relevant on first run or
  with `--regenerate-plan`). Default: 42.

### Dry-running

```bash
./eval_2/scripts/record_eval2.py --dry-run --plan-path /tmp/eval2_test/plan.json
```

Walks through the plan without touching the robot. Useful for sanity-checking
the prompt sequence before you start recording for real.

## Recording session tip

Aim for ~30 episodes per session = one arrangement batch. That's roughly
30 × (20 s record + ~10 s reset + brief positioning) ≈ 25 min. Press `p`
mid-session to confirm the family/color counters are progressing as expected.

## Training pipeline

The recipe mirrors Eval 1's v2 successful run, but trains
**from `lerobot/smolvla_base` from-scratch** (not warm-started from Eval 1) —
the Eval 1 base carries position→color and phrasing-overfit biases that this
training is specifically trying to avoid. `docs/PROJECT.md` §3 explicitly
allows different models per eval.

**Status: completed.** Trained on a Brev RTX Pro 6000, 25,000 steps in 9h42m,
zero `[WARN]` / errors. All five intermediates and the final 25k checkpoint
are uploaded to [`HBOrtiz/smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2).
The notes below are kept for reproducibility / re-training.

### Local prep (run before pushing to Brev)

```bash
# 1. Merge the 180 per-episode dirs into one LeRobot v3 dataset
./eval_2/scripts/merge_eval2_episodes.py
#   → ~/LeMonkey/datasets/eval2_merged/   (~600 MB)
```

### What to copy to Brev

```
~/LeMonkey/datasets/eval2_merged/                   ← merged dataset
~/LeMonkey/eval_2/scripts/brev/run_training.sh         → ~/run_training.sh
~/LeMonkey/eval_2/scripts/brev/start_training.sh       → ~/start_training.sh
~/LeMonkey/eval_2/scripts/brev/follow_training.sh      → ~/follow_training.sh
~/LeMonkey/eval_2/scripts/brev/training_status.sh      → ~/training_status.sh
~/LeMonkey/eval_1/scripts/brev_setup.sh             ← only if Brev VM is fresh
```

If the Brev VM is fresh, also rsync the `~/LeMonkey` repo (so paths inside
`run_training.sh` like `cd ~/LeMonkey` resolve), then run `brev_setup.sh`.

### On Brev

```bash
# 1. (only on a fresh VM) install miniconda + lerobot==0.5.1 + ffmpeg + HF auth
bash ~/LeMonkey/eval_1/scripts/brev_setup.sh
hf auth login   # paste write token

# 2. Linger so training survives SSH disconnect
sudo loginctl enable-linger $USER

# 3. Launch training as a systemd user service
chmod +x ~/run_training.sh ~/start_training.sh ~/follow_training.sh ~/training_status.sh
~/start_training.sh
~/follow_training.sh    # live progress (Ctrl-C to detach; training keeps running)
```

### Training config

| Param | Value | Why |
|---|---|---|
| `--policy.path` | `lerobot/smolvla_base` | from-scratch, not v2 (avoid bias) |
| `--dataset.repo_id` | `local/so101_eval2_all` | local-only (push_to_hub=false) |
| `--dataset.root` | `~/LeMonkey/datasets/eval2_merged` | the merged 180-ep dataset |
| `--dataset.image_transforms.enable` | `true` | color jitter only — no horizontal flip (would break spatial language) |
| `--policy.empty_cameras` | `2` | match v2 — pads two missing cameras with zeros |
| `--batch_size` | 192 | v2 used this on H100 |
| `--steps` | 25000 | v2 converged at 25k on a smaller dataset |
| `--save_freq` | 5000 | 5 intermediate checkpoints (5k/10k/15k/20k/25k) |
| (no `--rename_map`) | – | eval2 dataset already has `observation.images.camera1` natively |

### After training (completed)

1. Push to HF Hub: done — final at root, intermediates under `checkpoints/<step>/`.
2. Pull back locally for inference / probing: `hf download HBOrtiz/smolvla_eval2 --local-dir ~/LeMonkey/eval_2/train/smolvla_eval2`.
3. Verify language conditioning with `eval_1/scripts/probe_language_conditioning.py`
   (point `--model-path` at the Eval 2 checkpoint) and
   `probe_compositional.py` — pairwise distances should rise from the 5–11 Eval-1
   baseline to ≥ 20–30 if compositional reasoning was learned.
4. Run real-robot eval on shuffled bowl arrangements with held-out prompts.

## Hardware

Same as Eval 1 — see `eval_1/README.md`. Leader on `/dev/so101-leader`,
follower on `/dev/so101-follower`, camera on `/dev/video0`.
