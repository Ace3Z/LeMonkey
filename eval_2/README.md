# Eval 2: Compositional Instruction Following

The banana stays in the same position as Eval 1, but the three coloured bowls
are reshuffled across the three positions, and the prompt no longer names a
colour directly. The policy has to **reason about a compositional prompt** to
decide which bowl is the target, rather than mapping a colour word to a fixed
position.

> *"Put the banana into the 2nd bowl from the left."*
> *"Put the banana into the bowl on the right of the red bowl."*
> *"Put the banana into the bowl that is not green and not blue."*

<div align="center">
<table>
  <tr>
    <td align="center"><video src="../media/demos/eval2_01.mp4" autoplay loop muted playsinline width="320"></video></td>
    <td align="center"><video src="../media/demos/eval2_02.mp4" autoplay loop muted playsinline width="320"></video></td>
  </tr>
</table>
</div>

## Our approach

A policy trained only on Eval 1 fails here: our probes showed it conditioned on
the colour word but had essentially no response to spatial or relational
language. Eval 2 needs a dataset that *forces* compositional reasoning, so the
design has three parts:

**1. A balanced 180-episode dataset.** Every episode is teleoperated, and the
plan is balanced along two axes at once:

- **6 bowl arrangements** (every permutation of blue, red, green), 30 episodes each.
- **6 prompt families**, 30 episodes each:

  | Family | Example |
  |---|---|
  | direct | *"Drop the banana in the red bowl."* |
  | spatial absolute | *"Put the banana in the bowl furthest to the left."* |
  | spatial ordinal | *"Put the banana into the third bowl from the right."* |
  | relational (left/right) | *"the bowl directly to the right of the red bowl."* |
  | relational (between) | *"the bowl that sits between the blue and green bowls."* |
  | negation | *"the bowl that is not blue or green."* |

  Because arrangement and family are balanced independently, no single colour or
  position correlates with the answer: the policy can only succeed by parsing
  the prompt.

**2. Wide phrasing diversity.** Each family draws from a pool of phrasings, so
the 180 episodes carry 123 distinct prompt strings. The policy sees the
*concept* expressed many ways, not one template to memorise.

**3. Fine tune from the base model, not from Eval 1.** Training starts from
`lerobot/smolvla_base`, not the Eval 1 checkpoint, because the Eval 1 model
carries a position-to-colour bias and phrasing overfit that this task is
specifically trying to avoid. Image augmentation is colour jitter only: a
horizontal flip would invert left and right and break the spatial prompts.

The result is **`HBOrtiz/so101_smolvla_eval2`**, SmolVLA-450M trained for 25k steps.

## What is on the Hugging Face Hub

| Repo | Type | Contents |
|---|---|---|
| [`HBOrtiz/so101_smolvla_eval2`](https://huggingface.co/HBOrtiz/so101_smolvla_eval2) | model | Deployed Eval 2 policy: SmolVLA-450M, 25k steps from `lerobot/smolvla_base`, image augmentation. Final checkpoint at the repo root, intermediates under `checkpoints/`. |
| [`HBOrtiz/so101_eval2`](https://huggingface.co/datasets/HBOrtiz/so101_eval2) | dataset | 180 teleop episodes, 107,820 frames, 123 distinct compositional prompts, balanced over 6 arrangements and 6 prompt families. |

## Running a rollout

With the `lemonkey` conda environment active, the top-level wrapper
[`run_eval_2.sh`](../run_eval_2.sh) downloads the deployed checkpoint from
the Hub on first use and starts an interactive prompt loop:

```bash
./run_eval_2.sh                    # default = HF root (final 25k)
./run_eval_2.sh checkpoints/020000 # earlier intermediate
```

Reshuffle the bowls between rollouts and check that the policy follows the
prompt to the right bowl regardless of arrangement.

Three structured variants live under `scripts/`:

| Script | Mode |
|---|---|
| `scripts/run_rollout_structured.py` | Script picks the arrangement and samples a random prompt each iteration. |
| `scripts/run_rollout_freeplay.py` | You fix the arrangement once; the script samples a random prompt each iteration. |
| `scripts/filter_prompt.py` | Normalises eval-day prompts by stripping the OOD "from the robot perspective" qualifier before it reaches the policy (used by `run_eval_2.sh`). |

## Recording the dataset

`scripts/record_episodes.py` drives a fixed, balanced 180-episode recording plan
(persisted to `state/plan.json`, so progress survives restarts). It minimises
physical work by grouping episodes so the bowls only need to be reshuffled six
times across the whole collection. It announces each arrangement change, shows
the prompt and target bowl, and records one 20 s teleop episode per step.

```bash
./scripts/record_episodes.py             # resume (or create) the plan
./scripts/record_episodes.py --dry-run   # walk the plan without the robot
```

## Training pipeline

`scripts/merge_episodes.py` merges the 180 per-episode directories into one
LeRobot v3 dataset, which is then trained on a cloud GPU VM. The eval-2
specific recipe lives in `scripts/training_vm/run_training.sh`; the systemd-wrap
launcher, log tail, and status snapshot are the shared scripts under
[`../scripts/training_vm/`](../scripts/training_vm/) (driven via env vars, see that dir's
docstrings). Key settings: from `lerobot/smolvla_base`, batch size 192,
25k steps, colour-jitter augmentation only, `empty_cameras=2` to zero-pad
the unused camera slots.

To start a training run:

```bash
UNIT=lerobot-train-eval2 \
DESCRIPTION="LeRobot SmolVLA Eval 2 training (compositional, 180 ep)" \
TRAIN_SCRIPT=$PWD/eval_2/scripts/training_vm/run_training.sh \
LOG_FILE=$HOME/outputs/train/so101_smolvla_eval2.log \
    bash scripts/training_vm/start_training.sh

bash scripts/training_vm/follow_training.sh $HOME/outputs/train/so101_smolvla_eval2.log
```

## Layout

```
eval_2/
├── README.md                       this file
├── scripts/
│   ├── record_episodes.py          balanced 180-episode teleop recorder
│   ├── merge_episodes.py           merges episode dirs into one LeRobot v3 dataset
│   ├── run_rollout_structured.py   script-picked arrangement + sampled prompt loop
│   ├── run_rollout_freeplay.py     fixed arrangement + sampled prompt loop
│   ├── filter_prompt.py            strips OOD "from the robot perspective" qualifier
│   └── training_vm/run_training.sh eval-2 specific lerobot-train invocation
│                                   (systemd wrap + log tail + status: ../../scripts/training_vm/)
├── state/                          plan.json, persistent recording state (gitignored)
├── train/                          model checkpoints (gitignored)
├── rollouts/                       per-rollout dataset dumps (gitignored)
└── evals/                          per-session evaluation CSVs (gitignored)
```

## Hardware

Same as Eval 1: SO-101 follower on `/dev/so101-follower`, leader on
`/dev/so101-leader`, USB overhead camera (mounted above the workspace, looking down) on `/dev/video0`.
