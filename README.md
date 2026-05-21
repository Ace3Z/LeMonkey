The branch that was tested for **eval 3** is https://github.com/Ace3Z/LeMonkey/tree/feat/cotrain-smolvla-darius and trained on these datasets:
https://huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_baseline
https://huggingface.co/datasets/HBOrtiz/eval3_track3_vl_pairs

The branch that was tested and trained for eval 1 and 2 are sitting at the **main branch**

# LeMonkey

> Vision-Language-Action manipulation policy for the ETH Robot Learning FS26
> course (Project 1). Banana → bowl pick-and-place on a SO-101 6-DOF arm,
> driven by a natural-language prompt and a pretrained vision-language backbone.

[![Eval 1](https://img.shields.io/badge/Eval_1-deployed-success)](eval_1/)
[![Eval 2](https://img.shields.io/badge/Eval_2-deployed-success)](eval_2/)
[![Robot](https://img.shields.io/badge/robot-SO--101-blue)](https://huggingface.co/docs/lerobot/so101)
[![Backbone](https://img.shields.io/badge/backbone-SmolVLA_450M-orange)](https://huggingface.co/lerobot/smolvla_base)

## Table of contents

- [What's deployed](#whats-deployed)
- [The three evals](#the-three-evals)
- [Repository layout](#repository-layout)
- [How to use](#how-to-use)
- [Architecture](#architecture)
- [Hardware](#hardware)
- [Documentation index](#documentation-index)
- [Quick links](#quick-links)
- [Team & license](#team--license)

## What's deployed

All artifacts live under [`HBOrtiz/`](https://huggingface.co/HBOrtiz) on the Hugging Face Hub
(private — request access via Slack).

| Model | Eval | Recipe | Steps | Dataset |
|---|---|---|---|---|
| [`smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2) | Eval 1 | from `smolvla_base`, image augmentation | 25 000 | [`so101_eval1_all_v2`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_all_v2) — 153 ep |
| [`smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2) | Eval 2 | from `smolvla_base`, image augmentation | 25 000 | [`so101_eval2_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval2_all) — 180 ep |

Both models share the same `lerobot/smolvla_base` starting point and recipe,
but are trained on disjoint datasets and live in separate repos so the two
tasks never bleed into each other.

<details>
<summary>Source datasets (per-color BC + DAgger)</summary>

| Dataset | Type | Contents |
|---|---|---|
| [`so101_eval1_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_blue) | BC | 39 ep, banana → blue bowl |
| [`so101_eval1_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_red) | BC | 39 ep, banana → red bowl |
| [`so101_eval1_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_green) | BC | 40 ep, banana → green bowl |
| [`so101_eval1_dagger_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_blue) | DAgger | HG-DAgger corrections, blue corner positions |
| [`so101_eval1_dagger_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_red) | DAgger | red corner positions |
| [`so101_eval1_dagger_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_green) | DAgger | green corner positions |

</details>

## The three evals

<details open>
<summary><b>Eval 1 — Direct color-conditioned pick-and-place (50 pts)</b></summary>

Banana in a fixed position. Three colored bowls in fixed positions. The
policy must place the banana in the bowl named by the prompt:

> *"Put the banana in the blue colored bowl."*

Deployed: **[`HBOrtiz/smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2)**.
See [`eval_1/README.md`](eval_1/README.md) for run instructions.

</details>

<details open>
<summary><b>Eval 2 — Compositional instruction following (50 pts)</b></summary>

Banana in the **same** position as Eval 1, but the bowls are reshuffled across
positions and the prompts are compositional rather than direct:

> *"Put the banana into the 2nd bowl from the left."*<br>
> *"Put the banana into the bowl on the right of the red bowl."*<br>
> *"Put the banana into the bowl that is not green and not blue."*

Deployed: **[`HBOrtiz/smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2)**.
See [`eval_2/README.md`](eval_2/README.md) for the data-collection plan,
prompt families, and training pipeline.

</details>

<details>
<summary><b>Eval 3 — Coke can on celebrity image (50 pts)</b></summary>

A celebrity headshot is placed on the workspace. The policy must place a Coke
can in front of the matching face. Constraint: VLA-only — no separate face-ID
or VLM grounder is allowed in the inference pipeline. Not yet implemented.

</details>

<details>
<summary><b>Bonus — Smallest model (up to 50 pts)</b></summary>

Awarded to the team with the smallest VLA that still passes the evals. We
deploy SmolVLA-450M (the smallest off-the-shelf option), so we are competitive
on this dimension by default.

</details>

## Repository layout

```
.
├── README.md                                       ← this file
├── CLAUDE.md                                       behavioral guidelines for AI-pair coding
├── eval_1/                                         per-eval runtime (scripts + readme)
│   ├── README.md                                   eval 1 run instructions
│   └── scripts/
├── eval_2/                                         per-eval runtime (scripts + readme)
│   ├── README.md                                   eval 2 run instructions
│   ├── scripts/
│   └── scripts/brev/                               Brev VM training scripts
└── docs/
    ├── PROJECT.md                                  full course brief: eval spec, constraints, workflow, refs
    ├── RELATED_WORK.md                             prior work ranked per eval (repos, datasets, checkpoints)
    └── VLA_ARCHITECTURES.md                        VLA / VLM lit review, per-eval recommendation
```

`eval_X/{train,rollouts,evals,state}/` are gitignored — model checkpoints,
recordings, and per-session state stay local.

## How to use

### To deploy a model on the robot

1. Plug in the SO-101 (`/dev/so101-follower`, `/dev/so101-leader`) and camera (`/dev/video0`).
2. Either:
   - **Eval 1**: `cd eval_1 && ./scripts/run_rollout.sh` — see [`eval_1/README.md`](eval_1/README.md).
   - **Eval 2**: scripts mirror Eval 1's pattern; see [`eval_2/README.md`](eval_2/README.md).
3. Type or speak the prompt at the menu, then watch the rollout.

### To train from scratch on a fresh Brev VM

1. `bash ~/LeMonkey/eval_1/scripts/brev_setup.sh` — installs miniconda, lerobot 0.5.1, ffmpeg.
2. `hf auth login` with a write token.
3. `sudo loginctl enable-linger $USER` so training survives SSH disconnect.
4. Run the eval-specific training script (e.g. `~/start_training.sh` after
   copying it over per `eval_2/README.md`'s "On Brev" section).

### To collect more data

- **Eval 1**: `eval_1/scripts/dagger_record.py` for HG-DAgger corrections.
- **Eval 2**: `eval_2/scripts/record_eval2.py` runs a balanced 180-episode plan with persistent state.

## Architecture

```
        ┌──────────────────────┐
prompt  │  SmolVLA (450M)      │
  ┐     │  ─ frozen VLM        │ →  6-DOF action chunk
  ┘     │  ─ trainable expert  │    (50 steps × 6 joints)
image   └──────────────────────┘
```

Both deployed models are SmolVLA fine-tuned from `lerobot/smolvla_base`. The
SmolVLM2 vision-language backbone is frozen and the action expert is trained
on per-eval data (153 ep for Eval 1 v2, 180 ep for Eval 2). The action head
emits 50-step chunks; we execute the first action and replan every 50 frames.

## Hardware

- **Robot**: SO-101 6-DOF arm, follower at `/dev/so101-follower`, leader at `/dev/so101-leader` (udev-stable).
- **Camera**: USB camera at `/dev/video0`, 640×480 @ 30 fps, wrist-mounted.
- **GPU (inference)**: any NVIDIA GPU with ≥ 6 GB VRAM. GTX 1660 SUPER tested.
- **GPU (training)**: Brev H100 (80 GB) or RTX Pro 6000 (96 GB). ~10 hours per 25k-step run.

## Documentation index

| File | What it covers |
|---|---|
| [`docs/PROJECT.md`](docs/PROJECT.md) | The full course brief — read first |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Prior work ranked per eval |
| [`docs/VLA_ARCHITECTURES.md`](docs/VLA_ARCHITECTURES.md) | VLA / VLM choice, parameter counts, fine-tuning knobs |
| [`eval_1/README.md`](eval_1/README.md) | Eval 1 run instructions |
| [`eval_2/README.md`](eval_2/README.md) | Eval 2 plan, prompts, training pipeline |
| [`CLAUDE.md`](CLAUDE.md) | Behavioral guidelines for AI-pair coding on this repo |

## Quick links

| | |
|---|---|
| Slack — course channel | [`project-1-vla`](https://robot-course-ethz.slack.com/archives/C0AULTPSDHS) ([join workspace](https://join.slack.com/t/robotlearning-wht4341/shared_invite/zt-3vjghtb1w-K3k7b7amUr37y39IF9dL3g)) |
| GPU compute | [NVIDIA Brev](https://brev.nvidia.com) · [docs](https://docs.nvidia.com/brev/latest) |
| Robot | [SO-101 build & calibration](https://huggingface.co/docs/lerobot/so101) |
| Data format | [LeRobot dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) |
| Reference code | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| Course doc | [Google Docs](https://docs.google.com/document/d/1YsQ_Qe4vEwDp1dJdqn3l9vSt7oJBkc6JazjbmWLxAXg/edit?tab=t.0) |

## Team & license

See [`docs/PROJECT.md` §11](docs/PROJECT.md#11-team) for the team list.
Course coursework — internal use by the team only. Not for redistribution.
