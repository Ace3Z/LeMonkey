<div align="center">

<div align="center">
<table>
  <tr>
    <td align="center" valign="middle">
      <img src="media/figures/logos/cvg_logo_colour-white.png" height="40"/>
    </td>
    <td align="center" valign="middle">
      <img src="media/figures/logos/eth_logo_kurz_neg.png" height="80"/>
    </td>
    <td align="center" valign="middle">
      <img src="media/figures/logos/Microsoft-logo_rgb_c-gray.png" height="100"/>
    </td>
  </tr>
</table>
</div>


# LeMonkey — Language-Conditioned Robot Manipulation

**A vision-language-action (VLA) manipulation policy for the SO-101 arm that picks up an object and places it where a natural-language prompt tells it to — across three increasingly hard reasoning tasks.**

[Overview](#-overview) •
[The Three Evals](#-the-three-evals) •
[Installation](#-installation) •
[Running a Policy](#-running-a-policy) •
[Datasets & Models](#-datasets--models) •
[Team](#-team)

</div>

---

## 📖 Overview

**LeMonkey** is Team 7's project for the **Robot Learning course (ETH Zurich, FS26) — Project 1**. The goal: a single class of policy — a **VLA built on a pretrained vision-language backbone** — that observes a scene through the robot's camera, reads a natural-language prompt, and manipulates the right object accordingly. The grading rewards a policy that *reasons* about the prompt, not one that memorizes fixed positions.

We deploy **SmolVLA-450M** (the smallest off-the-shelf VLA), fine-tuned per task from `lerobot/smolvla_base`. The same 450M policy is used everywhere — keeping us competitive on the **smallest-model bonus**.

The project is split into **three evaluations**, each a harder reasoning problem on the same robot:

| Eval | Task | Prompt style | Status |
|---|---|---|---|
| **Eval 1** | Banana → colored bowl | Direct: *"Put the banana in the blue bowl."* | ✅ deployed |
| **Eval 2** | Banana → bowl, bowls reshuffled | Compositional: *"…the 2nd bowl from the left."* | ✅ deployed |
| **Eval 3** | Coke can → celebrity portrait | Identity: *"Put the coke on Barack Obama."* | ✅ deployed |

Each eval has its own runtime folder, README, deployed model, and dataset. This root README is the map; each `eval_N/README.md` is the detailed runbook.

---

## 🎬 Eval-Day Demos

Rollout recordings from the actual evaluation day are in
[`media/videos/videos_team7/`](media/videos/videos_team7/) — one folder per eval,
with a `script_of_prompts.md` listing the prompt used in each clip.

- [`eval1/`](media/videos/videos_team7/eval1/) — direct color-conditioned pick-and-place
- [`eval2/`](media/videos/videos_team7/eval2/) — compositional instruction following
- [`eval3/`](media/videos/videos_team7/eval3/) — coke can on the named celebrity (incl. an out-of-distribution celebrity)

---

## 🧩 The Three Evals

### Eval 1 — Direct color-conditioned pick-and-place

A banana sits in a fixed position; three colored bowls (blue / red / green) sit in fixed positions. The policy places the banana in the bowl named by the prompt.

> *"Put the banana in the blue colored bowl."*

- **Deployed:** [`HBOrtiz/smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2)
- **Trained on:** [`HBOrtiz/so101_eval1_all_v2`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_all_v2) — 153 teleop episodes (behavior-cloning demos + HG-DAgger corrections)
- **Runbook:** [`eval_1/README.md`](eval_1/README.md)

### Eval 2 — Compositional instruction following

The banana stays put, but the bowls are **reshuffled** across positions and the prompt no longer names a color directly — the policy must *work out* which bowl is meant.

> *"Put the banana into the 2nd bowl from the left."*
> *"…into the bowl on the right of the red bowl."*
> *"…into the bowl that is not green and not blue."*

- **Deployed:** [`HBOrtiz/smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2)
- **Trained on:** [`HBOrtiz/so101_eval2_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval2_all) — 180 teleop episodes, balanced over 6 bowl arrangements × 6 compositional prompt families
- **Runbook:** [`eval_2/README.md`](eval_2/README.md)

### Eval 3 — Coke can on a celebrity portrait

Three printed celebrity portraits are laid out on the workspace. The policy places a Coke can on the portrait of the **person named in the prompt** — including, in the hardest tier, celebrities never seen in training.

> *"Put the coke on Barack Obama."*

The catch (course rule): **no separate face-recognition model or external VLM may run at inference** — the deployed VLA must do the identity reasoning itself. We solve this by **co-training**: the SmolVLA policy is trained jointly on robot manipulation episodes *and* on a vision-language grounding dataset (portrait location + celebrity name), so celebrity knowledge ends up *inside the policy weights*.

- **Deployed (in-distribution celebrities):** [`HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1`](https://huggingface.co/HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1) — SmolVLA co-trained at a 5:1 robot-to-grounding ratio
- **Deployed (broad / out-of-distribution celebrities):** [`HBOrtiz/smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3) — SmolVLA trained on a 192-celebrity dataset
- **Trained on:** [`HBOrtiz/so101_eval3_track3_v3_baseline`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_baseline) (robot episodes) + [`HBOrtiz/eval3_track3_vl_pairs`](https://huggingface.co/datasets/HBOrtiz/eval3_track3_vl_pairs) (vision-language grounding pairs)
- **Runbook:** [`eval_3/README.md`](eval_3/README.md)

> Full dataset and model inventory: [`docs/DATASETS_AND_MODELS.md`](docs/DATASETS_AND_MODELS.md).

---

## 🔧 Installation

Everything runs in one conda environment, `lemonkey` (Python 3.12, PyTorch CUDA 12.8, `lerobot==0.5.1[smolvla]`).

### 1. Clone and create the environment

```bash
git clone https://github.com/Ace3Z/LeMonkey.git
cd LeMonkey
bash eval_1/scripts/brev_setup.sh      # canonical env recipe: miniconda + lerobot 0.5.1 + ffmpeg
conda activate lemonkey
```

`brev_setup.sh` is written for a fresh NVIDIA Brev training VM but the package set is the same on a laptop. On a consumer GPU also install: `feetech-servo-sdk`, `rerun-sdk==0.26.2`, `peft==0.19.1`, and `opencv-python` (not the `-headless` build).

### 2. Authenticate with Hugging Face

All models and datasets live under the private [`HBOrtiz/`](https://huggingface.co/HBOrtiz) org.

```bash
hf auth login          # paste a read token (write token if you will push)
```

### 3. Robot setup (only needed to run on real hardware)

- **SO-101 arm** — a udev rule pinning the follower to `/dev/so101-follower`
  (and `/dev/so101-leader` for the teleop arm); the user must be in the `dialout` group.
- **Camera** — a USB wrist camera at `/dev/video0`, 640×480 @ 30 fps.
- **Calibration** — per-arm calibration JSONs are checked in under [`calibration/`](calibration/);
  symlink them into LeRobot's cache:
  ```bash
  mkdir -p ~/.cache/huggingface/lerobot
  ln -s "$PWD/calibration" ~/.cache/huggingface/lerobot/calibration
  ```

---

## ▶️ Running a Policy

Each eval ships interactive rollout scripts. They download the deployed checkpoint from the Hub on first use, capture the arm's home pose, run the policy for one episode against a typed prompt, and drive the arm home for the next take.

```bash
conda activate lemonkey

# Eval 1 — direct color pick-and-place
cd eval_1 && ./scripts/run_rollout.sh

# Eval 2 — compositional instruction following
cd eval_2 && ./scripts/run_rollout.sh

# Eval 3 — coke can on a celebrity portrait
#   in-distribution celebrities (Swift / Obama / LeCun):
./eval_3/scripts/run_rollout_cotrain_track3_5to1.sh
#   broad / out-of-distribution celebrities:
./eval_3/scripts/run_rollout_smolvla_eval3.sh
```

Type the prompt at the menu (e.g. `Put the coke on Barack Obama.`), watch the rollout, then type the next one or `q` to quit. Each `eval_N/README.md` documents the per-eval scripts, checkpoints, and structured-evaluation tooling.

---

## 💾 Datasets & Models

Every trained policy and every teleop/augmentation dataset is published under
[`HBOrtiz/`](https://huggingface.co/HBOrtiz) on the Hugging Face Hub. The complete
inventory — what each artifact is, how it was built, and which to use — is in:

### → [`docs/DATASETS_AND_MODELS.md`](docs/DATASETS_AND_MODELS.md)

**Deployed models at a glance:**

| Eval | Model | Backbone | Trained on |
|---|---|---|---|
| 1 | [`smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2) | SmolVLA-450M | `so101_eval1_all_v2` (153 ep) |
| 2 | [`smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2) | SmolVLA-450M | `so101_eval2_all` (180 ep) |
| 3 | [`smolvla_eval3_cotrain_track3_5to1_cam1`](https://huggingface.co/HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1) | SmolVLA-450M | `so101_eval3_track3_v3_baseline` + `eval3_track3_vl_pairs` |
| 3 | [`smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3) | SmolVLA-450M | `so101_eval3_all` (192 celebs) |

---

## 🗂️ Repository Layout

```
LeMonkey/
├── README.md                  ← this file (the map)
├── eval_1/                    Eval 1 — runtime, scripts, README
│   ├── README.md
│   └── scripts/
├── eval_2/                    Eval 2 — runtime, scripts, README
│   ├── README.md
│   └── scripts/
├── eval_3/                    Eval 3 — runtime, scripts, README
│   ├── README.md
│   ├── aug/                   data-augmentation pipeline (celebrity portraits)
│   ├── scripts/               training, rollout, dataset-build, recording scripts
│   │   └── smolvla_cotrain/   the SmolVLA co-training trainer
│   └── tools/                 dataset verification tooling
├── calibration/               per-arm SO-101 calibration JSONs
├── media/                     logos, figures, eval-day videos
└── docs/                      project brief, datasets/models, experiment logs
    ├── DATASETS_AND_MODELS.md  full HF artifact inventory
    ├── PROJECT.md              the course brief
    └── experiments/            dated experiment logs
```

`eval_N/{train,rollouts,evals,state}/` are gitignored — checkpoints, recordings,
and per-session state stay local.

---

## 🖥️ Hardware

- **Robot** — SO-101 6-DOF arm (follower + leader for teleop), USB wrist camera, 640×480 @ 30 fps.
- **Inference** — any NVIDIA GPU with ≥ 6 GB VRAM (a laptop GPU is enough; SmolVLA-450M is small).
- **Training** — an NVIDIA Brev H100 / RTX PRO 6000, or a local RTX 5090. A 25–45k-step run is a few hours.

---

## 📚 Documentation Index

| Document | What it covers |
|---|---|
| [`docs/DATASETS_AND_MODELS.md`](docs/DATASETS_AND_MODELS.md) | Every dataset and model on the Hub — what, how built, which to use |
| [`docs/PROJECT.md`](docs/PROJECT.md) | The full course brief — task specs, constraints, grading |
| [`eval_1/README.md`](eval_1/README.md) · [`eval_2/README.md`](eval_2/README.md) · [`eval_3/README.md`](eval_3/README.md) | Per-eval runbooks |
| [`docs/VLA_ARCHITECTURES.md`](docs/VLA_ARCHITECTURES.md) | VLA / VLM background and the SmolVLA choice |
| [`docs/experiments/`](docs/experiments/) | Dated experiment logs |

---

## 👥 Team

**Team 7 — ETH Robot Learning FS26, Project 1**

[Roham Z. Nobari](https://github.com/rzninvo) · [Mahbod Tajdini](https://github.com/Ace3Z) · [Darius Foodeii](https://github.com/userdarius) · [Sejohn Uruthiralingam](https://github.com/SjohnU) · [Hans Baumann Oritz](https://github.com/katari16)

---

## 🙏 Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) — robot-learning framework, datasets, and `lerobot-record`
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — the 450M VLA backbone we fine-tune
- [NVIDIA Brev](https://brev.nvidia.com) — GPU compute for training
- The ETH Robot Learning FS26 course staff

---

<div align="center">

**[⬆ Back to Top](#lemonkey--language-conditioned-robot-manipulation)**

</div>
