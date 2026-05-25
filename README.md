<div align="center">

<div align="center">
<table>
  <tr>
    <td align="center" valign="middle">
      <a href="https://www.microsoft.com/en-us/research/lab/spatial-ai/"><img src="media/figures/Microsoft-logo_rgb_c-gray.png" height="120"/></a>
    </td>
    <td align="center" valign="middle">
      <a href="https://cvg.ethz.ch/"><img src="media/figures/cvg_logo_colour-white.png" height="40"/></a>
    </td>
    <td align="center" valign="middle">
      <a href="https://ethz.ch/"><img src="media/figures/eth_logo_kurz_neg.png" height="100"/></a>
    </td>
    <td align="center" valign="middle">
      <a href="https://www.ethrobotics.ch/"><img src="media/figures/ETHRC_primarywhite.svg" height="35"/></a>
    </td>
  </tr>
</table>
</div>


# LeMonkey: Language Conditioned Robot Manipulation

**A vision-language-action policy for the SO-101 arm that picks up an object and places it where a natural-language prompt tells it to, across three increasingly hard reasoning tasks.**

[Overview](#-overview) •
[The Three Evals](#-the-three-evals) •
[Installation](#-installation) •
[Running a Policy](#-running-a-policy) •
[Datasets and Models](#-datasets-and-models) •
[Team](#-team)

</div>

---

<div align="center">
<table>
  <tr>
    <td align="center"><video src="media/demos/eval1_blue.mp4" autoplay loop muted playsinline width="270"></video></td>
    <td align="center"><video src="media/demos/eval2_02.mp4" autoplay loop muted playsinline width="270"></video></td>
    <td align="center"><video src="media/demos/eval3_obama.mp4" autoplay loop muted playsinline width="270"></video></td>
  </tr>
  <tr>
    <td align="center"><b>Eval 1:</b> direct color bowl</td>
    <td align="center"><b>Eval 2:</b> compositional reasoning</td>
    <td align="center"><b>Eval 3:</b> celebrity portrait</td>
  </tr>
</table>

*The same 450M policy, fine-tuned per task, running on the real SO-101 arm.*

</div>

---

## 📖 Overview

**LeMonkey** is our project for the [Robot Learning course](https://cvg.ethz.ch/lectures/Robot-Learning/) at ETH Zurich. The goal is one class of policy, a vision-language-action (VLA) model built on a pretrained vision-language backbone, that observes a scene through the robot camera, reads a natural-language prompt, and manipulates the object the prompt refers to. The grading rewards a policy that genuinely *reasons* about the prompt rather than memorizing fixed positions.

Every task uses the same backbone: **SmolVLA-450M**, the smallest off-the-shelf VLA, fine-tuned per task from `lerobot/smolvla_base`. Using the smallest capable model keeps us competitive on the course smallest-model bonus.

The project is split into three evaluations, each a harder reasoning problem on the same robot:

| Eval | Task | Prompt style | Status |
|---|---|---|---|
| **Eval 1** | Banana into a colored bowl | Direct, e.g. *"Put the banana in the blue bowl."* | deployed |
| **Eval 2** | Banana into a bowl, bowls reshuffled | Compositional, e.g. *"the 2nd bowl from the left."* | deployed |
| **Eval 3** | Coke can onto a celebrity portrait | Identity, e.g. *"Put the Coke on Barack Obama."* | deployed |

Each eval has its own runtime folder, README, deployed model, and dataset. This root README is the map; each `eval_N/README.md` is the detailed runbook.

---

## 🧩 The Three Evals

| Eval | Task | Deployed model | Trained on | Runbook |
|---|---|---|---|---|
| **1** | Direct color-conditioned pick and place | [`so101_smolvla_eval1`](https://huggingface.co/HBOrtiz/so101_smolvla_eval1) | [`so101_eval1`](https://huggingface.co/datasets/HBOrtiz/so101_eval1), 153 episodes (BC + HG-DAgger) | [`eval_1/README.md`](eval_1/README.md) |
| **2** | Compositional instruction following | [`so101_smolvla_eval2`](https://huggingface.co/HBOrtiz/so101_smolvla_eval2) | [`so101_eval2`](https://huggingface.co/datasets/HBOrtiz/so101_eval2), 180 episodes, 6 bowl arrangements × 6 prompt families | [`eval_2/README.md`](eval_2/README.md) |
| **3 (IID)** | Coke can onto a celebrity portrait, 3 known celebrities | [`so101_smolvla_eval3_cotrain`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_cotrain) | [`so101_eval3_cotrain`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_cotrain) + [`so101_eval3_cotrain_grounding`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_cotrain_grounding) | [`eval_3/README.md`](eval_3/README.md) |
| **3 (broad)** | Same task, 192 celebrities, OOD at eval time | [`so101_smolvla_eval3_broad`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_broad) | [`so101_eval3_broad`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_broad) + [`so101_eval3_broad_grounding`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_broad_grounding) | [`eval_3/README.md`](eval_3/README.md) |

### Eval 1: direct color-conditioned pick and place

<video src="media/demos/eval1_red.mp4" autoplay loop muted playsinline align="right" width="300"></video>

A banana sits in a fixed position. Three colored bowls (blue, red, green) sit in fixed positions. The policy places the banana in the bowl named by the prompt.

> *"Put the banana in the blue colored bowl."*

### Eval 2: compositional instruction following

<video src="media/demos/eval2_01.mp4" autoplay loop muted playsinline align="right" width="300"></video>

The banana stays put, but the bowls are reshuffled across positions and the prompt no longer names a color directly. The policy has to work out which bowl is meant.

> *"Put the banana into the 2nd bowl from the left."*
> *"Put the banana into the bowl that is not green and not blue."*

### Eval 3: coke can onto a celebrity portrait

<video src="media/demos/eval3_obama.mp4" autoplay loop muted playsinline align="right" width="300"></video>

Three printed celebrity portraits are laid out on the workspace. The policy places a Coke can on the portrait of the person named in the prompt, including, in the hardest tier, celebrities never seen in training.

> *"Put the Coke on Barack Obama."*

The course rule: no separate face-recognition model or external VLM may run at inference, so the deployed VLA has to do the identity reasoning itself. We solve this with **co-training**, training the policy jointly on robot manipulation episodes and on a vision-language grounding dataset, so celebrity knowledge ends up inside the policy weights.

---

## 🔧 Installation

Everything runs in one conda environment, `lemonkey` (Python 3.12, PyTorch CUDA 12.8, `lerobot==0.5.1[smolvla]`).

### 1. Clone and create the environment

```bash
git clone https://github.com/Ace3Z/LeMonkey.git
cd LeMonkey
bash scripts/setup_smolvla_env.sh     # canonical env recipe: miniconda, lerobot 0.5.1, ffmpeg
conda activate lemonkey
```

`setup_smolvla_env.sh` is written for a fresh GPU host (Brev, Lambda, RunPod, etc.), but the package set is the same on a laptop. On a consumer GPU also install: `feetech-servo-sdk`, `rerun-sdk==0.26.2`, `peft==0.19.1`, and `opencv-python` (not the `-headless` build). Eval 3 Pi0.5 needs the vendored-lerobot variant under [`eval_3/scripts/training_vm/setup_pi05.sh`](eval_3/scripts/training_vm/setup_pi05.sh) instead.

### 2. Authenticate with Hugging Face

All models and datasets live under the [`HBOrtiz/`](https://huggingface.co/HBOrtiz) organization.

```bash
hf auth login          # paste a read token (write token if you will push)
```

### 3. Robot setup (only needed to run on real hardware)

- **SO-101 arm:** a udev rule pinning the follower to `/dev/so101-follower` (and `/dev/so101-leader` for the teleop arm). The user must be in the `dialout` group.
- **Camera:** a USB overhead camera (mounted above the workspace, looking down) at `/dev/video0`, 640x480 at 30 fps.
- **Calibration:** per-arm calibration JSONs are checked in under [`calibration/`](calibration/). Symlink them into the LeRobot cache:
  ```bash
  mkdir -p ~/.cache/huggingface/lerobot
  ln -s "$PWD/calibration" ~/.cache/huggingface/lerobot/calibration
  ```

---

## ▶️ Running a Policy

Each eval ships interactive rollout scripts. They download the deployed checkpoint from the Hub on first use, capture the arm home pose, run the policy for one episode against a typed prompt, and drive the arm home for the next take.

```bash
conda activate lemonkey

# Top-level wrappers (download checkpoints from HF on first use,
# or use a local copy if shipped under policy/<repo_name>/).
./run_eval_1.sh                  # Eval 1: direct color pick and place
./run_eval_2.sh                  # Eval 2: compositional instruction following
./run_eval_3.sh                  # Eval 3: in-distribution celebrities (default)
./run_eval_3.sh --broad          # Eval 3: broad / out-of-distribution celebrities
```

Type the prompt at the menu (for example `Put the Coke on Barack Obama.`), watch the rollout, then type the next one or `q` to quit. Each `eval_N/README.md` documents the per-eval scripts, checkpoints, and structured-evaluation tooling.

---

## 💾 Datasets and Models

Every trained policy and every dataset is published under the [`HBOrtiz/`](https://huggingface.co/HBOrtiz) organization on the Hugging Face Hub. The full inventory, what each artifact is and how it was built, is in **[`DATASETS_AND_MODELS.md`](DATASETS_AND_MODELS.md)**.

All four deployed policies use **SmolVLA-450M** initialised from `lerobot/smolvla_base`. See [`DATASETS_AND_MODELS.md`](DATASETS_AND_MODELS.md) for additional published variants (Pi0.5, KLAL, 10:1 cotrain ablations).

---

## 🗂️ Repository Layout

```
LeMonkey/
├── README.md                    this file (the map)
├── DATASETS_AND_MODELS.md       Hugging Face dataset and model inventory
├── run_eval_1.sh                top-level Eval 1 rollout launcher
├── run_eval_2.sh                top-level Eval 2 rollout launcher
├── run_eval_3.sh                top-level Eval 3 rollout launcher (--broad for OOD celebrities)
├── eval_1/                      Eval 1: runtime, scripts, README
├── eval_2/                      Eval 2: runtime, scripts, README
├── eval_3/                      Eval 3: runtime, scripts, README
│   ├── aug/                     data-augmentation pipeline (celebrity portraits)
│   ├── scripts/                 training, rollout, and dataset-build scripts
│   │   ├── rollout/             eval-day rollout runners (one per deployed policy)
│   │   ├── record/              teleop recording session scripts
│   │   ├── data/                dataset merge + validate + push + VL-pair builders
│   │   ├── celebs/              celebrity-photo bank builders
│   │   ├── smolvla_cotrain/     SmolVLA co-training trainer (deployed)
│   │   ├── pi05_vl_cotrain/     Pi0.5 + VL bbox-grounded VQA cotrain (published variant)
│   │   ├── warmstart/           PaliGemma VQA warm-start (init for Pi0.5)
│   │   └── training_vm/         training-VM entrypoints (env setup, sync, trainers, warmstart)
│   └── tools/                   dataset-verification tooling
├── scripts/                     shared, non-eval-specific scripts
│   ├── auto_home.py                arm-home capture / drive helper for rollouts
│   ├── rest_arms.py                release SO-101 follower + leader torque
│   ├── setup_smolvla_env.sh        training-VM bootstrap (miniconda + lerobot 0.5.1 PyPI)
│   └── training_vm/                systemd wrap + log tail + status (eval-agnostic; driven by env vars)
├── calibration/                 per-arm SO-101 calibration JSONs
├── media/                       logos (figures/) and demo GIFs (gifs/)
└── third_party/
    ├── lerobot/                    LeRobot framework as a git submodule
    ├── lerobot_patches/            patches applied to the lerobot submodule (env compat)
    └── sam2/                       SAM 2 (vendored for the aug pipeline's video predictor)
```

Per-eval `train/`, `rollouts/`, `evals/`, and `state/` folders stay local (checkpoints, recordings, and session state are not committed).

---

## 🖥️ Hardware

- **Robot:** SO-101 6-DOF arm (follower plus leader for teleop), USB overhead camera (mounted above the workspace, looking down) at 640x480, 30 fps.
- **Inference:** any NVIDIA GPU with at least 6 GB of VRAM. A laptop GPU is enough, since SmolVLA-450M is small.
- **Training:** an NVIDIA H100 or RTX PRO 6000, or a local RTX 5090. A 25k to 45k step run takes a few hours.

---

## 👥 Team

Built for the [Robot Learning course](https://cvg.ethz.ch/lectures/Robot-Learning/) at ETH Zurich.

- [Roham Z. Nobari](https://github.com/rzninvo) ([LinkedIn](https://www.linkedin.com/in/rohamzn/))
- [Mahbod Tajdini](https://github.com/Ace3Z) ([LinkedIn](https://www.linkedin.com/in/mahbodtajdini/))
- [Darius Foodeei](https://github.com/userdarius) ([LinkedIn](https://www.linkedin.com/in/darius-f-922447173/))
- [Sejohn Uruthiralingam](https://github.com/SjohnU) ([LinkedIn](https://www.linkedin.com/in/sejohn-uruthiralingam/))
- [Hans Baumann-Ortiz](https://github.com/katari16) ([LinkedIn](https://www.linkedin.com/in/hans-baumann-ortiz-854264248/))

---

## 🙏 Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot): robot-learning framework, dataset format, and `lerobot-record`
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base): the 450M VLA backbone we fine-tune
- [NVIDIA Brev](https://brev.nvidia.com): GPU compute for training
- The ETH Zurich Robot Learning course staff

---

<div align="center">

**[Back to top](#lemonkey-language-conditioned-robot-manipulation)**

</div>
