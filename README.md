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
    <td align="center"><img src="media/gifs/eval1_blue.gif" width="270"/></td>
    <td align="center"><img src="media/gifs/eval2_02.gif" width="270"/></td>
    <td align="center"><img src="media/gifs/eval3_obama.gif" width="270"/></td>
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
| **Eval 3** | Coke can onto a celebrity portrait | Identity, e.g. *"Put the coke on Barack Obama."* | deployed |

Each eval has its own runtime folder, README, deployed model, and dataset. This root README is the map; each `eval_N/README.md` is the detailed runbook.

---

## 🧩 The Three Evals

### Eval 1: direct color-conditioned pick and place

<img src="media/gifs/eval1_red.gif" align="right" width="300"/>

A banana sits in a fixed position. Three colored bowls (blue, red, green) sit in fixed positions. The policy places the banana in the bowl named by the prompt.

> *"Put the banana in the blue colored bowl."*

- **Deployed model:** [`HBOrtiz/smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2)
- **Trained on:** [`HBOrtiz/so101_eval1_all_v2`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_all_v2), 153 teleop episodes (behavior-cloning demos plus HG-DAgger corrections)
- **Runbook:** [`eval_1/README.md`](eval_1/README.md)

### Eval 2: compositional instruction following

<img src="media/gifs/eval2_01.gif" align="right" width="300"/>

The banana stays put, but the bowls are reshuffled across positions and the prompt no longer names a color directly. The policy has to work out which bowl is meant.

> *"Put the banana into the 2nd bowl from the left."*
> *"Put the banana into the bowl that is not green and not blue."*

- **Deployed model:** [`HBOrtiz/smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2)
- **Trained on:** [`HBOrtiz/so101_eval2_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval2_all), 180 teleop episodes balanced over 6 bowl arrangements and 6 compositional prompt families
- **Runbook:** [`eval_2/README.md`](eval_2/README.md)

### Eval 3: coke can onto a celebrity portrait

<img src="media/gifs/eval3_obama.gif" align="right" width="300"/>

Three printed celebrity portraits are laid out on the workspace. The policy places a Coke can on the portrait of the person named in the prompt, including, in the hardest tier, celebrities never seen in training.

> *"Put the coke on Barack Obama."*

The course rule: no separate face-recognition model or external VLM may run at inference, so the deployed VLA has to do the identity reasoning itself. We solve this with **co-training**, training the policy jointly on robot manipulation episodes and on a vision-language grounding dataset, so celebrity knowledge ends up inside the policy weights.

- **Deployed model (in-distribution celebrities):** [`HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1`](https://huggingface.co/HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1), trained on [`HBOrtiz/so101_eval3_track3_v3_baseline`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_baseline) (robot episodes) plus [`HBOrtiz/eval3_track3_vl_pairs`](https://huggingface.co/datasets/HBOrtiz/eval3_track3_vl_pairs) (grounding pairs)
- **Deployed model (broad celebrities):** [`HBOrtiz/smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3), trained on [`HBOrtiz/so101_eval3_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_all) (192-celebrity dataset)
- **Runbook:** [`eval_3/README.md`](eval_3/README.md)

---

## 🔧 Installation

Everything runs in one conda environment, `lemonkey` (Python 3.12, PyTorch CUDA 12.8, `lerobot==0.5.1[smolvla]`).

### 1. Clone and create the environment

```bash
git clone https://github.com/Ace3Z/LeMonkey.git
cd LeMonkey
bash eval_1/scripts/brev_setup.sh      # canonical env recipe: miniconda, lerobot 0.5.1, ffmpeg
conda activate lemonkey
```

`brev_setup.sh` is written for a fresh NVIDIA Brev training VM, but the package set is the same on a laptop. On a consumer GPU also install: `feetech-servo-sdk`, `rerun-sdk==0.26.2`, `peft==0.19.1`, and `opencv-python` (not the `-headless` build).

### 2. Authenticate with Hugging Face

All models and datasets live under the [`HBOrtiz/`](https://huggingface.co/HBOrtiz) organization.

```bash
hf auth login          # paste a read token (write token if you will push)
```

### 3. Robot setup (only needed to run on real hardware)

- **SO-101 arm:** a udev rule pinning the follower to `/dev/so101-follower` (and `/dev/so101-leader` for the teleop arm). The user must be in the `dialout` group.
- **Camera:** a USB wrist camera at `/dev/video0`, 640x480 at 30 fps.
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

# Eval 1: direct color pick and place
cd eval_1 && ./scripts/run_rollout.sh

# Eval 2: compositional instruction following
cd eval_2 && ./scripts/run_rollout.sh

# Eval 3: coke can onto a celebrity portrait
./eval_3/scripts/run_rollout_cotrain_track3_5to1.sh   # in-distribution celebrities
./eval_3/scripts/run_rollout_smolvla_eval3.sh         # broad celebrities
```

Type the prompt at the menu (for example `Put the coke on Barack Obama.`), watch the rollout, then type the next one or `q` to quit. Each `eval_N/README.md` documents the per-eval scripts, checkpoints, and structured-evaluation tooling.

---

## 💾 Datasets and Models

Every trained policy and every dataset is published under the [`HBOrtiz/`](https://huggingface.co/HBOrtiz) organization on the Hugging Face Hub. The full inventory, what each artifact is and how it was built, is in **[`DATASETS_AND_MODELS.md`](DATASETS_AND_MODELS.md)**.

Deployed models at a glance:

| Eval | Model | Backbone | Trained on |
|---|---|---|---|
| 1 | [`smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2) | SmolVLA-450M | `so101_eval1_all_v2` (153 ep) |
| 2 | [`smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2) | SmolVLA-450M | `so101_eval2_all` (180 ep) |
| 3 | [`smolvla_eval3_cotrain_track3_5to1_cam1`](https://huggingface.co/HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1) | SmolVLA-450M | `so101_eval3_track3_v3_baseline` plus `eval3_track3_vl_pairs` |
| 3 | [`smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3) | SmolVLA-450M | `so101_eval3_all` (192 celebrities) |

---

## 🗂️ Repository Layout

```
LeMonkey/
├── README.md                  this file (the map)
├── DATASETS_AND_MODELS.md      Hugging Face dataset and model inventory
├── eval_1/                    Eval 1: runtime, scripts, README
├── eval_2/                    Eval 2: runtime, scripts, README
├── eval_3/                    Eval 3: runtime, scripts, README
│   ├── aug/                   data-augmentation pipeline (celebrity portraits)
│   ├── scripts/               training, rollout, and dataset-build scripts
│   │   └── smolvla_cotrain/   the SmolVLA co-training trainer
│   └── tools/                 dataset-verification tooling
├── calibration/               per-arm SO-101 calibration JSONs
└── media/                     logos and demo GIFs
```

Per-eval `train/`, `rollouts/`, `evals/`, and `state/` folders stay local (checkpoints, recordings, and session state are not committed).

---

## 🖥️ Hardware

- **Robot:** SO-101 6-DOF arm (follower plus leader for teleop), USB wrist camera at 640x480, 30 fps.
- **Inference:** any NVIDIA GPU with at least 6 GB of VRAM. A laptop GPU is enough, since SmolVLA-450M is small.
- **Training:** an NVIDIA Brev H100 or RTX PRO 6000, or a local RTX 5090. A 25k to 45k step run takes a few hours.

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
