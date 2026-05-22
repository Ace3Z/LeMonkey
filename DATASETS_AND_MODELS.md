# Datasets and Models

Every trained policy and every dataset for LeMonkey is published under the
[`HBOrtiz/`](https://huggingface.co/HBOrtiz) organization on the Hugging Face Hub.
This file is the single inventory: what each artifact is, how it was built, and
which one is the deployed/recommended one.

All policies are **SmolVLA-450M** fine-tuned from
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base): the
SmolVLM2 vision-language backbone is kept and the flow-matching action expert
is trained on per-eval data. Datasets use the
[LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) format.

---

## Eval 1: direct color-conditioned pick and place

### Deployed model

| Repo | Description |
|---|---|
| [`HBOrtiz/smolvla_eval1`](https://huggingface.co/HBOrtiz/smolvla_eval1) | Deployed Eval 1 policy: SmolVLA-450M, 25k steps from `smolvla_base`, image augmentation on. Final checkpoint at the repo root, intermediates under `checkpoints/`. |

### Datasets

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval1`](https://huggingface.co/datasets/HBOrtiz/so101_eval1) | training set | Merged 153 episodes / 44.6k frames: 118 behavior-cloning demos plus 35 HG-DAgger correction demos. |

---

## Eval 2: compositional instruction following

### Deployed model

| Repo | Description |
|---|---|
| [`HBOrtiz/smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2) | Deployed Eval 2 policy: SmolVLA-450M, 25k steps from `smolvla_base`, image augmentation on. Final 25k checkpoint at the repo root, intermediates under `checkpoints/{005000..025000}/`. |

### Dataset

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval2`](https://huggingface.co/datasets/HBOrtiz/so101_eval2) | training set | 180 teleop episodes / 107,820 frames. 123 distinct compositional prompts, balanced over 6 bowl arrangements and 6 prompt families (direct, absolute spatial, ordinal spatial, left/right relational, between, negation). |

---

## Eval 3: coke can on a celebrity portrait

Eval 3 deploys **two** SmolVLA models: one specialized on the in-distribution
celebrities via co-training, one trained broad on 192 celebrities for
out-of-distribution coverage.

### Deployed models

| Repo | Use | Description |
|---|---|---|
| [`HBOrtiz/smolvla_eval3_cotrain`](https://huggingface.co/HBOrtiz/smolvla_eval3_cotrain) | in-distribution celebrities | SmolVLA-450M co-trained on robot episodes and vision-language grounding pairs at a 5:1 robot-to-vision-language ratio. Single-camera inference contract (`cam1`). Checkpoints nested under `step_NNNNNN/`. |
| [`HBOrtiz/smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3) | broad and out-of-distribution | SmolVLA-450M trained on the 192-celebrity dataset, 30k steps. Final 25k checkpoint at the repo root, intermediates under `checkpoints/`. |

### Datasets

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval3_cotrain`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_cotrain) | robot training stream | 9,394 episodes / 5,053,972 frames: real base teleops plus identity-preserving augmented variants of the can placed on Taylor Swift / Barack Obama / Yann LeCun portraits. 15 prompt templates (5 paraphrases per celebrity). |
| [`eval3_vl_pairs`](https://huggingface.co/datasets/HBOrtiz/eval3_vl_pairs) | vision-language stream | 56,202 vision-language pairs over 9,367 frames. Each pair links a portrait bounding box to the celebrity's name (two caption types: location-to-name and name-to-location). The grounding signal for co-training. |
| [`so101_eval3`](https://huggingface.co/datasets/HBOrtiz/so101_eval3) | training set (broad) | The 192-celebrity dataset: real base teleops plus augmented variants drawn from a 200-celebrity scraped photo bank. Training input for `smolvla_eval3`. |

### How the Eval 3 datasets were built

A few hundred real teleop episodes were multiplied into millions of frames by
an **identity-preserving augmentation pipeline** in [`eval_3/aug/`](eval_3/aug/):
each base episode is re-rendered with different celebrity faces inpainted onto
the printed portraits. The bounding box and identity of every portrait is known
by construction, so the **vision-language grounding pairs** are emitted
automatically alongside. Co-training SmolVLA on both streams puts the celebrity
knowledge into the policy weights themselves. See [`eval_3/README.md`](eval_3/README.md).

---

## Notes

- The Hugging Face repos are under the team organization `HBOrtiz` and are all public.
- Older or superseded artifacts from earlier iterations also exist on the Hub
  but are not listed here. The tables above are the current, deployed set.
- For the co-training trainer and recipe, see
  [`eval_3/scripts/smolvla_cotrain/`](eval_3/scripts/smolvla_cotrain/).
