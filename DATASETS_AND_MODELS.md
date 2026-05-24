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
| [`HBOrtiz/so101_smolvla_eval1`](https://huggingface.co/HBOrtiz/so101_smolvla_eval1) | Deployed Eval 1 policy: SmolVLA-450M, 25k steps from `smolvla_base`, image augmentation on. Final checkpoint at the repo root, intermediates under `checkpoints/`. |

### Datasets

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval1`](https://huggingface.co/datasets/HBOrtiz/so101_eval1) | training set | Merged 153 episodes / 44.6k frames: 118 behavior-cloning demos plus 35 HG-DAgger correction demos. |

---

## Eval 2: compositional instruction following

### Deployed model

| Repo | Description |
|---|---|
| [`HBOrtiz/so101_smolvla_eval2`](https://huggingface.co/HBOrtiz/so101_smolvla_eval2) | Deployed Eval 2 policy: SmolVLA-450M, 25k steps from `smolvla_base`, image augmentation on. Final 25k checkpoint at the repo root, intermediates under `checkpoints/{005000..025000}/`. |

### Dataset

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval2`](https://huggingface.co/datasets/HBOrtiz/so101_eval2) | training set | 180 teleop episodes / 107,820 frames. 123 distinct compositional prompts, balanced over 6 bowl arrangements and 6 prompt families (direct, absolute spatial, ordinal spatial, left/right relational, between, negation). |

---

## Eval 3: coke can on a celebrity portrait

Eval 3 publishes several models. The two deployed on eval day are the **5:1 cotrain** (for in-distribution celebrities) and the **broad** (for out-of-distribution). Three more variants are published for reproducibility and comparison: the **10:1 cotrain**, the **cotrain + KLAL** attention-supervised variant, and the **Pi0.5** variant. The **PaliGemma VQA warm-start** that initialises the Pi0.5 backbone is also published.

### Deployed (eval day)

| Repo | Use | Description |
|---|---|---|
| [`HBOrtiz/so101_smolvla_eval3_cotrain`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_cotrain) | in-distribution celebrities | SmolVLA-450M co-trained on robot episodes and vision-language grounding pairs at a 5:1 robot-to-vision-language ratio. Single-camera inference contract (`cam1`). Checkpoints nested under `step_NNNNNN/`. |
| [`HBOrtiz/so101_smolvla_eval3_broad`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_broad) | broad and out-of-distribution | SmolVLA-450M co-trained on the 192-celebrity robot dataset plus the 192-celebrity vision-language grounding pairs. The 25k checkpoint is deployed at the repo root, intermediates under `checkpoints/`. |

### Other published variants

| Repo | Recipe |
|---|---|
| [`HBOrtiz/so101_smolvla_eval3_cotrain_10to1`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_cotrain_10to1) | Same SmolVLA + robot + vision-language co-training as the deployed cotrain, but at the standard ObjectVLA 10:1 robot-to-vision-language ratio. Less VL pressure than the deployed 5:1 model. |
| [`HBOrtiz/so101_smolvla_eval3_cotrain_klal`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_cotrain_klal) | SmolVLA cotrain plus the KLAL (KL-divergence attention loss) attention-supervision objective. Steers the VLM attention toward the named celebrity's portrait bounding box during training. |
| [`HBOrtiz/so101_pi05_eval3`](https://huggingface.co/HBOrtiz/so101_pi05_eval3) | The Pi0.5 (PaliGemma-2B + Gemma-300M action expert) variant of Eval 3, fine-tuned via LoRA from the [`paligemma_vqa_warm`](https://huggingface.co/HBOrtiz/paligemma_vqa_warm) backbone init. |
| [`HBOrtiz/paligemma_vqa_warm`](https://huggingface.co/HBOrtiz/paligemma_vqa_warm) | PaliGemma backbone warm-started on VGGFace2 VQA. Init weights for the Pi0.5 variant above; not deployed as a policy on its own. |

### Datasets

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval3_cotrain`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_cotrain) | robot training stream | 9,394 episodes / 5,053,972 frames: real base teleops plus identity-preserving augmented variants of the can placed on Taylor Swift / Barack Obama / Yann LeCun portraits. 15 prompt templates (5 paraphrases per celebrity). |
| [`so101_eval3_cotrain_grounding`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_cotrain_grounding) | vision-language stream | 56,202 vision-language pairs over 9,367 frames. Each pair links a portrait bounding box to the celebrity's name (two caption types: location-to-name and name-to-location). The grounding signal for co-training. |
| [`so101_eval3_broad`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_broad) | robot training stream (broad) | 9,842 episodes / 5,294,800 frames: real base teleops plus identity-preserving augmented variants drawn from a 192-celebrity scraped photo bank. Robot half of the broad cotrain. |
| [`so101_eval3_broad_grounding`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_broad_grounding) | vision-language stream (broad) | 176,670 grounding pairs over 9,815 frames, covering 192 celebrities. The grounding half of the broad cotrain that produced `so101_smolvla_eval3_broad`. |

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
