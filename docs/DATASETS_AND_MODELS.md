# Datasets & Models

Every trained policy and every dataset for LeMonkey is published under the
[`HBOrtiz/`](https://huggingface.co/HBOrtiz) organization on the Hugging Face Hub.
This file is the single inventory — what each artifact is, how it was built, and
which one is the deployed/recommended one.

All policies are **SmolVLA-450M** fine-tuned from
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base): the
SmolVLM2 vision-language backbone is kept and the flow-matching action expert is
trained on per-eval data. Datasets are in the [LeRobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
format.

---

## Eval 1 — Direct color-conditioned pick-and-place

### Model — deployed

| Repo | Description |
|---|---|
| [`HBOrtiz/smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2) | **Deployed Eval 1 policy.** SmolVLA-450M, 25k steps from `smolvla_base`, image augmentation on. Final checkpoint at the repo root; intermediates under `checkpoints/`. |

### Datasets

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval1_all_v2`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_all_v2) | **training set** | Merged 153 episodes / 44.6k frames — 118 behavior-cloning demos + 35 HG-DAgger correction demos. |
| [`so101_eval1_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_blue) · [`_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_red) · [`_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_green) | source (BC) | Per-color behavior-cloning demos — 39 / 39 / 40 episodes. |
| [`so101_eval1_dagger_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_blue) · [`_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_red) · [`_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_green) | source (DAgger) | Per-color HG-DAgger correction demos, recorded against the BC policy's failure positions. |

---

## Eval 2 — Compositional instruction following

### Model — deployed

| Repo | Description |
|---|---|
| [`HBOrtiz/smolvla_eval2`](https://huggingface.co/HBOrtiz/smolvla_eval2) | **Deployed Eval 2 policy.** SmolVLA-450M, 25k steps from `smolvla_base`, image augmentation on. Final 25k checkpoint at the repo root; intermediates under `checkpoints/{005000…025000}/`. |

### Dataset

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval2_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval2_all) | **training set** | 180 teleop episodes / 107,820 frames. 123 distinct compositional prompts, balanced over 6 bowl arrangements × 6 prompt families (direct, absolute/ordinal spatial, left/right relational, between, negation). |

---

## Eval 3 — Coke can on a celebrity portrait

Eval 3 deploys **two** SmolVLA models — one specialized on the in-distribution
celebrities via co-training, one trained broad on 192 celebrities for
out-of-distribution coverage.

### Models — deployed

| Repo | Use | Description |
|---|---|---|
| [`HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1`](https://huggingface.co/HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1) | In-distribution celebrities | SmolVLA-450M **co-trained** on robot episodes + vision-language grounding pairs at a 5:1 robot-to-VL ratio. Single-camera inference contract (`cam1`). Checkpoints nested under `step_NNNNNN/`. |
| [`HBOrtiz/smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3) | Broad / out-of-distribution celebrities | SmolVLA-450M trained on the 192-celebrity dataset, 30k steps. Final 25k checkpoint at the repo root; intermediates under `checkpoints/`. |

### Datasets

| Repo | Type | Contents |
|---|---|---|
| [`so101_eval3_track3_v3_baseline`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_baseline) | **robot training stream** | 9,394 episodes / 5,053,972 frames — real base teleops + identity-preserving augmented variants of the can placed on Taylor Swift / Barack Obama / Yann LeCun portraits. 15 prompt templates (5 paraphrases × 3 celebrities). |
| [`eval3_track3_vl_pairs`](https://huggingface.co/datasets/HBOrtiz/eval3_track3_vl_pairs) | **VL grounding stream** | 56,202 vision-language pairs over 9,367 frames — each pair links a portrait's bounding box to the celebrity's name (two caption types: location→name and name→location). The grounding signal for co-training. |
| [`so101_eval3_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_all) | training set (broad) | The 192-celebrity dataset — real base teleops + augmented variants drawn from a ~200-celebrity scraped photo bank. Training input for `smolvla_eval3`. |

### How the Eval 3 datasets were built

A few hundred real teleop episodes were multiplied into millions of frames by an
**identity-preserving augmentation pipeline** ([`eval_3/aug/`](../eval_3/aug/)):
each base episode is re-rendered with different celebrity faces inpainted onto
the printed portraits. The bounding box and identity of every portrait is known
by construction, so the **VL grounding pairs** are emitted automatically
alongside. Co-training SmolVLA on both streams puts the celebrity knowledge into
the policy weights — see [`eval_3/README.md`](../eval_3/README.md).

---

## Notes

- The Hub repos are under the team org `HBOrtiz`; some are private — request
  access if a link 404s.
- Older / superseded artifacts from earlier iterations also exist on the Hub but
  are not listed here; the table above is the current, deployed set.
- For the co-training trainer and recipe, see
  [`eval_3/scripts/smolvla_cotrain/`](../eval_3/scripts/smolvla_cotrain/) and the
  experiment logs under [`docs/experiments/`](experiments/).
