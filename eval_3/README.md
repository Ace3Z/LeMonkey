# Eval 3 — Coke Can on a Celebrity Portrait

Place a Coke can on the printed portrait of the **person named in the prompt**.

> *"Put the coke on Barack Obama."*

This is the hardest of the three evals: the policy must connect a **name** in the
prompt to a **face** in the camera image — and do it for celebrities it was never
trained on — using only the deployed VLA, with no separate face-recognition model
or external VLM allowed at inference.

- [Task](#task)
- [How we solved the no-face-model constraint](#how-we-solved-the-no-face-model-constraint)
- [Deployed models](#deployed-models)
- [Datasets](#datasets)
- [Running a rollout](#running-a-rollout)
- [The data pipeline](#the-data-pipeline)
- [Folder layout](#folder-layout)

---

## Task

- Three printed celebrity portraits are placed on the workspace.
- An empty Coke can sits in front of the robot.
- Prompt: `"Put the coke on <celebrity name>."`
- 20 s per rollout. The policy must place the can on the matching portrait.
- Evaluated in three tiers of increasing difficulty:
  1. **In-distribution** — the three celebrities seen in training (Taylor Swift, Barack Obama, Yann LeCun).
  2. **Held-out photos** — the same three people, but different photos.
  3. **Out-of-distribution** — celebrities never seen in training.

**Inference constraint:** at demo time, only the deployed VLA may run — no YOLO,
no face-ID network, no cloud-VLM call. External models *are* allowed at *training*
time to label or synthesize data; only their effect on the VLA weights is used.

---

## How we solved the no-face-model constraint

Because no face model can run at inference, the celebrity knowledge has to live
**inside the policy weights**. We do this with **co-training**: SmolVLA is trained
on two interleaved data streams in one run —

| Stream | Data | Teaches |
|---|---|---|
| **Robot** | teleop episodes — the arm placing the can on a portrait | *how to move* the arm |
| **Vision-language** | image + portrait location + celebrity name | *who is where* in the image |

Mixed at a **5:1 robot-to-VL ratio** (ObjectVLA-style), the grounding signal
shapes the same backbone that drives the action head — so the deployed policy
recognizes the named person on its own. This is the
[`smolvla_cotrain/`](scripts/smolvla_cotrain/) trainer.

For broad / out-of-distribution coverage we also deploy a SmolVLA trained on a
**192-celebrity** dataset, giving the policy exposure to a much wider face
distribution.

---

## Deployed models

| Model | Use | Recipe |
|---|---|---|
| [`HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1`](https://huggingface.co/HBOrtiz/smolvla_eval3_cotrain_track3_5to1_cam1) | In-distribution celebrities | SmolVLA-450M, robot + VL co-training at 5:1, single-camera contract. Checkpoints under `step_NNNNNN/`. |
| [`HBOrtiz/smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3) | Broad / out-of-distribution | SmolVLA-450M trained on the 192-celebrity dataset (final 25k checkpoint at the repo root). |

Both are SmolVLA-450M fine-tuned from `lerobot/smolvla_base`; the SmolVLM2 vision-language
backbone is kept and the action expert is trained. Single live camera at inference;
the unused camera slots are zero-padded by SmolVLA's `empty_cameras` setting.

---

## Datasets

| Dataset | Role |
|---|---|
| [`HBOrtiz/so101_eval3_track3_v3_baseline`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_track3_v3_baseline) | Robot stream — 9,394 episodes (real base teleops + augmented variants) of the can being placed on Swift / Obama / LeCun portraits |
| [`HBOrtiz/eval3_track3_vl_pairs`](https://huggingface.co/datasets/HBOrtiz/eval3_track3_vl_pairs) | Vision-language stream — 56k grounding pairs (portrait bounding box ↔ celebrity name) for the 3-celebrity co-training |
| [`HBOrtiz/so101_eval3_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_all) | The 192-celebrity dataset behind `smolvla_eval3` |

Full provenance and build details: [`docs/DATASETS_AND_MODELS.md`](../docs/DATASETS_AND_MODELS.md).

---

## Running a rollout

With the `lemonkey` conda env active, the two eval-day rollout runners live in
[`scripts/`](scripts/):

```bash
# In-distribution celebrities (Swift / Obama / LeCun)
./eval_3/scripts/run_rollout_cotrain_track3_5to1.sh

# Broad / out-of-distribution celebrities
./eval_3/scripts/run_rollout_smolvla_eval3.sh
```

Each script downloads its checkpoint from the Hub on first use, then loops:
type a prompt (`Put the coke on <name>.`), the arm captures its home pose, runs
the policy for one 25 s episode against a single live wrist camera, and drives
back home. Type `q` to quit. Pass a checkpoint name as the first argument to use
an earlier step (e.g. `./eval_3/scripts/run_rollout_cotrain_track3_5to1.sh step_020000`).

The scripts assume the standard repo robot setup — SO-101 follower at
`/dev/so101-follower`, wrist camera at `/dev/video0` — and force HF offline mode
to avoid a chat-template rate-limit stall on first load.

---

## The data pipeline

The robot dataset was built so the policy sees **many celebrities on many
portrait layouts** without teleoperating thousands of episodes by hand:

1. **Base teleops** — a set of real teleop episodes of the can being placed on
   printed portraits.
2. **Identity-preserving augmentation** ([`aug/`](aug/)) — each base episode is
   re-rendered with different celebrity faces inpainted onto the printed
   portraits, multiplying a few real teleops into thousands of labeled variants.
3. **VL grounding pairs** — for every frame, a vision-language pair is emitted
   linking each portrait's location (bounding box) to the celebrity's name.
4. **Co-training** ([`scripts/smolvla_cotrain/`](scripts/smolvla_cotrain/)) — the
   robot episodes and the VL pairs are mixed 5:1 and SmolVLA is fine-tuned on both.

Dataset-verification tooling — render the bounding boxes and labels back onto the
videos to confirm correctness — lives in [`tools/`](tools/).

---

## Folder layout

```
eval_3/
├── README.md                ← this file
├── aug/                     identity-preserving portrait augmentation pipeline
├── scripts/
│   ├── smolvla_cotrain/      SmolVLA robot + VL co-training trainer
│   ├── warmstart/            PaliGemma VQA warm-start (Track-B Pi0.5 path)
│   ├── brev/                 training-VM launch scripts
│   ├── record_eval3*.py      teleop recorders
│   └── merge_*               dataset-merge scripts (LeRobot v3)
├── tools/                   dataset-verification renderers
├── train/  rollouts/  state/ ← gitignored: checkpoints, recordings, state
└── HANDOVER_TO_DEPLOY.md    deployment handover notes
```
