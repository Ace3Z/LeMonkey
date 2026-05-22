# Eval 3: Coke Can on a Celebrity Portrait

Place a coke can on the printed portrait of the person named in the prompt.

> *"Put the coke on Barack Obama."*

<div align="center">
<img src="../media/gifs/eval3_obama.gif" width="480"/>
</div>

This is the hardest of the three evals. The policy has to connect a **name** in
the prompt to a **face** in the camera image, and do it even for celebrities it
was never trained on, using only the deployed VLA. No separate face recognition
model or external VLM is allowed at inference.

- [Task](#task)
- [Our approach](#our-approach)
- [Deployed models](#deployed-models)
- [Datasets](#datasets)
- [Running a rollout](#running-a-rollout)
- [The data pipeline](#the-data-pipeline)
- [Folder layout](#folder-layout)

## Task

- Three printed celebrity portraits are placed on the workspace.
- An empty coke can sits in front of the robot.
- Prompt: `"Put the coke on <celebrity name>."`
- 20 s per rollout. The policy places the can on the matching portrait.
- Three tiers of increasing difficulty:
  1. **In distribution**: the three celebrities seen in training (Taylor Swift, Barack Obama, Yann LeCun).
  2. **Held-out photos**: the same three people, different photos.
  3. **Out of distribution**: celebrities never seen in training.

The inference constraint is the crux: at demo time only the deployed VLA may
run. No YOLO, no face-ID network, no cloud VLM call. External models are allowed
at *training* time to label or synthesize data; only their effect on the VLA
weights is used.

## Our approach

Because no face model can run at inference, the celebrity knowledge has to live
**inside the policy weights**. We put it there with **co-training**: SmolVLA is
trained on two interleaved data streams in one run.

| Stream | Data | What it teaches |
|---|---|---|
| Robot | teleop episodes of the arm placing the can on a portrait | how to move the arm |
| Vision language | an image plus a portrait location plus a celebrity name | who is where in the image |

Mixed at a **5:1 robot-to-vision-language ratio** (the ObjectVLA recipe), the
grounding stream shapes the same backbone that drives the action head, so the
deployed policy recognises the named person on its own. This is the
[`smolvla_cotrain/`](scripts/smolvla_cotrain/) trainer.

For broad and out-of-distribution coverage we also deploy a second SmolVLA
trained on a 192-celebrity dataset, which exposes the policy to a much wider
range of faces.

## Deployed models

| Model | Use | Recipe |
|---|---|---|
| [`HBOrtiz/so101_smolvla_eval3_cotrain`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_cotrain) | in-distribution celebrities | SmolVLA-450M, robot plus vision-language co-training at 5:1, single-camera contract. Checkpoints under `step_NNNNNN/`. |
| [`HBOrtiz/so101_smolvla_eval3_broad`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_broad) | broad and out-of-distribution | SmolVLA-450M, robot plus vision-language co-training on the 192-celebrity dataset. Final checkpoint at the repo root. |

Both are SmolVLA-450M fine tuned from `lerobot/smolvla_base`: the SmolVLM2
vision language backbone is kept and the action expert is trained. Inference
uses a single live camera; the unused camera slots are zero-padded by SmolVLA's
`empty_cameras` setting.

## Datasets

| Dataset | Role |
|---|---|
| [`HBOrtiz/so101_eval3_cotrain`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_cotrain) | Robot stream: 9,394 episodes (real base teleops plus augmented variants) of the can placed on Swift, Obama, and LeCun portraits. |
| [`HBOrtiz/eval3_vl_pairs`](https://huggingface.co/datasets/HBOrtiz/eval3_vl_pairs) | Vision-language stream: 56k grounding pairs linking a portrait bounding box to a celebrity name. |
| [`HBOrtiz/so101_eval3_broad`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_broad) | Broad robot stream: 9,842 episodes covering 192 celebrities. Robot half of the broad cotrain. |
| [`HBOrtiz/eval3_vl_pairs_broad`](https://huggingface.co/datasets/HBOrtiz/eval3_vl_pairs_broad) | Broad vision-language stream: 176,670 grounding pairs over 192 celebrities. Grounding half of the broad cotrain. |

Full provenance and build details: [`DATASETS_AND_MODELS.md`](../DATASETS_AND_MODELS.md).

## Running a rollout

With the `lemonkey` conda environment active, the two eval-day rollout runners
live in [`scripts/`](scripts/):

```bash
# in-distribution celebrities (Swift / Obama / LeCun)
./eval_3/scripts/run_rollout_cotrain_track3_5to1.sh

# broad and out-of-distribution celebrities
./eval_3/scripts/run_rollout_smolvla_eval3.sh
```

Each script downloads its checkpoint from the Hub on first use, then loops: type
a prompt (`Put the coke on <name>.`), the arm captures its home pose, runs the
policy for one 25 s episode against a single live wrist camera, and drives back
home. Type `q` to quit. Pass a checkpoint name as the first argument to use an
earlier step, for example `./eval_3/scripts/run_rollout_cotrain_track3_5to1.sh step_020000`.

The scripts assume the standard repo robot setup (SO-101 follower at
`/dev/so101-follower`, wrist camera at `/dev/video0`) and force Hugging Face
offline mode to avoid a chat-template rate-limit stall on first load.

## The data pipeline

The robot dataset is built so the policy sees many celebrities on many portrait
layouts without teleoperating thousands of episodes by hand:

1. **Base teleops**: a set of real teleop episodes of the can placed on printed portraits.
2. **Identity-preserving augmentation** ([`aug/`](aug/)): each base episode is re-rendered with different celebrity faces inpainted onto the printed portraits, multiplying a few real teleops into thousands of labelled variants.
3. **Vision-language grounding pairs**: for every frame, a pair is emitted linking each portrait's bounding box to the celebrity's name.
4. **Co-training** ([`scripts/smolvla_cotrain/`](scripts/smolvla_cotrain/)): the robot episodes and the grounding pairs are mixed 5:1 and SmolVLA is fine tuned on both.

Dataset-verification tooling, which renders the bounding boxes and labels back
onto the videos to confirm correctness, lives in [`tools/`](tools/).

## Folder layout

```
eval_3/
├── README.md          this file
├── aug/               identity-preserving portrait augmentation pipeline
├── scripts/
│   ├── smolvla_cotrain/   SmolVLA robot plus vision-language co-training trainer
│   ├── run_rollout_*      eval-day rollout runners
│   ├── warmstart/         PaliGemma VQA warm-start (Track-B Pi0.5 path)
│   └── brev/              training-VM launch scripts
├── tools/             dataset-verification renderers
└── train/, rollouts/, state/   gitignored: checkpoints, recordings, state
```
