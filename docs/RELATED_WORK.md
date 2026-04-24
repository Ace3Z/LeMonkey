# Related Work — LeMonkArm

Curated list of public code, datasets, model checkpoints, and writeups that map directly to the three LeMonkArm evaluations. Ordered by "how close is it to what we need to ship". Every link has been verified to resolve at the time of writing.

For the eval definitions, see [`PROJECT.md §2`](PROJECT.md#2-evaluation--150-pts-main--50-pts-bonus).

---

## TL;DR — what to copy first

| Need | Start here |
|------|-----------|
| End-to-end SO-101 + SmolVLA recipe | [Gota Ando's blog + `ggand0/vla-so101`](#gota-ando--smolvla-on-so-101-eval-1--2) |
| Reference SO-101 pick-place dataset | [`lerobot/svla_so101_pickplace`](#base-datasets) |
| Small-VLA architecture (size bonus) | [TinyVLA](#tinyvla), [SmolVLA](#smolvla-base), [FLOWER-VLA](#flower-vla-pretraining) |
| SO-101 data-collection / teleop scripts | [`vvrs/so101-playground`](#vvrsso101-playground) |
| Reasoning prompts (Eval 2) | [`lerobot/smolvla_vlabench`](#lerobotsmolvla_vlabench) |
| Official fine-tune CLI + docs | [huggingface/lerobot SmolVLA docs](#lerobot-smolvla-docs) |

---

## Eval 1 — "Put the banana in the [color] bowl" (direct color lookup)

### Gota Ando — SmolVLA on SO-101 (Eval 1 + 2)
- **Links:** Blog <https://ggando.com/blog/smolvla-so101/> · Code <https://github.com/ggand0/vla-so101> · Data `gtgando/so101_pick_place_10cm_*` on HF
- **What it is:** Full reproducible fine-tune of SmolVLA on SO-101 for cube pick-and-place. 75 demos, wrist + overhead cameras, ~10 h on a single RTX 3090, 60–80 % success. Tested language generalization to unseen colored cubes (1 / 5 — the motivation for our Eval 2).
- **Copy:** hparams (`batch_size=64`, `steps=20000`, cosine `1e-4`), data-collection protocol, training fork.
- **Caveat:** two cameras — we're limited to one. Their own ablation shows wrist-only was weaker, which is relevant to the shoulder-mount decision in [`PROJECT.md §4`](PROJECT.md#cameras-updated-project-rule).

### `lerobot/svla_so101_pickplace`
- **Link:** <https://huggingface.co/datasets/lerobot/svla_so101_pickplace>
- 50 episodes, 30 FPS, SO-101 follower, `up` + `side` cameras. Canonical dataset in the SmolVLA paper. Closest public match to our Eval 1 collection protocol.

### `markobrie/so101_banana`
- **Link:** <https://huggingface.co/datasets/markobrie/so101_banana>
- SO-101 data labelled literally *"Grasp a banana and put it on a drawing."* Same arm, same object, different target — the closest banana-centric SO-101 data in public.

### Other SO-101 pick-place community datasets (augmentation)
- `ud-smart-city/lerobot-so-101-manipulations` — <https://huggingface.co/datasets/ud-smart-city/lerobot-so-101-manipulations>
- `xinjiehu76/so101-pick-place-dataset`
- `kristaqp/so101-pick-place`
- `joung/so101-pick-place`

Useful for pretraining / augmentation. Almost all use two cameras; action-stats will differ from ours.

### Xavier O'Keefe — SmolVLA on SO-100 ("pick up the blue block")
- **Links:** Medium <https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632> · Code <https://github.com/xavier2933/smolVLA_finetune>
- 25 → 125 demos on SO-100, language-conditioned color pick. Worth reading for the **action-stats / denormalization gotcha** when starting from a non-SO-101 checkpoint.

---

## Eval 2 — Compositional instructions ("2nd bowl from the left", "red + blue → purple")

### `lerobot/smolvla_vlabench`
- **Link:** <https://huggingface.co/lerobot/smolvla_vlabench>
- SmolVLA fine-tuned on `lerobot/vlabench_unified` (3.11 M frames) covering reasoning-style manipulation prompts. Almost certainly a **better starting checkpoint than `smolvla_base` for Eval 2** — the compositional prompt distribution is closer.

### VLABench (OpenMOSS)
- **Repo:** <https://github.com/OpenMOSS/VLABench>
- Language-conditioned manipulation benchmark with long-horizon reasoning, world-knowledge, and spatial-relation prompts. Our [`PROJECT.md §12`](PROJECT.md#12-references--links) already flags it as the recommended starting read.

### FLOWER-VLA (pretraining)
- **Repo:** <https://github.com/intuitive-robots/flower_vla_pret> (user-provided)
- ~1B-param efficient VLA, OXE-pretrained, claims SOTA on CALVIN / LIBERO. **Pretraining code only** — this repo does not ship weights or fine-tune scripts. Fine-tuning lives in a separate IRL lab repo.
- **Inside:** `flower/training.py`, `conf/training.yaml`, OXE plumbing in `flower_vla/dataset/oxe/`, SLURM launchers. Launch: `accelerate launch flower/training.py`.
- **Caveat:** ~200 GPU-hours pretraining, 24 GB+ GPU for training (~8 GB inference). README explicitly warns val-loss ≠ success rate. Not plug-and-play.

---

## Eval 3 — "Place the coke on [celebrity name]"

No public VLA demo exists for *"place object on an image of a named person."* A practical recipe to assemble:

1. A **VLM** (SmolVLM, Qwen2.5-VL, PaliGemma) reads the prompt and picks the correct A5 print from the scene.
2. Output target coordinates → feed into a **generic "place the coke on the paper in front" SmolVLA** fine-tuned from our Eval 1 data.

Closest semantic analogs:
- **VLABench world-knowledge split** — above.
- **PaliGemma / Qwen2.5-VL zero-shot grounding** on celebrity images (OCR + face recognition live in the pretrained VLM; we do not need to train recognition).

No single copy-paste reference. Expect custom work here.

---

## Base models & recipes (all three evals)

### TinyVLA
- **Repo:** <https://github.com/liyaxuanliyaxuan/TinyVLA> (user-provided)
- Family of ~400 M / 700 M / 1.3 B-param VLAs pitched as fast and data-efficient. Explicitly listed as a recommended starting point in [`PROJECT.md §3`](PROJECT.md#3-architecture--procedure-constraints); the ~400 M variant is a strong candidate for the **smallest-model bonus**.
- **Inside:** `train_tinyvla.py`, `scripts/train.sh`, `scripts/process_ckpts.sh`, `eval_real_franka.py`, `llava-pythia/` backbone, `policy_heads/`, `data_utils/rlds_to_h5py.py`.
- **Checkpoints:** `lesjie/Llava-Pythia-400M`, `lesjie/Llava-Pythia-700M`, `lesjie/Llava-Pythia-1.3B` on HF.
- **Gotchas:** output dir must contain `"llava_pythia"` (and `"lora"` if LoRA); data must be HDF5 in a specific layout; post-processing pass required before eval; original evals were on Franka / ALOHA, not SO-101. MIT license.

### SmolVLA base
- **Model:** <https://huggingface.co/lerobot/smolvla_base>
- **Paper:** <https://arxiv.org/abs/2506.01844> (Shukor et al., 2025)
- 450 M params. Community-data pretraining lifts real SO-100 success 51.7 → 78.3 %. Strong default for Eval 1.

### LeRobot SmolVLA docs
- **Docs:** <https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx>
- Official fine-tune CLI:
  ```
  lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=$USER/ds \
    --batch_size=64 --steps=20000
  ```
- ~4 h on 1×A100. Recommended minimum ≈ 50 demos ("25 is not enough").
- Already available locally via our submodule: `third_party/lerobot/src/lerobot/policies/smolvla/`, `third_party/lerobot/docs/source/smolvla.mdx`, `third_party/lerobot/examples/tutorial/smolvla/using_smolvla_example.py`.

### Pi0 / Pi0.5
- **Docs:** <https://huggingface.co/docs/lerobot/pi0>
- **Tutorial:** <https://ghuijo.github.io/blog/2025/LeRobot-PI0-Finetuning-Tutorial/>
- Larger than SmolVLA — useful as a second baseline but hurts the size bonus. Available locally in `third_party/lerobot/src/lerobot/policies/{pi0,pi05}/`.

### NVIDIA GR00T-N1.5 on SO-101
- **Blog:** <https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning>
- Reference only — model size is too large for the bonus.

---

## Infrastructure & tutorials

### `vvrs/so101-playground`
- **Repo:** <https://github.com/vvrs/so101-playground> (user-provided)
- Personal SO-101 workspace wrapping the `lerobot` CLIs. Flagged WIP. **Not a model** — use as an **infrastructure reference** for data collection.
- **Inside:** `record.py`, `teleop.py`, `view_camera.py`, `eval.py`, `train_act.py`, `inspect_dataset.py`, `verify_data.py`, `CHEATSHEET.md`, `training.md`.
- Example commands (from the repo):
  ```
  python record.py --episodes 10 --task "Pick and place object"
  python view_camera.py --device 2
  ```
- Default device paths: `/dev/ttyACM0` (follower) / `/dev/ttyACM1` (leader), camera `/dev/video2`.
- **Caveat:** Linux-only device paths, very few commits, no tests, no weights — treat as recipe, not dependency.

### Phospho — "Train SmolVLA" tutorial
- **Docs:** <https://docs.phospho.ai/learn/train-smolvla>
- **YouTube:** <https://www.youtube.com/watch?v=vC7E6ZmXBT8>
- Full teleop → record → train → deploy walkthrough for SO-100 / SO-101. Good replica of the sanity-check step in [`PROJECT.md §7`](PROJECT.md#7-sanity-check-tasks-required--do-these-first).

### LeRobot Worldwide Hackathon — Firebreathing Rubber Duckies
- **Recap:** <https://www.hackster.io/news/embodied-ai-hackathon-winners-announced-2dc69c76942e>
- Two SO-ARM100s doing bowl-to-bowl pick-place (ACT / Pi0 / GR00T-N1). Good pacing reference for data collection.

---

## Base datasets

- **`lerobot/svla_so101_pickplace`** — <https://huggingface.co/datasets/lerobot/svla_so101_pickplace>
- **`lerobot/vlabench_unified`** — 3.11 M frames for VLABench reasoning tasks.
- **Community SO-101 pick-place** — see [Eval 1 section](#eval-1--put-the-banana-in-the-color-bowl-direct-color-lookup) above.

---

## Critical gotchas surfaced in prior work

- **Camera config (issue #1763).** Camera order must match between fine-tuning and deploy; no confirmed single-camera SO-101 SmolVLA success reported in public at the time of writing. This is directly relevant to our single-camera constraint — read before finalising the wrist-vs-shoulder decision.
  <https://github.com/huggingface/lerobot/issues/1763>
- **Action stats / denormalization.** Starting from a checkpoint trained on a different arm (SO-100 → SO-101, Franka → SO-101) without re-computing action stats silently produces degenerate policies. Xavier O'Keefe's writeup is the clearest walkthrough.
- **Val loss ≠ success rate.** Warned in the FLOWER-VLA README; applies to all VLA training runs. Always evaluate on the real robot (or at minimum in sim) before trusting a checkpoint.
- **Camera order at deploy.** Fine-tune-time camera ordering must be replicated at eval. Worth hard-coding a single source of truth in our training config.
