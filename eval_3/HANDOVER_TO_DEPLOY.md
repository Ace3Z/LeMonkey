# Handover from dev box → deploy machine

**Date:** 2026-05-17
**From:** the Claude session that ran on the dev box `rohamzn` (training + dataset + recipe work)
**To:** the Claude session reading this on whatever machine the SO-101 is connected to (Thor or the Predator deploy laptop)

> **First action:** `cd ~/LeMonkey && git pull origin main` — almost everything below assumes you're on the latest `main`. The work was on the dev box; pulling brings you to commit `bd5abf1` or newer.

## State of the world

We're at the point where Eval 3's **dataset is locked, the policy is trained, and the rollout script is the only thing left to build**.

| Artifact | Where it lives | Notes |
|---|---|---|
| Augmentation pipeline | `eval_3/aug/` | Done, ran 2026-05-14/15, produced 4017 inpainted variants. Don't re-run. |
| Augmented + base merged dataset | [`HBOrtiz/so101_eval3_all`](https://huggingface.co/datasets/HBOrtiz/so101_eval3_all) (public) | 4195 episodes, 933 unique prompts, 2.26M frames, 7.5 GB. Schema: `observation.images.camera1` (480×640 wrist) + `observation.images.reference` (480×480 constant-frame ref photo). |
| Trained SmolVLA policy | [`HBOrtiz/smolvla_eval3`](https://huggingface.co/HBOrtiz/smolvla_eval3) | 450M params, 30k steps, 6 checkpoints (5k–25k under `checkpoints/`, 30k at root for `from_pretrained()`). Final loss 0.018. |
| Training recipe + brev scripts | `eval_3/scripts/brev/` | Locked. Includes `TORCHCODEC_OOM_REPORT.md` capturing the only major incident we hit. |
| 50-celebrity sample PDF + zip | `docs/Eval_3_Sample_Celebrity_Images.{pdf,zip,index.txt}` | A5 portraits, includes the 3 IID OOD photos. Can be printed to physically extend the eval set. |

## What's NOT yet built

**The rollout script for Eval 3.** `eval_2/scripts/run_rollout.sh` is the natural template, but it loads a single-camera SmolVLA — Eval 3 needs two image streams. The model expects:

- `observation.images.camera1` ← live wrist camera (existing eval_2 wiring works as-is)
- `observation.images.camera2` ← **constant-frame reference photo of the target celebrity** (this is the new thing)
- `observation.images.camera3` ← empty (`--policy.empty_cameras=1` fills it at train time; at inference the policy expects 3 image slots and a zero-tensor for camera3 is fine)

The `rename_map` we used at training time was `{"observation.images.reference": "observation.images.camera2"}`. The policy's saved config in `HBOrtiz/smolvla_eval3` reflects this — it expects keys named `camera1`, `camera2`, `camera3` (the latter empty).

Per-rollout flow:
1. Operator picks a target celebrity (CLI arg / prompt input).
2. Script loads the reference photo for that celebrity (e.g. from `~/LeMonkey/docs/Eval_3_Sample_Celebrity_Images.zip` → `images/NN_<slug>.<ext>`, or any external photo).
3. Resize/letterbox the reference photo to 480×480 (the training resolution).
4. Build the text prompt — same format as training: `"Set the coke down on <Display Name>'s picture."` (or any of the 933 templates — they're synonymous to the model now).
5. Start a 20 s rollout: at every step, the policy gets (camera1 = live wrist frame, camera2 = the same reference photo, camera3 = zeros, prompt = text) → returns a 6-DOF action chunk → SO-101 follower follows.

The whole 9-rollout eval is then: rotate through 9 reference photos / prompts (3 known-IID + 3 held-out-IID + 3 OOD).

## What to do, in order

```bash
# 0. Pull
cd ~/LeMonkey && git pull origin main

# 1. (If env not already installed) install lemonkey env on this machine.
#    Use eval_1/scripts/brev_setup.sh ONLY if you're on a Brev VM — on Thor/Predator
#    the env should already exist from prior eval_1/eval_2 deployments.
#    Just activate it:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lemonkey

# 2. Pull the trained policy from HF. Use whatever HF token you already have
#    set up — the policy repo is public, so even an anonymous pull works.
hf download HBOrtiz/smolvla_eval3 --local-dir ~/LeMonkey/eval_3/train/smolvla_eval3
# Final checkpoint will be at ~/LeMonkey/eval_3/train/smolvla_eval3/ (root files)
# Intermediates under checkpoints/{005000..025000}/

# 3. Sanity-load the policy (no robot, no camera):
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
p = SmolVLAPolicy.from_pretrained('~/LeMonkey/eval_3/train/smolvla_eval3')
print('image_features:', list(p.config.image_features))
print('empty_cameras :', p.config.empty_cameras)
print('expects keys  :', p.config.image_features)
"
# expected keys: ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']
# empty_cameras=1 (camera3 is padded with zeros at inference)

# 4. (THE BIG TODO) write eval_3/scripts/run_rollout_eval3.py.
#    Best template: eval_2/scripts/run_rollout.sh + run_rollout_eval2.py.
#    Diffs vs eval_2:
#      - load + resize a reference photo (CLI arg) to 480x480
#      - construct observation dict with camera1 (live), camera2 (constant ref),
#        camera3 (torch.zeros)
#      - prompt template: "Set the coke down on <Display Name>'s picture."
#      - everything else (SO-101 wiring, fps, 20-s cap, etc.) is identical
```

## Files you should read on this machine (in order)

1. `~/.claude/projects/-home-<user>-LeMonkey/memory/MEMORY.md` — index of all memory entries
2. `~/.claude/projects/-home-<user>-LeMonkey/memory/project_eval3_handover_20260515.md` — the training-time handover; the recipe rationale and the SmolVLA-config gotchas (`empty_cameras=1`, `add_image_special_tokens=true`, the `rename_map`) are all in there
3. `~/.claude/projects/-home-<user>-LeMonkey/memory/project_eval3_status.md` — current phase status
4. `eval_3/README.md` — the project's source-of-truth doc (mostly still accurate, but the "Phase 4 (rollout)" section is what you're now actually building)
5. `eval_3/aug/STRATEGY_v3.md` — augmentation strategy. Useful for understanding what the model has actually seen.
6. `eval_2/scripts/run_rollout.sh` + `eval_2/scripts/run_rollout_eval2.py` — the rollout template to adapt
7. `eval_3/scripts/brev/run_training.sh` — the recipe we trained with (line-commented; tells you exactly which keys the policy expects)
8. `eval_3/scripts/brev/TORCHCODEC_OOM_REPORT.md` — only relevant if you re-train, but worth knowing about

## Reference photo source for rollouts

For the 9 official eval rollouts you'll need:
- **Runs 1–3 (known IID):** photos from `docs/Eval_3_TOY_Celebrity_Images.pdf` — the exact same images the TAs handed out and which the workspace prints are cut from. Use 1 per celeb.
- **Runs 4–6 (held-out IID):** photos from `datasets/eval3_celebs/heldout/{lecun,obama,swift}/` — none of these were in the training set's reference stream (they're held out specifically for this).
- **Runs 7–9 (OOD):** TBD by TAs (they publish on Slack). If they pick a celeb in our 192-celeb training pool, the model has seen *augmented* portraits of them but not *photos* of them as references — that's still mostly OOD. If they pick someone outside our pool, fully OOD.

For practice / smoke tests before the official eval, you can use any photo from `docs/Eval_3_Sample_Celebrity_Images.zip` — that bundle has 50 portraits including the 3 IID OOD photos already selected.

## Don'ts

- Don't re-run the augmentation pipeline — it took ~17h and the output is the dataset that's on HF now.
- Don't change `policy.empty_cameras` away from 1 at inference time — that's baked into the checkpoint's config and changes the expected image-feature layout.
- Don't try to use `torchcodec` for video loading on this machine without reading `TORCHCODEC_OOM_REPORT.md`. (`pyav` works fine.)
- Don't push experimental rollout-script work directly to `main` — branch off and PR.

## Open questions for the user

If anything below is unclear when you start, ask:
- Which celebrity reference photos do they want for the 9 rollouts (TA list published yet)?
- Single-rollout interactive mode vs scripted 9-rollout sequence?
- Where to save rollout videos / CSV traces (we have an `eval_3/rollouts/` convention but it's gitignored)?
- Smallest-model bonus measurement — are we counting the SigLIP + SmolLM2 + action expert all together, or just the action expert? Affects the 20-point bonus calculation; default per CLAUDE.md and the spec is *active inference parameters* = 450M total.

—

Good luck. Almost there.
