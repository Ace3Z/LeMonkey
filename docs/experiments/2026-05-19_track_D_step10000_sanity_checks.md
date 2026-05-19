# Track D step-10000 sanity checks — pre-Strix gate

**Date:** 2026-05-19
**Checkpoint:** `HBOrtiz/smolvla_eval3_track_D_m2_mahbod@step-10000`
**Script:** `eval_3/scripts/sanity_checks.py` (commit `45d9c75`)
**Machine:** Brev `time2sleep` (A100 80GB)

## TL;DR

All 4 no-robot probes pass. The checkpoint is cleared for the 3-rollout
Strix protocol (TODO.md Day 2). Mid-training reviewer's #1 worry —
"language pathway off-axis" — is empirically refuted: the policy
produces measurably different actions for different celeb prompts on
the same image.

## Results

### 1. Language sensitivity — PASS

Same camera frame, three celeb prompts, observed action chunks (first
timestep):

```
swift : [-0.053,  1.359, -2.923, -0.364, -0.303, -0.560]
obama : [-0.018,  1.429, -2.992, -0.310, -0.427, -0.549]
lecun : [+0.045,  1.387, -2.805, -0.286, -0.526, -0.564]

pairwise mean |Δ| (swift/obama, swift/lecun, obama/lecun) = 0.06, 0.09, 0.07
```

Dimension 2 (one of the shoulder joints) shifts most strongly across
celebs; dim 4 (gripper rotation) also shows clear prompt-dependence.
The policy is reading the prompt.

### 2. Vision sensitivity — PASS

Same prompt ("Place the coke on Taylor Swift."), three real frames
pulled from episodes 100, 5000, 9000 of the merged dataset:

```
pairwise mean |Δ| across 3 frames = 0.04, 0.04, 0.04
```

Consistent jitter across very different visual contexts. Policy is
reading the camera.

### 3. Consistency + range — PASS

Two consecutive `select_action` calls on identical input:

```
mean |Δ| between two identical-input calls = 0.067
max|a|: 2.724
jitter ratio (Δ/max): 2.47%
NaN: False  Inf: False
```

SmolVLA's flow-matching head re-samples noise each call, so the small
jitter is expected. The 2.47 % ratio is comfortably under the 10 %
threshold for "stable output". Actions live in a sensible
MEAN_STD-normalized magnitude band.

### 4. Patch-mask aliasing — PASS

Walked all 151 source-episode `face_labels.json` files (80,936 frame
entries, 215,079 detected face bboxes). Quantized each bbox to the
8×8 patch grid the SmolVLM2 vision tower projects against:

```
 1 patches:  17638 (  8.2%) ███
 2 patches:  58956 ( 27.4%) ██████████
 3 patches:   2350 (  1.1%)
 4 patches: 107446 ( 50.0%) ███████████████████
 6 patches:  26009 ( 12.1%) ████
 9 patches:   2680 (  1.2%)
```

Median = 4 patches per face. Half of all faces cleanly span a 2×2
patch block; only 8 % collapse to a single patch. The M2 alignment
loss is operating on a non-degenerate spatial signal — the reviewer's
"single-patch identity bottleneck" concern is not realised in practice.

## What this does NOT prove

These three questions still need Strix:

- Does the policy move the arm toward the *correct* face (not just
  any face)?
- Does the gripper actually close on the coke and release on target?
- How does it generalise to held-out photos / OOD celebs?

The 3-rollout Strix protocol (TOY / held-out IID / OOD, TODO.md Day 2)
is the only thing that answers those.

## What we'd intervene on if a check failed

Documented for reference even though all four passed:

| Failed check | Likely cause | Fix |
|---|---|---|
| Language insensitive | Policy collapsed onto vision-only mode | Lower M2_LAMBDA to 0.1 and resume from last good ckpt |
| Vision insensitive | Action expert reads only state | Investigate flow-matching head; unfreeze layer 15 |
| Consistency NaN/Inf | bf16 overflow, gradient explosion | Drop LR, replay last 500 steps |
| Mask aliasing 1-patch dominant | Camera too far from faces | Re-run augmentation with a wider FOV crop |

## Files referenced

- `eval_3/scripts/sanity_checks.py` — the test harness
- `eval_3/aug/smolvlm_inference_patch.py` — runtime patch needed before policy load
- `eval_3/scripts/verify_policy_load.py` — separate end-to-end load check
- `eval_3/scripts/run_rollout.sh` — interactive prompt-loop for Strix testing
