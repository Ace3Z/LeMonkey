# HANDOVER — Brev VM (residual SmolVLA training)

You are picking up an in-progress robotics project. The previous session set up
this Brev H100 VM, transferred the source code, and installed the env. **Your
job is to train a residual policy head, evaluate it locally, and report back.**
This document tells you exactly what to do.

This project follows `~/LeMonkey/CLAUDE.md`. Read it first. Two rules are
load-bearing:
- **Surface assumptions before coding** — when uncertain, ask the user.
- **No silent fallbacks** — every fallback path must `print("[WARN] ...")`
  with what was expected, what was got, what was chosen.

---

## 1. What this project is

**Eval 1 of a course robotics project.** A SO-101 6-DOF arm has to pick a
squishy banana and place it in a colored bowl (blue / red / green) following a
natural-language prompt:

> *"Put the banana in the blue colored bowl."*

The policy is **SmolVLA** (HuggingFace, 450M params), fine-tuned from
`lerobot/smolvla_base` on 118 SO-101 teleoperation episodes. That's the **base
model** (`HBOrtiz/smolvla_eval1` on HF Hub). It works in the central case but
**fails when the banana is placed in corner positions** — the policy approaches
but can't grasp from those off-distribution states.

This is canonical **compounding-error** failure (Ross et al. 2011). To fix it,
we collected ~35 HG-DAgger correction episodes (where a human took over the
leader arm at failure points) and we want to train a small **residual policy**
on top of the frozen base, per CR-DAgger (arXiv 2506.16685, Wang et al. 2025).

The recent literature finding that drove the architecture choice:

> Fine-tuning the base policy on DAgger data **dropped success by 30%**.
> Retraining from scratch on union data was tied with residual.
> **Residual policy was the best approach**, beating both alternatives.

So: keep the base frozen, train a small additive correction net.

---

## 2. The strategy you're executing

Full design doc: **`~/LeMonkey/docs/report/residual_strategy.md`** — read it
before doing anything. It covers architecture decisions, hyperparameters, the
literature it's based on, risks, and the verification plan.

**Architecture**: the residual is a small MLP (~384K params).

```
inputs:
  image_features (960)  ← mean-pooled SmolVLM2 vision-encoder output
  state          (6)    ← current joint positions
  base_action    (6)    ← what the base SmolVLA wants to do next
output:
  residual       (6)    ← joint-position delta, clipped to ±5°/joint, ±10 gripper

inference:
  final_action = base_action + clip(residual)
```

**Trained only on DAgger data** (35 episodes / ~13k frames). Per-step residual.
Frozen base. AdamW + cosine LR. Intervention frames upweighted 2× in the loss.

---

## 3. Where things are on this VM

```
~/LeMonkey/                                 source code (rsync'd from laptop)
├── CLAUDE.md                               behavioral guidelines (READ FIRST)
├── HANDOVER.md                             this file
├── docs/
│   ├── SETUP.md                            (if present) earlier-session VM runbook — useful reference
│   ├── PROJECT.md                          course project spec
│   └── report/                             (gitignored locally) strategy docs
│       ├── residual_strategy.md            ← READ THIS
│       └── dagger_strategy.md              earlier-stage strategy
├── eval_1/                                 the working dir for this eval
│   ├── README.md                           per-folder layout
│   └── scripts/                            ← all runnables
│       ├── brev_setup.sh                   bootstrap installer (already ran)
│       ├── compare_evals.py                CSV aggregator
│       ├── (other eval+rollout scripts, robot-side — irrelevant on this VM)
│       └── residual/                       ← the residual subsystem
│           ├── residual_head.py            MLP module
│           ├── train_residual.py           THE TRAINING SCRIPT YOU'LL RUN
│           ├── inference_residual.py       ResidualWrapper for rollouts
│           └── eval_residual.py            30-rollout evaluator (used on laptop)
├── third_party/lerobot/                    DO NOT pip install -e from this. It's
│                                           an incomplete copy (missing
│                                           lerobot.datasets package). PyPI 0.5.1
│                                           is what's actually installed.
└── datasets/                               will be created by you when you
                                            download the dagger datasets

~/miniconda3/                                conda installation (env: lemonkey)
~/outputs/residual/                          will be created by training output
~/.cache/huggingface/                        HF Hub cache + token (already there)
```

**Always do `conda activate lemonkey`** before running anything. The python
binary you want is `~/miniconda3/envs/lemonkey/bin/python`.

---

## 4. Hardware + software state (verified at handover)

```
GPU       : NVIDIA H100 PCIe (80 GB)
VM        : brev-3qdr52lzl, Ubuntu 22.04, 28 vCPU, 177 GB RAM, 700 GB /ephemeral
Python    : 3.12.13 (env: lemonkey)
Torch     : 2.10.0+cu128 (CUDA available)
LeRobot   : 0.5.1 from PyPI
HF auth   : logged in as `HBOrtiz`
```

Verify all of the above with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lemonkey
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
hf auth whoami
```

If any of those fail, run `bash ~/LeMonkey/eval_1/scripts/brev_setup.sh` to
re-bootstrap (it's idempotent).

---

## 5. WHAT YOU NEED TO DO

### Step 1 — Pull data + base checkpoint from HF Hub (~10 min, ~1 GB total)

```bash
cd ~/LeMonkey
mkdir -p datasets/eval1_dagger

# 3 dagger datasets (the correction episodes)
for c in blue red green; do
  hf download HBOrtiz/so101_eval1_dagger_${c} --repo-type=dataset \
    --local-dir datasets/eval1_dagger/${c}
done

# Base SmolVLA model (the frozen one we'll add the residual to)
mkdir -p eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model
hf download HBOrtiz/smolvla_eval1 --repo-type=model \
  --local-dir eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model
```

**Verify:**

```bash
for c in blue red green; do
  python -c "
import json
m = json.load(open('datasets/eval1_dagger/$c/meta/info.json'))
print(f'  $c: {m[\"total_episodes\"]} episodes, {m[\"total_frames\"]} frames')
"
done
ls -la eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model/model.safetensors
```

Expected output:
```
  blue: 11 episodes, 4211 frames
  red: 12 episodes, 4572 frames
  green: 12 episodes, 4507 frames
... model.safetensors  ~907M
```

### Step 2 — Train the residual (~25-30 min on H100)

```bash
mkdir -p ~/outputs/residual

python eval_1/scripts/residual/train_residual.py \
  --dataset-root ~/LeMonkey/datasets/eval1_dagger/blue \
  --dataset-root ~/LeMonkey/datasets/eval1_dagger/red \
  --dataset-root ~/LeMonkey/datasets/eval1_dagger/green \
  --policy-path ~/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model \
  --out ~/outputs/residual \
  --steps 5000 \
  --batch-size 32 \
  --lr 3e-4 \
  --intervention-weight 2.0
```

**What to watch for in the logs:**

- A header lists the 3 datasets, ~13,290 total frames, ~384K trainable params.
- Each `--log-every` step (default 20) prints loss + per-class MSE:
  - `int_mse` (intervention frames) starts much higher than `noint_mse`
    (non-intervention frames). They should both DECREASE over training.
  - If `int_mse` doesn't decrease → bug or bad data.
  - If `noint_mse` stays at 0 → maybe no non-intervention frames in batches
    (could be fine, but suspicious — every 20 steps you should see n>0 in
    `noint_mse=… (n=N)`).
- Checkpoints save every `--save-every 500` steps (default) to
  `~/outputs/residual/step_NNNNNN/`. Final saved as `~/outputs/residual/last/`.
- Total runtime ~25-30 min on H100. **If a step takes >10s, something's
  wrong** — the H100 should do bs=32 in 200-500ms.

**Failure modes to watch for and report:**

- `KeyError: 'observation.images.front'` → image-key mismatch with dataset. The
  script handles `camera1` and `front` automatically — if it errors, add a
  `[WARN]` print and report which dataset's keys are unexpected.
- `RuntimeError: CUDA out of memory` → drop `--batch-size` to 16 or 8.
- `[WARN]` lines appearing every step → likely real bugs; collect and report.
- Loss explodes (NaN, or huge spikes at every step) → bad data or LR too high.
  Try `--lr 1e-4`.

### Step 3 — Report results and stop the VM

After training finishes, **don't run further automated steps.** Report back to
the user with:

```bash
# 1. Final checkpoint location + size
ls -la ~/outputs/residual/last/
du -sh ~/outputs/residual/last/

# 2. Last 30 log lines from training (capture them in a tee or check the
#    saved log if you ran with one). Useful for the user to see convergence
#    pattern.

# 3. Brief summary: did int_mse drop? did noint_mse stay near 0?
#    Any [WARN] lines that fired?
```

**The user will pull the checkpoint back to their laptop and evaluate on the
real SO-101 robot** (you don't have hardware here). The eval script is
`eval_1/scripts/eval_residual.py` which runs on their laptop with the robot
plugged in.

---

## 6. Important context & gotchas

**Don't `pip install -e third_party/lerobot/`.** That submodule is a partial
copy missing `lerobot.datasets`. The PyPI install is what works (already done).

**The base SmolVLA is frozen.** `train_residual.py` sets all base params to
`requires_grad=False` and the optimizer only sees `residual.parameters()`. If
you find a path that updates the base by accident, that's a bug — flag it.

**Image preprocessing must match SmolVLA exactly.** `_pad_to_512` in
`train_residual.py` and `_extract_image_features` in `inference_residual.py`
both right+bottom-pad (matching `lerobot/policies/smolvla/modeling_smolvla.py:134`).
DO NOT change these to center-pad — that was a bug that an earlier review
caught.

**`base.reset()` is called per-frame at training AND inference.** This is
critical — it ensures the residual sees `chunk_idx=0` base predictions
consistently. Don't optimize this away.

**Per-frame reset costs ~700ms on a 1660 SUPER (deployment).** The user knows
this. On the H100 here, training does ~200-500ms/step which is fine.

**The user's HF token sits at `~/.cache/huggingface/token`.** It's a write
token. Don't print it. Don't push it anywhere. Don't `git add` it. The user
will rotate it when they're done.

**Disk usage matters.** `/ephemeral` has 700 GB free, `/` has 80 GB. Outputs
go under `~` which is on `/`. If you need to cache big things, use
`/ephemeral`.

---

## 7. Things you should NOT do without the user's ok

- **Don't push to HF Hub** unless explicitly asked. The user uploads carefully.
- **Don't modify** `third_party/lerobot/` or `eval_1/scripts/` files unless
  fixing a bug found during training. The user reviewed and committed them.
- **Don't terminate the VM.** That's the user's call (it's billed by hour).
- **Don't run the eval rollout scripts** on this VM — there's no robot
  attached. They'll fail.
- **Don't do exploratory training runs** without telling the user first. Each
  run costs H100 credit.

---

## 8. References

- **Project repo:** https://github.com/Ace3Z/LeMonkey
- **Residual strategy doc:** `~/LeMonkey/docs/report/residual_strategy.md`
- **DAgger strategy doc:** `~/LeMonkey/docs/report/dagger_strategy.md`
- **Course project spec:** `~/LeMonkey/docs/PROJECT.md`
- **Earlier-session VM runbook:** `~/LeMonkey/docs/SETUP.md` (if rsync'd)
- **HF Hub model:** https://huggingface.co/HBOrtiz/smolvla_eval1 (private)
- **HF Hub datasets:** `HBOrtiz/so101_eval1_{blue,red,green,dagger_blue,dagger_red,dagger_green}`
- **CR-DAgger paper:** https://arxiv.org/html/2506.16685v5
- **Reference implementations:**
  - https://github.com/yifan-hou/cr-dagger
  - https://github.com/tongzhoumu/policy_decorator
  - https://github.com/ankile/robust-rearrangement

---

## 9. Quick-start one-liner for the impatient

After verifying you're in the env (`conda activate lemonkey`):

```bash
cd ~/LeMonkey && \
mkdir -p datasets/eval1_dagger eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model ~/outputs/residual && \
for c in blue red green; do hf download HBOrtiz/so101_eval1_dagger_${c} --repo-type=dataset --local-dir datasets/eval1_dagger/${c}; done && \
hf download HBOrtiz/smolvla_eval1 --repo-type=model --local-dir eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model && \
python eval_1/scripts/residual/train_residual.py \
  --dataset-root datasets/eval1_dagger/blue \
  --dataset-root datasets/eval1_dagger/red \
  --dataset-root datasets/eval1_dagger/green \
  --policy-path eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model \
  --out ~/outputs/residual \
  --steps 5000 --batch-size 32 --lr 3e-4 --intervention-weight 2.0 \
  2>&1 | tee ~/outputs/residual/train.log
```

(That's data download + train + log capture in one shot.)

After it finishes, run:

```bash
echo "=== final checkpoint ==="
ls -la ~/outputs/residual/last/
du -sh ~/outputs/residual/last/

echo "=== last 30 log lines ==="
tail -30 ~/outputs/residual/train.log

echo "=== any WARN lines? ==="
grep '\[WARN\]' ~/outputs/residual/train.log | head -20
```

…and report the output to the user.
