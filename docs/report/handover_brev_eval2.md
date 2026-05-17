# HANDOVER — Brev VM (Eval 2 SmolVLA training, RETRAIN on relabeled data)

You are picking up an in-progress robotics project. The previous session set up
this Brev VM (`daddy-sejohn`), transferred the dataset + scripts, and (if it's
a fresh VM) installed the env. **Your job is to fine-tune SmolVLA on a
compositional 180-episode dataset and report back.** This document tells you
exactly what to do.

**IMPORTANT (this is a retrain):** the first training run produced a model
(`HBOrtiz/smolvla_eval2`) that worked only ~50% of the time on left/right
spatial prompts because the original prompts were authored in user-frame but
the camera sees the workspace mirrored. The merged dataset's `meta/tasks.parquet`
has now been relabeled to camera-frame convention (left ↔ right swapped on the
57 prompts that contain those tokens; "middle"/"center"/colour-only prompts
unchanged). Local + HF Hub (`HBOrtiz/so101_eval2_all`) have the new prompts.
**The trajectories, videos, actions, and `task_index` mapping are unchanged**
— the demos themselves were always correct, only the words attached to them
were wrong. This run trains on the same demos but with corrected language.

This project follows `~/LeMonkey/CLAUDE.md`. Read it first. Two rules are
load-bearing:
- **Surface assumptions before coding** — when uncertain, ask the user.
- **No silent fallbacks** — every fallback path must `print("[WARN] ...")`
  with what was expected, what was got, what was chosen.

---

## 1. What this project is

**Eval 2 of a course robotics project.** A SO-101 6-DOF arm has to pick a
squishy banana and place it into one of 3 colored bowls (blue / red / green),
**but the bowls are reshuffled across positions** between trials and the
prompts are *compositional* rather than direct color lookups. Examples from
the course brief (`docs/PROJECT.md` §2):

> *"Put the banana into the 2nd bowl from the left."*
> *"Put the banana into the bowl on the right of the red bowl."*
> *"Put the banana into the bowl that is not green and not blue."*

The Eval 1 policy (`HBOrtiz/smolvla_eval1_v2`) does NOT solve this. We have
empirical evidence:

- `probe_language_conditioning.py` on v2/25k: strong color-word conditioning
  (`wrong_color ≈ 57`) but bad phrasing-overfit (`paraphrase ≈ 61`).
- `probe_compositional.py` on v2/25k: pairwise distances on spatial / ordinal /
  relational / negation prompts are 5–11 — i.e. **v2 is essentially blind to
  compositional structure**, 5–10× weaker than its color-word signal.

Per `PROJECT.md` §3 different models are explicitly allowed across evals. So
Eval 2 trains its own policy.

---

## 2. The strategy you're executing

**Fine-tune SmolVLA from `lerobot/smolvla_base`** (NOT a warm-start from
v2/25k). Reasons:

1. v2 was trained on 153 episodes with bowls in fixed positions — it likely
   learned `color → position` shortcuts. Eval 2 demands `color → visual
   appearance`, which means the position bias has to be unlearned. Starting
   from a clean prior avoids that fight.
2. v2 is overfit to its 13 specific Eval-1 phrasings. The probe shows even
   paraphrases are as disruptive as wrong-color prompts. From-scratch on the
   180-ep diverse dataset gives the model a clean phrasing prior.
3. The frozen `SmolVLM2` backbone is identical between `smolvla_base` and
   v2 — there's no language-understanding gain to inherit from v2.

**Dataset**: 180 teleop episodes, 107,820 frames, 123 distinct compositional
prompts. Balanced 6 arrangements × 6 prompt families × 5 reps. Already merged
into one LeRobot v3 dataset and uploaded to HF Hub. See §3 below.

**Recipe** mirrors `eval_1/v2`'s successful run (which converged smoothly to
loss 0.023 in 25k steps):
- `lerobot/smolvla_base` from-scratch
- 25k steps, batch 192, save_freq 5000
- `image_transforms.enable=true` — color jitter only (DO NOT ADD horizontal
  flip — it would invert "leftmost"/"rightmost" semantics)
- `empty_cameras=2`
- No `--rename_map` (eval2 dataset already uses `observation.images.camera1`)

Expected runtime: **~10 hours on H100**, ~$10–15 of Brev credit.

---

## 3. Where things are on this VM

```
~/LeMonkey/                                 source code (rsync'd from laptop)
├── CLAUDE.md                               behavioral guidelines (READ FIRST)
├── HANDOVER.md                             this file (rsync'd to home)
├── docs/
│   ├── PROJECT.md                          course project spec
│   └── report/
│       └── handover_brev_eval2.md          (same content as ~/HANDOVER.md)
├── eval_1/                                 prior task — leave alone
└── eval_2/                                 the working dir for this eval
    ├── README.md                           per-folder layout & rationale
    └── scripts/
        ├── record_eval2.py                 robot-side, irrelevant on this VM
        ├── merge_eval2_episodes.py         already ran on the laptop — output's on HF
        └── brev/                           ← scripts you'll run here
            ├── run_training.sh             THE TRAINING COMMAND
            ├── start_training.sh           wrap it as a systemd user service
            ├── follow_training.sh          live-tail the log
            └── training_status.sh          one-shot snapshot

~/run_training.sh, ~/start_training.sh, ~/follow_training.sh, ~/training_status.sh
                                            (the same 4 scripts, copied to home for convenience)

~/miniconda3/                                conda installation (env: lemonkey)
~/outputs/train/smolvla_eval2/               will be created by training output
~/.cache/huggingface/                        HF Hub cache + token (already there)
```

Conda env: **always `conda activate lemonkey`** before running anything.
Python binary: `~/miniconda3/envs/lemonkey/bin/python`.

**Username note**: `run_training.sh` hardcodes `/home/shadeform/...` paths
(matches Brev's default username on most images). If your VM user is **NOT**
`shadeform` (run `whoami` to check), edit those two paths in
`~/run_training.sh` to match `$HOME` before launching.

---

## 4. Hardware + software state to verify

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lemonkey
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
hf auth whoami
```

Expected:
```
NVIDIA H100 (or H100 PCIe, 80 GB)
2.10.0+cu128 True 0.5.1
user=HBOrtiz
```

If any of those fail:
```bash
bash ~/LeMonkey/eval_1/scripts/brev_setup.sh   # idempotent — installs miniconda + lerobot==0.5.1 + ffmpeg
hf auth login                                   # paste write token if prompted
```

Also enable lingering so training survives SSH disconnect:
```bash
sudo loginctl enable-linger $USER
```

---

## 5. WHAT YOU NEED TO DO

### Step 1 — Pull the dataset from HF Hub (~5 min, ~600 MB)

```bash
mkdir -p ~/LeMonkey/datasets/eval2_merged
hf download HBOrtiz/so101_eval2_all \
  --repo-type=dataset \
  --local-dir ~/LeMonkey/datasets/eval2_merged
```

**Verify:**
```bash
python -c "
import json
m = json.load(open('/home/$USER/LeMonkey/datasets/eval2_merged/meta/info.json'))
print(f'  episodes : {m[\"total_episodes\"]}')
print(f'  frames   : {m[\"total_frames\"]}')
print(f'  fps      : {m[\"fps\"]}')
print(f'  features : {list(m[\"features\"].keys())}')
"
```

Expected:
```
  episodes : 180
  frames   : 107820
  fps      : 30
  features : ['action', 'observation.state', 'observation.images.camera1', ...]
```

### Step 2 — Launch training (~10h on H100, runs in background)

```bash
chmod +x ~/run_training.sh ~/start_training.sh ~/follow_training.sh ~/training_status.sh
~/start_training.sh
```

This launches `lerobot-train` as a transient systemd user service named
`lerobot-train-eval2`, so it survives SSH disconnect. Output goes to:
- `~/outputs/train/smolvla_eval2.log` — full training log
- `~/outputs/train/smolvla_eval2/checkpoints/` — saved every 5000 steps
- `~/outputs/train/smolvla_eval2/checkpoints/last/` — symlink to the latest

### Step 3 — Monitor

```bash
~/follow_training.sh        # live colored progress (Ctrl-C to detach; training keeps going)
~/training_status.sh        # one-shot snapshot
```

**What to watch for:**
- Step time should be ~1.4 s/step on H100. Total ~25,000 × 1.4 ≈ 9.7 h.
- `loss:` should drop smoothly. v2's reference curve was: step 5k=0.075,
  10k=0.041, 15k=0.028, 20k=0.024, 25k=0.023. Eval 2's curve may differ
  (different data) but should plateau by step 20k.
- Checkpoints `005000/`, `010000/`, ..., `025000/` appear every ~2h.
- **If a step takes >10 s, something's wrong** (CPU bottleneck, swap, etc.).

### Step 4 — Report and stop

After training finishes (or after step 25k, whichever first), report:
```bash
echo "=== final checkpoints ==="
ls -d ~/outputs/train/smolvla_eval2/checkpoints/*/

echo "=== last 30 log lines ==="
tr '\r' '\n' < ~/outputs/train/smolvla_eval2.log | tail -30

echo "=== any [WARN] / Error / Traceback lines? ==="
tr '\r' '\n' < ~/outputs/train/smolvla_eval2.log | grep -E '\[WARN\]|Traceback|Error|RuntimeError|CUDA out of memory' | head -30
```

The user will rsync the checkpoints back to their laptop, push them to HF Hub
as `HBOrtiz/smolvla_eval2`, and run the real-robot evals.

**Don't push to HF Hub yourself.** The user uploads carefully.

---

## 6. Important context & gotchas

**Don't `pip install -e third_party/lerobot/`.** That submodule is a partial
copy missing `lerobot.datasets`. The PyPI install (`lerobot[smolvla]==0.5.1`,
already done by `brev_setup.sh`) is what works.

**`run_training.sh` hardcodes `/home/shadeform/...`.** If `whoami` returns a
different name on your VM, edit the two lines that reference
`/home/shadeform/LeMonkey/datasets/eval2_merged` and
`/home/shadeform/outputs/train/smolvla_eval2` before launching.

**Image augmentation: color-only, no flip.** The training config has
`image_transforms.enable=true` which turns on brightness/contrast/saturation
jitter (the same config v2 used). Do NOT add horizontal flip or large
rotations — Eval 2 prompts contain spatial language ("leftmost", "right of
red") that flips would invert. Same for large random crops that could remove
a bowl.

**The user's HF token sits at `~/.cache/huggingface/token`.** It's a write
token. Don't print it. Don't push it anywhere. Don't `git add` it.

**Disk usage matters.** `/ephemeral` typically has 700 GB free, `/` ~80 GB.
Outputs land under `~` which is on `/`. The training output dir grows to
~5 GB (5 checkpoints × ~907 MB each). If `df` shows `/` filling up, move
older checkpoints to `/ephemeral` and update the symlink.

---

## 7. Things you should NOT do without the user's ok

- **Don't push to HF Hub** — the user uploads carefully and verifies.
- **Don't modify** `eval_1/scripts/`, `eval_2/scripts/`, or `third_party/lerobot/`
  unless fixing a bug found during training.
- **Don't terminate the VM** — that's the user's call (it's billed by hour).
- **Don't run rollout / eval scripts** here — there's no robot attached.
- **Don't do exploratory training runs** without telling the user. Each run
  costs Brev credit.

---

## 8. References

- **Project repo:** https://github.com/Ace3Z/LeMonkey
- **Course project spec:** `~/LeMonkey/docs/PROJECT.md`
- **Eval 2 README:** `~/LeMonkey/eval_2/README.md`
- **HF Hub dataset:** [HBOrtiz/so101_eval2_all](https://huggingface.co/datasets/HBOrtiz/so101_eval2_all) (private)
- **HF Hub model (v2 reference, NOT used here):**
  [HBOrtiz/smolvla_eval1_v2](https://huggingface.co/HBOrtiz/smolvla_eval1_v2)
- **HF Hub model (will be created after this run):** `HBOrtiz/smolvla_eval2`

---

## 9. Quick-start one-liner for the impatient

After verifying you're in the env (`conda activate lemonkey`):

```bash
# 1. fetch the dataset
mkdir -p ~/LeMonkey/datasets/eval2_merged && \
hf download HBOrtiz/so101_eval2_all --repo-type=dataset \
  --local-dir ~/LeMonkey/datasets/eval2_merged && \

# 2. enable lingering so training survives ssh disconnect
sudo loginctl enable-linger $USER && \

# 3. launch training as a systemd user service
chmod +x ~/run_training.sh ~/start_training.sh ~/follow_training.sh ~/training_status.sh && \
~/start_training.sh
```

Then watch progress with `~/follow_training.sh`.

After it finishes (~10h), run:
```bash
echo "=== final checkpoints ==="
ls -d ~/outputs/train/smolvla_eval2/checkpoints/*/
echo "=== last 30 log lines ==="
tr '\r' '\n' < ~/outputs/train/smolvla_eval2.log | tail -30
echo "=== any WARN/Error? ==="
tr '\r' '\n' < ~/outputs/train/smolvla_eval2.log | grep -E '\[WARN\]|Traceback|Error|CUDA out of memory' | head
```

…and report the output to the user.
