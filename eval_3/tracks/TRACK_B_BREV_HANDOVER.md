# Track B — Brev handover (step-by-step)

Owner: Roham · Goal: launch Pi0.5-LoRA training on Brev's RTX PRO 6000 Blackwell VM.
Total wall-time once running: ~24 h.

This doc is the **end-to-end runbook** for spinning up a fresh Brev VM, syncing
code/token, and launching Track B. Everything is copy-paste.

## Why a separate handover doc

You ([`eval_3/tracks/TRACK_B.md`](TRACK_B.md)) are the *what + why*. This file is
the *exact commands in order*. The 24 h Pi0.5 run on Brev is the longest single
job in our 4-day sprint; we shouldn't lose ~30 min on Brev fumbling.

---

## 0 · Prerequisites (on dev box, before anything)

```bash
cd /home/rohamzn/ETH_Uni/LeMonkey
git checkout track-b-pi05
git pull origin track-b-pi05      # be sure you have the latest commit
```

Verify the four pieces are in place:

```bash
ls -1 eval_3/tracks/TRACK_B.md \
       eval_3/scripts/brev/run_training_track_B.sh \
       eval_3/scripts/brev/sync_to_brev.sh \
       secrets/huggingface/token_hbortiz
```

All four should exist. If any are missing, you're on the wrong branch.

Verify the HF token has `write` access to `HBOrtiz/*`:

```bash
conda run -n lemonkey python -c "
from huggingface_hub import HfApi
t = open('secrets/huggingface/token_hbortiz').read().strip()
print(HfApi(token=t).whoami())
"
```

Expected output should include `'auth': {'accessToken': {'role': 'write', ...}}`.

---

## 1 · Provision the Brev VM

**Spec:** RTX PRO 6000 Blackwell, 96 GB VRAM. ~$2/h. Region: EU if available
(lower latency for HF Hub uploads).

In the Brev dashboard ([console.brev.dev](https://console.brev.dev)):

1. **Create instance** → "Custom" → GPU = RTX PRO 6000 Blackwell, RAM ≥ 64 GB,
   disk ≥ 100 GB.
2. **OS:** Ubuntu 22.04 (default).
3. **Pre-installed**: select the "PyTorch 2.x + CUDA 12.4" image if available;
   otherwise use the bare Ubuntu and we'll install CUDA via the
   `brev_setup.sh` helper.
4. Wait ~3–5 min for the instance to come up.
5. Note the SSH endpoint: `shadeform@<brev-host>`. Test:
   ```bash
   ssh shadeform@<brev-host> 'nvidia-smi'
   ```
   You should see one RTX PRO 6000 Blackwell row with 96 GB VRAM.

**On the Brev VM (one-time):**

```bash
# Enable systemd user lingering so background services survive your SSH disconnect.
sudo loginctl enable-linger shadeform
loginctl show-user shadeform --property=Linger      # expect: Linger=yes
```

---

## 2 · Sync repo + token from dev box → Brev

This pushes the LeMonkey code, the lerobot fork (submodule), and the HF token
to the VM. Skips heavy artefacts (datasets, outputs, .git).

```bash
# from dev box, repo root
bash eval_3/scripts/brev/sync_to_brev.sh shadeform@<brev-host>:~/LeMonkey
```

The script (verified at [`eval_3/scripts/brev/sync_to_brev.sh`](../scripts/brev/sync_to_brev.sh)):

- syncs the LeMonkey code (rsync, excludes `.git/`, `datasets/`, `outputs/`, `wandb/`)
- syncs the merged eval3 dataset locally (this dates from earlier image-as-prompt work and is NOT what Track B needs; harmless to leave)
- syncs `secrets/huggingface/token_hbortiz` so the policy push at end-of-training works

**Important:** Track B does NOT need any local dataset — Pi0.5 training reads
`HBOrtiz/so101_eval3_track3_v3_pi05` directly from HF. So if you want to skip
the dataset rsync to save time, you can comment out the `[2/3]` step in
`sync_to_brev.sh`. Not required.

---

## 3 · Environment setup on Brev (one-time per VM)

```bash
ssh shadeform@<brev-host>
cd ~/LeMonkey

# Install conda env + lerobot in editable mode (idempotent)
bash eval_1/scripts/brev_setup.sh
```

`brev_setup.sh` installs miniconda if missing, creates the `lemonkey` env from
`environment.yml`, and pip-installs `third_party/lerobot` in editable mode.
Takes ~5–10 min on a fresh VM.

Verify after install:

```bash
conda activate lemonkey
python -c "import lerobot; print(lerobot.__file__)"
nvidia-smi | head -20
```

The lerobot import should resolve to `/home/shadeform/LeMonkey/third_party/lerobot/src/lerobot/...`.

---

## 4 · Pre-flight on Brev (validate the training won't crash in 10 sec)

```bash
cd ~/LeMonkey
conda activate lemonkey

# Quick dataset probe — pulls the meta files from HF, verifies the shape we want
python -c "
from huggingface_hub import HfApi
api = HfApi(token=open('secrets/huggingface/token_hbortiz').read().strip())
info = api.dataset_info('HBOrtiz/so101_eval3_track3_v3_pi05')
print(f'dataset: {info.id} (public={not info.private}, files={len(info.siblings)})')
"
```

Should print `dataset: HBOrtiz/so101_eval3_track3_v3_pi05 (public=True, files=18798)`.

---

## 5 · Launch training

```bash
cd ~/LeMonkey
conda activate lemonkey

# Foreground (smoke-test, ~2 min just to see the first log line)
HF_TOKEN="$(cat secrets/huggingface/token_hbortiz)" \
    bash eval_3/scripts/brev/run_training_track_B.sh 2>&1 | head -50
# CTRL-C after you see "Compiling model..." or the first loss log

# Background / persistent (the real 24h run — survives your SSH disconnect)
nohup env HF_TOKEN="$(cat secrets/huggingface/token_hbortiz)" \
    bash eval_3/scripts/brev/run_training_track_B.sh \
    > ~/track_B.log 2>&1 &
echo $! > ~/track_B.pid
echo "Launched. PID=$(cat ~/track_B.pid). Log: ~/track_B.log"
```

The launch script ([`run_training_track_B.sh`](../scripts/brev/run_training_track_B.sh))
encodes the corrected recipe:

- `--policy.pretrained_path=lerobot/pi05_base` (Pi0.5 PaliGemma-2B start point)
- `--policy.dtype=bfloat16` (fits Brev VRAM with grad_ckpt + LoRA)
- `--policy.freeze_vision_encoder=True` (keep SigLIP frozen)
- `--policy.train_expert_only=False` (don't freeze VLM — contradicted by paper)
- `--policy.empty_cameras=2` (pi05_base has 3 cam slots; we have 1)
- `--policy.optimizer_lr=1e-5` (half of Pi0.5 default 2.5e-5)
- `--peft.method_type=LORA --peft.r=16 --peft.lora_alpha=32` (LoRA on Gemma-2B attention)
- `--dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_pi05` (exact-quantile dataset)
- `--dataset.rename_map=...camera1→right_wrist_0_rgb` (lines up with pretrained slot)
- `--steps=30000 --batch_size=24`
- pushes to `HBOrtiz/pi05_eval3_track_B` at end of training

---

## 6 · Monitor

From the dev box (no need to be on the VM):

```bash
# tail the log
ssh shadeform@<brev-host> 'tail -f ~/track_B.log'

# one-shot status snapshot
ssh shadeform@<brev-host> 'ls -lh ~/track_B.log; ps -p $(cat ~/track_B.pid) || echo dead'

# nvidia-smi sanity check
ssh shadeform@<brev-host> 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'
```

Expected milestones:
| t   | event                                                  |
|-----|--------------------------------------------------------|
| 0   | conda activate, lerobot-train fires                    |
| ~1m | "Compiling model..." (first compile pass)              |
| ~3m | first batch loaded, first forward+backward             |
| ~5m | first loss log line. should be 2-5 (flow-matching MSE) |
| ~24h| training finishes, policy pushed to HF                 |

**Red-flag signs you should KILL and fix:**

- Loss is NaN within 100 steps → likely quantile clipping issue.
  Inspect `meta/stats.json` on hub vs local, restart with `--policy.optimizer_lr=5e-6`.
- VRAM usage hits 96 GB → OOM coming. Drop `--batch_size=16` and restart.
- "missing key observation.images.camera1" → rename_map didn't apply. Check the script's CLI quoting.

---

## 7 · After training finishes

```bash
# verify push succeeded
conda run -n lemonkey python -c "
from huggingface_hub import HfApi
import os
t = open('/home/rohamzn/ETH_Uni/LeMonkey/secrets/huggingface/token_hbortiz').read().strip()
print(HfApi(token=t).model_info('HBOrtiz/pi05_eval3_track_B'))
"
```

Then hand off to **Darius** for Strix deployment per the standard 3-rollout
protocol ([`TODO.md`](../../TODO.md#strix-testing-protocol-darius)).

---

## 8 · Things you can do in parallel while training runs (24 h)

- **Ping Hans + Sejohn** to launch Tracks A and C — both can use the SmolVLA
  baseline dataset directly (no quantile recompute needed for them). Confirmed
  via code-+-data check at [`docs/experiments/2026-05-19_track_b_validations.md`](../../docs/experiments/2026-05-19_track_b_validations.md).
- **Upload the Drive backups**: `datasets/eval3_track3_aug.tar.zst` (14 GB) and
  `datasets/eval3_track3_celebs.tar.zst` (19 MB).
- **Update the team in Slack**: dataset URLs, Brev VM started, ETA.
- **Coordinate with Darius** on Strix prep — he needs the SO-101 + wrist cam set
  up tomorrow morning for testing the SmolVLA tracks first, then Pi0.5 Track B
  ~24 h later.

---

## Citations

- [`TRACK_B.md`](TRACK_B.md) §3 (validation findings), §4 (per-flag reasoning)
- [`docs/experiments/2026-05-19_track_b_validations.md`](../../docs/experiments/2026-05-19_track_b_validations.md) — agent transcripts
- [`docs/report/EVAL_3_FINAL_PLAN.html`](../../docs/report/EVAL_3_FINAL_PLAN.html) §3 (canonical 4-day plan)

---

*Last updated 2026-05-19. Status: ready to launch. Branch: `track-b-pi05`.*
