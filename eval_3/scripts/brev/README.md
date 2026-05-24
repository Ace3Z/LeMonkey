# brev - cloud-VM launcher kit

These scripts run **on a Brev VM** (not on the dev box). They are the launchers that produced every Eval 3 training checkpoint published under [`HBOrtiz/`](https://huggingface.co/HBOrtiz) on the Hub.

The dev-box to Brev workflow:

```
dev box                       Brev VM
-------                       -------
sync_to_brev.sh   --rsync-->  ~/LeMonkey/
                              setup_pi05.sh                 (one-time env install)
                              start_training.sh             (systemd-user wrap of one train_*.sh)
                              follow_training.sh            (live log tail)
                              training_status.sh            (one-shot snapshot)
```

## Files

### Sync + environment setup

| File | What it does |
|---|---|
| `sync_to_brev.sh` | rsync the repo + the merged eval3 dataset + the HF token to a fresh Brev VM. Run **on the dev box**. Usage: `bash sync_to_brev.sh user@host:~/LeMonkey`. |
| `setup_pi05.sh` | Conda env install. Pi0.5 needs the vendored `third_party/lerobot[smolvla,pi]`, not the PyPI build. Idempotent. |
| `setup_paligemma_warmstart.sh` | Same as above, plus `datasets` + `Pillow` for VGGFace2 loading, plus a cu128 PyTorch override for Blackwell GPUs (RTX PRO 6000 / 5090 / B100 / B200). |

### Run + monitor

| File | What it does |
|---|---|
| `start_training.sh` | Wraps the active `train_*.sh` in a transient systemd user unit (`lerobot-train-eval3.service`). Survives SSH disconnect and laptop close. Requires user lingering. |
| `follow_training.sh` | Colourised live `tail -F` of the training log + a `nvidia-smi` snapshot header. |
| `training_status.sh` | One-shot snapshot: systemd unit state, GPU util, latest progress line, last 5 events, checkpoints saved. |

### Trainers (the actual `lerobot-train` invocations)

| File | Produces (HF repo) | Notes |
|---|---|---|
| `train_smolvla_broad.sh` | [`HBOrtiz/so101_smolvla_eval3_broad`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_broad) | 192-celeb cotrain. Sets `--dataset.video_backend=pyav` to dodge the torchcodec host-RAM leak on the broad dataset (see Known issues). |
| `train_pi05.sh` | [`HBOrtiz/so101_pi05_eval3`](https://huggingface.co/HBOrtiz/so101_pi05_eval3) | Pi0.5 LoRA fine-tune from `lerobot/pi05_base` (or from the PaliGemma warm-start). 30k steps. |
| `train_paligemma_warmstart.sh` | [`HBOrtiz/paligemma_vqa_warm`](https://huggingface.co/HBOrtiz/paligemma_vqa_warm) | PaliGemma VQA LoRA warm-start (see [`../warmstart/`](../warmstart/)). |

## Quickstart (Pi0.5 reference policy, end-to-end)

```bash
# 1. Dev box: sync to the VM.
bash eval_3/scripts/brev/sync_to_brev.sh user@brev-h100-foo:~/LeMonkey

# 2. Brev VM: one-time env setup.
ssh user@brev-h100-foo
cd ~/LeMonkey
bash eval_3/scripts/brev/setup_pi05.sh                # ~15 min, idempotent

# 3. Optional warm-start track (on a second VM): see ../warmstart/README.md.

# 4. Pi0.5 action fine-tune.
bash eval_3/scripts/brev/start_training.sh            # systemd wrap of train_pi05.sh
bash eval_3/scripts/brev/follow_training.sh           # live log
```

The checkpoint is auto-pushed to `$PUSH_REPO` (default `HBOrtiz/so101_pi05_eval3`).

## Pre-flight gates (before launching any train_*.sh)

| Check | One-liner |
|---|---|
| Repo synced | `[ -d ~/LeMonkey/third_party/lerobot ] && echo ok` |
| GPU visible | `nvidia-smi` |
| `lemonkey` env active | `[ "$CONDA_DEFAULT_ENV" = lemonkey ] && echo ok` |
| HF token present | `[ -n "$HF_TOKEN" ] && echo ok` |
| User lingering on (only for `start_training.sh`) | `loginctl show-user $USER --property=Linger` reports `Linger=yes` |

## Smoke before the 24h run

SmolVLA broad:

```bash
STEPS=200 BATCH_SIZE=8 bash train_smolvla_broad.sh > smoke.log 2>&1
```

Gates: both `flow_loss` and `vqa_loss` lines appear; no host-RAM growth past 20 GB/worker in the first 10 minutes (that would be the torchcodec leak signature - confirm `--dataset.video_backend=pyav` is in the command line).

Pi0.5:

```bash
STEPS=200 BATCH_SIZE=4 bash train_pi05.sh > smoke.log 2>&1
```

Gate: `flow_loss` decreases at least 30% over the 200 steps; VRAM peak under 75 GB on an 80 GB H100.

## Known issues

- **torchcodec host-RAM leak on multi-mp4 datasets.** lerobot 0.5.1's default video backend leaks host RAM per distinct mp4 opened (roughly 10 MB / 100 iterations). On the broad dataset (8,390 mp4s) the kernel OOM-kills a DataLoader worker after about 30 minutes. Workaround: `--dataset.video_backend=pyav`. `train_smolvla_broad.sh` already sets this. Upstream report drafted but not filed.
- **Blackwell sm_120 GPUs need cu128 PyTorch.** `setup_paligemma_warmstart.sh` detects the card name and force-pins the cu128 wheel. `setup_pi05.sh` assumes Hopper or Ampere; on Blackwell, copy the cu128 install block from `setup_paligemma_warmstart.sh` or `torch.cuda.is_available()` returns `False`.
- **File-descriptor limit.** PyTorch DataLoader workers + mmapped parquet/video shards hit the 1024 default at around step 40. `start_training.sh` raises `LimitNOFILE=524288` on the systemd unit. If launching `bash train_*.sh` directly, `ulimit -n 524288` first.
