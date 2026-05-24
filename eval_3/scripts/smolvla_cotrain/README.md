# SmolVLA cotrain (deployed Eval 3 trainer)

This is the trainer that produced the two SmolVLA models deployed on Eval 3 day:

| HF repo | Recipe | Launcher |
|---|---|---|
| [`HBOrtiz/so101_smolvla_eval3_cotrain`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_cotrain) | 5:1 robot:VL, 3 in-distribution celebrities | `launch_single_gpu.sh` or `launch_multi_gpu.sh` |
| [`HBOrtiz/so101_smolvla_eval3_broad`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_broad) | 10:1 robot:VL, 192 celebrities | `eval_3/scripts/training_vm/train_smolvla_broad.sh` |

Both are SmolVLA-450M from `lerobot/smolvla_base` with the SmolVLM2 backbone trainable and SigLIP frozen.

## What the trainer does

Every `vl_ratio + 1`-th step is a vision-language VQA batch with CE loss on the SmolVLM2 LM head; the rest are robot batches with SmolVLA's flow-matching action loss. Both gradients flow into the same VLM body - the mechanism that keeps the celebrity-name prior alive across the action fine-tune. The two streams come from paired HF datasets: `HBOrtiz/so101_eval3_cotrain` (robot) and `HBOrtiz/so101_eval3_cotrain_grounding` (vision-language).

## File layout

```
train_smolvla_cotrain.py        the training script (single-process or torchrun multi-GPU)
klal_core.py                    KLALConfig + klal_loss + gaussian_target_from_mask
klal_smolvla_action.py          KLAL hookset on the robot-action forward
klal_smolvla_vl.py              KLAL hookset on the VL co-training forward (deployed)
lora_smolvla.py                 LoRA on SmolVLA VLM attention projections
launch_single_gpu.sh            single-GPU env-var-driven launcher
launch_multi_gpu.sh             multi-GPU launcher; autodetects every GPU on the node
setup_env.sh                    one-time conda env bootstrap
predownload_vl_dataset.sh       one-time VL dataset pre-download (HF rate-limit workaround)
tests/                          smoke test (test_klal_lora_smoke.py)
```

## Quickstart (single GPU)

```bash
bash setup_env.sh                  # one-time, ~15 min (idempotent)
conda activate lemonkey
export HF_TOKEN=hf_...             # read + write

# Smoke first (200 steps, small batch, no push):
STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2 LOG_EVERY=1 bash launch_single_gpu.sh 2>&1 | tee smoke.log

# Real run:
PUSH_REPO=youruser/smolvla_eval3_cotrain bash launch_single_gpu.sh
```

## Quickstart (multi-GPU cluster)

`third_party/lerobot` is a git submodule. Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/Ace3Z/LeMonkey.git
cd LeMonkey
git submodule update --init --recursive
```

Env (Python 3.12 required by lerobot 0.5.1):

```bash
conda create -y -n cotrain python=3.12 && conda activate cotrain
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e "third_party/lerobot[smolvla,dataset,av-dep]" zstandard
```

HF token + push target (write access required - checkpoints are pushed every `SAVE_FREQ` steps). `.hf_token` is gitignored:

```bash
export HF_TOKEN=hf_...                              # or echo to .hf_token
export PUSH_REPO=youruser/smolvla_cotrain_run       # auto-created
```

Run:

```bash
nohup bash launch_multi_gpu.sh > cotrain.log 2>&1 &
tail -f cotrain.log
```

First run downloads ~15 GB of datasets before training starts. Checkpoints push to HF every `SAVE_FREQ` steps as `PUSH_REPO/step_NNNNNN`.

## Pre-flight gates (all must be green before any launch)

| Check | One-liner |
|---|---|
| `HF_TOKEN` set | `[ -n "$HF_TOKEN" ] && echo ok` |
| GPU visible | `nvidia-smi` |
| lerobot SmolVLA imports | `python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy"` |
| Robot dataset loads | `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; print(len(D('HBOrtiz/so101_eval3_cotrain')))"` |
| VL manifest loads | `python -c "from huggingface_hub import hf_hub_download as f; import pandas as pd; print(len(pd.read_parquet(f('HBOrtiz/so101_eval3_cotrain_grounding','manifest.parquet',repo_type='dataset'))))"` |

## Smoke-test gates (200 steps)

| Gate | Check | Pass |
|---|---|---|
| Both losses fire | `grep -c flow_loss smoke.log` and `grep -c vqa_loss smoke.log` | both > 0; ratio approximately `vl_ratio:1` |
| No silent NaN | `grep "non-finite" smoke.log` | no matches after step 5 |
| VRAM headroom | `nvidia-smi` during run | under 90% of card |
| Flow loss trends down | step 10 vs step 190 | step 190 not greater than 70% of step 10 |
| VL loss not stuck | step 11 vs step 187 | step 187 not greater than 85% of step 11 |

## Tunables (env-var overrides)

| Var | Default (single GPU) | Default (multi-GPU cluster) | Notes |
|---|---|---|---|
| `STEPS` | 30000 | 50000 | total training steps |
| `SAVE_FREQ` | 5000 | 5000 | checkpoint + push interval |
| `BATCH_SIZE` | 32 | 200 per GPU (sized for 141 GB H200) | scale to VRAM: `(0.80 * VRAM_GB - 1.9) / 0.55` |
| `VL_BATCH_SIZE` | 8 | `BATCH_SIZE` per GPU (200) | equal to `BATCH_SIZE` on cluster (true 5:1 effective ratio); `BATCH_SIZE / 4` on single GPU |
| `VL_RATIO` | 5 | 5 | matches the deployed cotrain (5:1 robot:VL); ObjectVLA used 10:1 for broad |
| `LR` | 5e-5 | 5e-5 | half the LeRobot default; protects pretrained features |
| `NUM_WORKERS` | 4 | 16 | dataloader workers per process |
| `OUT_DIR` | `outputs/smolvla_cotrain_${VL_RATIO}to1` | `outputs/smolvla_cotrain_klal_lora_${STEPS}` | local checkpoint dir (parametrised so the path stays accurate when you override) |

For a multi-node job, replace `--standalone` in `launch_multi_gpu.sh` with your cluster's torchrun rendezvous arguments.

## Abort gates during the run

| Symptom | Action |
|---|---|
| `flow_loss` flat after step 2k | abort; check dataset stats |
| `vqa_loss` plateaued > 3.0 after step 5k | abort; check VL collator label masking |
| GPU OOM | halve `BATCH_SIZE`, restart |
| Random NaN losses | confirm `DTYPE=bfloat16`; SmolVLM2 weights are bf16, fp16 NaN-outs |
| `vqa_loss` count not approximately `flow_loss / VL_RATIO` | VL batches not actually firing - check log routing |

