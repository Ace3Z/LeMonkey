# SmolVLA cotrain (deployed Eval 3 trainer)

This is the trainer that produced the two SmolVLA models deployed on Eval 3 day:

| HF repo | Recipe | Launcher |
|---|---|---|
| [`HBOrtiz/so101_smolvla_eval3_cotrain`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_cotrain) | 5:1 robot:VL, 3 in-distribution celebrities | `launch.sh` (single GPU) or `run_cluster.sh` (multi-GPU) |
| [`HBOrtiz/so101_smolvla_eval3_broad`](https://huggingface.co/HBOrtiz/so101_smolvla_eval3_broad) | 10:1 robot:VL, 192 celebrities | `eval_3/scripts/brev/train_smolvla_broad.sh` |

Both are SmolVLA-450M from `lerobot/smolvla_base` with the SmolVLM2 backbone trainable and SigLIP frozen.

## What the trainer does

Every `vl_ratio + 1`-th step is a vision-language VQA batch with CE loss on the SmolVLM2 LM head; the rest are robot batches with SmolVLA's flow-matching action loss. Both gradients flow into the same VLM body — the mechanism that keeps the celebrity-name prior alive across the action fine-tune. The two streams come from paired HF datasets: `HBOrtiz/so101_eval3_cotrain` (robot) and `HBOrtiz/so101_eval3_cotrain_grounding` (vision-language).

## File layout

```
cotrain.py        the training script (single-process or torchrun multi-GPU)
launch.sh         single-GPU env-var-driven launcher
run_cluster.sh    multi-GPU launcher; autodetects every GPU on the node
setup_env.sh      one-time conda env bootstrap
predl_vl.sh       one-time VL dataset pre-download (HF rate-limit workaround)
```

## Quickstart (single GPU)

```bash
bash setup_env.sh                  # one-time, ~15 min (idempotent)
conda activate lemonkey
export HF_TOKEN=hf_...             # read + write

# Smoke first (200 steps, small batch, no push):
STEPS=200 BATCH_SIZE=4 VL_BATCH_SIZE=2 LOG_EVERY=1 bash launch.sh 2>&1 | tee smoke.log

# Real run:
PUSH_REPO=youruser/smolvla_eval3_cotrain bash launch.sh
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

HF token + push target (write access required — checkpoints are pushed every `SAVE_FREQ` steps). `.hf_token` is gitignored:

```bash
export HF_TOKEN=hf_...                              # or echo to .hf_token
export PUSH_REPO=youruser/smolvla_cotrain_run       # auto-created
```

Run:

```bash
nohup bash run_cluster.sh > cotrain.log 2>&1 &
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
| `VL_BATCH_SIZE` | 8 | 100 per GPU | keep at `BATCH_SIZE / 2` for cluster, equal for single GPU |
| `VL_RATIO` | 10 | 5 | the deployed cotrain used 5; ObjectVLA default is 10 |
| `LR` | 5e-5 | 5e-5 | half the LeRobot default; protects pretrained features |
| `NUM_WORKERS` | 4 | 16 | dataloader workers per process |
| `OUT_DIR` | `outputs/smolvla_cotrain` | `outputs/smolvla_klal_lora_25k` | local checkpoint dir |

For a multi-node job, replace `--standalone` in `run_cluster.sh` with your cluster's torchrun rendezvous arguments.

## Abort gates during the run

| Symptom | Action |
|---|---|
| `flow_loss` flat after step 2k | abort; check dataset stats |
| `vqa_loss` plateaued > 3.0 after step 5k | abort; check VL collator label masking |
| GPU OOM | halve `BATCH_SIZE`, restart |
| Random NaN losses | confirm `DTYPE=bfloat16`; SmolVLM2 weights are bf16, fp16 NaN-outs |
| `vqa_loss` count not approximately `flow_loss / VL_RATIO` | VL batches not actually firing — check log routing |

## Known caveats

- **torchcodec host-RAM leak on large multi-mp4 datasets.** lerobot 0.5.1's default video backend leaks host RAM per distinct mp4 opened (~10 MB / 100 iterations); the broad set (8,390 mp4s) OOMs a DataLoader worker after about 30 minutes with the default 4 workers. Workaround: `--dataset.video_backend=pyav`. The broad launcher `eval_3/scripts/brev/train_smolvla_broad.sh` already sets this; pass the flag manually if running `cotrain.py` on a dataset with thousands of mp4 files.
- **bf16 only.** SmolVLM2 weights are bf16; fp16 NaN-outs the loss.
- **`torch.compile` off by default.** Flaky over SmolVLA's custom forward in some lerobot versions; enable only after smoke passes.
- **VL collator runs the SmolVLM2 processor twice per batch** (prompt-only and prompt+target) to get correct label-masking around the `<image>` tokens. If `vqa_loss` plateaus at an elevated floor, dump a couple of `labels` rows and verify the `-100` boundary lands exactly where the target starts; tokenizer leading-special drift would shift the boundary by 1-2 tokens.
- **Action-head dim mismatch (low-probability).** If checkpoint `action_in_proj` / `action_out_proj` dims disagree with the config's `max_action_dim`, lerobot's `strict=False` silently keeps random-init projections — step 0-10 `flow_loss` will be very large.
- **HF push failures are non-fatal** — they emit `[WARN]` and continue; the local checkpoint under `OUT_DIR` is kept.

## Troubleshooting

- `third_party/lerobot` empty → `git submodule update --init --recursive`.
- `cannot import lerobot SmolVLA` → env not active or pip install incomplete.
- `nvidia-smi not found` → run on a GPU node.
