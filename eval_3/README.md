# Eval 3 — SmolVLM2 LoRA Fine-Tune

End-to-end pipeline that fine-tunes the SmolVLM2-500M VLM (the same backbone
SmolVLA freezes) on celebrity face → name grounding, so the downstream VLA can
identify celebrities for Eval 3.

> **Read first:** [`docs/lora_vlm_finetuning.md`](../docs/lora_vlm_finetuning.md)
> §2.A explains why LoRA is constrained to LM layers 0–15 + the full vision
> tower (the layer-truncation footgun) and documents Approach A (what we ship)
> vs Approach B (the fallback).

---

## Contents

1. [TL;DR](#tldr)
2. [Setup (one-time)](#setup-one-time)
3. [Data preparation](#data-preparation)
4. [Smoke test (always run before real training)](#smoke-test-always-run-before-real-training)
5. [Real training](#real-training)
6. [All flags](#all-flags)
7. [Outputs](#outputs)
8. [Troubleshooting](#troubleshooting)

---

## TL;DR

```bash
# from a clean machine with conda + git installed:
git clone --recursive git@github.com:Ace3Z/LeMonkey.git
cd LeMonkey

conda create -n vlm_finetune python=3.12 -y
conda activate vlm_finetune
pip install -r eval_3/requirements.txt
# Linux + CUDA only (overrides the Mac/CPU torch wheel pip just installed):
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121

hf auth login    # paste a HF token; needed for SmolVLM2 download
wandb login      # paste a wandb API key from https://wandb.ai/authorize

# Kaggle creds for VGGFace2 download — do one of:
#   (a) drop kaggle.json at ~/.kaggle/kaggle.json (chmod 600), OR
#   (b) export KAGGLE_USERNAME=... KAGGLE_KEY=... in your shell

# data ----------------------------------------------------------------------
DATA_ROOT=$HOME/datasets/vggface2_hearfool   # adjust to taste
mkdir -p $DATA_ROOT/meta && cd $DATA_ROOT
kaggle datasets download -d hearfool/vggface2 --unzip
curl -sSfL -o meta/identity_meta_with_estimated_age.csv \
  https://raw.githubusercontent.com/abars/VGGFace2AgeLabel/master/estimated/identity_meta_with_estimated_age.csv
cd -

# manifest + training JSONL -------------------------------------------------
python eval_3/scripts/build_celeb_dataset.py --data-root $DATA_ROOT
python eval_3/scripts/build_llava_json.py    --manifest $DATA_ROOT/manifests/manifest.parquet \
                                             --out-dir  $DATA_ROOT/manifests

# smoke (validates pipeline; 7 min on Mac MPS, ~30 s on A10G) ---------------
python eval_3/scripts/build_llava_json.py --manifest $DATA_ROOT/manifests/manifest.parquet \
                                          --out-dir  $DATA_ROOT/manifests \
                                          --max-identities 5 --train-imgs-per-id 20 \
                                          --val-imgs-per-id 5 --out-suffix .smoke
python eval_3/scripts/train_smolvlm2_lora.py --smoke --data-root $DATA_ROOT/manifests

# real training (A10G, ~8-12h depending on flags) ---------------------------
python eval_3/scripts/train_smolvlm2_lora.py --data-root $DATA_ROOT/manifests \
                                             --out-dir   $DATA_ROOT/lora_celeb_v0
```

---

## Setup (one-time)

### 1. Conda env

```bash
conda create -n vlm_finetune python=3.12 -y
conda activate vlm_finetune
pip install -r eval_3/requirements.txt
```

`eval_3/requirements.txt` pins `transformers==5.3.0` because newer versions
(5.4+) break SmolVLM2's `AutoProcessor` auto-detect. Don't bump it without
re-validating that `AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")` still works.

### 2. CUDA torch (AWS A10G only)

`pip install -r requirements.txt` installs the default torch wheel. On Linux
+ CUDA you need to explicitly install the CUDA build:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
# verify:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected on A10G: True  NVIDIA A10G
```

For other GPUs swap `cu121` → matching CUDA version (cu118, cu124, …).

### 3. Auth

| Service | Why | Command |
|---|---|---|
| HuggingFace | downloading SmolVLM2 weights | `hf auth login` (paste token from <https://huggingface.co/settings/tokens>) |
| Weights & Biases | training run logs | `wandb login` (paste key from <https://wandb.ai/authorize>) |
| Kaggle | VGGFace2 dataset download | `~/.kaggle/kaggle.json` (chmod 600) **or** `export KAGGLE_USERNAME=... KAGGLE_KEY=...` |

For the HF token to be visible inside Python (not just to the `hf` CLI), also
`export HF_TOKEN=$(hf auth token)` (and add to `~/.bashrc` to persist).

---

## Data preparation

### Download VGGFace2 (hearfool mirror, ~2.5 GB)

```bash
DATA_ROOT=$HOME/datasets/vggface2_hearfool   # any path with ~5 GB free
mkdir -p $DATA_ROOT && cd $DATA_ROOT
kaggle datasets download -d hearfool/vggface2 --unzip
```

This produces:
```
$DATA_ROOT/
├── train/  (480 identity dirs: n000002/, n000003/, ...)
└── val/    (60 identity dirs, disjoint from train)
```

The hearfool mirror is **images-only** — it ships **no metadata CSV**. Without
the `n00xxxx → real name` mapping, the pipeline can't run. Get it from the
`abars/VGGFace2AgeLabel` mirror:

```bash
mkdir -p $DATA_ROOT/meta
curl -sSfL -o $DATA_ROOT/meta/identity_meta_with_estimated_age.csv \
  https://raw.githubusercontent.com/abars/VGGFace2AgeLabel/master/estimated/identity_meta_with_estimated_age.csv
```

The CSV format is `class_id, name, sample_count, train_flag, gender, est_age`.
9,131 rows total; we intersect with the 480+60 IDs we actually have on disk.

### macOS only: clean shadow files

The hearfool zip carries `._*` macOS resource-fork files that bloat disk usage
~10× on exFAT. Drop them after extraction:

```bash
find $DATA_ROOT -name "._*" -delete
rm -f $DATA_ROOT/vggface2.zip   # post-extract; saves another 2.3 GB
```

### Build the manifest

```bash
python eval_3/scripts/build_celeb_dataset.py --data-root $DATA_ROOT
```

Outputs `$DATA_ROOT/manifests/manifest.parquet` with one row per identity:
`class_id, name, n_images, gender, source_split, image_paths`.

Filters: drops identities with `<--min-images` images (default 30) and any
identity not present in the metadata CSV. Expected output: 540 IDs / ~197k
images, 0 dropped.

### Build the training JSONL

```bash
# full set: 100 imgs/id train, 30 imgs/id val (default)
python eval_3/scripts/build_llava_json.py --manifest $DATA_ROOT/manifests/manifest.parquet \
                                          --out-dir  $DATA_ROOT/manifests
```

Outputs `$DATA_ROOT/manifests/{train,val}.jsonl`. Each line:
```json
{"class_id": "n000002", "name": "A Fine Frenzy",
 "prompt": "Who is shown in this photo?", "response": "A Fine Frenzy",
 "image_path": "/.../train/n000002/0001_01.jpg"}
```

The hearfool train/val identity split is naturally disjoint; we preserve it
so val measures **name generalization to unseen identities**, not
memorization.

Prompts are randomized across 5 grounding templates (see
`PROMPT_TEMPLATES` in the script). Eval-style phrasings ("place coke on …")
are deliberately excluded — that's the VLA's job to learn at downstream
training time.

---

## Smoke test (always run before real training)

The smoke build is 5 IDs × 20 imgs, run for 10 steps, no wandb. Validates the
pipeline end-to-end (model load, LoRA placement, data collation, gradient
flow, adapter save) **without** spending GPU hours on something that doesn't
even compile.

```bash
# build the tiny dataset (writes train.smoke.jsonl + val.smoke.jsonl)
python eval_3/scripts/build_llava_json.py --manifest $DATA_ROOT/manifests/manifest.parquet \
                                          --out-dir  $DATA_ROOT/manifests \
                                          --max-identities 5 --train-imgs-per-id 20 \
                                          --val-imgs-per-id 5 --out-suffix .smoke

# run smoke
python eval_3/scripts/train_smolvlm2_lora.py --smoke --data-root $DATA_ROOT/manifests
```

What you should see:
```
=== LoRA module placement audit ===
  text_model layers with LoRA:   [0, 1, 2, ..., 15]
  vision_model layers with LoRA: [0, 1, 2, ..., 11]
  total LoRA-adapted modules:    184
  ✓ no LoRA on text_model layers 16-31 (truncation-safe)

trainable params: 6,995,968 || all params: 514,478,272 || trainable%: 1.3598
[collator] pad_id=2  image_token_ids=[49190, 49189, 49152]

10 steps trained
loss: ~3.0–4.0    (NOT ~16; if you see ~16 the image-token mask is broken)
```

Runtime: ~7 min on Mac MPS (fp32), ~30 s on A10G (bf16). If the smoke takes
10× longer than expected, check that `device=cuda dtype=torch.bfloat16` was
printed (and not `device=cpu`).

---

## Real training

```bash
python eval_3/scripts/train_smolvlm2_lora.py \
  --data-root  $DATA_ROOT/manifests \
  --out-dir    $DATA_ROOT/lora_celeb_v0 \
  --epochs     3 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr         1e-4
```

Effective batch = `batch-size × grad-accum`. With the values above (4×4=16),
the full 48k-example train set takes ~9k optimizer steps per epoch.

**Time-budget table on A10G (bf16, batch=4, grad_accum=4):**

| Configuration | Steps/epoch | Total time (3 epochs) |
|---|---|---|
| `--train-imgs-per-id 100` (default, 48k examples) | ~9k | **~12 h** |
| `--train-imgs-per-id 50` (24k examples) | ~4.5k | ~6 h |
| `--train-imgs-per-id 30` (14.4k examples) | ~2.7k | ~4 h |
| `--epochs 2` (instead of 3) | — | × 2/3 |

Re-build the JSONL with `--train-imgs-per-id N` to change dataset size:

```bash
python eval_3/scripts/build_llava_json.py --manifest $DATA_ROOT/manifests/manifest.parquet \
                                          --out-dir  $DATA_ROOT/manifests \
                                          --train-imgs-per-id 50
```

WandB runs land under project `lemonkey-eval3-smolvlm` (override with
`--wandb-project`). Check live training at <https://wandb.ai/baumann-ortiz-eth-zurich/lemonkey-eval3-smolvlm>.

---

## Diagnostic eval — name accuracy on held-out images of training identities

After training, run this to find out whether the LoRA actually learned face → name:
```bash
python eval_3/scripts/eval_lora_train_id_accuracy.py \
  --adapter   $DATA_ROOT/lora_celeb_v0 \
  --data-root $DATA_ROOT \
  --n-identities 50 --n-imgs-per-id 5
```

This samples 50 training identities, finds 5 images per identity that were **not** in `train.jsonl`, runs inference (`"Who is shown in this photo?"`), and reports normalized name accuracy. Add `--no-lora` for the BASE-model baseline. ~5 min on A10G.

> **Why this script exists:** the eval loss reported during training measures performance on identity-disjoint val identities (zero-shot identification, structurally impossible). This script measures the actually-meaningful signal — can the model name a *new photo of an identity it WAS trained on*. See `docs/lora_vlm_finetuning.md` for context.

For runs going forward, the `build_llava_json.py` default is now `--val-strategy per-identity`, so `eval_loss` during training will track this signal directly.

---

## All flags

### `build_celeb_dataset.py`

| Flag | Default | Effect |
|---|---|---|
| `--data-root` | `/Volumes/externalSSD/datasets/vggface2_hearfool` | Where the `train/`, `val/`, `meta/` dirs live |
| `--meta-csv` | `<data-root>/meta/identity_meta_with_estimated_age.csv` | Identity → name mapping |
| `--out` | `<data-root>/manifests/manifest.parquet` | Output parquet path |
| `--min-images` | 30 | Drop identities with fewer images |

### `build_llava_json.py`

| Flag | Default | Effect |
|---|---|---|
| `--manifest` | `<DATA_ROOT>/manifests/manifest.parquet` | Input from previous step |
| `--out-dir` | `<DATA_ROOT>/manifests` | Where `train.jsonl` / `val.jsonl` are written |
| `--train-imgs-per-id` | 100 | Cap per identity (smaller → faster training, less coverage) |
| `--val-imgs-per-id` | 30 | Cap per val identity |
| `--max-identities` | None | Limit identities per split (smoke testing) |
| `--out-suffix` | `""` | Suffix for output files (`.smoke` → `train.smoke.jsonl`) |
| `--val-strategy` | `per-identity` | `per-identity` (recommended): same IDs in train+val, different images — `eval_loss` measures whether LoRA learned trained identities. `disjoint` (legacy): hearfool's identity-disjoint partition — measures impossible zero-shot ID, use only as an OOD probe. |
| `--seed` | 42 | RNG seed for image sampling + prompt selection |

### `eval_lora_train_id_accuracy.py`

| Flag | Default | Effect |
|---|---|---|
| `--adapter` | (required) | Path to the LoRA adapter dir |
| `--data-root` | (required) | Root with `manifests/` subdir |
| `--manifest` | `<data-root>/manifests/manifest.parquet` | Identity → name + image_paths |
| `--train-jsonl` | `<data-root>/manifests/train.jsonl` | Used to identify *held-out* images per identity |
| `--n-identities` | 50 | How many training identities to sample |
| `--n-imgs-per-id` | 5 | Held-out images per sampled identity |
| `--prompt` | `"Who is shown in this photo?"` | Prompt format |
| `--no-lora` | False | Skip adapter, eval base model (BASELINE) |
| `--out-jsonl` | None | Optional dump of every (img, expected, predicted, correct) row |
| `--seed` | 0 | RNG seed for identity + image sampling |

### `train_smolvlm2_lora.py`

| Flag | Default | Effect |
|---|---|---|
| `--smoke` | False | Use `train.smoke.jsonl`, 1 epoch, batch=1, max-steps=10, no wandb |
| `--data-root` | `/Volumes/externalSSD/datasets/vggface2_hearfool/manifests` | Where the JSONL files live |
| `--out-dir` | `/Volumes/externalSSD/lemonkey/eval_3/lora_celeb_v0` | Where the LoRA adapter is saved |
| `--epochs` | 3.0 | Training epochs (real mode only) |
| `--batch-size` | 2 | Per-device batch size |
| `--grad-accum` | 8 | Gradient accumulation (effective batch = `batch-size × grad-accum`) |
| `--lr` | 1e-4 | Learning rate |
| `--wandb-project` | `lemonkey-eval3-smolvlm` | WandB project name |

LoRA hyperparameters (rank 16, alpha 32, dropout 0.05) and `target_modules`
regex are hard-coded — see `LORA_TARGET_REGEX` at the top of the script.
Change these inside the script (not via CLI) so the choice is committed
explicitly to the trainer's source.

---

## Outputs

```
<DATA_ROOT>/manifests/
├── manifest.parquet          (540 rows, ~1.5 MB)
├── train.jsonl               (~48k lines)
├── val.jsonl                 (~1.8k lines)
├── train.smoke.jsonl         (100 lines, smoke only)
└── val.smoke.jsonl           (25 lines, smoke only)

<OUT_DIR> (default <DATA_ROOT>/lora_celeb_v0)/
├── adapter_config.json       (peft LoRA config)
├── adapter_model.safetensors (LoRA weights, ~30 MB)
├── tokenizer.json            (full processor for re-loading)
├── preprocessor_config.json
├── chat_template.jinja
├── special_tokens_map.json
├── added_tokens.json
└── checkpoint-XXXX/          (intermediate checkpoints if --save-steps fires)
```

Load the adapter on top of the base model later with:
```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
model = PeftModel.from_pretrained(base, "<OUT_DIR>")
proc = AutoProcessor.from_pretrained("<OUT_DIR>")
```

To merge LoRA into the backbone permanently (for SmolVLA retraining, Step 6 of
[`docs/lora_vlm_finetuning.md`](../docs/lora_vlm_finetuning.md)):
```python
merged = model.merge_and_unload()
merged.save_pretrained("<OUT_DIR>_merged")
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unrecognized image processor in HuggingFaceTB/SmolVLM2-...` | transformers ≥ 5.4 | `pip install transformers==5.3.0` |
| `ModuleNotFoundError: torchvision` / `num2words` | requirements not installed | `pip install -r eval_3/requirements.txt` |
| Loss starts at ~16 and never falls | image tokens not masked in labels | Check `[collator] image_token_ids=[...]` is non-empty in the log; if empty, your processor doesn't expose those token IDs and the masking didn't fire |
| `ValueError: Mismatch in image token count` | trl/processor truncating mid-image-tokens | We replaced trl's collator; if you see this you're running an older script — `git pull` |
| `device=cpu` printed instead of `cuda` | torch CUDA build not installed | See "CUDA torch" in Setup |
| OOM on A10G | batch too large for image-token-heavy sequences | Lower `--batch-size` to 2 or 1, raise `--grad-accum` to keep effective batch |
| Training runs but loss plateaus | LR too low, or LoRA rank too small | Try `--lr 2e-4`; or edit `r=16` → `r=32` in `attach_lora()` |
| WandB run doesn't appear | login expired | `wandb login --relogin` |
| First HF download fails / rate-limited | unauth'd | `export HF_TOKEN=$(hf auth token)` |
