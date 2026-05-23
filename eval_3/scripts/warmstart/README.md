# warmstart — PaliGemma VQA warm-start for Pi0.5

Produces the backbone init for the Pi0.5 Eval 3 variant: [`HBOrtiz/paligemma_vqa_warm`](https://huggingface.co/HBOrtiz/paligemma_vqa_warm).

Why a warm-start: PaliGemma's WebLI pretrain was DLP-filtered — celebrity faces and names were stripped. A Pi0.5 action fine-tune that starts from the raw backbone has no celebrity-name prior to lean on, so it degenerates into a positional-shortcut policy. LoRA-tuning PaliGemma on VGGFace2 VQA *before* the action expert fine-tune gives the policy something to preserve.

## Files

| File | Role |
|---|---|
| `prepare_vggface2_vqa.py` | Builds a VQA parquet from a VGGFace2 raw directory — one row per (identity, photo) pair with `image_path`, prompt `"<image>Who is the person in this image?"`, target name, identity id. |
| `train_paligemma_vqa.py` | LoRA fine-tune (r=32, alpha=64, dropout=0.05) on PaliGemma's `q/k/v/o + gate/up/down` projections inside `lerobot/pi05_base`. Freezes vision_tower, multi_modal_projector, lm_head, and the whole Gemma-300M action expert. Pushes the merged adapter as a Pi0.5 checkpoint. |

The Brev-side launcher is [`../brev/train_paligemma_warmstart.sh`](../brev/train_paligemma_warmstart.sh).

## Build the manifest (one-time)

```bash
python eval_3/scripts/warmstart/prepare_vggface2_vqa.py \
    --vggface2-root /path/to/vggface2_train \
    --out datasets/vggface2_vqa_train.parquet
```

Schema: `image_path`, `prompt`, `target`, `identity_id`.

## Train (on a Brev H100 80 GB)

```bash
ssh <brev-host>
cd ~/LeMonkey
bash eval_3/scripts/brev/setup_paligemma_warmstart.sh   # idempotent env install
nohup bash eval_3/scripts/brev/train_paligemma_warmstart.sh \
    > ~/outputs/paligemma_warm.log 2>&1 &
```

Defaults: 1 epoch, batch 8 with grad-accum 4, LR 1e-5, approximately 6 h on H100 80 GB. On completion the merged checkpoint is pushed to `$PUSH_REPO` (default `HBOrtiz/paligemma_vqa_warm`).

## Pre-flight gates

| Check | One-liner |
|---|---|
| `HF_TOKEN` set with write access | `[ -n "$HF_TOKEN" ] && echo ok` |
| VGGFace2 manifest exists | `[ -f "$MANIFEST_PATH" ] && head -1 "$MANIFEST_PATH"` |
| Pi0.5 base loads | `python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; PI05Policy.from_pretrained('lerobot/pi05_base')"` |
| PEFT >= 0.10 installed | `python -c "import peft; print(peft.__version__)"` |

## Smoke gate (200 steps)

- `vqa_loss` decreases at least 20% from step 0 to step 200.
- No `non-finite loss` lines after step 5.
- VRAM peak under 90% of card.

If `vqa_loss` is flat at around 10 after step 100, the label-masking boundary is wrong — dump a label row and verify the `-100` mask ends at the target's first token, not before the `<image>` features get spliced.
