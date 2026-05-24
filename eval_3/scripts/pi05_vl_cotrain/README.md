# pi05_vl_cotrain - Pi0.5 + ObjectVLA VL cotrain (published variant)

Pi0.5 sibling to `../smolvla_cotrain/`. Same RT-2 §3.2 recipe (interleaved robot + bbox-grounded VQA batches) but targets Pi0.5-3B (PaliGemma-2B + Gemma-300M action expert) instead of SmolVLA-450M.

The Pi0.5 Eval 3 variant is published as [`HBOrtiz/so101_pi05_eval3`](https://huggingface.co/HBOrtiz/so101_pi05_eval3) (one of the "Other published variants" in [`DATASETS_AND_MODELS.md`](../../../DATASETS_AND_MODELS.md)). The eval-day deployed policies are the two SmolVLA checkpoints (`so101_smolvla_eval3_cotrain` and `_broad`), not this Pi0.5 path.

## Files

### Data prep (ran end-to-end)

| File | Purpose |
|---|---|
| `verify_bbox_schema.py` | Pre-flight: validate the bbox parquet's column names. |
| `arcface_audit_200celeb.py` | Per-frame `target_cos`, `max_distractor_cos`, `hardneg_gap` audit over the 200-celebrity inpainted dataset. |
| `build_keep_list_and_weights.py` | Audit parquet -> `keep_episodes.txt` + `hardneg_weights.npy`. |
| `build_task_to_centroid.py` | Map task strings -> celeb slug -> ArcFace centroid. Produces `precomputed/task_index_to_centroid.json`. |
| `build_confusion_matrix.py` | 192×192 celeb-vs-celeb ArcFace cosine matrix. Produces `precomputed/{confusion_matrix.npy, confusion_slugs.json, confusable_topk.json}`. |
| `generate_vl_pairs.py` | RetinaFace-based VL-pair generator for the 193-celeb bank. |
| `run_audit_pipeline.sh` | One-shot orchestrator: schema-verify -> audit -> build keep_list + weights. |
| `precomputed/` | Static audit artifacts checked in (used by the wrapper below). |

### Training wrapper (canonical scaffold)

| File | Purpose |
|---|---|
| `lerobot_train_with_vl_cotrain.py` | Mixed-batch Pi0.5 + VL cotrain wrapper around `lerobot-train`. Includes warm-PG start, audit/filter, hard-negative weights, per-layer LoRA rank (`layer_rank.json`), two-phase curriculum (`curriculum_sampler.py`), EMA. |
| `curriculum_sampler.py` | Two-phase weighted sampler (easy variants until step 5000, then full distribution). |
| `precomputed/layer_rank.json` | Per-layer LoRA rank config (r=64 on layers 8-12, r=48 on layers 15-17). |
| `probe_pi05_inference.py` | Standalone Pi0.5 VRAM + latency probe (pass criteria: peak under 14 GB, p95 forward under 20 s). |

## How `HBOrtiz/so101_pi05_eval3` was produced

1. PaliGemma backbone warm-started on VGGFace2 VQA -> [`HBOrtiz/paligemma_vqa_warm`](https://huggingface.co/HBOrtiz/paligemma_vqa_warm) (see [`../warmstart/`](../warmstart/)).
2. Pi0.5 LoRA fine-tune from that init on `HBOrtiz/so101_eval3_broad`, launched via [`../brev/train_pi05.sh`](../brev/train_pi05.sh), the vanilla LoRA path.

The ObjectVLA enhancements (mixed batches, hard-neg curriculum, per-layer LoRA, EMA) listed in the file table above are documented here as the design intent; the published checkpoint is the vanilla-LoRA result. The wrapper here is preserved so the enhanced recipe can be revived from the precomputed artifacts.

## Face-name binding rationale (design intent)

- **Warm PaliGemma start.** Load `HBOrtiz/paligemma_vqa_warm` instead of `lerobot/pi05_base`, so PaliGemma already has a celebrity-name prior.
- **Audit & filter inpainted variants.** Drop inpainted variants where the painted face fails ArcFace `cos >= 0.50` against the celeb centroid (removes noise that would weaken the face-binding gradient).
- **Hard-negative oversampling.** 2x weight on variants where a visually confusable distractor is visible.
- **Per-layer LoRA rank.** `r=64` on Gemma layers 8-12 (BlindVLA face-discrim zone) + `r=48` on layers 15-17 (top-LM name-token alignment).
- **Two-phase curriculum.** Easy variants first (high `hardneg_gap`); switch to full distribution at step 5000.
- **EMA shadow weights.** `alpha=0.999` shadow weights reduce gradient-noise oscillation late in training.

## Outputs

| Repo | Description |
|---|---|
| [`HBOrtiz/so101_pi05_eval3`](https://huggingface.co/HBOrtiz/so101_pi05_eval3) | Published Pi0.5 variant (warm-start + LoRA on broad dataset). |
| [`HBOrtiz/paligemma_vqa_warm`](https://huggingface.co/HBOrtiz/paligemma_vqa_warm) | PaliGemma backbone warm-started on VGGFace2 VQA (init for Pi0.5). |

Full provenance: [`DATASETS_AND_MODELS.md`](../../../DATASETS_AND_MODELS.md).

The rollout runner for the published checkpoint is [`../rollout/pi05_vl_cotrain.sh`](../rollout/pi05_vl_cotrain.sh).
