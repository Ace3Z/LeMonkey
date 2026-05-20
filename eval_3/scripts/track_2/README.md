# Track 2 — ObjectVLA enhanced data prep & training

Owner: Sejohn · Branch: `dev/SjohnU/track_2_objectvla`

This directory contains the data-side pipeline and training wrapper for Track 2
(ObjectVLA bbox-grounded VQA co-train on Pi0.5 + 200-celeb dataset).

Full spec: [`../tracks/TRACK_OBJECTVLA_ENHANCED.md`](../tracks/TRACK_OBJECTVLA_ENHANCED.md)

---

## Files at a glance

| File | Purpose | Status |
|---|---|---|
| `build_task_to_centroid.py` | Parse 960 task strings → map task_index → celeb slug → ArcFace centroid | ✓ tested (192/192 covered) |
| `task_index_to_centroid.json` | Output of above; used by audit script | ✓ |
| `build_confusion_matrix.py` | Compute 192×192 celeb-vs-celeb cosines for hard-neg analysis | ✓ tested |
| `confusion_matrix.npy` / `confusion_slugs.json` / `confusable_topk.json` | Outputs of above | ✓ |
| `verify_bbox_schema.py` | Pre-flight: validate Roham's bbox parquet schema | ✓ |
| `arcface_audit_200celeb.py` | Per-frame target_cos + max_distractor_cos + hardneg_gap | ✓ syntax + argparse smoke |
| `build_keep_list_and_weights.py` | Audit parquet → keep_episodes.txt + hardneg_weights.npy | ✓ syntax + argparse smoke |
| `curriculum_sampler.py` | Two-phase weighted sampler (easy → full at step 5000) | ✓ smoke-tested with fake data |
| `layer_rank_track2.json` | Per-layer LoRA rank config (Enhancement B-4) | ✓ JSON valid |
| `lerobot_train_with_vl_cotrain.py` | Mixed-batch lerobot-train wrapper (canonical + B-1/B-4/B-5/B-7) | scaffolded; [BREV_INTEGRATE] markers |
| `run_audit_pipeline.sh` | One-shot pipeline: schema verify → audit → build_keep_list | ✓ |
| `SMOKE_TEST.md` | Gate-by-gate verification protocol before 24 h Brev launch | ✓ |

Launch script lives at `../scripts/brev/run_training_track_2.sh`.
Rollout runner at `../scripts/run_rollout_track_2.sh`.

---

## When Roham's bboxes land

One command runs the entire data prep:

```bash
bash eval_3/scripts/track_2/run_audit_pipeline.sh <path-or-HF-repo>
```

This does:
1. Schema verify (~5 sec) — fails loud if column names differ from expected
2. ArcFace audit (~1 h CPU / ~10 min GPU) — per-frame target_cos + hardneg_gap
3. Build keep_episodes.txt + hardneg_weights.npy (~5 min)

After this completes, the Brev launch script has all artifacts it needs.

---

## When Darius's VL manifest lands

The launch script picks up `HBOrtiz/eval3_objectvla_vl_pairs` automatically
via the `VL_MANIFEST` env var (default).

Schema expected by `VLPairsDataset` (in `lerobot_train_with_vl_cotrain.py`):
```
image_path   (str)
prompt       (str)
target       (str)
bbox         (list[float], 4 elements, normalized xyxy)
celeb        (str slug)
caption_type (str: location_explicit / qa_grounded / qa_open / caption)
```

If Darius uses different column names, the wrapper will fail loudly at
load time; adjust `VLPairsDataset.__init__` column-resolution accordingly.

---

## Smoke test gates (BEFORE 24 h launch)

See [`SMOKE_TEST.md`](SMOKE_TEST.md) for the full gate-by-gate protocol.
Critical:

1. Darius's Strix VRAM probe (universal Pi0.5 kill switch)
2. Data prep ran successfully (this directory has `keep_episodes.txt`)
3. 200-step run on brev_instance2 fires BOTH loss types (flow + VQA CE)
4. No transformers ≥5.0 dict-attention-mask crash (or fallback splice works)

---

## Face-name binding rationale (the "why" for the enhanced spec)

Every script in this directory exists to improve image-to-name binding +
face recognition during Track 2 training:

- **Audit + filter (B-2)**: drops inpainted variants where the painted face
  doesn't actually look like the target celeb (cos < 0.50). Removes noise
  that would otherwise weaken the face-binding gradient.
- **Hard-negative oversampling (B-3)**: 2× weight on variants whose target
  celeb has a visually-confusable distractor visible. Forces the model to
  develop fine-grained features instead of coarse ones.
- **Layer-wise LoRA rank (B-4)**: r=64 on layers 8–12 (BlindVLA Table 12
  face-discrim zone) + r=48 on 15–17 (top-LM name-token alignment).
  Concentrates trainable capacity in face-relevant layers.
- **Curriculum (B-5)**: easy variants first (high hardneg_gap), then full
  distribution at step 5 k. Lets face features form before action loss
  settles.
- **Warm-PG starting (B-1)**: load from `HBOrtiz/pi05_paligemma_celeb_warm_v2`
  which already has VGGFace2 celeb knowledge LoRA-merged into PaliGemma.
- **EMA (B-7)**: α=0.999 shadow weights reduce gradient-noise oscillation
  late in training.

Multi-prompt N=3 inference ensemble (B-6) was previously considered but
removed per the user's call.
