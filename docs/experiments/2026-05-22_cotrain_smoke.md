# 2026-05-22 — Co-training smoke test (it works)

## Setup

RTX 5090 (31 GB free), conda env `lemonkey` (transformers 5.3.0, lerobot 0.5.1).
Robot dataset `HBOrtiz/so101_eval3_track3_v3_baseline` served locally by
symlinking `datasets/eval3_track3_v3_merged` → `$HF_LEROBOT_HOME/HBOrtiz/...`
(the HF copy was not cached — only metadata stubs). VL dataset
`HBOrtiz/eval3_track3_vl_pairs` + SmolVLA base downloaded.

## Runs

`cotrain.py --steps=30 --batch_size=4 --vl_batch_size=4 --vl_ratio=5`

| run | mode | result |
|---|---|---|
| 1 | plain co-training | **EXIT 0** — 30 steps, checkpoint saved. `flow_loss` 0.1–1.0 (healthy), `vqa_loss` 12.2 → 10.5 (decreasing). |
| 2 | `--enable_lora --enable_klal --klal_layers=10,12,14` | **EXIT 0** — 30 steps, checkpoint saved. `flow_loss` healthy, `vqa_loss` computed, `klal ≈ 1.5` live. LoRA mode used less VRAM (2.2 vs 3.8 GiB — base frozen). |

Plus the KLAL+LoRA component test `test_klal_lora_smoke.py`: **33/33 PASS** —
including a real end-to-end check that KLAL's gradient reaches a LoRA adapter.

## Verdict

Both co-training modes **work**. This empirically confirms mahbod's `cotrain.py`
does not have Darius's C3/C4 blockers — it reaches step 0 and runs VQA steps
without crashing.

## Caveats (still true)

This is a 30-step, batch-4, single-GPU smoke — it proves basic functionality,
not a production run. The 5 missing Darius fixes (see
`2026-05-21_klal_import_review.md`) concern long-run / multi-GPU robustness
(partial-cache handling, `HF_HUB_OFFLINE`, `empty_cameras`/camera-rename, etc.),
not basic function. A real run still needs those reconciled + a longer
multi-GPU smoke on the target node.
