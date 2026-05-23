# eval_3/aug

The identity-preserving augmentation pipeline + training-time loss/adapter
modules that produced the deployed Eval 3 SmolVLA policies on the Hub
(`HBOrtiz/so101_smolvla_eval3_cotrain`, `…_broad`, `…_cotrain_klal`).

A few hundred real teleop episodes are multiplied into millions of frames by
re-rendering each base episode with different celebrity faces inpainted onto
the printed portraits. The bounding box and identity of every portrait is
known by construction, so vision-language grounding pairs are emitted
automatically alongside. Co-training SmolVLA on both streams puts the
celebrity knowledge into the policy weights.

For the design rationale and numerical defaults, see [`STRATEGY.md`](STRATEGY.md)
and [`VALIDATION.md`](VALIDATION.md).

## Layout

```
eval_3/aug/
├── stages/               per-stage pipeline primitives (libraries + CLIs)
│   ├── detect_static.py    static-camera portrait detection + per-frame occluder masks
│   ├── refine_paper_quad.py sub-pixel paper-edge rectangle refit (Canny + Hough + cornerSubPix)
│   ├── inpaint_video.py    composite engine (warp + Reinhard + MTF + seamlessClone)
│   └── video_io.py         AV1 -> H.264 sidecar transcode + frame iterators
│
├── mining/
│   └── mine_celeb_photos.py  Wikimedia + icrawler scrape + ArcFace verifier
│
├── generators/           variant dataset builders (call into stages/)
│   ├── broad.py            195-celeb out-of-distribution generator (`so101_eval3_broad`)
│   ├── broad_topup.py      patch run for celebs missed by broad.py
│   ├── cotrain.py          3-IID-celeb full-enumeration generator (`so101_eval3_cotrain`)
│   └── build_cotrain_bank.py 8-photo-per-celeb bank for cotrain.py
│
├── merge_prep/           post-augmentation LeRobotDataset fixups (run before
│   │                     and after `eval_3/scripts/data/merge_episodes.py`)
│   ├── prep_for_merge.py        break info.json hardlinks; add reference stream
│   ├── patch_episodes_parquet.py add the video columns LeRobotDataset expects
│   ├── relabel_cotrain_prompts.py rewrite tasks.parquet per variant
│   ├── fix_merged_tasks.py      rebuild global tasks.parquet post-merge
│   └── validate_merged.py       13-check sanity audit of the merged dataset
│
├── training/             KLAL + LoRA modules consumed by
│   │                     `eval_3/scripts/smolvla_cotrain/cotrain.py`
│   ├── klal_core.py            KLALConfig + klal_loss + gaussian_target_from_mask
│   ├── klal_smolvla_action.py  KLAL hookset on the robot-action forward path
│   ├── klal_smolvla_vl.py      KLAL hookset on the VL co-training forward path (deployed)
│   └── lora_smolvla.py         minimal LoRA on SmolVLA VLM attention proj
│
├── dbg/                  visual-gate scripts (manual + programmatic --dbg use)
│   ├── compare_gif.py
│   ├── mask_overlay.py
│   ├── segmentation_video.py
│   └── stage2_panels.py
│
├── tests/
│   ├── test_replace_portrait.py  synthetic regression on inpaint_video.replace_portrait
│   └── test_klal_lora_smoke.py   two-tier pure-logic + real-SmolVLA forward gate
│
└── _legacy/              v1 pipeline + superseded research; not part of the
                          deployed recipe. See _legacy/README.md.
```

## Deployed recipe (what actually built the released datasets)

1. **Record base teleops**: `eval_3/scripts/record/record_guided.sh`
   (the 180-episode operator-facing session) -> `datasets/eval3/ep_NNNN_*/`.
2. **Stage 2 portrait detection**: `stages/detect_static.py` ->
   `<ep>/portrait_corners.json` + `<ep>/portrait_masks.pkl` per episode.
3. **Augmentation variants**:
   - `generators/broad.py` for the 192-celebrity OOD dataset.
   - `generators/cotrain.py` for the 3-IID-celebrity (Swift / Obama / LeCun) dataset.
   Both call `stages/inpaint_video.replace_portrait()` per variant.
4. **Merge + fix-up**:
   - `merge_prep/prep_for_merge.py` and `patch_episodes_parquet.py`
   - `eval_3/scripts/data/merge_episodes.py` (the actual aggregator)
   - `merge_prep/fix_merged_tasks.py` and `validate_merged.py`
5. **Push to HF**: `eval_3/scripts/data/push_dataset_to_hf.py`.
6. **Train**: `eval_3/scripts/smolvla_cotrain/cotrain.py` consumes
   `training/{klal_core, klal_smolvla_vl, klal_smolvla_action, lora_smolvla}.py`.

## Outputs

`stages/*` and `generators/*` write under `datasets/` (gitignored).
`merge_prep/*` modifies the merged dataset in place. The deployed datasets and
models live on the Hugging Face Hub under
[`HBOrtiz/`](https://huggingface.co/HBOrtiz); see the top-level
[`DATASETS_AND_MODELS.md`](../../DATASETS_AND_MODELS.md) for the full
inventory.
