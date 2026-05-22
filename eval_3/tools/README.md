# eval_3/tools

Visual-verification tooling for the Track-3 co-training datasets. These scripts
were used to audit and validate the `eval3_track3_vl_pairs` (VL grounding) and
`so101_eval3_track3_v3_baseline` (robot action) datasets before training.

| script | what it does |
|---|---|
| `verify_robot_episodes.py` | Renders N random robot episodes as annotated videos — full trajectory, 3 portrait quads tracked per-frame, target highlighted, prompt shown. Confirms the can lands on the labelled target portrait. |
| `verify_vl_pairs.py` | Renders panels from a VL-pairs manifest — frame-0 image with `quad_corners_norm` + `celeb_name` overlaid, plus the reference photo. Confirms label ↔ bbox ↔ face agree. |
| `render_quad_overlay_videos.py` | Overlays portrait quads on augmented episode videos; can filter to degenerate-quad episodes. General quad-overlay video renderer. |

All three take `--help`. Paths default to `datasets/...` relative to the repo
root; run from the repo root. Outputs (jpg/mp4/gif) are written under
`eval_3/attention_steering/` by default and are gitignored — re-render anytime.

## Provenance

These came out of the 2026-05-21 dataset audit that found and fixed two bugs in
`eval3_track3_vl_pairs` (image↔label mispairing, degenerate quads). Full record:

- `docs/experiments/2026-05-21_vl_pairs_image_mispairing.md`
- `docs/experiments/2026-05-21_track3_robot_dataset_visual_verify.md`
