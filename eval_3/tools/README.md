# eval_3/tools

Visual-verification tooling for the Track-3 co-training datasets. These scripts
were used to audit and validate the `so101_eval3_cotrain_grounding` (VL grounding) and
`so101_eval3_cotrain` (robot action) datasets before training.

| script | what it does |
| --- | --- |
| `verify_robot_episodes.py` | Renders N random robot episodes as annotated videos - full trajectory, 3 portrait quads tracked per-frame, target highlighted, prompt shown. Confirms the can lands on the labelled target portrait. |
| `verify_vl_pairs.py` | Renders panels from a VL-pairs manifest - frame-0 image with `quad_corners_norm` + `celeb_name` overlaid, plus the reference photo. Confirms label ↔ bbox ↔ face agree. |
| `render_quad_overlay_videos.py` | Overlays portrait quads on augmented episode videos; can filter to degenerate-quad episodes. General quad-overlay video renderer. |

All three take `--help`. Paths default to `datasets/...` relative to the repo
root; run from the repo root. **Render outputs (jpg/mp4/gif) go to
`eval_3/outputs/<tool>/` by default** - that folder is gitignored (kept local;
re-render anytime via the scripts). Current contents:

- `eval_3/outputs/dataset_verify/` - 20 annotated robot-episode videos + montage
- `eval_3/outputs/vl_pairs_audit/` - VL-pairs label↔bbox↔face verification panels
- `eval_3/outputs/bbox_check*/`, `eval_3/outputs/degenerate_videos/` - earlier audit renders

## Provenance

These came out of the 2026-05-21 dataset audit that found and fixed two bugs in
`so101_eval3_cotrain_grounding`: image-to-label mispairing, and degenerate bounding-box
quads.
