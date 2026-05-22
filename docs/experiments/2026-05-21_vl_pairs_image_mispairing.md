# 2026-05-21 — `eval3_track3_vl_pairs` audit: image↔label mispairing

## What was run

Re-downloaded `HBOrtiz/eval3_track3_vl_pairs` (manifest fresh, MD5 unchanged
from prior download — the "fix again" never reached HF; latest HF commit
`f058da2654`). Re-validated bboxes and celeb labels against the **local
ground-truth cache** on this dev box.

## Local ground-truth cache (it exists, it is correct)

The dataset was generated on this PC, so the authoritative per-episode cache
is present:

- `datasets/eval3/<base_ep>/portrait_corners.json` — correct **4-corner**
  paper quads, per pid, per frame, with `occluded`/`score`/`interpolated`
  flags. `pipeline_version: v9.5_face_aware_corner_anchor`.
- `datasets/eval3/<base_ep>/portrait_seeds.json` — **ArcFace-verified**
  `celebs` per pid (`arcface_trusted: true`, `arcface_cosines`,
  `arcface_full`). Also carries `layout_celebs` (operator sidecar) which
  can disagree with ArcFace — ArcFace is the trusted source.
- `datasets/eval3_track3_aug/<variant>/augmentation.json` — `pid_to_celeb_full`
  (the inpainted identity) + `new_layout_camera_lmr` for each t3 variant,
  plus the actual inpainted video under `videos/observation.images.camera1/`.

## Findings

### 1. Celeb labels are NOT scrambled (earlier claim retracted)

A previous pass claimed "labels catastrophically scrambled" — that was wrong;
it compared against the layout string, which itself disagrees with ArcFace.
Verified properly:

- **151/151 base-teleop rows**: manifest `celeb_name` matches
  `portrait_seeds.json` ArcFace `celebs` exactly.
- **t3 rows**: manifest `celeb_name` matches the named variant's
  `augmentation.json` `pid_to_celeb_full`.

The labels are sourced correctly.

### 2. THE REAL BUG — t3 frame-0 images are mispaired (98.4% of dataset)

The manifest has 9367 episodes: 151 base teleops + **9216 t3 variants**.
For the t3 rows, the manifest labels/quads are internally consistent with
the named variant, **but the stored frame-0 JPEG is from a different
variant**.

Evidence (`eval_3/attention_steering/vl_pairs_audit/`):
- `t3_compare.jpg` — `quick_swift_SOL_ep02_..._t3_0126_v53`: the t3 variant
  video frame 0 is Swift/Obama/LeCun (L→R, matches its augmentation.json);
  the VL-pairs stored image is LeCun/Swift/Obama — a different variant.
- `base_compare.jpg` — same episode: the stored image is also NOT the base
  teleop `frame_0.png` (LeCun/Obama/Swift).
- `t3_mismatch_grid.jpg` — 4 more random t3 variants, **4/4 mismatched**.
- Batch: 30/30 random t3 variants — stored VL image is pixel-distinct from
  both the t3 variant video frame 0 and the base teleop frame 0.

Net: for ~98% of rows you would train on an image that does not contain the
celebs / portraits the label and bbox describe.

### 3. Degenerate quads — separate writer bug (~21%)

11,744 / 56,202 rows (20.9%) have `quad_corners_norm` with only 3 distinct
corners (4th = duplicate of the 3rd). Cross-checked one against the local
cache: `quick_lecun_LSO_ep02_210147` pid0 — `portrait_corners.json` frame 0
has 4 distinct corners `[176,206],[264,294],[127.5,430.5],[39.5,342.5]`; the
manifest dropped `[264,294]` and duplicated `[127.5,430.5]`. The true quad
is intact in the cache; the generator's quad writer drops a corner.

The augmented videos themselves are fine — inpainting is clean and stable
across all 538 frames (see `eval_3/attention_steering/degenerate_videos/`).
The degenerate quad is annotation-only, not a video defect.

### 4. README ↔ manifest mismatch

README documents column `bbox_xyxy_norm` (4-number); actual parquet column
is `quad_corners_norm` (8-number flattened prompts).

## Next steps

Regenerate `eval3_track3_vl_pairs` locally from the cache:
- t3 row image = frame 0 of `eval3_track3_aug/<variant>/videos/.../file-000.mp4`
- labels = `augmentation.json pid_to_celeb_full`
- quads = base episode `portrait_corners.json` frame-0 corners (paper does not
  move under inpainting; only the face is swapped)
- base-teleop rows: image = `eval3/<ep>/frame_0.png`, labels = ArcFace
  `celebs`, quads = `portrait_corners.json` frame 0

Then re-push to HF and re-run the visual gate before any attention-steering work.
