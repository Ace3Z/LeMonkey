# 2026-05-19 — M2 data audit: join correctness across face_labels x augmentation.json x celeb_embeddings.json

**Status:** complete. Reproduce with `eval_3/aug/dbg/dbg_m2_data_audit.py`
(`lemonkey-arcface` conda env).

## What we audited

For five representative-variant face_labels files (one per layout: LSO, SLO,
SOL, OSL, OLS — LOS was requested but is not present in the face_labels
collection, so SOL was substituted), we joined:

- `eval_3/aug/stats/celeb_embeddings.json` (manifest, 192 celebs, 1,445 photos)
- `eval_3/aug/stats/face_labels/<src>.face_labels.json` (per-frame bboxes from
  rep variant's camera1)
- `~/Downloads/eval3_track3_aug/<rep>/augmentation.json` (per-variant
  celeb-at-slot assignment via `new_layout_camera_lmr`)

…using `eval_3/aug/m2_alignment.py:build_supervision_for_frame` and asked
whether the per-slot `target_centroid` matches what's actually rendered in
the camera1 video at frame 0.

## Verification dimensions

| Check | Result |
|---|---|
| Centroid recompute (mean+normalize over per-photo .npys) matches manifest centroid for Obama, LeCun, Swift | cos = 1.000000 for all three; `|manifest_centroid|` = 1.0 (already normalized in JSON) |
| `target_centroids` from `build_supervision_for_frame` are L2-normalized to unit length | min=1.000000, max=1.000000 across all 15 frame-0 slots |
| `n_visible_faces < 3` correctly yields `valid[s]=False` and zero-vector targets for missing slots | n_keep ∈ {0,1,2,3}: all produce expected `valid` mask |
| Slot 0 = camera-POV left, slot 2 = camera-POV right (positive-x in image) | confirmed by visual inspection; matches `face_labels` left-to-right sort and `new_layout_camera_lmr` letter index |
| For each on-screen slot, ArcFace top-1 over all 192 manifest centroids matches augmentation.json's claim | 14/15 slots OK; 1/15 mismatch (see below) |

## The 5-row audit table

| Source | Slot | bbox | Expected celeb | ArcFace top-1 | Match | `|centroid|` |
|---|---|---|---|---|---|---|
| `lecun_LSO_ep01_20260511_205000` | L | (140,243,207,325) | barack_obama | barack_obama (cos=0.766) | OK | 1.000000 |
| `lecun_LSO_ep01_20260511_205000` | M | (312,312,379,388) | yann_lecun | yann_lecun (cos=0.762) | OK | 1.000000 |
| `lecun_LSO_ep01_20260511_205000` | R | (470,253,537,315) | taylor_swift | taylor_swift (cos=0.434) | OK | 1.000000 |
| `lecun_SLO_ep01_20260511_210355` | L | (145,246,213,333) | barack_obama | barack_obama (cos=0.776) | OK | 1.000000 |
| `lecun_SLO_ep01_20260511_210355` | M | (312,304,371,366) | taylor_swift | taylor_swift (cos=0.580) | OK | 1.000000 |
| `lecun_SLO_ep01_20260511_210355` | R | (475,260,552,337) | yann_lecun | yann_lecun (cos=0.762) | OK | 1.000000 |
| `lecun_SOL_ep01_20260511_212006` | L | (158,257,221,320) | taylor_swift | taylor_swift (cos=0.589) | OK | 1.000000 |
| `lecun_SOL_ep01_20260511_212006` | M | (322,298,388,388) | barack_obama | barack_obama (cos=0.811) | OK | 1.000000 |
| `lecun_SOL_ep01_20260511_212006` | **R** | (484,257,563,334) | **yann_lecun** | **taylor_swift (cos=0.632)** | **MISMATCH** | 1.000000 |
| `obama_OSL_ep01_20260511_200348` | L | (164,245,207,292) | barack_obama | barack_obama (cos=0.713) | OK | 1.000000 |
| `obama_OSL_ep01_20260511_200348` | M | (294,297,359,376) | yann_lecun | yann_lecun (cos=0.753) | OK | 1.000000 |
| `obama_OSL_ep01_20260511_200348` | R | (459,237,540,315) | taylor_swift | taylor_swift (cos=0.557) | OK | 1.000000 |
| `swift_OLS_ep01_20260511_192524` | L | (147,262,207,324) | taylor_swift | taylor_swift (cos=0.559) | OK | 1.000000 |
| `swift_OLS_ep01_20260511_192524` | M | (300,304,378,396) | barack_obama | barack_obama (cos=0.772) | OK | 1.000000 |
| `swift_OLS_ep01_20260511_192524` | R | (482,249,584,350) | yann_lecun | yann_lecun (cos=0.825) | OK | 1.000000 |

## The one mismatch — analysis

For source `quick_lecun_SOL_ep01_20260511_212006`, rep variant `__t3_0000_v00`:
- `augmentation.json` claims `new_layout_camera_lmr = "SOL"` → slot R should be
  `yann_lecun`.
- The camera1.mp4 at frame 0 (and frames 50, 100, 200, 300, 400) shows a
  **Taylor Swift portrait** at the right slot. ArcFace agrees: cos(slot_R,
  taylor_swift) = 0.632 vs cos(slot_R, yann_lecun) = 0.014.
- This is **not** a noisy single-frame detection; the inpainted photo is
  Taylor Swift throughout the episode.

### Wider sweep

Auditing all 151 representative-variant face_labels files (frame 0) against
their augmentation.json revealed **4 mismatching source episodes (2.6%)**:

- `quick_lecun_SLO_ep01_20260511_211540` (slot R: expected lecun, got swift)
- `quick_lecun_SLO_ep04_20260511_211734` (slot R: expected lecun, got swift)
- `quick_lecun_SOL_ep01_20260511_212006` (slot R: expected lecun, got swift)
- `quick_obama_SLO_ep05_20260511_204851` (slot R: expected lecun, got swift)

All four have the same pattern:
- `orig_layout_camera_lmr[2] = "S"` (original right-slot photo was Swift)
- `new_layout_camera_lmr[2] = "L"` (the variant should have LeCun replace it)
- The rendered video still shows Swift at slot R

For source `quick_lecun_SOL_ep01_20260511_212006` (63 variants total), 17/17
variants where `new_lmr[2]=L` are corrupted — i.e. **100% of LeCun-at-R
variants for this source render Swift instead**. Within the 66 source
episodes whose rep variant has this orig=S/new=L right-slot replacement
configuration, 4 fail — sporadic, not 100%, so the failure mode is some
property of the specific variant rather than a deterministic layout-pair
rule.

The same failure mode is plausible for non-rep variants of the same source
episodes (each source has up to ~64 variants pasting different reference
photos).

## Root-cause hypothesis

The augmentation pipeline (`generate_aug_track3.py`) sometimes fails to
**replace** a right-slot photo when going from a Swift-at-R original to a
LeCun-at-R inpainted variant. The original Swift photo is left in place.
This is **not** a problem in `m2_alignment.py:build_supervision_for_frame`
itself — the join logic is correct; the upstream data is corrupted.

A `[SHORTCUT]` likely sits in `generate_aug_track3.py`'s right-slot replace
path; whether it's `4_inpaint_video.py` or the segmentation/quad re-projection
step is to be determined.

## Impact on M2 training

- **Magnitude**: at least 4/151 source episodes (≥ 2.6%) have at least one
  slot with miscoded supervision at frame 0. Across all 9,216 variants the
  rate could be similar or higher (TBD by a per-variant sweep).
- **Effect**: in mismatched frames, the loss pulls the slot-R hidden state
  toward `yann_lecun`'s centroid while the patches show `taylor_swift`. A
  small fraction (≤ 3%) is unlikely to break training; with `λ = 0.2` the
  net pull from corrupted slots is ~0.6% of the total loss. Not catastrophic,
  but logged so we can correct later.
- **Action items**:
  1. (P2) Per-variant ArcFace sanity sweep over all 9,216 variants —
     identify exact corrupted set and either drop those variants from the
     M2-loss frames or regenerate them with the fixed inpaint.
  2. (P3) Root-cause `generate_aug_track3.py`'s right-slot Swift→LeCun
     replacement bug.
  3. (P2) Add a `bbox_valid` override: if `top1` from ArcFace disagrees with
     `new_layout_camera_lmr[s]` at training time (offline-checkable since
     bboxes are static per source), set `valid[b, s] = False` for that
     frame. Costs one offline pass over all bbox crops.

## Conclusion

**The M2 join logic in `build_supervision_for_frame` is correct.** The
manifest, the L2 normalisation, the slot ordering (camera POV left→right
maps to L→M→R letter index), and the partial-data path (`n_visible_faces<3`)
all check out.

**The training data is mostly trustworthy, with a small known systematic
bug.** ~2.6% of source episodes have slot-R supervision corrupted when the
original right-slot photo was Swift and the variant wanted LeCun there.

For the 4-day sprint, the M2 toolkit (Track D / Mahbod) can ship the loss
implementation as-is. Track D ships behind a guard that warns on this
specific pattern and recommends running the action-item sweep before the
loss is wired into a production training run.
