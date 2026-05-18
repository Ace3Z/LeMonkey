# 2026-05-19 — M2 data foundation: ArcFace cache + face_labels build

**Status:** in progress (re-run with `det_size=640` queued); first-run with
`det_size=320` complete and audited. Successor doc will report the re-run.

**Branch:** [`dev/m2-arcface-toolkit`](https://github.com/Ace3Z/LeMonkey/tree/dev/m2-arcface-toolkit)

## Goal

Stand up the two data artifacts the M2 ArcFace cosine-distillation training
loss needs:

1. **`celeb_embeddings.json` + `<photo>.arcface.npy`** — per-photo and
   per-celeb-centroid ArcFace embeddings covering *all* photos in the bank
   (`heldout/` + `scraped/`).
2. **`<source>.face_labels.json`** (151 files) — per-frame face bboxes for
   one representative variant per source episode (`v00`-style). The fixed-
   camera setup means variants of the same source share pixel positions, so
   151 detections cover the whole 9,216-variant aug set.

These are training-time-only artifacts. The eval-day policy contract is
unchanged (`camera1 + state + text_prompt` only).

## What we built (new files)

- [eval_3/aug/cache_arcface_embeddings.py](../../eval_3/aug/cache_arcface_embeddings.py) — caches
  `buffalo_l` ArcFace embeddings for the celeb bank. Idempotent (skips if
  `.arcface.npy` already exists). Writes a manifest.
- [eval_3/aug/build_face_labels.py](../../eval_3/aug/build_face_labels.py) — groups variant
  directories by source-episode prefix (regex `(.+)__t3_\d+_v\d+$`), runs
  RetinaFace on one representative per group, interpolates between
  keyframes (`--stride`). Outputs one `*.face_labels.json` per source.
- [eval_3/aug/dbg/dbg_face_labels.py](../../eval_3/aug/dbg/dbg_face_labels.py) — visual gate:
  bbox overlays on selected frames, centroid-similarity heatmap, matcher
  sanity grid.
- [eval_3/aug/dbg/dbg_ood_matcher.py](../../eval_3/aug/dbg/dbg_ood_matcher.py) — leave-one-out
  validation of the bank: for each photo, recompute its celeb's centroid
  without it and check nearest-centroid identity recall.

## Environment

`conda create -n lemonkey-arcface python=3.11`, then
`pip install "numpy<2.0" opencv-python onnxruntime insightface matplotlib`.
On Apple Silicon (arm64, macOS 26.3.1, conda 25.1.1), CPU-only execution.
buffalo_l detection at `det_size=640` ≈ 6-7 s per 536-frame video at
stride=5.

## What we measured

### Cache step (1,445 photos across heldout + scraped)

- 192 celebs ingested; 0 failed photos. Median 8 photos/celeb, range 2-29.
- All embeddings shape `(512,)` float32, L2-norm = 1.000000.
- IID same-pair cosines: Swift 0.530, Obama 0.678, LeCun 0.717.
- Cross-pair cosines: all in [-0.13, +0.15].
- Per-celeb centroid quality: photo↔centroid cos 0.73-0.96 (mean 0.83).
- **Leave-one-out top-1 accuracy across the entire 1,445-photo bank:
  1,438/1,445 = 99.5%**. Per bucket: 100% at 2-7 photos, 99.2% at 8-12
  photos, 100% at 13+ photos. The 99.2% bucket has the 7 misses, 6 of which
  are all `oier_mees` (8 photos) — his scraped bank is structurally broken
  (intra-celeb own_cos 0.05-0.13, essentially noise). One `will_ferrell`
  in-character miss. Both are bank-curation issues, not ArcFace issues.

### Build step (151 sources, first run with `det_size=320`)

- All 151 files present, schema valid, frame count 536 each, 39 MB total.
- Bbox position stability per source (fixed-camera invariant check):
  median std 0.10-0.16 px across slots 0/1/2 — confirms the camera is
  effectively static and portraits don't move.
- **Issue found:** mean pct-of-frames-with-3-faces is only 50.4%; 98/151
  sources have <70% coverage; 5 have 0% (never all-3 visible).
- Root cause (probed): not occlusion. At `det_size=320` one face is
  routinely below the detection-size threshold for SOL-layout recordings.
  Re-probed at `det_size={480, 640, 800}`: all 3 faces detected with
  scores 0.7-0.9. Re-running with `det_size=640`.

### Visual gates produced

`eval_3/aug/stats/face_labels_dbg/`:
- `centroid_similarity_iid_plus_ood.png` (8×8 cosine heatmap, diagonal-bright)
- `source_overlays/<src>__{frame0,mid,occlN}.png` (18 panels across 6 sources)
- `matcher_sanity__<src>.png` (3-row grid: bbox crops vs reference photos + top-3 predictions)
- `ood_loo_summary.png` (bucket accuracy + per-photo cosine/margin scatter)
- `ood_sample_grid.png` (12 random OOD celebs with LOO predictions)
- `coverage_issue/<src>_frame0_overlay_ds640.png` (probe overlay showing det_size matters)

In the 6 sources sampled (3 frames each = 18 overlays), the matcher
identified 18/18 bboxes correctly against the full 192-celeb gallery
(cosines 0.538-0.833 vs cross-celeb max 0.15 — ~5× margin).

## Surprises

1. **`det_size=320` undersizes for our recordings.** InsightFace's default
   works on portrait photos because the face fills the frame; here the
   faces are ~10% of a 640×480 frame and need a larger det resolution.
   Bump to 640.
2. **`oier_mees` scraped bank is broken.** 6/8 photos misclassify under
   LOO. Probably a name-collision scrape or low-quality web hits. Should
   be removed from the OOD pool (or re-scraped) before any reliance on it.
3. **The fixed-camera invariant is real.** Slot 0/1 x_center std ≈ 0.1 px
   across 100+ keyframes per source. The original docs (and my initial
   plan) assumed a wrist-mounted moving camera — that's wrong, the camera
   is stationary. Simplifies per-frame face-region computation
   significantly.

## Next steps

After the `det_size=640` re-run finishes:

1. **Re-validate** the new face_labels: expect mean pct-of-3-faces ≥ 80%
   and 0 sources at <50%.
2. **Refresh the dbg visuals** on the same 6 sources + the 5 previously-0%
   sources for confirmation.
3. **Write the M2 alignment head** (`eval_3/aug/m2_alignment_head.py`):
   per-bbox feature pooling + frozen MLP projector + contrastive cosine
   against per-pid ArcFace target. Reads face_labels + celeb_embeddings +
   augmentation.json.
4. **Write the dataloader hook** (`eval_3/aug/m2_dataloader.py`): maps
   pixel-space bboxes to SmolVLM patch indices for the SmolVLA 512×512
   resize → 2×2 pixel-shuffle → 16×16 patch grid.
5. **Patch `modeling_smolvla.py`** minimally via `register_forward_hook`
   on the SmolLM2 layer of choice (probably 12 of 16 — not 8 as the docs
   say; BlindVLA used layer 16 of 28 in OpenVLA's LLM, the analogous
   late-mid position in a 16-layer truncation is ~12).
6. **Local smoke test**: load 5 frames, attach hook, verify
   `loss.backward()` produces non-zero gradients in unfrozen layers
   8-15 of the VLM (catches the `train_expert_only=True` silent-no-op
   blocker).
7. **Brev re-train** mirroring Hans's Track A launch with
   `train_expert_only=False` + partial freeze, output to
   `HBOrtiz/smolvla_eval3_track_A_m2_mahbod`. Blocked on Roham green-
   lighting the flag change (or shipping under a distinct repo name).

## Open questions for Roham

1. Confirm `train_expert_only=False` (load-bearing — the M2 loss is a
   no-op otherwise).
2. Whether to add a Path C+ stage on `scraped/` for post-2017 OOD
   coverage (VGGFace2 ends in 2017; figures like Aravind Srinivas,
   Chappell Roan, Liang Wenfeng, Daniela Amodei need this stage).
3. The 151 source `portrait_corners.json` from his original SAM-2 pass
   (we now have RetinaFace bboxes which are smaller and identity-
   focused, but his SAM quads would be useful if we ever extend to
   per-paper masking).
