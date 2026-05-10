# eval_3/aug — identity-preserving inpainting augmentation

Turns recorded SO-101 demos into ~10× as many effective training episodes
by replacing the printed celebrity portraits in each frame with different
photos of the same celebrities, mined from the web. Action labels stay
byte-identical; only the camera video changes. The companion strategy
write-up is in [`STRATEGY.md`](STRATEGY.md).

## Module map

```
1_mine_celeb_photos.py     stage 1 — Wikimedia + icrawler + ArcFace verifier (web → photo bank)
2_segment_video.py         stage 2 — GroundingDINO/click + SAM 2.1 video propagator (mp4 → masks.pkl)
3_extract_corners.py       stage 3 — masks → ordered quadrilateral + occlusion interpolation
4_inpaint_video.py         stage 4 — homography + Reinhard + Poisson NORMAL_CLONE + encode (Recommended tier)
5_verify_identity.py       stage 5 — ArcFace cosine ≥ 0.4 quality gate
pipeline.py                orchestrator: runs 2 → 3 → 4 → 5 in order
dbg/
  dbg_mask_overlay.py      after stage 2: overlay masks on frame 0 (sanity check segmentation)
  dbg_compare_gif.py       after stage 4: side-by-side animated GIF of original vs augmented
```

## Install (Brev x86 with discrete GPU recommended)

```bash
pip install torch==2.5 torchvision opencv-contrib-python-headless==4.10.*
pip install "git+https://github.com/facebookresearch/sam2.git"
mkdir -p ~/checkpoints
wget -P ~/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
pip install insightface onnxruntime-gpu icrawler wikipedia-api requests pillow
pip install pycocotools decord ffmpeg-python tqdm
```

## Quickstart — process the 5 quick-record episodes end-to-end

```bash
# 0. Unpack the bundle (assumes you've already scp'd it home)
mkdir -p ~/eval3_at_home && cd ~/eval3_at_home
tar -xzf eval3_quick_episodes.tar.gz
ln -s ~/eval3_at_home ~/LeMonkey/datasets/eval3_quick   # so default paths work

# 1. Mine 30 verified photos per IID celeb (one-time, ~10 min)
python eval_3/aug/1_mine_celeb_photos.py --celebs swift obama lecun --num 30

# 2-5. Run the pipeline. First run uses --interactive so you click the 3
#      portrait centres on frame 0 of each episode (~30 s human time total).
python eval_3/aug/pipeline.py --root ~/LeMonkey/datasets/eval3_quick \
                              --num-variants 5 --interactive

# Spot-check: side-by-side GIFs of the first few augmented variants
python eval_3/aug/dbg/dbg_compare_gif.py --root ~/LeMonkey/datasets/eval3_aug --first 5
```

After the first interactive run, the click points are saved to
`<episode_dir>/portrait_seeds.json`. Subsequent runs (or runs on the same
episodes from a different machine) need no UI.

## What each stage produces (and where)

| Stage | Reads | Writes |
|---|---|---|
| 1 | (web) | `~/LeMonkey/datasets/eval3_celebs/web/<celeb>/<id>_cos<NNN>.jpg` |
| 2 | `<ep>/videos/.../file-000.mp4` | `<ep>/portrait_masks.pkl` (~3 MB), `<ep>/portrait_seeds.json` |
| 3 | `<ep>/portrait_masks.pkl` | `<ep>/portrait_corners.json` |
| 4 | corners + masks + photo bank | `<out>/{ep_name}__var<NN>/` (full LeRobot v3 dataset, but with augmented mp4 + new sidecar) |
| 5 | augmented variants + reference photos | `<variant>/verification.json` |

## Failure-mode cheat sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Stage 2 errors `no portrait_seeds.json` | First run, no clicks yet | Add `--interactive` to pipeline.py |
| Mask overlay (`dbg_mask_overlay.py`) shows mask catching the table or fingers | Click was off-centre or near a hand | Edit `<ep>/portrait_seeds.json` and re-run stage 2 with `--force` |
| Stage 4 errors `no photos for celeb 'X'` | Stage 1 hasn't been run for that celeb yet | `python 1_mine_celeb_photos.py --celebs X --num 30` |
| Stage 4 fast-fails on `layout 'SOL' is required` | Episode's reference.json has `layout: "-"` | Edit the sidecar to specify SOL/SLO/OSL/OLS/LSO/LOS |
| Augmented mp4 looks crisp but original looks blurry | Camera MTF mismatch | Increase `mtf_sigma` in `4_inpaint_video.py:replace_portrait` (try 1.0) |
| Stage 5 rejects everything | Identity threshold too strict OR wrong reference photo selected | Lower `--threshold` to 0.35; verify `--photo-bank` is the right dir |
| Visible seam at portrait boundary in `dbg_compare_gif.py` | Reinhard ring under-sampled (deep shadow at boundary) | Increase `ring_dilate_px` in `4_inpaint_video.py:replace_portrait` (try 17) |

## Smoke milestones (per STRATEGY.md §8)

- M1 — `1_mine_celeb_photos.py` produces 30 verified Swift photos.
- M2 — `2_segment_video.py` produces correct masks on quick_swift_SOL_ep01.
  Verify with `dbg/dbg_mask_overlay.py`.
- M3 — `3_extract_corners.py` produces stable corners across all 600
  frames. Verify with a quick numpy plot of corner trajectories vs time.
- M4 — `4_inpaint_video.py` produces a single augmented variant that
  looks natural via `dbg/dbg_compare_gif.py`.
- M5 — `5_verify_identity.py` confirms the M4 variant's ArcFace
  cosine ≥ 0.4 across all 5 sampled frames.
- M6 — `pipeline.py` produces 25 augmented variants (5 episodes × 5
  variants). End-to-end smoke test passes.

After M6, scale to the main 144-ep collection.
