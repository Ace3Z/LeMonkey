# Track D (M2 ArcFace cosine distillation) — Brev launch log

**Date:** 2026-05-19
**Operator:** Mahbod (laptop) → Brev VM `time2sleep` (IP 185.216.22.170, A100 80GB)
**Branch:** `dev/m2-arcface-toolkit` @ commit `b8009b1`
**Output:** `HBOrtiz/smolvla_eval3_track_D_m2_mahbod` (push enabled)

## TL;DR

5-step smoke test green, full 30k-step run launched in tmux session `m2` on
`time2sleep`. Reconnect with `ssh -t time2sleep tmux attach -t m2`.

## What we trained

- **Policy:** `lerobot/smolvla_base` (SmolVLA-450M, 16 of 32 VLM layers used)
- **VLM:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` — *vanilla*, not Hans's
  warm-VLM (his repo `HansOrtiz/smolvlm2_celeb_warm` was 404 on HF at launch
  time; track A/C are responsible for the warm-VLM path).
- **Dataset:** `HBOrtiz/so101_eval3_track3_v3_baseline` v3.0 (9,394 episodes /
  5,053,972 frames / ~15 GB camera1 + reference).
- **M2 supervision:** the toolkit we built earlier — face_labels/ (151
  sources), celeb_embeddings.json (192 celeb centroids), arcface_embeddings/
  (per-photo embeddings), episode_mapping.json (9,394 entries: 178 base +
  9,216 aug).
- **Hyperparameters:** layer-9 capture (depth-matched to BlindVLA Eq. 9),
  λ=0.2, lr=5e-5, bs=64, steps=30,000, save_freq=5,000.

## Smoke-test result (5 steps, bs=8)

```
[m2 launcher] wrapped policy: 88.5M frozen, 68.8M trainable
[m2 wrapper] built episode_starts lookup (9394 entries)
[m2] step=     0  m2_loss=+0.0143  n_valid=23/24  mean_cos=-0.0143  base=0
[m2] step=     1  m2_loss=+0.0063  n_valid=22/24  mean_cos=-0.0064  base=0
[m2] step=     2  m2_loss=+0.0082  n_valid=22/24  mean_cos=-0.0082  base=0
[m2] step=     3  m2_loss=+0.0045  n_valid=21/24  mean_cos=-0.0046  base=0
[m2] step=     4  m2_loss=+0.0171  n_valid=18/24  mean_cos=-0.0171  base=1
Training: 100%|██████████| 5/5 [00:21<00:00,  2.05s/step]
```

Reading the numbers:
- **`88.5M frozen / 68.8M trainable`** — partial-freeze: VLM layers 0-8
  frozen (Hans's prior path), layers 9-15 + action expert trainable. Matches
  the BlindVLA recipe split.
- **`n_valid=18-23 / 24`** — 75-96 % slot validity per batch. Three slots
  per sample × 8 samples = 24 candidate alignment targets; we lose the
  ones whose source is in the excluded-sources list or whose face_labels
  frame has no detection.
- **`mean_cos ≈ -0.01`** — expected at init: the frozen projector head
  (LN → 960→2048→2048→512 with random weights) outputs near-zero cosine
  against ArcFace centroids before any training.
- **`base=0-1`** — a handful of base teleop samples land in each batch; M2
  doesn't apply supervision to those (no face-swap to align), and the
  launcher correctly reports them as a no-op.

## Sequence of fixes that got the smoke test green

1. **Dataset try_load → False** because 5 camera1 video files were missing
   in `~/.cache/huggingface/lerobot/HBOrtiz/so101_eval3_track3_v3_baseline/videos/observation.images.camera1/chunk-002/` (file-960/962/965/968/969.mp4). Cache only had 9,389 of 9,394 mp4s.
   - Fix: enumerated missing files from `meta/episodes/*.parquet`, downloaded the 5 directly with `hf_hub_download` into `/tmp/hf_dl` then moved them into the cache (worked around the huggingface_hub Windows-long-path bug on Linux).
2. **`has_legacy_hub_download_metadata` → re-download path** because
   `hf_hub_download` leaves behind a `.cache/huggingface/download/` marker.
   - Fix: launcher (`eval_3/scripts/lerobot_train_with_m2.py`) now wipes
     `<cache>/<repo>/.cache` before any LeRobotDataset is constructed.
     Commit `e32e069`.
3. **`SmolVLMVisionEmbeddings.forward` → boundaries on CPU vs CUDA**
   (transformers 4.55.0 bug).
   - Fix: launcher monkey-patches that method to build `boundaries` and
     `position_ids` on `pixel_values.device`. No site-packages edit.
     Commits `27626e6` + `fd7ab38`.
4. **`KeyError: 'frame_index'` from `M2WrappedPolicy._extract_indices`**
   — LeRobot v3 batches emit only `index` (global) + `episode_index`, no
   per-episode `frame_index`.
   - Fix: wrapper lazily reads `meta/episodes/*.parquet`, builds an
     `episode_index → dataset_from_index` lookup, and derives
     `frame_idx = global_index - dataset_from_index[episode_index]`. Commit
     `07d4f49`.
5. **`HansOrtiz/smolvlm2_celeb_warm` 404 on HF** — Hans hasn't pushed yet.
   - Decision: vanilla `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
     (Mahbod's track is M2-toolkit-validation, not warm-VLM evaluation).
     Comment in `run_training_track_D_m2.sh` flags the swap so we can
     revert when the warm checkpoint lands. Commit `b8009b1`.
6. **`augmentation.json` missing on Brev** — the 9,216 aug variant
   directories were never uploaded; only the merged HF dataset was.
   - Fix: tarred just the 9,216 `augmentation.json` files (918 KB total)
     from `~/Downloads/eval3_track3_aug` on the laptop, scp'd to
     `~/LeMonkey/datasets/eval3_track3_aug/` on Brev, extracted preserving
     variant dirs. Inpainted frames already live in the merged dataset, so
     this metadata-only bundle is the only piece M2 still needs at
     train-time.

## Full launch command (running now inside tmux `m2`)

`bash eval_3/scripts/brev/run_training_track_D_m2.sh` — see that file for
the exact flags. Wall-time estimate: 5-8 h on A100 80GB. Log:
`~/outputs/train/smolvla_track_D_m2.log`. Reconnect:
`ssh -t time2sleep tmux attach -t m2`.

## What's next

- Monitor the run periodically (m2_loss should drift downward over the
  first ~2 k steps; mean_cos should climb from near-zero to positive once
  the projector + late SmolLM2 layers find the alignment subspace).
- On finish, push autoland to `HBOrtiz/smolvla_eval3_track_D_m2_mahbod`
  (already wired via `--policy.push_to_hub=True`).
- Hand to Darius for Strix deployment + the 3-rollout protocol (TODO.md
  Day 2 / Day 3 plan).
