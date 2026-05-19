# Edna handover — 200-celeb v3 augmentation

**Machine:** edna (`ites-elelnx01`), 128 cores, 503 GB RAM, 573 GB free disk.
**User on edna:** `rzendehdel` (working dir `/home/rzendehdel`).
**Goal:** generate ~10 000 image-as-prompt augmented episodes spanning all 193 celebs in the photo bank, using the 178 base teleops + per-base cached masks/corners.

This complements (not replaces) the existing 3-celeb Track 3 dataset
(`HBOrtiz/so101_eval3_track3_v3_baseline`). It's the insurance bet for the
Day-3 fallback path described in `eval_3/tracks/TRACK_B.md` §8 option 1.

---

## 0 — Where everything lives

| Item | Path on edna | Size |
|---|---|---|
| Repo | `~/LeMonkey/` | ~500 MB code + 7 GB datasets |
| Base teleops (178 episodes + caches) | `~/LeMonkey/datasets/eval3/` | 6.0 GB |
| Photo bank (193 celebs, 810 photos) | `~/LeMonkey/datasets/eval3_celebs/scraped/` | 892 MB |
| Aug pipeline source | `~/LeMonkey/eval_3/aug/generate_aug_v3.py` (+ `4_inpaint_video.py`, `_video_io.py`) | — |
| HF tokens | `~/LeMonkey/secrets/huggingface/token_hbortiz` | 12 KB |
| Conda env | `~/miniconda3/envs/aug/` (python 3.11, opencv 4.13, ffmpeg 8.1.1, insightface, pycocotools) | 8 GB |
| InsightFace `buffalo_l` model | `~/.insightface/models/buffalo_l/` | 281 MB |
| **Output (will be created)** | `~/LeMonkey/datasets/eval3_aug_v3_200celebs/` | ~15-18 GB |

Claude session/memory was also synced from the dev box — see §5 below.

---

## 1 — Smoke test (~2 min)

Always do this first on a fresh edna shell:

```bash
bash ~/LeMonkey/eval_3/scripts/edna/run_aug_v3_200celebs.sh smoke
```

Expected: 4 variants written under `/tmp/aug_v3_smoke_<timestamp>/` in ~90 s.
Worker logs should end with `✓ <variant_name> target=<celeb> bucket=<...>`.
If you see `IndexError` or `ModuleNotFoundError`, stop and inspect logs in
`/tmp/aug_v3_smoke_*/\_logs/`.

---

## 2 — Production run (~1.5-2 h wall)

Foreground (Ctrl-C kills, doesn't survive SSH disconnect):

```bash
bash ~/LeMonkey/eval_3/scripts/edna/run_aug_v3_200celebs.sh full
```

Background (survives SSH disconnect — **recommended**):

```bash
nohup bash ~/LeMonkey/eval_3/scripts/edna/run_aug_v3_200celebs.sh full \
    > ~/aug_v3.log 2>&1 &
echo $! > ~/aug_v3.pid
echo "Launched PID=$(cat ~/aug_v3.pid). Tail log: tail -f ~/aug_v3.log"
```

Defaults inside the script:

| Variable | Default | What |
|---|---|---|
| `OUT_ROOT` | `~/LeMonkey/datasets/eval3_aug_v3_200celebs` | output dir |
| `NUM_VARIANTS` | 56 | per base ep → 178 × 56 = ~10 000 total |
| `NUM_WORKERS` | 64 | parallel processes |
| `SEED` | 42 | RNG |

Override any with env var, e.g.:

```bash
NUM_VARIANTS=25 NUM_WORKERS=32 bash ~/LeMonkey/eval_3/scripts/edna/run_aug_v3_200celebs.sh full
```

---

## 3 — Monitor + verify

```bash
# Live tail one worker
tail -f ~/LeMonkey/datasets/eval3_aug_v3_200celebs/_logs/worker_00.log

# Count completed variants so far
ls ~/LeMonkey/datasets/eval3_aug_v3_200celebs | grep -v _logs | grep -v _run_summary | wc -l

# CPU + RAM check
top -bn1 | head -20
free -h | head -2

# Per-worker error count (should be 0)
grep -c '"error"' ~/LeMonkey/datasets/eval3_aug_v3_200celebs/_run_summary_w*.json | sort -t: -k2 -nr | head -10
```

When done you should see roughly:

```
==> done in ~6800s (~115 min)
    variants     : ~9968  (mp4 count ~19936)
    workers with errors in summary: 0
```

Each variant dir contains:

- `videos/observation.images.camera1/chunk-000/file-000.mp4` (inpainted wrist cam)
- `videos/observation.images.reference/chunk-000/file-000.mp4` (constant target portrait)
- `data/`, `meta/` (hardlinked from base)
- `reference.json` (target celeb, prompt, etc.)
- `augmentation.json` (provenance + workspace photo assignments)

---

## 4 — After the run

The output is in raw per-variant LeRobot-v2-ish layout. To turn it into a
single trainable LeRobot v3 dataset you'd:

1. **Merge** with the existing 178 base teleops:
   ```bash
   python ~/LeMonkey/eval_3/scripts/merge_track3_custom.py \
       --base ~/LeMonkey/datasets/eval3 \
       --aug  ~/LeMonkey/datasets/eval3_aug_v3_200celebs \
       --out  ~/LeMonkey/datasets/eval3_aug_v3_200celebs_merged
   ```
   (the merger uses hardlinks + pyarrow; ~85 s for 9 394 eps; expect similar
   for ~10 000 eps here).

2. **Push to HF** (only after you're sure of the dataset):
   ```bash
   export HF_TOKEN=$(cat ~/LeMonkey/secrets/huggingface/token_hbortiz)
   python ~/LeMonkey/eval_3/scripts/push_dataset_to_hf.py \
       --local ~/LeMonkey/datasets/eval3_aug_v3_200celebs_merged \
       --repo  HBOrtiz/so101_eval3_aug_v3_200celebs
   ```

3. **(If you'll use it for Pi0.5)** also re-run the fast quantile recompute
   (see `eval_3/scripts/fast_recompute_quantiles.py`) and push a
   `_pi05`-suffixed repo with the corrected `stats.json`.

Don't push anything until you're sure — overwriting Hans's or Sejohn's
shared dataset repos by accident is the failure mode we already pay attention
to (separate `_baseline` vs `_pi05` repos exist for exactly this reason).

---

## 5 — Resuming the Claude session on edna

The dev-box Claude project dir (sessions + memory + tool-results cache) was
rsynced to:

```
~/.claude/projects/-home-rzendehdel-LeMonkey/
├── 16ec5533-eb3a-4fc8-b854-aa8a6577e69c.jsonl   (33 MB, current session)
├── b3c2b2c1-aa13-4532-af60-85c11f3b57cd.jsonl   (200 KB, older session)
├── 16ec5533-eb3a-4fc8-b854-aa8a6577e69c/        (tool-results cache, 106 MB)
├── memory/                                       (project memory, 80 KB)
└── memory_brev_backup/                           (Brev memory backup, 52 KB)
```

To pick up where the dev-box session left off:

```bash
cd ~/LeMonkey
claude                  # starts fresh on edna BUT loads project memory
```

Claude on edna will load `memory/*.md` automatically (the index is in
`MEMORY.md`). All the project facts (Track B, Brev VM details, dataset
inventory, Pi0.5 validations, etc.) carry over.

**Caveat:** the session JSONLs reference dev-box paths like
`/home/rohamzn/ETH_Uni/LeMonkey/...`. If you ask Claude to re-read a path
from a past conversation, it'll have to translate to `~/LeMonkey/...` on
edna. Memory entries are path-agnostic and travel cleanly.

To explicitly resume the most recent session, run `claude --resume` and
pick the `16ec5533-…` one from the list.

---

## 6 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cv2` | conda env not active | `source ~/miniconda3/etc/profile.d/conda.sh && conda activate aug` |
| `ModuleNotFoundError: insightface` | not installed | `pip install insightface onnxruntime` (in the aug env) |
| `OSError: libavutil.so.X` | ffmpeg version mismatch with cv2 | env should already have conda-forge ffmpeg 8.x — `which ffmpeg` should print under miniconda |
| `IndexError: boolean index did not match … axis 1: 480 vs 640` | `find_video()` picked the reference mp4 not camera1 | already fixed in `4_inpaint_video.py:find_video`; pull latest if you see this |
| Workers idle / 0% CPU | Maybe waiting on filesystem | check `iotop`, `dmesg | tail` for kernel messages |
| RAM climbing past 200 GB | leak in torchcodec-style frame buffers | drop `NUM_WORKERS` to 32 and rerun |
| ffmpeg encoding errors | libx264 / pix_fmt issue on this ffmpeg version | check `_logs/worker_XX.log` — the script prefers `h264_nvenc` then falls back to libx264 |

---

## 7 — One-line tldr

```bash
nohup bash ~/LeMonkey/eval_3/scripts/edna/run_aug_v3_200celebs.sh full > ~/aug_v3.log 2>&1 &
tail -f ~/aug_v3.log
```

Expected wall: 1.5-2 h. Output: `~/LeMonkey/datasets/eval3_aug_v3_200celebs/` (~10 000 variants, ~15-18 GB).

---

*Written 2026-05-19 after porting + bug-fixing the aug pipeline on edna.
Two fixes vs the dev-box version: (a) worker-id striping added to v3,
(b) `find_video()` now prefers camera1 (was returning the reference mp4
on edna's filesystem ordering).*
