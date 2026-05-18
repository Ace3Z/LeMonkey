#!/usr/bin/env python3
"""Custom fast Track 3 v3 merger.

Bypasses lerobot's aggregate_datasets() which is single-threaded and
spends most of its time calling ffprobe on every mp4. We:

  1. Skip ffprobe entirely (we know every video is 538 frames @ 30 fps =
     17.93s — hardcoded after user confirmation).
  2. Hardlink videos instead of copy (~ 0 I/O cost on same filesystem).
  3. Use pyarrow.concat_tables for parquet aggregation (vectorised; way
     faster than pandas-loop per-ep).
  4. Aggregate per-ep stats (from each ep's meta/episodes/.../parquet)
     into a global stats.json using pooled-variance formulas — also
     vectorised.

Produces a LeRobot v3 dataset structure byte-compatible with what
aggregate_datasets() produces: same meta/info.json schema, same
data/chunk-NNN/file-NNN.parquet layout, same meta/episodes columns.

Expected wall time on dev box: 2–5 minutes for 9,394 episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FRAMES_PER_EP = 538          # user-confirmed; we don't ffprobe
FPS = 30
DURATION_S = FRAMES_PER_EP / FPS    # 17.9333…
CHUNKS_SIZE = 1000           # files per chunk dir (same as existing merged datasets)
DATA_FILES_SIZE_MB = 100     # split data parquet when bigger
VIDEO_FILES_SIZE_MB = 0.01   # → one mp4 per episode


def discover(base_root: Path, aug_root: Path, aug_pattern: str) -> list[Path]:
    base = sorted(p for p in base_root.iterdir()
                    if p.is_dir() and (p / "meta" / "info.json").is_file()
                    and (p / "reference.json").is_file()) if base_root.is_dir() else []
    aug = sorted(p for p in aug_root.iterdir()
                   if p.is_dir() and aug_pattern in p.name
                   and (p / "meta" / "info.json").is_file()) if aug_root.is_dir() else []
    return base + aug


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3"))
    p.add_argument("--aug-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_track3_aug"))
    p.add_argument("--aug-pattern", default="__t3_")
    p.add_argument("--dst", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_track3_v3_merged"))
    p.add_argument("--robot-type", default="so_follower")
    p.add_argument("--video-keys", nargs="+",
                   default=["observation.images.camera1", "observation.images.reference"])
    args = p.parse_args()

    t_start = time.time()
    ep_dirs = discover(args.base_root, args.aug_root, args.aug_pattern)
    n_ep = len(ep_dirs)
    n_base = sum(1 for d in ep_dirs if d.parent == args.base_root)
    print(f"[1/8] discovered {n_ep} eps (base={n_base}, aug={n_ep - n_base})", flush=True)
    if n_ep == 0:
        return 1

    # ── 2. Copy info.json/features schema from first ep, update top-level fields ──
    src_info = json.loads((ep_dirs[0] / "meta" / "info.json").read_text())
    total_frames = n_ep * FRAMES_PER_EP
    out_info = {
        "codebase_version": src_info.get("codebase_version", "v3.0"),
        "robot_type": args.robot_type,
        "total_episodes": n_ep,
        "total_frames": total_frames,
        "total_tasks": 0,         # filled later
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": DATA_FILES_SIZE_MB,
        "video_files_size_in_mb": VIDEO_FILES_SIZE_MB,
        "fps": FPS,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": src_info.get("features", {}),
    }
    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / "meta").mkdir(parents=True, exist_ok=True)
    print(f"[2/8] info.json scaffold ready ({total_frames} total frames)", flush=True)

    # ── 3. Hardlink videos. Each ep gets global episode_index → chunk = idx // CHUNKS_SIZE, file = idx % CHUNKS_SIZE ──
    print(f"[3/8] hardlinking {n_ep * len(args.video_keys)} videos...", flush=True)
    t = time.time()
    for global_idx, src_dir in enumerate(ep_dirs):
        chunk_idx = global_idx // CHUNKS_SIZE
        file_idx  = global_idx %  CHUNKS_SIZE
        for vkey in args.video_keys:
            src_mp4 = src_dir / "videos" / vkey / "chunk-000" / "file-000.mp4"
            if not src_mp4.is_file():
                print(f"  [WARN] missing video: {src_mp4}", flush=True)
                continue
            dst_mp4 = args.dst / "videos" / vkey / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            if dst_mp4.exists():
                dst_mp4.unlink()
            try:
                os.link(src_mp4, dst_mp4)
            except OSError:
                shutil.copy(src_mp4, dst_mp4)
        if (global_idx + 1) % 2000 == 0:
            print(f"     {global_idx + 1}/{n_ep}", flush=True)
    print(f"     done in {time.time() - t:.1f}s", flush=True)

    # ── 4. Build global tasks.parquet (union of unique prompts) ──
    print(f"[4/8] collecting prompts...", flush=True)
    t = time.time()
    task_strings: dict[str, int] = {}
    per_ep_task: list[str] = []
    for src_dir in ep_dirs:
        tdf = pq.read_table(src_dir / "meta" / "tasks.parquet").to_pandas().reset_index()
        task = tdf["task"].iloc[0]
        per_ep_task.append(task)
        if task not in task_strings:
            task_strings[task] = len(task_strings)
    out_info["total_tasks"] = len(task_strings)
    sorted_tasks = sorted(task_strings.items(), key=lambda kv: kv[1])
    pa.table({
        "task_index": pa.array([i for _, i in sorted_tasks], type=pa.int64()),
        "task":       pa.array([t for t, _ in sorted_tasks], type=pa.string()),
    }).to_pandas().set_index("task").to_parquet(args.dst / "meta" / "tasks.parquet")
    print(f"     {len(task_strings)} unique prompts ({time.time() - t:.1f}s)", flush=True)

    # ── 5. Concat data parquets — re-index episode_index / index / task_index ──
    print(f"[5/8] merging data parquets...", flush=True)
    t = time.time()
    data_chunks: list[pa.Table] = []
    cur_size_b = 0
    cur_chunk_idx = 0
    cur_file_idx = 0
    target_bytes = int(DATA_FILES_SIZE_MB * 1024 * 1024)
    # ep → (data_chunk_idx, data_file_idx); records into episodes parquet later
    ep_to_data_chunkfile: list[tuple[int, int]] = []
    out_data_dir = args.dst / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    global_index = 0
    for global_idx, src_dir in enumerate(ep_dirs):
        t_ep = pq.read_table(src_dir / "data" / "chunk-000" / "file-000.parquet")
        n = t_ep.num_rows
        # Re-index: episode_index → global_idx; index → global_index..global_index+n; task_index → global task id
        new_episode_index = pa.array([global_idx] * n, type=pa.int64())
        new_index         = pa.array(list(range(global_index, global_index + n)), type=pa.int64())
        new_task_index    = pa.array([task_strings[per_ep_task[global_idx]]] * n, type=pa.int64())
        t_ep = t_ep.set_column(t_ep.schema.get_field_index("episode_index"),
                                "episode_index", new_episode_index)
        t_ep = t_ep.set_column(t_ep.schema.get_field_index("index"),
                                "index", new_index)
        t_ep = t_ep.set_column(t_ep.schema.get_field_index("task_index"),
                                "task_index", new_task_index)
        data_chunks.append(t_ep)
        global_index += n
        ep_to_data_chunkfile.append((cur_chunk_idx, cur_file_idx))
        cur_size_b += t_ep.nbytes
        # Flush data file when we exceed target size
        if cur_size_b >= target_bytes:
            big = pa.concat_tables(data_chunks, promote_options="default")
            out_file = out_data_dir / f"chunk-{cur_chunk_idx:03d}" / f"file-{cur_file_idx:03d}.parquet"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(big, out_file, compression="zstd")
            data_chunks = []
            cur_size_b = 0
            cur_file_idx += 1
            if cur_file_idx >= CHUNKS_SIZE:
                cur_file_idx = 0
                cur_chunk_idx += 1
    # Flush remaining
    if data_chunks:
        big = pa.concat_tables(data_chunks, promote_options="default")
        out_file = out_data_dir / f"chunk-{cur_chunk_idx:03d}" / f"file-{cur_file_idx:03d}.parquet"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(big, out_file, compression="zstd")
    print(f"     done in {time.time() - t:.1f}s (final ({cur_chunk_idx},{cur_file_idx}) split point)", flush=True)

    # ── 6. Build global episodes parquet by concatenating per-ep ones, re-indexing ──
    print(f"[6/8] merging episodes parquet (per-ep stats preserved)...", flush=True)
    t = time.time()
    ep_tables: list[pa.Table] = []
    cumulative_frames = 0
    for global_idx, src_dir in enumerate(ep_dirs):
        eps_pq = src_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        t_eps = pq.read_table(eps_pq)
        n_frames = t_eps.column("length").to_pylist()[0]  # should be 538
        # Override episode_index, tasks, dataset_from/to_index, data/chunk + data/file,
        # videos/<k>/chunk_index + file_index + from_timestamp + to_timestamp,
        # meta/episodes/chunk_index + file_index
        dc, df = ep_to_data_chunkfile[global_idx]
        vc = global_idx // CHUNKS_SIZE
        vf = global_idx %  CHUNKS_SIZE
        replacements: dict[str, pa.Array] = {
            "episode_index": pa.array([global_idx], type=pa.int64()),
            "tasks": pa.array([[per_ep_task[global_idx]]]),   # list-of-string
            "data/chunk_index": pa.array([dc], type=pa.int64()),
            "data/file_index":  pa.array([df], type=pa.int64()),
            "dataset_from_index": pa.array([cumulative_frames], type=pa.int64()),
            "dataset_to_index":   pa.array([cumulative_frames + n_frames], type=pa.int64()),
            "meta/episodes/chunk_index": pa.array([0], type=pa.int64()),
            "meta/episodes/file_index":  pa.array([0], type=pa.int64()),
        }
        for vkey in args.video_keys:
            replacements[f"videos/{vkey}/chunk_index"] = pa.array([vc], type=pa.int64())
            replacements[f"videos/{vkey}/file_index"]  = pa.array([vf], type=pa.int64())
            replacements[f"videos/{vkey}/from_timestamp"] = pa.array([0.0], type=pa.float64())
            replacements[f"videos/{vkey}/to_timestamp"]   = pa.array([float(DURATION_S)], type=pa.float64())

        cols = []
        for name in t_eps.schema.names:
            if name in replacements:
                cols.append(replacements[name])
            else:
                cols.append(t_eps.column(name))
        t_eps = pa.table(cols, names=t_eps.schema.names)
        ep_tables.append(t_eps)
        cumulative_frames += n_frames

    big_eps = pa.concat_tables(ep_tables, promote_options="default")
    out_eps = args.dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_eps.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(big_eps, out_eps, compression="zstd")
    print(f"     {big_eps.num_rows} eps; {time.time() - t:.1f}s", flush=True)

    # ── 7. Aggregate global stats.json from per-ep stats columns in big_eps ──
    print(f"[7/8] aggregating global stats from per-ep stats columns...", flush=True)
    t = time.time()
    stats: dict[str, dict[str, list[float] | float | int]] = {}
    # discover which features have stats
    feature_keys: set[str] = set()
    for name in big_eps.schema.names:
        if name.startswith("stats/") and name.endswith("/count"):
            fk = name[len("stats/"):-len("/count")]
            feature_keys.add(fk)
    for fk in sorted(feature_keys):
        cnt = np.asarray(big_eps.column(f"stats/{fk}/count").to_pylist())
        mean = np.asarray(big_eps.column(f"stats/{fk}/mean").to_pylist())
        std  = np.asarray(big_eps.column(f"stats/{fk}/std").to_pylist())
        mn   = np.asarray(big_eps.column(f"stats/{fk}/min").to_pylist())
        mx   = np.asarray(big_eps.column(f"stats/{fk}/max").to_pylist())
        q01  = np.asarray(big_eps.column(f"stats/{fk}/q01").to_pylist())
        q10  = np.asarray(big_eps.column(f"stats/{fk}/q10").to_pylist())
        q50  = np.asarray(big_eps.column(f"stats/{fk}/q50").to_pylist())
        q90  = np.asarray(big_eps.column(f"stats/{fk}/q90").to_pylist())
        q99  = np.asarray(big_eps.column(f"stats/{fk}/q99").to_pylist())
        # cnt is [n_eps] or [n_eps, dims]; broadcast as needed
        if cnt.ndim == 1 and mean.ndim == 2:
            cnt_b = cnt[:, None]
        else:
            cnt_b = cnt
        total_count = float(cnt.sum() if cnt.ndim == 1 else cnt[:, 0].sum())
        # global mean
        global_mean = (mean * cnt_b).sum(axis=0) / cnt_b.sum(axis=0)
        # pooled variance: V = (Σ (n_i - 1) σ_i² + Σ n_i (μ_i - μ)²) / (Σ n_i - 1)
        # for our use case, treat each ep's std as its sample std
        n_minus_1 = np.maximum(cnt_b - 1, 1)
        between = (cnt_b * (mean - global_mean)**2).sum(axis=0)
        within = ((cnt_b - 1) * std**2).sum(axis=0)
        denom = max(cnt_b.sum(axis=0).item() if cnt_b.ndim == 1 else (cnt_b.sum(axis=0).max() - 1), 1)
        # broadcast denom for vector features
        global_var = (between + within) / max(int(total_count) - 1, 1)
        global_std = np.sqrt(np.clip(global_var, 0, None))
        stats[fk] = {
            "min":   mn.min(axis=0).tolist()  if mn.ndim  > 0 else float(mn.min()),
            "max":   mx.max(axis=0).tolist()  if mx.ndim  > 0 else float(mx.max()),
            "mean":  global_mean.tolist()      if global_mean.ndim > 0 else float(global_mean),
            "std":   global_std.tolist()       if global_std.ndim > 0 else float(global_std),
            "count": int(total_count),
            "q01":   q01.min(axis=0).tolist() if q01.ndim > 0 else float(q01.min()),
            "q10":   q10.min(axis=0).tolist() if q10.ndim > 0 else float(q10.min()),
            "q50":   np.median(q50, axis=0).tolist() if q50.ndim > 0 else float(np.median(q50)),
            "q90":   q90.max(axis=0).tolist() if q90.ndim > 0 else float(q90.max()),
            "q99":   q99.max(axis=0).tolist() if q99.ndim > 0 else float(q99.max()),
        }
    (args.dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"     {len(stats)} features; {time.time() - t:.1f}s", flush=True)

    # ── 8. Write info.json ──
    (args.dst / "meta" / "info.json").write_text(json.dumps(out_info, indent=2))
    print(f"[8/8] info.json written. Total: {time.time() - t_start:.1f}s", flush=True)

    # ── final summary ──
    out_du = sum(p.stat().st_size for p in args.dst.rglob("*") if p.is_file())
    print(f"\n  ✓ merged dataset: {args.dst}")
    print(f"     total_episodes: {out_info['total_episodes']}")
    print(f"     total_frames  : {out_info['total_frames']}")
    print(f"     total_tasks   : {out_info['total_tasks']}")
    print(f"     on-disk size  : {out_du / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
