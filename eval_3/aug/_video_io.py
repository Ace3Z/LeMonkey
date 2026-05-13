"""Video I/O helpers — handle AV1-encoded LeRobot videos on Thor.

Why this module exists:
  - lerobot-record on thor encodes camera videos with libsvtav1 by default.
  - Thor's bundled OpenCV cannot decode AV1 (the system ffmpeg has dav1d
    but the OpenCV-bundled ffmpeg does not).
  - Workaround: lazy-transcode each AV1 mp4 to a sidecar H.264 file the
    first time we need to open it, then return the H.264 path. Original
    file is never touched. cv2 / SAM 2 / everything downstream then reads
    the H.264 sidecar.
  - The sidecar is named `<original-stem>__h264.mp4` and lives next to
    the original. Subsequent runs detect it and skip the transcode.

The transcode is fast: ~1.5 s for a 20 s 640×480 30 fps clip on Thor's
ARM Neoverse cores using libx264. Total cost across the 5 quick episodes:
~7 s. For the future 144-ep main collection: ~3.5 min, one-time.

Usage:
    from _video_io import ensure_h264, read_frame_zero, iter_frames

    h264_path = ensure_h264(av1_path)
    cap = cv2.VideoCapture(str(h264_path))     # works
    frame0 = read_frame_zero(av1_path)          # handles AV1 + h264 alike
    for frame in iter_frames(av1_path): ...     # ditto
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


_LEGACY_DATASET_PREFIX = "/home/lemonkey/LeMonkey"


def _local_lemonkey_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "LeMonkey" and (parent / "datasets").exists():
            return parent
    return None


_LOCAL_ROOT_CACHE: list = []
_REMAP_LOGGED: set = set()


def resolve_lemonkey_path(p) -> Path:
    """Rewrite a /home/lemonkey/LeMonkey/... path to its local equivalent when
    running outside Thor. No-op if the path exists, doesn't start with the
    legacy prefix, or no local LeMonkey root is detectable.

    Override the auto-detected root by setting the LEMONKEY_ROOT env var.
    Logs once per remapped path so the fallback is never silent.
    """
    p = Path(p)
    if p.exists():
        return p
    s = str(p)
    if not s.startswith(_LEGACY_DATASET_PREFIX + "/"):
        return p

    if not _LOCAL_ROOT_CACHE:
        env = os.environ.get("LEMONKEY_ROOT")
        _LOCAL_ROOT_CACHE.append(Path(env) if env else _local_lemonkey_root())
    root = _LOCAL_ROOT_CACHE[0]
    if root is None:
        return p

    rewritten = root / s[len(_LEGACY_DATASET_PREFIX) + 1 :]
    if str(rewritten) not in _REMAP_LOGGED:
        _REMAP_LOGGED.add(str(rewritten))
        print(
            f"[WARN] path_remap: expected={p}, got=not-found, fallback={rewritten}",
            flush=True,
        )
    return rewritten


def _is_h264_or_decodable(video_path: Path) -> bool:
    """Quickly decide whether cv2 can read this file. On AV1 mp4s it can't."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    return bool(ok and frame is not None and frame.size > 0)


def ensure_h264(video_path: Path | str, *, force: bool = False) -> Path:
    """Return a Path to an H.264-encoded mp4 of `video_path`.

    If the original is already cv2-readable (H.264, etc.), returns it
    as-is. Otherwise transcodes once to a sidecar `<stem>__h264.mp4`
    next to the original and returns that path.

    `force=True` re-transcodes even if a cached sidecar exists.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    sidecar = video_path.with_name(f"{video_path.stem}__h264.mp4")
    if sidecar.is_file() and not force:
        return sidecar

    # If cv2 can read the original directly, no transcode needed.
    if _is_h264_or_decodable(video_path):
        return video_path

    # Transcode via system ffmpeg. We use libx264 CRF 18 (visually
    # lossless) so downstream stages see effectively the same content
    # as the original.
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH; can't transcode AV1")

    print(f"[_video_io] transcoding AV1 → H.264 (one-time): {video_path.name}", flush=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",                                   # drop audio (LeRobot videos have none)
        str(sidecar),
    ]
    # `-nostdin` is critical when this runs inside a bash while-read loop:
    # without it, ffmpeg consumes characters from the parent shell's stdin,
    # corrupting the next iteration's input AND can return non-zero on the
    # garbage. Diagnosed 2026-05-12 during the 60-sample batch.
    rc = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL)
    if rc.returncode != 0 or not sidecar.is_file():
        raise RuntimeError(f"ffmpeg transcode failed for {video_path}")
    return sidecar


def read_frame_zero(video_path: Path | str) -> np.ndarray:
    """Decode and return frame 0 as BGR uint8 ndarray.

    Routes through ensure_h264 so AV1 inputs work transparently.
    """
    h264 = ensure_h264(video_path)
    cap = cv2.VideoCapture(str(h264))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame 0 of {h264}")
    return frame


def iter_frames(video_path: Path | str):
    """Generator yielding (frame_idx, frame_bgr_uint8) tuples."""
    h264 = ensure_h264(video_path)
    cap = cv2.VideoCapture(str(h264))
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield fi, frame
            fi += 1
    finally:
        cap.release()


def ensure_frame_dir(video_path: Path | str, *, force: bool = False) -> Path:
    """Extract `video_path` to a sibling dir of zero-padded JPEG frames.

    SAM 2's video predictor accepts either an mp4 path (decoded via decord,
    which has no aarch64 wheel on Thor) OR a directory of JPEG frames named
    in lexicographic-sortable order. We use the latter to bypass the decord
    requirement.

    Returns the path to the frames dir (e.g. <video>.parent / `<stem>__frames/`).
    Caches between runs — re-running is a no-op unless `force=True`.

    Filenames: `00000.jpg`, `00001.jpg`, ... matches SAM 2's expectation
    (it sorts alphabetically, so zero-padding is required).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    frames_dir = video_path.with_name(f"{video_path.stem}__frames")
    sentinel = frames_dir / ".extraction_complete"
    if sentinel.is_file() and not force:
        return frames_dir

    frames_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg has libdav1d so it handles AV1 directly, no need to transcode first.
    print(f"[_video_io] extracting frames → {frames_dir.name}/ (one-time)", flush=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-q:v", "2",                              # high-quality JPEG
        str(frames_dir / "%05d.jpg"),
    ]
    rc = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL)
    if rc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed for {video_path}")
    sentinel.touch()
    return frames_dir


def video_metadata(video_path: Path | str) -> dict:
    """Return basic metadata: {n_frames, fps, width, height}."""
    h264 = ensure_h264(video_path)
    cap = cv2.VideoCapture(str(h264))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"n_frames": n_frames, "fps": fps, "width": width, "height": height}
