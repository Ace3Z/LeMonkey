#!/usr/bin/env python3
"""Pre-merge prep: fix schema + add reference video to base teleops.

Two phases:

  PHASE A (augmented variants): for every datasets/eval3_aug_v3/*__var*/:
      - The variant's meta/info.json is hardlinked to the base teleop's
        info.json (so editing in-place would corrupt the base). Break the
        link, then add `observation.images.reference` as a video feature
        (SECOND in dict order — SigLIP-prefix position-sensitive per
        Interleave-VLA §A.1 / SmolVLA modeling_smolvla.py:404-444).

  PHASE B (base teleops): for every datasets/eval3/quick_*/:
      - Pick a portrait+color reference photo from
        datasets/eval3_celebs/scraped/{yann_lecun|barack_obama|taylor_swift}/
        deterministically by ep name hash.
      - Generate a constant-frame H.264 mp4 at 480×480 / 30 fps / N frames
        (N = base ep's total_frames) at
        videos/observation.images.reference/chunk-000/file-000.mp4.
      - Break the info.json hardlink (if any) and declare the reference
        camera feature (SECOND in order).

Validate after: every produced dataset dir should `LeRobotDataset.from_root`
cleanly with both camera streams visible.

Usage:
    prep_for_merge.py --aug-root datasets/eval3_aug_v3 \
                      --base-root datasets/eval3 \
                      --photo-bank datasets/eval3_celebs/scraped
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Map base teleop's short celeb tag -> scraped/<slug>/ dir
CELEB_SLUG_MAP = {
    "lecun":  "yann_lecun",
    "obama":  "barack_obama",
    "swift":  "taylor_swift",
}


def reference_feature_spec(height: int = 480, width: int = 480, fps: int = 30) -> dict:
    """Mirror of the camera1 video feature spec but at the reference dims.
    Per Interleave-VLA conventions, this is just a constant-frame video so
    it slots in alongside camera1 with identical metadata except dims."""
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def add_reference_feature(info: dict, *, height: int = 480, width: int = 480,
                          fps: int = 30) -> dict:
    """Insert observation.images.reference into features dict, RIGHT AFTER
    observation.images.camera1 (preserve SigLIP-prefix order). Idempotent —
    no-op if already present."""
    feats = info.get("features", {})
    if "observation.images.reference" in feats:
        return info        # already declared
    new_feats = {}
    inserted = False
    for k, v in feats.items():
        new_feats[k] = v
        if k == "observation.images.camera1" and not inserted:
            new_feats["observation.images.reference"] = reference_feature_spec(
                height=height, width=width, fps=fps,
            )
            inserted = True
    if not inserted:
        # camera1 not found — append at end (shouldn't happen for our data)
        print(f"  [WARN] add_reference_feature: camera1 not in features dict, "
              f"appending reference at end")
        new_feats["observation.images.reference"] = reference_feature_spec(
            height=height, width=width, fps=fps,
        )
    info["features"] = new_feats
    return info


def break_hardlink_and_write_json(path: Path, content: dict) -> None:
    """If `path` is hardlinked, replace it with a fresh independent file
    containing `content`. Safe for non-hardlinked paths too."""
    text = json.dumps(content, indent=4)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    # Atomic rename — drops the hardlink
    os.replace(tmp, path)


def phase_a_patch_augmented(aug_root: Path) -> tuple[int, int]:
    """Patch info.json on every augmented variant.
    Returns (n_patched, n_already_ok)."""
    var_dirs = sorted(p for p in aug_root.iterdir()
                        if p.is_dir() and "__var" in p.name)
    n_patched = 0
    n_already = 0
    for d in var_dirs:
        info_path = d / "meta" / "info.json"
        if not info_path.is_file():
            print(f"  [WARN] phase_a: missing {info_path}, skip")
            continue
        info = json.loads(info_path.read_text())
        if "observation.images.reference" in info.get("features", {}):
            n_already += 1
            continue
        # Read reference video dims to keep schema honest
        ref_mp4 = d / "videos/observation.images.reference/chunk-000/file-000.mp4"
        h, w, fps = 480, 480, 30
        if ref_mp4.is_file():
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v",
                     "-show_entries", "stream=width,height,r_frame_rate",
                     "-of", "default=nw=1:nk=1", str(ref_mp4)],
                    capture_output=True, text=True, check=True,
                )
                lines = r.stdout.strip().splitlines()
                if len(lines) >= 3:
                    w, h = int(lines[0]), int(lines[1])
                    num, den = lines[2].split("/")
                    fps = int(round(int(num) / int(den)))
            except Exception as e:
                print(f"  [WARN] phase_a: ffprobe failed on {ref_mp4}: {e}, "
                      f"using defaults 480x480 / 30 fps")
        info = add_reference_feature(info, height=h, width=w, fps=fps)
        break_hardlink_and_write_json(info_path, info)
        n_patched += 1
    return n_patched, n_already


def pick_ref_photo(celeb_slug: str, ep_name: str, photo_bank: Path) -> Path:
    """Deterministic pick of a portrait+color reference photo for `ep_name`
    from photo_bank/<celeb_slug>/."""
    import cv2
    cdir = photo_bank / celeb_slug
    if not cdir.is_dir():
        raise FileNotFoundError(f"photo dir missing: {cdir}")
    candidates = sorted(p for p in cdir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                          and not p.name.startswith("__"))
    # Apply the same portrait+color filter the bank loader uses
    keep = []
    for p in candidates:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w >= h:                     # not portrait
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if float(hsv[..., 1].mean()) < 60.0:    # not color enough
            continue
        keep.append(p)
    if not keep:
        raise FileNotFoundError(
            f"no portrait+color photos in {cdir} (have {len(candidates)} total)"
        )
    # Deterministic pick by ep_name hash
    idx = int(hashlib.md5(ep_name.encode()).hexdigest(), 16) % len(keep)
    return keep[idx]


def write_reference_mp4(photo: Path, n_frames: int, fps: int,
                        size: int, out_path: Path) -> None:
    """Write a constant-frame mp4 of `photo` resized to size×size, length n_frames at fps.
    Uses ffmpeg's -loop 1 -t (or -frames:v)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-framerate", str(fps), "-i", str(photo),
        "-frames:v", str(n_frames),
        "-vf", f"scale={size}:{size}:force_original_aspect_ratio=increase,"
               f"crop={size}:{size}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def phase_b_prep_base(base_root: Path, photo_bank: Path,
                      ref_size: int = 480) -> tuple[int, int, int]:
    """For every base teleop with reference.json, generate the reference mp4
    and patch info.json. Returns (n_prepared, n_already, n_skipped)."""
    ep_dirs = sorted(p for p in base_root.iterdir()
                       if p.is_dir() and (p / "reference.json").is_file())
    n_prep = 0
    n_already = 0
    n_skip = 0
    for ep in ep_dirs:
        info_path = ep / "meta" / "info.json"
        ref_mp4 = ep / "videos/observation.images.reference/chunk-000/file-000.mp4"
        sidecar = json.loads((ep / "reference.json").read_text())
        celeb = sidecar.get("target_celeb", "")
        slug = CELEB_SLUG_MAP.get(celeb)
        if slug is None:
            print(f"  [WARN] phase_b: unknown target_celeb={celeb!r} in {ep.name}, skip")
            n_skip += 1
            continue
        info = json.loads(info_path.read_text())
        already_in_schema = "observation.images.reference" in info.get("features", {})
        if ref_mp4.is_file() and already_in_schema:
            n_already += 1
            continue
        # Write the reference mp4 (idempotent — overwrite if exists, since
        # ref_mp4.is_file() might be False or schema might be missing)
        n_frames = info.get("total_frames", 538)
        fps = info.get("fps", 30)
        try:
            photo = pick_ref_photo(slug, ep.name, photo_bank)
        except FileNotFoundError as e:
            print(f"  [WARN] phase_b: {ep.name}: {e}, skip")
            n_skip += 1
            continue
        try:
            write_reference_mp4(photo, n_frames=n_frames, fps=fps,
                                size=ref_size, out_path=ref_mp4)
        except subprocess.CalledProcessError as e:
            print(f"  [WARN] phase_b: ffmpeg failed for {ep.name}: {e}, skip")
            n_skip += 1
            continue
        info = add_reference_feature(info, height=ref_size, width=ref_size,
                                     fps=fps)
        break_hardlink_and_write_json(info_path, info)
        # Also update reference.json to point at the new ref photo path
        sidecar["reference_photo"] = str(photo)
        (ep / "reference.json").write_text(json.dumps(sidecar, indent=2))
        n_prep += 1
        if n_prep % 20 == 0:
            print(f"  phase_b: {n_prep} prepped so far...", flush=True)
    return n_prep, n_already, n_skip


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--aug-root",
                   default="datasets/eval3_aug_v3", type=Path)
    p.add_argument("--base-root",
                   default="datasets/eval3", type=Path)
    p.add_argument("--photo-bank",
                   default="datasets/eval3_celebs/scraped", type=Path)
    p.add_argument("--phase", choices=["a", "b", "both"], default="both")
    p.add_argument("--ref-size", type=int, default=480,
                   help="Reference video resolution (square)")
    args = p.parse_args()

    if args.phase in ("a", "both"):
        print(f"== PHASE A: patch info.json on augmented variants under "
              f"{args.aug_root} ==", flush=True)
        n_a, n_a_ok = phase_a_patch_augmented(args.aug_root)
        print(f"  → patched {n_a}, already-ok {n_a_ok}", flush=True)

    if args.phase in ("b", "both"):
        print(f"\n== PHASE B: generate reference mp4 + patch info.json for "
              f"base teleops under {args.base_root} ==", flush=True)
        n_b, n_b_ok, n_b_skip = phase_b_prep_base(
            args.base_root, args.photo_bank, ref_size=args.ref_size,
        )
        print(f"  → prepped {n_b}, already-ok {n_b_ok}, skipped {n_b_skip}",
              flush=True)

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
