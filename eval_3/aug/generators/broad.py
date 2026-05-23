"""
Eval 3 broad augmentation generator: replaces each printed portrait with
another photo of the same celebrity and writes a reference-image stream;
supports up to ~200 celebrities.

Per the locked strategy in `eval_3/aug/STRATEGY.md`:

  For each base teleop episode, generate M variants. Each variant:
    1. Samples 3 distinct celebs from the bank (target + 2 distractors).
    2. Inpaints ALL 3 visible portraits in the wrist camera.
    3. Writes a reference video stream `observation.images.reference`
       containing a DIFFERENT photo of the target celeb than the one
       painted on the table (face-verification setup).
    4. Sets the prompt via a 75 / 15 / 10 mixture (default name+action /
       reference-only / counterfactual wrong-name).

Reuses helpers from `eval_3/aug/stages/inpaint_video.py`
(face_centered_aspect_crop, render_variant, load_photo_bank). This file
is the orchestrator: it holds the random sampling, the prompt mixture,
and the reference video writer.

Run:
    python eval_3/aug/generators/broad.py --root ~/LeMonkey/datasets/eval3 \
                              --photo-bank ~/LeMonkey/datasets/eval3_celebs/scraped \
                              --out-root ~/LeMonkey/datasets/eval3_aug_v3 \
                              --num-variants 25 \
                              --seed 42
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# ─── Local imports: reuse helpers from eval_3/aug/stages/inpaint_video.py ───
_HERE = Path(__file__).resolve().parent
_spec = _ilu.spec_from_file_location("_v4", str(_HERE.parent / "stages" / "inpaint_video.py"))
_v4 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_v4)
load_photo_bank        = _v4.load_photo_bank
face_centered_aspect_crop = _v4.face_centered_aspect_crop
render_variant         = _v4.render_variant
hardlink_meta          = _v4.hardlink_meta
find_video             = _v4.find_video

# ─── Prompt mixture (see STRATEGY.md §8) ─────────────────────────────
PROMPT_PARAPHRASES = [
    "Place the coke on {name}.",
    "Put the coke on {name}.",
    "Place the can on the photo of {name}.",
    "Set the coke down on {name}'s picture.",
    "Put the can on {name}'s photo.",
]
PROMPT_REFERENCE_ONLY = [
    "Place the coke on the person in the reference image.",
    "Put the coke on whoever is in the reference photo.",
    "Place the can on the celebrity shown in the reference image.",
]
# Mix is 75/15/10 — see STRATEGY.md §8.1
BUCKET_PROBS = {"default": 0.75, "ref_only": 0.15, "counterfactual": 0.10}


def slug_to_name(slug: str) -> str:
    """taylor_swift -> Taylor Swift. Keeps capitalisation simple
    (Title Case on underscore split). Acceptable for prompts."""
    return " ".join(w.capitalize() for w in slug.split("_"))


def pick_prompt(target_slug: str, all_slugs: list[str], rng: random.Random
                  ) -> tuple[str, str]:
    """Returns (prompt_text, bucket_label) per the 75/15/10 mixture."""
    r = rng.random()
    target_name = slug_to_name(target_slug)
    if r < BUCKET_PROBS["default"]:
        return rng.choice(PROMPT_PARAPHRASES).format(name=target_name), "default"
    if r < BUCKET_PROBS["default"] + BUCKET_PROBS["ref_only"]:
        return rng.choice(PROMPT_REFERENCE_ONLY), "ref_only"
    # counterfactual — wrong-name + correct reference. Reference channel
    # still points to the correct celeb, the action label is unchanged.
    wrong = rng.choice([s for s in all_slugs if s != target_slug])
    return f"Place the coke on {slug_to_name(wrong)}.", f"counterfactual:{wrong}"


# ─── Reference video stream writer (constant-frame MP4) ─────────────────
def write_reference_video(ref_photo_path: Path, n_frames: int, fps: int,
                            out_mp4: Path, *, size: tuple[int, int] = (480, 480),
                            ) -> int:
    """Write a constant-frame MP4 where every frame is `ref_photo_path`
    resized to `size`. LeRobot's data loader will read this as
    `observation.images.reference` at training time.

    Strategy v3 §4 option A: trivially compatible with the existing
    LeRobot schema; no policy-config tweak needed beyond declaring the
    new image stream in `input_features`."""
    img = cv2.imread(str(ref_photo_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read reference photo: {ref_photo_path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        # Write one frame, then use ffmpeg's -loop to replicate.
        tmp_frame = Path(td) / "f.png"
        cv2.imwrite(str(tmp_frame), img)
        duration_s = float(n_frames) / float(fps)
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-loop", "1", "-i", str(tmp_frame),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-t", f"{duration_s:.4f}",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        rc = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL)
        if rc.returncode != 0 or not out_mp4.is_file():
            raise RuntimeError(f"ffmpeg reference-encode failed for {out_mp4}")
    return n_frames


# ─── Per-variant celeb assignment ────────────────────────────────────────
def precompute_target_assignment(
    ep_names: list[str], num_variants: int, bank_celebs: list[str], seed: int,
) -> dict[tuple[str, int], str]:
    """Deterministic uniform target assignment across ALL (episode, variant_idx)
    tuples. Each celeb gets exactly floor(N/K) or ceil(N/K) target appearances
    where N = len(ep_names) * num_variants and K = len(bank_celebs).

    For the planned 179 base episodes × M=25 = 4475 total variants over a
    195-celeb bank: 185 celebs each appear as target 23 times, 10 celebs
    each appear as target 22 times (185×23 + 10×22 = 4475). This is what
    STRATEGY.md §5 calls for and what the user mandated.

    Returns {(episode_name, variant_idx): target_celeb_slug}.
    """
    n_total = len(ep_names) * num_variants
    n_celebs = len(bank_celebs)
    bank_celebs = sorted(bank_celebs)   # reproducibility
    base, extra = divmod(n_total, n_celebs)
    # First `extra` celebs (in sorted order) get (base+1) target appearances;
    # the rest get `base`. After shuffling, this distinction is invisible —
    # the only invariant is per-celeb COUNT.
    target_seq: list[str] = []
    for i, c in enumerate(bank_celebs):
        target_seq.extend([c] * (base + (1 if i < extra else 0)))
    assert len(target_seq) == n_total, (len(target_seq), n_total)
    rng = random.Random(seed)
    rng.shuffle(target_seq)
    # Lay out into a (ep_name, var_idx) → target dict in canonical order.
    assignment: dict[tuple[str, int], str] = {}
    flat_idx = 0
    for ep in sorted(ep_names):
        for v in range(num_variants):
            assignment[(ep, v)] = target_seq[flat_idx]
            flat_idx += 1
    return assignment


def assign_celebs_for_variant(
    pid_to_orig_celeb: dict[str, str],
    orig_target_celeb: str,
    bank: dict[str, list[Path]],
    rng: random.Random,
    *,
    forced_target: str | None = None,
) -> tuple[dict[str, str], str]:
    """Sample 3 distinct celebs from the bank and assign them to the 3
    pids. The new TARGET celeb goes to the same pid that originally held
    the target (so the recorded trajectory still ends on the target
    paper). The other 2 pids get the 2 distractor celebs in random order.

    If `forced_target` is provided (from precompute_target_assignment),
    use it; otherwise sample uniformly random.

    Returns (new pid_to_celeb mapping, target_celeb_slug).
    """
    celeb_pool = list(bank.keys())
    if len(celeb_pool) < 3:
        raise ValueError(f"bank has only {len(celeb_pool)} celebs; need ≥3")
    if forced_target is not None:
        if forced_target not in celeb_pool:
            raise ValueError(f"forced target {forced_target!r} not in bank")
        non_target = [c for c in celeb_pool if c != forced_target]
        d1, d2 = rng.sample(non_target, 2)
        new_target = forced_target
    else:
        new_target, d1, d2 = rng.sample(celeb_pool, 3)
    # Find pid that originally held the target
    target_pid = next(pid for pid, c in pid_to_orig_celeb.items()
                      if c == orig_target_celeb)
    other_pids = [pid for pid in pid_to_orig_celeb if pid != target_pid]
    rng.shuffle(other_pids)
    new_assignment = {
        target_pid: new_target,
        other_pids[0]: d1,
        other_pids[1]: d2,
    }
    return new_assignment, new_target


def pick_photos_v3(
    pid_to_celeb: dict[str, str],
    target_celeb: str,
    bank: dict[str, list[Path]],
    rng: random.Random,
) -> tuple[dict[str, Path], Path]:
    """Pick 4 distinct photos per variant: one per workspace slot + one
    reference for the target (must be DIFFERENT photo than the workspace
    target photo). Per STRATEGY.md §3."""
    used: set[Path] = set()
    workspace: dict[str, Path] = {}
    for pid, celeb in pid_to_celeb.items():
        pool = bank.get(celeb, [])
        if not pool:
            raise ValueError(f"bank has no photos for {celeb!r}")
        # Avoid re-using a photo across slots when possible
        candidates = [p for p in pool if p not in used] or pool
        choice = rng.choice(candidates)
        workspace[pid] = choice
        used.add(choice)
    # Reference: different from workspace[target]
    target_pool = bank.get(target_celeb, [])
    target_workspace = next(p for pid, p in workspace.items()
                            if pid_to_celeb[pid] == target_celeb)
    ref_candidates = [p for p in target_pool if p != target_workspace]
    if not ref_candidates:
        # Edge case: celeb has only 1 photo. Use it anyway (will fail
        # the "different photo" rule but won't crash). Log [WARN].
        print(f"[WARN] ref_photo_collision: celeb={target_celeb!r}, "
              f"only_n_photos=1, fallback=reuse_workspace_photo", flush=True)
        ref_photo = target_workspace
    else:
        ref_photo = rng.choice(ref_candidates)
    return workspace, ref_photo


# ─── Per-episode orchestrator ────────────────────────────────────────────
def process_episode(
    ep_dir: Path,
    out_root: Path,
    bank: dict[str, list[Path]],
    num_variants: int,
    seed: int,
    fps: int,
    *,
    force: bool = False,
    debug: bool = False,
    target_assignment: dict[tuple[str, int], str] | None = None,
) -> dict:
    """Generate `num_variants` augmented variants for one base
    teleop episode. Each variant writes:
      - videos/observation.images.camera1/...  (inpainted wrist)
      - videos/observation.images.reference/...  (constant ref photo)
      - data/chunk-000/file-000.parquet  (hardlinked from base)
      - meta/                              (hardlinked from base)
      - reference.json  (prompt, target_celeb, etc.)
      - augmentation.json  (provenance + per-variant random seed)
      - debug/  (only if debug=True — compare frames, etc.)
    """
    corners_json = ep_dir / "portrait_corners.json"
    masks_pkl = ep_dir / "portrait_masks.pkl"
    if not masks_pkl.is_file():
        masks_pkl = None
    ref_json = ep_dir / "reference.json"
    if not corners_json.is_file():
        return {"ep": ep_dir.name, "error": "portrait_corners.json missing"}
    if not ref_json.is_file():
        return {"ep": ep_dir.name, "error": "reference.json missing"}

    corners_data = json.loads(corners_json.read_text())
    orig_sidecar = json.loads(ref_json.read_text())
    layout = orig_sidecar.get("layout", "-")
    orig_target = orig_sidecar["target_celeb"]
    layout_celebs = _v4.decode_layout(layout)
    if not layout_celebs:
        return {"ep": ep_dir.name, "error": f"bad layout {layout!r}"}

    # Re-derive the original pid_to_celeb so we know which pid was the target.
    seeds_path = ep_dir / "portrait_seeds.json"
    seeds_dict = json.loads(seeds_path.read_text()) if seeds_path.is_file() else None
    pid_to_orig = _v4.assign_celebs_to_portraits(corners_data, layout_celebs, seeds_dict)
    if len(pid_to_orig) != 3:
        return {"ep": ep_dir.name, "error": f"orig pid_to_celeb has {len(pid_to_orig)} entries"}

    src_video = find_video(ep_dir)
    rng = random.Random(seed + hash(ep_dir.name) % 1_000_000)
    frame_0_path = ep_dir / "frame_0.png"
    frame_0_img = cv2.imread(str(frame_0_path)) if frame_0_path.is_file() else None

    work_dir = Path("/tmp") / f"_aug_v3_{ep_dir.name}_{time.time_ns()}"
    work_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict] = []
    all_slugs = list(bank.keys())

    try:
        for var_idx in range(num_variants):
            var_name = f"{ep_dir.name}__var{var_idx:02d}"
            var_out = out_root / var_name
            var_video = var_out / "videos/observation.images.camera1/chunk-000/file-000.mp4"
            ref_video = var_out / "videos/observation.images.reference/chunk-000/file-000.mp4"
            if var_video.is_file() and ref_video.is_file() and not force:
                rendered.append({"variant": var_idx, "skipped": True})
                continue

            # 1. Sample celebs + photos for this variant.
            # `target_assignment` (if given) carries the deterministic uniform
            # target for this (episode, variant_idx). Distractors are still
            # rng-sampled from the remaining pool.
            forced = target_assignment.get((ep_dir.name, var_idx)) \
                     if target_assignment is not None else None
            new_pid_to_celeb, new_target_slug = assign_celebs_for_variant(
                pid_to_orig, orig_target, bank, rng, forced_target=forced,
            )
            workspace_photos, ref_photo = pick_photos_v3(
                new_pid_to_celeb, new_target_slug, bank, rng,
            )

            # 2. Pre-process photos for inpainting (v9.6 face-centered crop)
            pid_photos: dict[str, np.ndarray] = {}
            for pid, p in workspace_photos.items():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"cannot read {p}")
                pts = np.asarray(corners_data["portraits"][pid]["0"]["corners"], dtype=np.float32)
                paper_top = float(np.linalg.norm(pts[1] - pts[0]))
                paper_left = float(np.linalg.norm(pts[3] - pts[0]))
                target_aspect = paper_top / max(paper_left, 1e-6)
                pid_photos[pid] = face_centered_aspect_crop(img, target_aspect)

            # 3. Inpaint the wrist video
            n_written = render_variant(
                src_video, corners_data, masks_pkl, pid_photos,
                out_video=var_video, fps=fps, work_dir=work_dir,
                frame_0=frame_0_img,
            )

            # 4. Write the reference video stream
            write_reference_video(ref_photo, n_written, fps, ref_video)

            # 5. Pick prompt from 75/15/10 mixture
            prompt, bucket = pick_prompt(new_target_slug, all_slugs, rng)

            # 6. Hardlink data + meta + write sidecars
            hardlink_meta(ep_dir, var_out)
            new_sidecar = {**orig_sidecar}
            new_sidecar["source"] = "augmented_v3"
            new_sidecar["augmented_from"] = ep_dir.name
            new_sidecar["variant_idx"] = var_idx
            new_sidecar["target_celeb"] = new_target_slug
            new_sidecar["target_celeb_name"] = slug_to_name(new_target_slug)
            new_sidecar["reference_photo"] = str(ref_photo)
            new_sidecar["prompt"] = prompt
            new_sidecar["prompt_bucket"] = bucket
            (var_out / "reference.json").write_text(json.dumps(new_sidecar, indent=2))
            (var_out / "augmentation.json").write_text(json.dumps({
                "src_episode": ep_dir.name,
                "variant_idx": var_idx,
                "strategy_version": "v3",
                "pid_to_celeb": new_pid_to_celeb,
                "orig_target_celeb": orig_target,
                "new_target_celeb": new_target_slug,
                "workspace_photos": {pid: str(p) for pid, p in workspace_photos.items()},
                "reference_photo": str(ref_photo),
                "prompt": prompt,
                "prompt_bucket": bucket,
                "n_frames": n_written,
            }, indent=2))

            # 7. Optional debug bundle: compare PNG + stage2 panels per variant
            if debug:
                from importlib import import_module
                _dbg = _ilu.spec_from_file_location(
                    "_dbg_panels", str(_HERE.parent / "dbg" / "stage2_panels.py")
                )
                _dbg_mod = _ilu.module_from_spec(_dbg)
                _dbg.loader.exec_module(_dbg_mod)
                _dbg_mod.render_one(var_out)
                # And the compare gif
                _cmp = _ilu.spec_from_file_location(
                    "_dbg_compare", str(_HERE.parent / "dbg" / "compare_gif.py")
                )
                _cmp_mod = _ilu.module_from_spec(_cmp)
                _cmp.loader.exec_module(_cmp_mod)
                _cmp_mod.make_compare(var_out)

            rendered.append({
                "variant": var_idx, "target": new_target_slug,
                "bucket": bucket, "frames": n_written,
            })
            print(f"  ✓ {var_name}  target={new_target_slug:<25}  "
                  f"bucket={bucket}  ({n_written} frames)", flush=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {"ep": ep_dir.name, "rendered": rendered}


# ─── Main ────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help="Process all teleop dirs under this root")
    p.add_argument("--episode-dirs", nargs="+", default=None,
                   help="Explicit list of teleop episode dirs")
    p.add_argument("--photo-bank", required=True,
                   help="Path to the scraped/ photo bank (one subdir per celeb)")
    p.add_argument("--out-root", required=True,
                   help="Where augmented variants will be written")
    p.add_argument("--num-variants", type=int, default=25,
                   help="Number of augmented variants to generate per episode")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (combined with episode name for reproducibility)")
    p.add_argument("--fps", type=int, default=30,
                   help="Output mp4 frame rate; must match the source episode fps")
    p.add_argument("--force", action="store_true",
                   help="Re-render variants whose output directory already exists")
    p.add_argument("--debug", action="store_true",
                   help="Also produce dbg_compare.mp4 + dbg_stage2_panels.png per variant")
    args = p.parse_args()

    if not args.root and not args.episode_dirs:
        p.error("either --root or --episode-dirs required")

    # Load 195-celeb photo bank
    bank = load_photo_bank(Path(args.photo_bank), portrait_only=True, color_only=True)
    if len(bank) < 3:
        print(f"[FATAL] photo bank has only {len(bank)} celebs", file=sys.stderr)
        return 2
    print(f"loaded photo bank: {len(bank)} celebs, "
          f"{sum(len(v) for v in bank.values())} photos total", flush=True)

    if args.episode_dirs:
        ep_dirs = [Path(d) for d in args.episode_dirs]
    else:
        root = Path(args.root)
        ep_dirs = sorted(p for p in root.iterdir()
                          if p.is_dir() and (p / "reference.json").is_file())
    print(f"processing {len(ep_dirs)} episodes × {args.num_variants} variants = "
          f"{len(ep_dirs) * args.num_variants} variants total", flush=True)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-compute deterministic uniform target assignment across ALL
    # (episode, variant_idx) tuples. Each of the 195 celebs gets exactly
    # 22 or 23 target appearances across the full 4 475-variant corpus.
    target_assignment = precompute_target_assignment(
        [e.name for e in ep_dirs], args.num_variants,
        list(bank.keys()), args.seed,
    )
    from collections import Counter
    counts = Counter(target_assignment.values())
    n_at_max = sum(1 for v in counts.values() if v == max(counts.values()))
    n_at_min = sum(1 for v in counts.values() if v == min(counts.values()))
    print(f"target assignment: {len(counts)} celebs covered, "
          f"min={min(counts.values())} (×{n_at_min} celebs) "
          f"max={max(counts.values())} (×{n_at_max} celebs)", flush=True)

    results = []
    for i, ep in enumerate(ep_dirs, start=1):
        print(f"\n[{i}/{len(ep_dirs)}] {ep.name}", flush=True)
        try:
            r = process_episode(
                ep, out_root, bank,
                num_variants=args.num_variants,
                seed=args.seed,
                fps=args.fps,
                force=args.force,
                debug=args.debug,
                target_assignment=target_assignment,
            )
            results.append(r)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            traceback.print_exc()
            results.append({"ep": ep.name, "error": f"{type(e).__name__}: {e}"})

    summary = {
        "n_episodes": len(ep_dirs),
        "num_variants_per_episode": args.num_variants,
        "results": results,
    }
    (out_root / "_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary: {out_root / '_run_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
