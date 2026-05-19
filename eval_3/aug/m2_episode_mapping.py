"""Reconstruct `episode_index → variant_name` for the merged HF dataset.

The merger (`eval_3/scripts/merge_eval3_episodes.py:50-60`) builds the
episode order deterministically:
    base = sorted base teleop dirs under `--base-root`
    aug  = sorted aug variant dirs under `--aug-root` (filtered by `__t3_`)
    merged = base + aug

So merged `episode_index` reconstructs as:
    if episode_index < len(base):
        is_base = True; variant = base[episode_index].name
    else:
        is_base = False; variant = aug[episode_index - len(base)].name

For M2 supervision:
- **Aug variants**: we have face_labels.json keyed by source episode
  (the variant's name prefix before `__t3_`), and the variant's own
  augmentation.json gives the slot→celeb mapping. Both feed
  `M2SupervisionBuilder.build_batch`.
- **Base teleops**: we don't have face_labels for these (only generated
  for aug-variant cameras above), so `bbox_valid=False` for all 3 slots
  and the action loss trains alone on those episodes. ~2 % of total
  episodes (178 of 9,394) — small effect.

The mapping is built once at training start and cached. Building takes
< 1 s (just sorted directory listings).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EpisodeInfo:
    episode_index: int
    is_base: bool
    variant_name: str             # full directory name on disk
    source_episode: str           # prefix before "__t3_" for aug; same as variant for base


def discover_merge_order(base_root: Path, aug_root: Path,
                         aug_pattern: str = "__t3_") -> tuple[list[Path], list[Path]]:
    """Mirror `merge_eval3_episodes.discover_episode_dirs`.

    Base teleops: directories under `base_root` that have
    `meta/info.json` AND `reference.json`. Sorted alphabetically.

    Aug variants: directories under `aug_root` with `aug_pattern` in the
    name AND `meta/info.json` present. Sorted alphabetically.
    """
    base = []
    if base_root.is_dir():
        base = sorted(p for p in base_root.iterdir()
                      if p.is_dir()
                      and (p / "meta" / "info.json").is_file()
                      and (p / "reference.json").is_file())
    aug = []
    if aug_root.is_dir():
        aug = sorted(p for p in aug_root.iterdir()
                     if p.is_dir()
                     and aug_pattern in p.name
                     and (p / "meta" / "info.json").is_file())
    return base, aug


def variant_to_source(variant_name: str) -> str:
    """Aug variant `quick_X__t3_NNNN_vXX` → source `quick_X`.

    Base teleop `quick_X` (no `__t3_`) → returns the name unchanged
    (callers should use is_base to branch behaviour anyway).
    """
    sep = "__t3_"
    return variant_name.split(sep)[0] if sep in variant_name else variant_name


def build_episode_mapping(base_root: Path,
                          aug_root: Path,
                          aug_pattern: str = "__t3_") -> list[EpisodeInfo]:
    """Build [EpisodeInfo, …] indexed by merged episode_index."""
    base_dirs, aug_dirs = discover_merge_order(base_root, aug_root, aug_pattern)
    out: list[EpisodeInfo] = []
    for i, p in enumerate(base_dirs):
        out.append(EpisodeInfo(
            episode_index=i,
            is_base=True,
            variant_name=p.name,
            source_episode=variant_to_source(p.name),
        ))
    n_base = len(base_dirs)
    for j, p in enumerate(aug_dirs):
        out.append(EpisodeInfo(
            episode_index=n_base + j,
            is_base=False,
            variant_name=p.name,
            source_episode=variant_to_source(p.name),
        ))
    return out


def save_mapping_json(mapping: list[EpisodeInfo], dest: Path) -> None:
    rows = [
        {
            "episode_index": ep.episode_index,
            "is_base": ep.is_base,
            "variant_name": ep.variant_name,
            "source_episode": ep.source_episode,
        }
        for ep in mapping
    ]
    dest.write_text(json.dumps({"schema_version": 1, "episodes": rows}, indent=2))


def load_mapping_json(p: Path) -> list[EpisodeInfo]:
    rows = json.loads(p.read_text())["episodes"]
    return [EpisodeInfo(**r) for r in rows]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path("datasets/eval3"),
                        help="Directory with base teleop dirs (one per recording).")
    parser.add_argument("--aug-root", type=Path,
                        default=Path.home() / "Downloads/eval3_track3_aug",
                        help="Directory with augmented variant dirs.")
    parser.add_argument("--output", type=Path,
                        default=Path("eval_3/aug/stats/episode_mapping.json"))
    parser.add_argument("--allow-missing-base", action="store_true",
                        help="If the base-root doesn't exist locally (e.g. on Mahbod's "
                             "Mac), proceed with N_base=178 from the docs and only aug "
                             "indices are concrete; base indices get placeholder names.")
    args = parser.parse_args()

    base_dirs, aug_dirs = discover_merge_order(args.base_root, args.aug_root)
    print(f"[discover] base teleops: {len(base_dirs)}  aug variants: {len(aug_dirs)}")

    if not base_dirs and not args.allow_missing_base:
        raise SystemExit(
            f"--base-root={args.base_root} has no base teleops. Use "
            "--allow-missing-base to placehold them (only aug indices will be "
            "concrete; you'll need to fill in base names on Brev where they live)."
        )

    if not base_dirs and args.allow_missing_base:
        n_base = 178
        print(f"[WARN] base teleops missing locally — placeholding with N_base={n_base}, "
              f"fallback=base names emitted as 'BASE_<i>', will need rebuild on Brev.",
              flush=True)
        out = []
        for i in range(n_base):
            out.append(EpisodeInfo(
                episode_index=i, is_base=True,
                variant_name=f"BASE_{i:03d}", source_episode=f"BASE_{i:03d}",
            ))
        for j, p in enumerate(aug_dirs):
            out.append(EpisodeInfo(
                episode_index=n_base + j, is_base=False,
                variant_name=p.name, source_episode=variant_to_source(p.name),
            ))
    else:
        out = build_episode_mapping(args.base_root, args.aug_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_mapping_json(out, args.output)
    n_base_real = sum(1 for ep in out if ep.is_base)
    n_aug = sum(1 for ep in out if not ep.is_base)
    print(f"[save] {args.output}  n_base={n_base_real}  n_aug={n_aug}  total={len(out)}")
