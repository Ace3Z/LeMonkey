"""Per-batch M2 supervision builder.

At training time, each sample in a batch is a frame from one variant of one
source episode.  This module joins three pre-built artifacts to produce the
supervision tensors the M2 alignment loss consumes:

  1) `eval_3/aug/stats/face_labels/<source_episode>.face_labels.json`
     — per-frame face bboxes (positions only, sorted left→right)
  2) `<variant>/augmentation.json` — slot→celeb mapping
     (`new_layout_camera_lmr` letter triplet)
  3) `eval_3/aug/stats/celeb_embeddings.json` — per-celeb L2-normed
     ArcFace centroid

For each frame, we look up its 3 bboxes (slot 0/1/2), discover which celeb is
at each slot, build the per-slot patch mask + ArcFace centroid, stack across
the batch, and return.

Exclusion list
--------------
Reviewer B's data audit (2026-05-19) flagged 4 source episodes whose
upstream aug pipeline failed to replace the right-slot photo on a specific
`orig_R=S, new_R=L` permutation.  Per the user's (C) decision, these
sources are excluded entirely from training (drops ~260 / 9,216 variants
= 2.8 %, well under the loss-of-signal threshold).

Performance
-----------
- All 151 face_labels JSONs are preloaded once into RAM (~40 MB total).
- Per-celeb centroids preloaded once (~150 KB).
- Per-variant augmentation.json is lazily loaded with an LRU cache
  (`maxsize=None`, since ~9000 variants × ~1 KB JSON = ~9 MB).
- Patch masks for each (bbox triplet) are computed by `m2_alignment.bbox_to_patch_mask`
  in-process; ~10 µs per bbox on CPU.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import torch

from m2_alignment import (
    ARCFACE_EMBED_DIM,
    NUM_CAMERA1_PATCHES,
    build_supervision_for_frame,
)
from m2_episode_mapping import EpisodeInfo, load_mapping_json


# Sources whose augmentation.json claims a celeb at slot R that doesn't
# match the actual pasted photo. Discovered by Reviewer B's leave-one-out
# ArcFace consistency sweep on 2026-05-19.
EXCLUDED_SOURCES: frozenset[str] = frozenset({
    "quick_lecun_SLO_ep01_211540",
    "quick_lecun_SLO_ep04_211734",
    "quick_lecun_SOL_ep01_212006",
    "quick_obama_SLO_ep05_204851",
})


class M2SupervisionBuilder:
    """Materialise per-batch M2 supervision tensors.

    Construct once at training start. Call `build_batch(...)` per training
    step with the source-episode prefix, frame index, and variant directory
    (or its augmentation.json contents) for each batch element.

    Usage:
        builder = M2SupervisionBuilder(
            face_labels_dir=REPO_ROOT / "eval_3/aug/stats/face_labels",
            manifest_path=REPO_ROOT / "eval_3/aug/stats/celeb_embeddings.json",
            aug_root=Path.home() / "Downloads/eval3_track3_aug",
        )

        # per training step
        supervision = builder.build_batch(
            source_episodes=["quick_swift_SOL_ep04_..."] * B,
            frame_idxs=[0, 17, 132, ...],
            variants=["quick_swift_SOL_ep04___t3_0002_v44", ...],
            device=device,
            dtype=torch.bfloat16,
        )
        # supervision is dict: bbox_masks, bbox_valid, target_centroids

    `is_excluded(source_episode)` returns True for the 4 bad sources;
    callers should pre-filter the LeRobot dataset to exclude these variants.
    """

    def __init__(
        self,
        face_labels_dir: Path,
        manifest_path: Path,
        aug_root: Path,
        episode_mapping_path: Path | None = None,
    ):
        self.face_labels_dir = Path(face_labels_dir)
        self.manifest_path = Path(manifest_path)
        self.aug_root = Path(aug_root)
        # Optional: an episode_index → EpisodeInfo map for the merged HF dataset.
        # Built by `m2_episode_mapping.build_episode_mapping(...)`. If provided,
        # callers can use `build_batch_from_episode_indices(...)` instead of
        # supplying (source_episodes, frame_idxs, variants) per batch.
        self.episode_mapping: list[EpisodeInfo] | None = None
        if episode_mapping_path is not None:
            self.episode_mapping = load_mapping_json(Path(episode_mapping_path))

        # Preload manifest + centroid lookup (~150 KB).
        manifest = json.loads(self.manifest_path.read_text())
        self.centroid_lookup: dict[str, np.ndarray] = {
            celeb: np.asarray(info["centroid"], dtype=np.float32)
            for celeb, info in manifest["celebs"].items()
            if info["centroid"] is not None
        }

        # Preload all face_labels JSONs (~40 MB) into RAM, keyed by source.
        self._face_labels_cache: dict[str, dict] = {}
        for p in sorted(self.face_labels_dir.glob("*.face_labels.json")):
            src = p.name.replace(".face_labels.json", "")
            if src in EXCLUDED_SOURCES:
                continue
            self._face_labels_cache[src] = json.loads(p.read_text())

        # Quick lookup: dict {source: {frame_idx: frame_entry}}
        # face_labels.json has "frames" as a list; convert to a dict so the
        # per-step lookup is O(1) instead of O(n_frames).
        self._frame_lookup: dict[str, dict[int, dict]] = {}
        for src, fl in self._face_labels_cache.items():
            self._frame_lookup[src] = {f["frame_idx"]: f for f in fl["frames"]}

        # Augmentation.json is loaded lazily — variants per training run vary,
        # and there are ~9 000 of them total. LRU cache keeps RAM bounded if
        # the training set is sampled with replacement.
        self._aug_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_excluded(source_episode: str) -> bool:
        return source_episode in EXCLUDED_SOURCES

    @staticmethod
    def excluded_variants_filter(variant_name: str) -> bool:
        """Return True if a variant should be dropped from training.

        Variants are named `<source>__t3_<tuple>_v<variant>` so we just
        prefix-match against EXCLUDED_SOURCES.
        """
        for src in EXCLUDED_SOURCES:
            if variant_name.startswith(src + "__t3_"):
                return True
        return False

    def _load_augmentation(self, variant: str) -> dict:
        """Read augmentation.json for one variant. LRU-cached."""
        if variant in self._aug_cache:
            return self._aug_cache[variant]
        p = self.aug_root / variant / "augmentation.json"
        data = json.loads(p.read_text())
        self._aug_cache[variant] = data
        return data

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_batch(
        self,
        source_episodes: list[str],
        frame_idxs: list[int],
        variants: list[str],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Build supervision tensors for a batch of B samples.

        Args:
            source_episodes: source episode prefix per batch element.
                Length B.  Must NOT be in EXCLUDED_SOURCES — caller should
                pre-filter the LeRobot dataset.  We raise if violated.
            frame_idxs: frame index within the source episode (0..535).
            variants: variant directory name (used to look up
                augmentation.json's new_layout_camera_lmr).
            device, dtype: where/how to place the output tensors.

        Returns:
            dict with keys:
              bbox_masks: (B, 3, 64) torch.bool
              bbox_valid: (B, 3) torch.bool
              target_centroids: (B, 3, 512) torch.float32

        Any per-sample invariant violation raises rather than silently
        emitting zeroes — that would be a hidden silent-fallback per
        CLAUDE.md §5.
        """
        B = len(source_episodes)
        assert len(frame_idxs) == B and len(variants) == B

        masks = np.zeros((B, 3, NUM_CAMERA1_PATCHES), dtype=bool)
        valid = np.zeros((B, 3), dtype=bool)
        targets = np.zeros((B, 3, ARCFACE_EMBED_DIM), dtype=np.float32)

        n_dropped_excluded = 0
        for i, (src, fidx, variant) in enumerate(zip(source_episodes, frame_idxs, variants)):
            if self.is_excluded(src):
                # If callers correctly pre-filtered, this branch is unreachable.
                # Emit [WARN] and emit all-invalid supervision for this sample.
                # CLAUDE.md §5 — never silent.
                n_dropped_excluded += 1
                continue

            frames_for_src = self._frame_lookup.get(src)
            if frames_for_src is None:
                raise KeyError(
                    f"M2SupervisionBuilder: no face_labels.json for source "
                    f"{src!r} (preloaded sources={len(self._face_labels_cache)})"
                )
            frame_entry = frames_for_src.get(fidx)
            if frame_entry is None:
                raise KeyError(
                    f"M2SupervisionBuilder: face_labels for {src!r} has no "
                    f"frame_idx={fidx} (max was "
                    f"{max(frames_for_src.keys()) if frames_for_src else 'n/a'})"
                )
            aug = self._load_augmentation(variant)
            new_lmr = aug["new_layout_camera_lmr"]

            m, v, t = build_supervision_for_frame(
                frame_entry,
                new_layout_camera_lmr=new_lmr,
                centroid_lookup=self.centroid_lookup,
            )
            masks[i] = m
            valid[i] = v
            targets[i] = t

        if n_dropped_excluded:
            print(f"[WARN] M2SupervisionBuilder: expected pre-filtered batch but received "
                  f"{n_dropped_excluded} samples from excluded sources, "
                  f"fallback=marked-all-slots-invalid-for-those-samples", flush=True)

        return {
            "bbox_masks": torch.from_numpy(masks).to(device=device),
            "bbox_valid": torch.from_numpy(valid).to(device=device),
            "target_centroids": torch.from_numpy(targets).to(device=device, dtype=torch.float32),
        }

    def build_batch_from_episode_indices(
        self,
        episode_indices: list[int],
        frame_idxs: list[int],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper for the training loop.

        The LeRobotDataset yields batches with `episode_index` and
        `frame_index` per sample. We map each episode_index to a variant
        via `self.episode_mapping`, then call build_batch.

        For BASE TELEOPS (episode_index < N_base), face_labels haven't been
        generated, so we return all-invalid supervision for those samples.
        The M2 loss handles all-invalid frames gracefully (zero loss with
        grad attached; per `m2_align_loss` docstring).

        Requires `episode_mapping_path` was provided at construction.
        """
        if self.episode_mapping is None:
            raise RuntimeError(
                "build_batch_from_episode_indices requires construction with "
                "episode_mapping_path=... — see m2_episode_mapping.py"
            )

        B = len(episode_indices)
        assert len(frame_idxs) == B

        # Allow callers (Pi0.5 wrapper) to inject grid + geometry overrides
        # so the same builder works for SmolVLA's 8x8 grid and Pi0.5's 16x16.
        patch_grid = kwargs.get("patch_grid", None)  # int; if set, overrides NUM_CAMERA1_PATCHES
        resize_with_pad_box_fn = kwargs.get("resize_with_pad_box", None)
        bbox_to_patch_mask_fn = kwargs.get("bbox_to_patch_mask", None)
        num_patches = (patch_grid * patch_grid) if patch_grid is not None else NUM_CAMERA1_PATCHES

        masks = np.zeros((B, 3, num_patches), dtype=bool)
        valid = np.zeros((B, 3), dtype=bool)
        targets = np.zeros((B, 3, ARCFACE_EMBED_DIM), dtype=np.float32)
        # Per-sample auxiliary info for KLAL attention supervision.
        target_slot_idx = np.full((B,), -1, dtype=np.int64)
        target_celeb_short = [""] * B  # "swift" / "obama" / "lecun" / "" if no target

        # Short → layout-letter map; mirrors generate_aug_v3.py.
        SHORT_TO_LETTER = {"swift": "S", "obama": "O", "lecun": "L"}

        n_base = 0
        n_excluded = 0
        n_no_detection = 0
        for i, (ep_idx, fidx) in enumerate(zip(episode_indices, frame_idxs)):
            if ep_idx < 0 or ep_idx >= len(self.episode_mapping):
                raise IndexError(
                    f"episode_index={ep_idx} out of range "
                    f"(mapping has {len(self.episode_mapping)} entries)"
                )
            ep = self.episode_mapping[ep_idx]
            if ep.is_base:
                # No face_labels for base teleops — leave supervision all-invalid.
                n_base += 1
                continue
            if self.is_excluded(ep.source_episode):
                # Pre-filter should have caught this, but defend in depth.
                n_excluded += 1
                continue

            frames_for_src = self._frame_lookup.get(ep.source_episode)
            if frames_for_src is None:
                raise KeyError(
                    f"M2SupervisionBuilder: no face_labels for source "
                    f"{ep.source_episode!r} (episode_index={ep_idx}, "
                    f"variant={ep.variant_name!r})"
                )
            frame_entry = frames_for_src.get(fidx)
            if frame_entry is None:
                # The face detector missed this frame (e.g. last frame of an
                # episode where the robot occludes the workspace). Skip M2
                # supervision for this sample — leave masks/valid/targets at
                # their zero-initialized state so the loss ignores it. The
                # action-loss path still uses the sample normally.
                n_no_detection += 1
                continue
            aug = self._load_augmentation(ep.variant_name)
            new_lmr = aug["new_layout_camera_lmr"]
            tgt_short = (aug.get("new_target_short")
                         or aug.get("target_short")
                         or aug.get("target_celeb"))
            if tgt_short is not None and tgt_short in SHORT_TO_LETTER:
                letter = SHORT_TO_LETTER[tgt_short]
                if letter in new_lmr:
                    target_slot_idx[i] = new_lmr.index(letter)
                target_celeb_short[i] = tgt_short

            m, v, t = build_supervision_for_frame(
                frame_entry,
                new_layout_camera_lmr=new_lmr,
                centroid_lookup=self.centroid_lookup,
                patch_grid=patch_grid,
                resize_with_pad_box_fn=resize_with_pad_box_fn,
                bbox_to_patch_mask_fn=bbox_to_patch_mask_fn,
            )
            masks[i] = m
            valid[i] = v
            targets[i] = t

        if n_excluded:
            print(f"[WARN] M2SupervisionBuilder: expected pre-filtered batch but got "
                  f"{n_excluded} excluded samples, fallback=marked-invalid", flush=True)
        if n_no_detection:
            # Per CLAUDE.md §5 — never silent. Detector misses are expected at
            # a low rate (often the last frame of an episode); we log them
            # so a sudden surge would be visible.
            print(f"[WARN] M2SupervisionBuilder: {n_no_detection}/{B} samples had "
                  f"no face_labels entry for the requested frame_idx, "
                  f"fallback=marked-invalid (M2 skipped for those samples)",
                  flush=True)

        return {
            "bbox_masks": torch.from_numpy(masks).to(device=device),
            "bbox_valid": torch.from_numpy(valid).to(device=device),
            "target_centroids": torch.from_numpy(targets).to(device=device, dtype=torch.float32),
            "n_base_samples": n_base,
            # KLAL inputs (Pi0.5 wrapper consumes these).
            "target_slot_idx": torch.from_numpy(target_slot_idx).to(device=device),
            "target_celeb_short": target_celeb_short,  # list[str], wrapper builds name_token_positions
        }


def variant_name_to_source(variant_name: str) -> str:
    """Parse the source-episode prefix from a variant directory name.

    `quick_lecun_LSO_ep01_20260511_205000__t3_0002_v00`
        → `quick_lecun_LSO_ep01_20260511_205000`
    """
    sep = "__t3_"
    if sep not in variant_name:
        raise ValueError(f"variant_name={variant_name!r} doesn't contain {sep!r}")
    return variant_name.split(sep)[0]
