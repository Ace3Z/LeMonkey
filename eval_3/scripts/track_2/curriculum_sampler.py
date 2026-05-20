"""Curriculum learning sampler for Track 2 (Enhancement B-5).

Step < switch_step:  sample only "easy" episodes (mean hardneg_gap >= 0.10)
Step >= switch_step: sample from the full keep_list distribution

Per Bengio 2009 "Curriculum Learning" — the model learns coarse face features
first on easy variants, then refines fine-grained discrimination on confusable
pairs once the basic features are formed.

Integration point: drop-in replacement for the lerobot dataloader's sampler.
The training loop calls `set_step(step)` once per step; the sampler internally
flips its weight tensor at the switch point.

Per CLAUDE.md §5: no silent fallbacks — emit [WARN] if config is inconsistent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


class CurriculumWeightedSampler(Sampler[int]):
    """Two-phase weighted sampler.

    Phase 1 (step < switch_step):
        weights = base_weights * easy_mask   (only easy episodes can be sampled)

    Phase 2 (step >= switch_step):
        weights = base_weights               (full distribution per hard-neg weighting)

    Args:
        base_weights: np.ndarray of shape (max_episode_idx+1,) — per-episode sampling
            weights from build_keep_list_and_weights.py (hardneg_weights.npy).
            Zero for filtered-out episodes; 1.0 for normal; HARD_WEIGHT (default 2.0)
            for confusable hard variants.
        easy_mask: boolean np.ndarray of same shape — True if episode's mean
            hardneg_gap >= threshold (i.e. clearly distinguishable identity).
            Built from the audit parquet's per-episode aggregate.
        switch_step: int — training step at which curriculum switches from
            phase 1 to phase 2. Default 5000.
        num_samples: int — total samples per epoch. Default len(base_weights).
        seed: int — RNG seed.
    """

    def __init__(
        self,
        base_weights: np.ndarray,
        easy_mask: np.ndarray,
        switch_step: int = 5000,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        if base_weights.shape != easy_mask.shape:
            raise ValueError(
                f"shape mismatch: base_weights={base_weights.shape} "
                f"easy_mask={easy_mask.shape}"
            )
        n_easy = int((easy_mask & (base_weights > 0)).sum())
        n_all = int((base_weights > 0).sum())
        if n_easy == 0:
            print("[WARN] CurriculumWeightedSampler: expected>=1 easy episode, "
                  "got=0, fallback=phase 2 only (no curriculum)", flush=True)
        else:
            print(f"[curriculum] phase 1: {n_easy}/{n_all} episodes "
                  f"({n_easy/max(n_all,1)*100:.1f}% of keep_list are 'easy')")

        self.base_weights = torch.from_numpy(base_weights.astype(np.float32))
        self.easy_mask = torch.from_numpy(easy_mask.astype(bool))
        self.switch_step = int(switch_step)
        self.num_samples = num_samples or int((self.base_weights > 0).sum().item())
        self._step = 0
        self._g = torch.Generator()
        self._g.manual_seed(seed)
        # Phase 1 weights pre-computed (mask × base, zeros stay zero).
        self._phase1_weights = self.base_weights * self.easy_mask.float()
        self._phase2_weights = self.base_weights
        if self._phase1_weights.sum() <= 0:
            print("[WARN] CurriculumWeightedSampler: phase 1 weights all zero; "
                  "expected>0, got=0, fallback=phase 2 from start", flush=True)
            self._phase1_weights = self._phase2_weights

    def set_step(self, step: int) -> None:
        """Training loop calls this once per step to advance curriculum phase."""
        self._step = int(step)

    @property
    def in_phase_1(self) -> bool:
        return self._step < self.switch_step

    def __iter__(self):
        w = self._phase1_weights if self.in_phase_1 else self._phase2_weights
        # multinomial with replacement, single batch of num_samples indices.
        idx = torch.multinomial(
            w, num_samples=self.num_samples,
            replacement=True, generator=self._g,
        )
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_curriculum_sampler(
    hardneg_weights_path: Path,
    audit_parquet_path: Path,
    switch_step: int = 5000,
    hardneg_gap_threshold: float = 0.10,
    num_samples: int | None = None,
    seed: int = 42,
) -> CurriculumWeightedSampler:
    """Construct the sampler from the two artifacts produced by data prep.

    Args:
        hardneg_weights_path: from build_keep_list_and_weights.py (.npy file).
        audit_parquet_path: from arcface_audit_200celeb.py (the audit parquet).
        switch_step: when to switch from phase 1 to phase 2.
        hardneg_gap_threshold: mean hardneg_gap below which an episode is "hard"
            (i.e. NOT easy → masked out in phase 1).
        num_samples: samples per epoch.
        seed: RNG seed.
    """
    import pandas as pd

    if not hardneg_weights_path.is_file():
        raise FileNotFoundError(f"hardneg weights not found: {hardneg_weights_path}")
    if not audit_parquet_path.is_file():
        raise FileNotFoundError(f"audit parquet not found: {audit_parquet_path}")

    base_weights = np.load(hardneg_weights_path)
    df = pd.read_parquet(audit_parquet_path)
    if "episode_idx" not in df.columns or "hardneg_gap" not in df.columns:
        raise ValueError(f"audit parquet missing required columns: "
                         f"{set(df.columns)}")

    # Per-episode mean hardneg_gap (drop NaN rows from the agg).
    per_ep = df.dropna(subset=["hardneg_gap"]).groupby("episode_idx").agg(
        mean_hardneg_gap=("hardneg_gap", "mean")
    )
    easy_mask = np.zeros_like(base_weights, dtype=bool)
    for ep_idx, row in per_ep.iterrows():
        ep_idx_int = int(ep_idx)
        if ep_idx_int < easy_mask.shape[0]:
            easy_mask[ep_idx_int] = row["mean_hardneg_gap"] >= hardneg_gap_threshold

    return CurriculumWeightedSampler(
        base_weights=base_weights,
        easy_mask=easy_mask,
        switch_step=switch_step,
        num_samples=num_samples,
        seed=seed,
    )
