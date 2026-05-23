#!/usr/bin/env python3
"""Top-up generator: produce ~22 variants per previously-missing celeb.

Background. The main run (generate_aug_broad.py, 5/15) used a 181-celeb bank
because load_photo_bank dropped 13 celebs whose only photos were
landscape or B&W. We dropped Eastwood + Brando (intrinsically B&W) and
backfilled the remaining 11 with face-cropped landscape + sat-boosted
ETH portraits, so the bank is now 192 celebs.

This script generates 22 extra variants per missing celeb, using NEW
variant indices (>=25) so the existing variants 0-24 stay untouched.
Distractors are drawn from the full 192-celeb pool automatically by
assign_celebs_for_variant.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Re-use the orchestrator's helpers
sys.path.insert(0, str(Path(__file__).parent))
from generate_aug_broad import process_episode    # noqa: E402

import importlib.util                          # noqa: E402
spec = importlib.util.spec_from_file_location(
    "inp", Path(__file__).parent.parent / "stages" / "inpaint_video.py"
)
_v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(_v4)
load_photo_bank = _v4.load_photo_bank


# Identified from datasets/eval3_aug_v3/_run_summary.json audit (5/15):
# 13 celebs had zero renders. Two (Eastwood, Brando) intentionally dropped.
MISSING_CELEBS = [
    "andrej_karpathy", "clement_delangue", "drake",
    "lebron_james", "marc_pollefeys", "marco_hutter",
    "oier_mees", "roland_siegwart", "sergey_levine",
    "stan_wawrinka", "yann_lecun",
]
VARIANTS_PER_MISSING = 22
START_VAR_IDX = 25     # variants 0..24 already rendered by the main run


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="datasets/eval3")
    p.add_argument("--photo-bank", required=True,
                   help="datasets/eval3_celebs/scraped")
    p.add_argument("--out-root", required=True,
                   help="datasets/eval3_aug_v3 (will add new variants here)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    # 1. Load bank — should be 192 celebs after backfill
    bank = load_photo_bank(Path(args.photo_bank),
                           portrait_only=True, color_only=True)
    print(f"loaded photo bank: {len(bank)} celebs", flush=True)
    missing_in_bank = [c for c in MISSING_CELEBS if c not in bank]
    if missing_in_bank:
        print(f"[FATAL] still missing from bank: {missing_in_bank}",
              file=sys.stderr)
        return 2
    print(f"all {len(MISSING_CELEBS)} target celebs are in the bank", flush=True)

    # 2. List eps with corners (the 151 that succeeded stage 2)
    root = Path(args.root)
    all_eps = sorted(p for p in root.iterdir()
                       if p.is_dir() and (p / "reference.json").is_file())
    corner_eps = [p for p in all_eps
                    if (p / "portrait_corners.json").is_file()]
    print(f"corner-eps available: {len(corner_eps)} / {len(all_eps)}",
          flush=True)

    # 3. Build target_assignment via round-robin across corner_eps.
    #    Each missing celeb gets VARIANTS_PER_MISSING new variant slots.
    next_var_idx: dict[str, int] = {ep.name: START_VAR_IDX for ep in corner_eps}
    target_assignment: dict[tuple[str, int], str] = {}
    ep_names = [ep.name for ep in corner_eps]
    for round_i in range(VARIANTS_PER_MISSING):
        for ci, celeb in enumerate(MISSING_CELEBS):
            ep_idx = (round_i * len(MISSING_CELEBS) + ci) % len(ep_names)
            ep_name = ep_names[ep_idx]
            v = next_var_idx[ep_name]
            target_assignment[(ep_name, v)] = celeb
            next_var_idx[ep_name] = v + 1

    assert len(target_assignment) == len(MISSING_CELEBS) * VARIANTS_PER_MISSING

    # 4. Per-episode dispatch — only eps that received at least one new slot.
    eps_to_run = sorted({ep_name for ep_name, _ in target_assignment})
    print(f"running {len(eps_to_run)} episodes "
          f"({len(target_assignment)} new variants total)", flush=True)
    from collections import Counter
    celeb_counts = Counter(target_assignment.values())
    print(f"per-celeb counts (should all equal {VARIANTS_PER_MISSING}):")
    for c, n in sorted(celeb_counts.items()):
        print(f"   {c:25}  {n}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for i, ep_name in enumerate(eps_to_run, start=1):
        ep_dir = root / ep_name
        # Find the max var_idx assigned to this ep — pass it as num_variants
        max_v_here = max(v for (en, v) in target_assignment if en == ep_name)
        # process_episode iterates range(num_variants); existing 0..24 will be
        # skipped via the "already exists" check, only START_VAR_IDX..max_v
        # will actually render.
        num_variants = max_v_here + 1
        print(f"\n[{i}/{len(eps_to_run)}] {ep_name} "
              f"(rendering up to var{max_v_here:02d})", flush=True)
        try:
            r = process_episode(
                ep_dir, out_root, bank,
                num_variants=num_variants,
                seed=args.seed, fps=args.fps,
                force=False, debug=False,
                target_assignment=target_assignment,
            )
        except Exception as e:
            r = {"ep": ep_name, "error": f"exception: {e}"}
            print(f"  [ERROR] {ep_name}: {e}", flush=True)
        results.append(r)

    summary = {
        "missing_celebs": MISSING_CELEBS,
        "variants_per_missing": VARIANTS_PER_MISSING,
        "start_var_idx": START_VAR_IDX,
        "n_eps_run": len(eps_to_run),
        "elapsed_sec": time.time() - t0,
        "results": results,
    }
    summary_path = out_root / "_topup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
