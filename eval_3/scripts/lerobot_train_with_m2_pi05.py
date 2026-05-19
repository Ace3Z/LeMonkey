#!/usr/bin/env python3
"""Launcher: run `lerobot.scripts.lerobot_train` with M2 + KLAL on Pi0.5.

Same monkey-patch trick as `lerobot_train_with_m2.py` (SmolVLA), but
targets `PI05Policy` and stacks the KLAL attention-supervision loss on
top of M2 ArcFace alignment.

Env vars (all required unless noted):
    M2_FACE_LABELS_DIR     — eval3_m2_toolkit/face_labels/
    M2_MANIFEST_PATH       — eval3_m2_toolkit/celeb_embeddings.json
    M2_AUG_ROOT            — directory containing augmentation.json files
    M2_EPISODE_MAPPING     — eval3_m2_toolkit/episode_mapping.json
    M2_DATASET_REPO_ID     — used by wrapper to derive frame_index lookup
    M2_LAMBDA              — float, default 0.2 (BlindVLA Eq.9 lambda)
    M2_CAPTURE_LAYER       — int, default 10 (gemma_2b 18-layer, ~57% depth)
    KLAL_LAMBDA            — float, default 1.0 (WACV 2026 KLAL weight)
    KLAL_LAYERS            — comma-separated, default "6,10,14,17"
    KLAL_SIGMA_PATCHES     — Gaussian sigma in patch units, default 1.5
    M2_LOG_EVERY           — int, default 100
    M2_DISABLE             — set "1" to skip the wrap (debug)
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval_3/aug"))


def _env(key, default=None, required=False, cast=str):
    v = os.environ.get(key)
    if v is None:
        if required:
            raise SystemExit(f"required env var {key!r} not set")
        return default
    return cast(v)


def _patch_get_safe_version():
    """Same stub as the SmolVLA launcher — skip HF tag lookup."""
    import lerobot.datasets.utils as ldu
    import lerobot.datasets.dataset_metadata as ldmm
    import lerobot.datasets.lerobot_dataset as lldataset
    def _stub(repo_id, version):
        v = str(version)
        return v if v.startswith("v") else f"v{v}"
    ldu.get_safe_version = ldmm.get_safe_version = lldataset.get_safe_version = _stub
    print("[m2-pi05] patched get_safe_version", flush=True)

    # Same .cache marker removal as SmolVLA launcher.
    import shutil
    hf_home = os.environ.get("HF_LEROBOT_HOME") or str(Path.home() / ".cache/huggingface/lerobot")
    repo = os.environ.get("M2_DATASET_REPO_ID")
    if not repo:
        for a in sys.argv:
            if a.startswith("--dataset.repo_id="):
                repo = a.split("=", 1)[1]; break
    if repo:
        marker = Path(hf_home) / repo / ".cache"
        if marker.exists():
            shutil.rmtree(marker, ignore_errors=True)
            print(f"[m2-pi05] removed stale download marker {marker}", flush=True)


def _patch_make_policy():
    _patch_get_safe_version()
    if _env("M2_DISABLE", default="0") == "1":
        print("[m2-pi05] M2_DISABLE=1 — skipping wrap", flush=True)
        return

    face_labels_dir = Path(_env("M2_FACE_LABELS_DIR", required=True))
    manifest_path = Path(_env("M2_MANIFEST_PATH", required=True))
    aug_root = Path(_env("M2_AUG_ROOT", required=True))
    episode_mapping = Path(_env("M2_EPISODE_MAPPING", required=True))
    lam_m2 = _env("M2_LAMBDA", default=0.2, cast=float)
    capture_layer = _env("M2_CAPTURE_LAYER", default=10, cast=int)
    log_every = _env("M2_LOG_EVERY", default=100, cast=int)

    klal_lam = _env("KLAL_LAMBDA", default=1.0, cast=float)
    klal_layers = tuple(int(x) for x in _env("KLAL_LAYERS", default="6,10,14,17").split(","))
    klal_sigma = _env("KLAL_SIGMA_PATCHES", default=1.5, cast=float)

    for p in [face_labels_dir, manifest_path, aug_root, episode_mapping]:
        if not p.exists():
            raise SystemExit(f"[m2-pi05] required path does not exist: {p}")

    from m2_alignment import FrozenProjector
    from m2_dataloader import M2SupervisionBuilder
    from m2_klal import KLALConfig
    from m2_pi05_policy_wrapper import M2Pi05WrappedPolicy

    print("[m2-pi05] building M2SupervisionBuilder", flush=True)
    builder = M2SupervisionBuilder(
        face_labels_dir=face_labels_dir,
        manifest_path=manifest_path,
        aug_root=aug_root,
        episode_mapping_path=episode_mapping,
    )
    print(f"[m2-pi05] face_labels={len(builder._face_labels_cache)} "
          f"celebs={len(builder.centroid_lookup)} "
          f"episodes={len(builder.episode_mapping) if builder.episode_mapping else 0} "
          f"lam_m2={lam_m2} klal_lam={klal_lam} capture={capture_layer} "
          f"klal_layers={klal_layers}", flush=True)

    projector = FrozenProjector(in_dim=2048)
    klal_cfg = KLALConfig(
        capture_layers=klal_layers,
        target_sigma_patches=klal_sigma,
        lam=klal_lam,
    )

    factory = importlib.import_module("lerobot.policies.factory")
    original_make_policy = factory.make_policy

    def make_policy_with_m2_pi05(*args, **kwargs):
        policy = original_make_policy(*args, **kwargs)
        wrapped = M2Pi05WrappedPolicy(
            policy=policy, builder=builder, projector=projector,
            capture_layer=capture_layer, lam_m2=lam_m2,
            klal_cfg=klal_cfg, log_every=log_every,
        )
        n_frozen, n_trainable = wrapped.apply_partial_freeze()
        print(f"[m2-pi05] wrapped policy: {n_frozen/1e6:.1f}M frozen, "
              f"{n_trainable/1e6:.1f}M trainable", flush=True)

        # Eligibility sanity check (mirror the SmolVLA one).
        save_self = wrapped.save_pretrained.__self__.__class__.__name__
        push_self = wrapped.push_model_to_hub.__self__.__class__.__name__
        if save_self != "PI05Policy" or push_self != "PI05Policy":
            raise SystemExit(
                f"[m2-pi05] ELIGIBILITY FAIL: save_pretrained→{save_self}, "
                f"push_model_to_hub→{push_self} (expected PI05Policy)"
            )
        print("[m2-pi05] eligibility OK: save/push bind to PI05Policy", flush=True)
        return wrapped

    factory.make_policy = make_policy_with_m2_pi05
    train_mod = importlib.import_module("lerobot.scripts.lerobot_train")
    if hasattr(train_mod, "make_policy"):
        train_mod.make_policy = make_policy_with_m2_pi05


def main() -> int:
    _patch_make_policy()
    from lerobot.scripts.lerobot_train import main as lerobot_main
    return lerobot_main() or 0


if __name__ == "__main__":
    sys.exit(main())
