#!/usr/bin/env python3
"""SmolVLA + VL co-training, RT-2 §3.2 style.

Mixed-batch trainer: every (vl_ratio+1)-th step is a VL batch (CE loss on
SmolVLM2's LM head), the rest are robot batches (SmolVLA flow-matching
action loss). Both losses share one optimizer; both gradients flow into
the SmolVLM2 body.

This is the SmolVLA-450M sibling of eval_3/scripts/track_2/lerobot_train_with_vl_cotrain.py
(which is Pi0.5-3B). Unlike that one, this is NOT a scaffold — the training loop
is integrated end-to-end so it runs as-is on a single GPU AWS node.

Per RT-2 §3.2: keeping web/VQA data alongside robot data during fine-tuning
prevents catastrophic forgetting of the VLM's web knowledge (celeb-name
recognition, in our case). Diagnosed need: sequential VLM-then-action
fine-tuning produces a positional-shortcut policy that ignores the prompt
celeb name.

Per CLAUDE.md §5: every fallback emits [WARN] with context.
Per CLAUDE.md §7: written but UNSMOKED locally — must be smoke-tested on
the user's AWS node before being trusted for a 24h run.

Usage
=====

    python eval_3/scripts/smolvla_cotrain/cotrain.py \\
        --robot_dataset=HBOrtiz/so101_eval3_track3_v3_baseline \\
        --vl_manifest=HBOrtiz/eval3_track3_vl_pairs \\
        --vl_ratio=5 \\
        --output_dir=outputs/smolvla_cotrain_5to1 \\
        --steps=30000 \\
        --batch_size=32 \\
        --vl_batch_size=8 \\
        --lr=5e-5 \\
        --push_to_hub_repo=HBOrtiz/smolvla_eval3_cotrain_5to1

For a smoke test, drop --steps=200 and --batch_size=4.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# ---------------------------------------------------------------------------
# Lerobot HF-API rate-limit bypass
#
# lerobot v0.5.2 calls get_safe_version() (→ list_repo_refs API call) in two
# places: LeRobotDatasetMetadata.__init__ and LeRobotDataset.__init__.  Both
# fire on every process start, even when the dataset is fully cached, because
# snapshot_download enumerates the entire repo via the HF API before applying
# allow_patterns.  On this instance this reliably triggers 429 rate-limit
# stalls.
#
# Fix: patch get_safe_version and snapshot_download in lerobot's namespaces
# before any lerobot import.  get_safe_version becomes a no-op (returns the
# revision unchanged).  snapshot_download returns the existing lerobot hub
# cache path immediately if it already exists — otherwise falls through to the
# real download.
# ---------------------------------------------------------------------------
def _patch_lerobot_network_calls() -> None:
    import importlib
    import huggingface_hub as _hfhub

    _orig_snapshot = _hfhub.snapshot_download

    _LEROBOT_HUB = Path(os.path.expanduser("~/.cache/huggingface/lerobot/hub"))

    def _fast_snapshot(repo_id: str, repo_type: str = "model",
                       cache_dir=None, **kw) -> str:
        if repo_type == "dataset":
            _cache = Path(cache_dir) if cache_dir else _LEROBOT_HUB
            slug = "datasets--" + repo_id.replace("/", "--")
            snap_dir = _cache / slug / "snapshots"
            if snap_dir.is_dir():
                snaps = sorted(snap_dir.iterdir())
                if snaps and (snaps[0] / "meta").is_dir():
                    print(f"[cotrain] snapshot_download bypassed — using cache: {snaps[0]}",
                          flush=True)
                    return str(snaps[0])
        return _orig_snapshot(repo_id=repo_id, repo_type=repo_type,
                              cache_dir=cache_dir, **kw)

    def _noop_get_safe_version(repo_id: str, version: str) -> str:
        return version

    # Patch in all modules that import these names directly.
    for mod_name in (
        "lerobot.datasets.dataset_metadata",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.datasets.utils",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "snapshot_download"):
            mod.snapshot_download = _fast_snapshot
        if hasattr(mod, "get_safe_version"):
            mod.get_safe_version = _noop_get_safe_version

    print("[cotrain] lerobot HF-API calls patched (snapshot_download + get_safe_version)",
          flush=True)


_patch_lerobot_network_calls()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--robot_dataset", required=True,
                   help="LeRobotDataset HF repo id (e.g. HBOrtiz/so101_eval3_track3_v3_baseline)")
    p.add_argument("--robot_max_episodes", type=int, default=None,
                   help="Cap dataset to the first N episodes. Use when only a subset of video "
                        "files are cached locally to avoid FileNotFoundError mid-training.")
    p.add_argument("--vl_manifest", required=True,
                   help="VL pairs HF repo id (e.g. HBOrtiz/eval3_track3_vl_pairs) "
                        "OR local parquet path")
    p.add_argument("--vl_image_root", default=None,
                   help="Override path to pre-extracted VL images dir. "
                        "If None, snapshot_download the whole VL repo (~1 GB).")
    # Model
    p.add_argument("--pretrained_path", default="lerobot/smolvla_base",
                   help="SmolVLA starting checkpoint. Use HansOrtiz/smolvlm2_celeb_warm "
                        "for warm-VLM, or lerobot/smolvla_base for cold.")
    p.add_argument("--vlm_model_name", default=None,
                   help="Optional override for the inner VLM checkpoint. "
                        "If set, replaces config.vlm_model_name before loading.")
    # Training
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Robot batch size (per step on robot-batch steps)")
    p.add_argument("--vl_batch_size", type=int, default=8,
                   help="VL batch size (per step on VL-batch steps). Usually smaller.")
    p.add_argument("--vl_ratio", type=int, default=5,
                   help="Number of robot batches per 1 VL batch. RT-2 spec is "
                        "'robot >> web'; ObjectVLA used 10. We ran the eval-day "
                        "checkpoint at 5:1 (more VQA pressure than ObjectVLA).")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Optimizer LR. SmolVLA default; RT-2 says 'use VLM-paper hparams'.")
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_freq", type=int, default=5000,
                   help="Save a checkpoint every N steps")
    p.add_argument("--push_to_hub_repo", default=None,
                   help="If set, push final checkpoint to this HF repo id")
    # Smoke / dev
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--compile_model", action="store_true",
                   help="torch.compile the policy. Recommend OFF for smoke, ON for full run.")
    p.add_argument("--video_backend", default="pyav",
                   help="LeRobotDataset video backend. Default pyav — torchcodec leaks "
                        "~35 GB/worker over long runs (see TORCHCODEC_OOM_REPORT.md).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# VL dataset — reads Roham's `eval3_objectvla_vl_pairs` parquet schema.
# Schema (verified 2026-05-20): image_path, prompt, target, bbox_xyxy_norm,
#                                celeb_name, celeb_slug, caption_type, episode,
#                                frame_idx, pid
# -----------------------------------------------------------------------------

class VLPairsDataset(Dataset):
    """VQA pairs: (image, prompt) → target. Used for the LM-head CE loss."""

    REQUIRED_COLS = {"image_path", "prompt", "target", "celeb_slug", "caption_type"}

    def __init__(self, manifest_path_or_id: str, image_root: Path | None = None):
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("pandas required for VL dataset") from e

        # Local parquet OR HF repo id.
        if Path(manifest_path_or_id).is_file():
            self.df = pd.read_parquet(manifest_path_or_id)
            dataset_root = Path(manifest_path_or_id).parent
        else:
            from huggingface_hub import hf_hub_download, snapshot_download
            mf_path = hf_hub_download(
                repo_id=manifest_path_or_id, repo_type="dataset",
                filename="manifest.parquet",
                token=os.environ.get("HF_TOKEN"),
            )
            self.df = pd.read_parquet(mf_path)
            dataset_root = Path(mf_path).parent
            if image_root is None:
                if (dataset_root / "images").is_dir():
                    image_root = dataset_root / "images"
                else:
                    print(f"[WARN] vl_dataset: expected pre-extracted images/ under "
                          f"{dataset_root}, got=missing, "
                          f"fallback=snapshot_download(repo_id={manifest_path_or_id})",
                          flush=True)
                    full = snapshot_download(
                        repo_id=manifest_path_or_id, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"),
                        max_workers=1,  # avoid HF API rate-limit from parallel requests
                    )
                    image_root = Path(full)
                    # The snapshot may have images.tar.zst still packed; unpack if so.
                    # Repo may use either images.tar.zst or data.tar.zst
                    for _archive_name in ("images.tar.zst", "data.tar.zst"):
                        tar_zst = image_root / _archive_name
                        if tar_zst.is_file() and not (image_root / "images").is_dir():
                            print(f"[vl_dataset] extracting {tar_zst} ...", flush=True)
                            import subprocess
                            subprocess.run(
                                ["tar", "--use-compress-program=unzstd",
                                 "-xf", str(tar_zst), "-C", str(image_root)],
                                check=True,
                            )
                            break

        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise ValueError(f"VL manifest missing columns: {missing}; "
                             f"got {list(self.df.columns)}")

        self.image_root = image_root
        print(f"[vl_dataset] {len(self.df)} pairs, {self.df['celeb_slug'].nunique()} celebs, "
              f"image_root={image_root}", flush=True)
        print(f"[vl_dataset] caption mix: "
              f"{dict(self.df['caption_type'].value_counts())}", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        row = self.df.iloc[idx]
        rel = Path(str(row["image_path"]))
        # Try several path layouts the manifest may produce.
        candidates = [
            self.image_root / rel,
            self.image_root.parent / rel,
        ]
        if str(rel).startswith("images/") and self.image_root.name == "images":
            candidates.insert(0, self.image_root / rel.relative_to("images"))

        img_path = next((c for c in candidates if c.is_file()), None)
        if img_path is None:
            print(f"[WARN] vl_dataset row {idx}: image not found at "
                  f"{[str(c) for c in candidates]}, fallback=gray 224x224",
                  flush=True)
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] vl_dataset row {idx}: read failed for "
                      f"{img_path}: {e}, fallback=gray", flush=True)
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return {
            "image": img,
            "prompt": str(row["prompt"]),
            "target": str(row["target"]),
            "celeb_slug": str(row["celeb_slug"]),
        }


def make_vl_collator(processor, max_text_len: int = 1024):
    """Builds VL batches for SmolVLM2's LM head.

    SmolVLM2 uses an Idefics3-style processor that expands the `<image>`
    placeholder into N_image actual image tokens when `images=` is passed.
    To find the prompt/target boundary for label masking, we must run BOTH
    the prompt-only text AND the prompt+target text through the *same*
    `processor(text=..., images=...)` call so the image expansion happens
    identically in both. We then take the prompt-only sequence length as
    the mask boundary in the full input_ids.

    (The earlier shortcut of using `processor.tokenizer(...)` alone was
    wrong — it does NOT run the image-token expansion, so prompt_lens were
    off by ~80-170 tokens and the model would have trained to predict
    image-placeholder tokens.)
    """
    image_token = "<image>"

    def _ensure_image_placeholder(text: str) -> str:
        return text if image_token in text else f"{image_token}{text}"

    def collate(batch: list[dict]) -> dict:
        prompts_only = [_ensure_image_placeholder(ex["prompt"]) for ex in batch]
        prompts_full = [f"{p} {ex['target']}" for p, ex in zip(prompts_only, batch)]
        # SmolVLM2 processor expects a nested list: one inner list of images per text.
        images = [[ex["image"]] for ex in batch]

        # 1. Process full (prompt + target) text + images. This is what we
        #    feed to the VLM forward.
        full_inputs = processor(
            text=prompts_full,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_text_len,
        )

        # 2. Process prompt-only with the SAME images so image-token
        #    expansion matches identically. We discard the resulting
        #    pixel_values; we only need the input_ids length.
        prompt_inputs = processor(
            text=prompts_only,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_text_len,
        )
        # Per-sample prompt token length (number of non-pad tokens).
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

        # 3. Build labels: -100 on prompt portion + pad; target tokens elsewhere.
        input_ids = full_inputs["input_ids"]
        attn = full_inputs["attention_mask"]
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            # Always leave at least one position to predict — otherwise the
            # CE loss is undefined for that sample.
            cap = max(1, min(int(plen), input_ids.shape[1] - 1))
            labels[i, :cap] = -100
        labels = labels.masked_fill(attn == 0, -100)

        out = dict(full_inputs)
        out["labels"] = labels
        return out

    return collate


# -----------------------------------------------------------------------------
# Loss paths
# -----------------------------------------------------------------------------

def smolvla_vqa_loss(policy, batch: dict, device: torch.device) -> torch.Tensor:
    """CE loss on SmolVLM2's LM head for a VL batch.

    Bypasses the action expert entirely — we call the underlying SmolVLM2
    `AutoModelForImageTextToText` instance, which has its own .lm_head and
    returns CausalLMOutputWithPast(.loss, .logits, ...) when labels= is given.
    """
    vlm = policy.model.vlm_with_expert.vlm
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Match the VLM's parameter dtype for pixel values (vision tower is bf16).
    pv_dtype = next(vlm.model.vision_model.parameters()).dtype
    if pixel_values.dtype != pv_dtype:
        pixel_values = pixel_values.to(pv_dtype)

    out = vlm(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
    )
    return out.loss


def smolvla_action_loss(policy, batch: dict) -> torch.Tensor:
    """Standard SmolVLA flow-matching loss on a robot batch."""
    loss, _ = policy.forward(batch)
    return loss


# -----------------------------------------------------------------------------
# Policy + dataset construction (reuse lerobot factories)
# -----------------------------------------------------------------------------

def load_policy_and_processor(args, device: torch.device):
    """Constructs SmolVLAPolicy with cotrain-friendly flags."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # Load config via the ChoiceRegistry base so draccus can dispatch on the
    # "type" field stored in the checkpoint's config.json.  Calling
    # SmolVLAConfig.from_pretrained() directly fails because draccus tries to
    # decode "type" as a dataclass field of the concrete subclass.
    cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    assert isinstance(cfg, SmolVLAConfig), (
        f"Expected SmolVLAConfig from {args.pretrained_path}, got {type(cfg)}"
    )
    cfg.train_expert_only = False    # CRITICAL: VLM body must be trainable for VQA CE.
    cfg.freeze_vision_encoder = True # Keep SigLIP frozen — RT-2 doesn't tune it either.
    cfg.empty_cameras = 1  # camera3 zero-padded (training recipe --policy.empty_cameras=1)
    if args.vlm_model_name is not None:
        cfg.vlm_model_name = args.vlm_model_name
    # The cfg.device read in subordinate constructors should match our device.
    # Use the full string ("cuda:1" under DDP) so DeviceProcessorStep moves
    # tensors to *this* rank's GPU, not always cuda:0.
    cfg.device = str(device)

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path, config=cfg)
    policy = policy.to(device)

    # Re-apply requires_grad in case the policy load reset them.
    policy.model.vlm_with_expert.set_requires_grad()

    # SmolVLM2 processor (Idefics3Processor under the hood) for the VL collator.
    vl_processor = policy.model.vlm_with_expert.processor
    return policy, vl_processor, cfg


def _episodes_with_complete_files(meta) -> list[int]:
    """Episode indices whose every data + video file is present in the cache.

    The lerobot cache may hold only a *partial* download. LeRobotDataset builds
    fine from metadata regardless, but raises FileNotFoundError inside a
    DataLoader worker the first time `__getitem__` touches an episode whose
    file is absent (observed: reference/file-934.mp4 missing from an otherwise
    near-contiguous cache, crashing the run ~100 steps in). We pre-filter to
    fully-covered episodes so training never hits a missing file mid-run.

    Each episode's metadata row records the (chunk_index, file_index) of its
    data parquet and of every video stream; we resolve those against the
    on-disk path templates and keep only episodes where all files exist.
    """
    import re

    root = Path(meta.root)
    ep = meta.episodes.to_pandas()

    # (path_template, video_key_or_None) for every file an episode references.
    file_specs: list[tuple[str, str | None]] = [(meta.info.data_path, None)]
    for col in ep.columns:
        m = re.fullmatch(r"videos/(.+)/file_index", col)
        if m:
            file_specs.append((meta.info.video_path, m.group(1)))

    valid: list[int] = []
    missing: set[str] = set()
    for _, row in ep.iterrows():
        complete = True
        for tmpl, vkey in file_specs:
            prefix = f"videos/{vkey}/" if vkey else "data/"
            rel = tmpl.format(
                video_key=vkey,
                chunk_index=int(row[f"{prefix}chunk_index"]),
                file_index=int(row[f"{prefix}file_index"]),
            )
            if not (root / rel).is_file():
                complete = False
                missing.add(rel)
        if complete:
            valid.append(int(row["episode_index"]))

    if missing:
        print(f"[WARN] robot_dataset: partial cache — expected all {len(ep)} "
              f"episodes covered, got={len(valid)} fully-covered, "
              f"fallback=train on the {len(valid)}-episode subset "
              f"({len(missing)} files missing, e.g. {sorted(missing)[:2]})",
              flush=True)
    return valid


def load_robot_dataset(args, policy_cfg):
    """Loads the LeRobot dataset.

    We use LeRobotDataset directly (not make_dataset()) to avoid the full
    TrainPipelineConfig surface. The dataset's per-frame items are RAW —
    they need to be put through the policy's preprocessor (see
    `build_robot_preprocessor` below) before `policy.forward()`.

    CRITICAL: `delta_timestamps` must be passed. SmolVLA's flow-matching loss
    operates on an *action chunk* of shape (chunk_size, action_dim); without
    delta_timestamps each frame yields a single action (action_dim,) and
    `policy.forward()` blows up inside `embed_suffix` (att/pad mask length
    mismatch). We build delta_timestamps from the policy config exactly as
    lerobot's own `make_dataset()` factory does (resolve_delta_timestamps):
    action → chunk_size future steps, observations → [0].

    Episodes are filtered to those with complete data+video file coverage in
    the local cache (see `_episodes_with_complete_files`) — a partial download
    otherwise crashes mid-run on the first missing file.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    meta = LeRobotDatasetMetadata(repo_id=args.robot_dataset)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    if delta_timestamps is None or "action" not in delta_timestamps:
        raise RuntimeError(
            f"resolve_delta_timestamps produced {delta_timestamps!r} — expected an "
            f"'action' chunk of length chunk_size={policy_cfg.chunk_size}. SmolVLA's "
            f"flow-matching loss cannot run on single-step actions."
        )

    episodes = _episodes_with_complete_files(meta)
    if not episodes:
        raise RuntimeError(
            f"robot_dataset {args.robot_dataset}: no episode has a complete set of "
            f"data+video files in the local cache — cannot train. Re-download the "
            f"dataset or check the cache path."
        )
    if args.robot_max_episodes:
        episodes = episodes[: args.robot_max_episodes]

    ds = LeRobotDataset(repo_id=args.robot_dataset, delta_timestamps=delta_timestamps,
                        video_backend=args.video_backend, episodes=episodes)
    print(f"[robot_dataset] {len(ds)} frames across {ds.num_episodes} episodes "
          f"(video_backend={args.video_backend}, "
          f"action chunk={len(delta_timestamps['action'])}, "
          f"{len(episodes)} episodes with complete file coverage"
          + (f", capped at {args.robot_max_episodes}" if args.robot_max_episodes else "")
          + ")", flush=True)
    return ds


def build_robot_preprocessor(policy_cfg, dataset, pretrained_path: str, device: torch.device):  # noqa: ARG001
    """Builds the SmolVLA pre/post-processor pipeline from lerobot's factory.

    The preprocessor is responsible for:
      1. Renaming observation keys to match the policy's expected schema.
      2. Adding a batch dimension (no-op for already-batched DataLoader output).
      3. Adding a trailing newline to `task` strings (SmolVLA tokenizer quirk).
      4. Tokenizing `task` → `observation.language.tokens` / `..._attention_mask`.
      5. Moving everything to `device`.
      6. Normalizing state/action features using dataset stats.

    Without this, `policy.forward(batch)` raises KeyError on the language-token
    keys. Returns (preprocessor, postprocessor) — the postprocessor is
    saved/pushed alongside but not used during training itself.
    """
    from lerobot.policies.factory import make_pre_post_processors

    # empty_cameras=1 pads camera3 with zeros (matching the training recipe).
    # The base checkpoint has empty_cameras=0; we must set it before building
    # the preprocessor so the pipeline allocates the right number of image slots.
    policy_cfg.empty_cameras = 1

    # Do NOT pass pretrained_path — lerobot/smolvla_base has no policy_preprocessor.json.
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=dataset.meta.stats,
    )

    # Inject the rename map: dataset uses observation.images.reference but the
    # policy expects observation.images.camera2 (training recipe --rename_map).
    from lerobot.processor.rename_processor import RenameObservationsProcessorStep
    for step in pre.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            step.rename_map = {"observation.images.reference": "observation.images.camera2"}
            print("[cotrain] preprocessor rename_map set: reference → camera2", flush=True)
            break

    return pre, post


# -----------------------------------------------------------------------------
# Main training loop
#
# Multi-GPU note: we do NOT wrap the policy in torch DistributedDataParallel.
# DDP syncs gradients via hooks tied to the wrapped module's forward(); but this
# trainer has TWO forward paths — robot batches call policy.forward(), VL
# batches call policy.model.vlm_with_expert.vlm(...) directly — and the VL path
# would bypass DDP entirely. Instead we keep plain modules and average gradients
# across ranks by hand after backward() (see the all_reduce below). This is
# correct for both paths and matches the script's hand-rolled training loop.
# Launch multi-GPU via `torchrun --nproc_per_node=N`; with no torchrun env vars
# it runs single-GPU exactly as before.
# -----------------------------------------------------------------------------

def upload_checkpoint(local_dir: Path, repo_id: str, path_in_repo: str | None) -> bool:
    """Upload a checkpoint dir to the HF Hub. Best-effort: never raises.

    Returns True on success, False on failure (after logging a [WARN]).

    launch.sh exports HF_HUB_OFFLINE=1 once both datasets are cached, which
    would block every Hub HTTP call. By training time the datasets are loaded
    and per-item reads hit local disk, so we flip the offline flag off just
    for the upload and restore it afterwards (workers have their own copy).
    A failed upload must never kill a multi-hour run — hence the [WARN].
    """
    import huggingface_hub.constants as hf_const
    from huggingface_hub import HfApi

    prev_offline = hf_const.HF_HUB_OFFLINE
    hf_const.HF_HUB_OFFLINE = False
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=f"checkpoint: {path_in_repo or 'final'}",
        )
        dest = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
        print(f"[cotrain] uploaded {local_dir.name} → "
              f"https://huggingface.co/{dest}", flush=True)
        return True
    except Exception as e:
        print(f"[WARN] checkpoint upload: expected=upload {local_dir} to "
              f"{repo_id}/{path_in_repo or '(root)'}, got={type(e).__name__}: {e}, "
              f"fallback=local-only checkpoint at {local_dir}", flush=True)
        return False
    finally:
        hf_const.HF_HUB_OFFLINE = prev_offline


def main() -> int:
    args = parse_args()

    # --- Distributed setup. torchrun sets these env vars; absent => single GPU.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main = rank == 0

    if is_distributed:
        torch.cuda.set_device(local_rank)  # must precede init_process_group
        # Generous timeout: the periodic-save barrier and final barrier must
        # not trip the NCCL watchdog while rank 0 writes a checkpoint.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

    def log(msg: str) -> None:
        """Print on rank 0 only — keeps multi-GPU logs readable."""
        if is_main:
            print(msg, flush=True)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility — same seed on every rank so model init matches.
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(f"[cotrain] device={device}, dtype={args.dtype}, world_size={world_size} "
        f"({'DDP' if is_distributed else 'single-GPU'})")
    if device.type == "cuda":
        log(f"[cotrain] gpu={torch.cuda.get_device_name(local_rank)}, "
            f"vram={torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")

    # batch_size / vl_batch_size are GLOBAL — split evenly across ranks so the
    # optimization math is identical to the single-GPU runs we validated.
    if args.batch_size % world_size != 0:
        raise ValueError(f"batch_size {args.batch_size} not divisible by "
                         f"world_size {world_size}")
    if args.vl_batch_size % world_size != 0:
        raise ValueError(f"vl_batch_size {args.vl_batch_size} not divisible by "
                         f"world_size {world_size}")
    per_gpu_bs = args.batch_size // world_size
    per_gpu_vl_bs = args.vl_batch_size // world_size

    # 1. Policy + processor.
    log(f"[cotrain] loading SmolVLA policy from {args.pretrained_path} ...")
    policy, vl_processor, policy_cfg = load_policy_and_processor(args, device)
    policy.train()

    # Make every rank bit-identical before the first step (insurance — same
    # seed + same checkpoint should already guarantee this).
    if is_distributed:
        for p in policy.parameters():
            dist.broadcast(p.data, src=0)
        for b in policy.buffers():
            dist.broadcast(b.data, src=0)

    trainable_n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in policy.parameters())
    log(f"[cotrain] trainable params: {trainable_n / 1e6:.1f}M / {total_n / 1e6:.1f}M "
        f"({100 * trainable_n / total_n:.1f}%)")

    # 2. Robot dataset + preprocessor + dataloader.
    log(f"[cotrain] loading robot dataset {args.robot_dataset} ...")
    robot_ds = load_robot_dataset(args, policy_cfg)
    log("[cotrain] building robot preprocessor (tokenizer + normalizer + device) ...")
    preprocessor, postprocessor = build_robot_preprocessor(
        policy_cfg=policy_cfg,
        dataset=robot_ds,
        pretrained_path=args.pretrained_path,
        device=device,
    )
    robot_sampler = (
        DistributedSampler(robot_ds, num_replicas=world_size, rank=rank,
                           shuffle=True, seed=args.seed, drop_last=True)
        if is_distributed else None
    )
    robot_loader = DataLoader(
        robot_ds,
        batch_size=per_gpu_bs,
        shuffle=(robot_sampler is None),
        sampler=robot_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    robot_epoch = 0
    if robot_sampler is not None:
        robot_sampler.set_epoch(robot_epoch)
    robot_iter = iter(robot_loader)

    # 3. VL dataset + dataloader.
    log(f"[cotrain] loading VL pairs from {args.vl_manifest} ...")
    vl_image_root = Path(args.vl_image_root) if args.vl_image_root else None
    vl_ds = VLPairsDataset(args.vl_manifest, image_root=vl_image_root)
    vl_sampler = (
        DistributedSampler(vl_ds, num_replicas=world_size, rank=rank,
                           shuffle=True, seed=args.seed, drop_last=True)
        if is_distributed else None
    )
    vl_loader = DataLoader(
        vl_ds,
        batch_size=per_gpu_vl_bs,
        shuffle=(vl_sampler is None),
        sampler=vl_sampler,
        num_workers=args.num_workers,
        collate_fn=make_vl_collator(vl_processor),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    vl_epoch = 0
    if vl_sampler is not None:
        vl_sampler.set_epoch(vl_epoch)
    vl_iter = iter(vl_loader)

    # 4. Optimizer — one AdamW over all trainable params.
    optim = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )

    # 5. Training loop.
    log(f"[cotrain] starting training: steps={args.steps}, "
        f"vl_ratio={args.vl_ratio} (one VL batch per {args.vl_ratio} robot batches), "
        f"lr={args.lr}, global batch={args.batch_size}/{args.vl_batch_size} "
        f"({per_gpu_bs}/{per_gpu_vl_bs} per GPU)")

    period = args.vl_ratio + 1   # vl_ratio=10 → VL hits at step%11==0
    last_log_time = time.perf_counter()
    last_flow_loss = float("nan")
    last_vqa_loss = float("nan")

    for step in range(args.steps):
        # is_vl_step is deterministic => every rank runs the same step type, so
        # the set of params carrying a grad matches across ranks and the
        # per-tensor all_reduce below stays collective-safe.
        is_vl_step = (step % period == 0)

        if is_vl_step:
            try:
                batch = next(vl_iter)
            except StopIteration:
                vl_epoch += 1
                if vl_sampler is not None:
                    vl_sampler.set_epoch(vl_epoch)
                vl_iter = iter(vl_loader)
                batch = next(vl_iter)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                loss = smolvla_vqa_loss(policy, batch, device)
            loss_name = "vqa_loss"
        else:
            try:
                batch = next(robot_iter)
            except StopIteration:
                robot_epoch += 1
                if robot_sampler is not None:
                    robot_sampler.set_epoch(robot_epoch)
                robot_iter = iter(robot_loader)
                batch = next(robot_iter)
            # Apply the SmolVLA preprocessor: tokenizes the language task,
            # normalizes state/action features, moves to device. Without
            # this the policy's forward raises KeyError on language tokens.
            batch = preprocessor(batch)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                loss = smolvla_action_loss(policy, batch)
            loss_name = "flow_loss"

        # Global-mean loss for logging + a rank-consistent finite check.
        loss_report = loss.detach().clone()
        if is_distributed:
            dist.all_reduce(loss_report, op=dist.ReduceOp.SUM)
            loss_report /= world_size
        loss_val = loss_report.item()
        if is_vl_step:
            last_vqa_loss = loss_val
        else:
            last_flow_loss = loss_val

        if not torch.isfinite(loss_report):
            # loss_report is identical on every rank => all ranks skip together.
            log(f"[WARN] step {step}: non-finite loss ({loss_name}={loss_val}), "
                f"skipping optimizer step")
            optim.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # Manual DDP gradient averaging. We iterate ALL trainable params in a
        # fixed order and materialise a zero grad wherever backward produced
        # none — the VQA path leaves the action expert with no grad, the robot
        # path leaves the LM head with none. This keeps the all_reduce call
        # sequence byte-identical on every rank: NCCL matches collectives
        # positionally, so a rank-dependent param set could deadlock or mismatch
        # silently. Stock DistributedDataParallel does the same (it reduces and
        # steps every parameter, unused ones included).
        #
        # SUM/world_size is standard DDP averaging: each rank backprops a mean
        # loss over its own equal-sized sub-batch (per_gpu_bs is fixed,
        # drop_last=True), so the averaged gradient is the global-batch mean.
        if is_distributed:
            for p in policy.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in policy.parameters() if p.requires_grad),
                max_norm=args.grad_clip,
            ).item()
        else:
            grad_norm = float("nan")
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step < 50:
            dt = time.perf_counter() - last_log_time
            last_log_time = time.perf_counter()
            steps_per_sec = args.log_every / dt if dt > 0 else 0
            log(f"step {step:6d}  {loss_name}={loss_val:.4f}  "
                f"(last flow={last_flow_loss:.4f} vqa={last_vqa_loss:.4f})  "
                f"grad={grad_norm:.2f}  steps/s={steps_per_sec:.2f}")

        if step > 0 and step % args.save_freq == 0:
            if is_main:
                ckpt_dir = out_dir / f"step_{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                # Save preprocessor/postprocessor alongside so eval-day
                # inference can reproduce normalization + tokenization.
                # Without these, the checkpoint cannot be deployed.
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                log(f"[cotrain] checkpoint saved → {ckpt_dir}")
                # Auto-upload this checkpoint to its own step_NNNNNN/ subfolder.
                if args.push_to_hub_repo:
                    upload_checkpoint(ckpt_dir, args.push_to_hub_repo,
                                      f"step_{step:06d}")
            # All ranks meet here so non-main ranks do not run ahead into the
            # next step's collectives while rank 0 is still writing the ckpt.
            if is_distributed:
                dist.barrier()

    # 6. Final save (rank 0 only — all ranks hold identical weights).
    final_dir = out_dir / "final"
    rc = 0
    if is_main:
        final_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(final_dir)
        preprocessor.save_pretrained(final_dir)
        postprocessor.save_pretrained(final_dir)
        log(f"[cotrain] final checkpoint → {final_dir}")

    # Sync, then tear the process group DOWN before the (possibly slow) HF
    # upload. If the barrier stayed live during the push, non-main ranks would
    # sit in a collective and trip the NCCL watchdog timeout while rank 0
    # uploads. After this point rank 0 runs alone with no live group.
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    # 7. HF push (final checkpoint dir) — rank 0 only, no live process group.
    if is_main and args.push_to_hub_repo:
        if not upload_checkpoint(final_dir, args.push_to_hub_repo, None):
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
