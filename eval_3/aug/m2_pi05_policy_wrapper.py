"""M2Pi05WrappedPolicy — drop-in wrapper for `PI05Policy` that adds the
M2 ArcFace alignment loss AND (optionally) the KLAL attention-supervision
loss the SmolVLA attention probe says we need.

Mirrors `eval_3/aug/m2_policy_wrapper.py` but adapted for PI05Policy:
- different hook target (PaliGemma text LM instead of SmolLM2)
- 16x16 = 256 patch grid (instead of 8x8 = 64)
- 2048-dim hidden state (instead of 960)
- center-padded image preprocessing
- prefix layout `[img × n_cams, lang_tokens]` (no state token in prefix)

`forward(batch)` returns `(total_loss, output_dict_with_m2_and_klal_metrics)`.
Everything else delegates to the inner PI05Policy via `__getattr__` so the
HF push is clean (verified by the same `save_pretrained` audit we did for
SmolVLA — only the inner PI05Policy weights hit the safetensors file).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from m2_alignment import FrozenProjector, m2_align_loss
from m2_dataloader import M2SupervisionBuilder
from m2_pi05_geometry import (
    NUM_PI05_PATCHES,
    PI05_PATCH_GRID,
    bbox_to_patch_mask_pi05,
    resize_with_pad_box_pi05,
)
from m2_pi05_hook import (
    DEFAULT_PI05_CAPTURE_LAYER,
    apply_m2_pi05_partial_freeze,
    attach_m2_pi05_hook,
)
from m2_klal import KLALConfig, KLALHookSet, klal_loss


class M2Pi05WrappedPolicy(nn.Module):
    def __init__(
        self,
        policy: nn.Module,
        builder: M2SupervisionBuilder,
        projector: FrozenProjector | None = None,
        capture_layer: int = DEFAULT_PI05_CAPTURE_LAYER,
        lam_m2: float = 0.2,
        klal_cfg: KLALConfig | None = None,
        log_every: int = 100,
    ):
        super().__init__()
        self.policy = policy
        self.builder = builder
        # PaliGemma hidden dim is 2048 (gemma_2b). Our SmolVLA projector default
        # is in_dim=960; we instantiate explicitly here.
        self.projector = projector or FrozenProjector(in_dim=2048)
        self.capture_layer = capture_layer
        self.lam_m2 = lam_m2
        self.klal_cfg = klal_cfg
        self.log_every = log_every
        self._step = 0

        # 1. M2 hook on layer (capture_layer + 1).input_layernorm.
        self.hook = attach_m2_pi05_hook(policy, capture_layer=capture_layer)

        # 2. KLAL multi-layer Q/K hookset (lazily built on first forward
        #    so we can read head counts from the loaded model).
        self._klal_hookset: KLALHookSet | None = None

    # ─── public API ───────────────────────────────────────────────────

    def apply_partial_freeze(self, freeze_below: int | None = None) -> tuple[int, int]:
        freeze_below = freeze_below if freeze_below is not None else self.capture_layer
        return apply_m2_pi05_partial_freeze(self.policy, freeze_below=freeze_below)

    # ─── forward ─────────────────────────────────────────────────────

    def _extract_indices(self, batch: dict) -> tuple[list[int], list[int]]:
        """Same logic as the SmolVLA wrapper — LeRobot v3 batches don't
        carry per-episode frame_index, only global `index`. Derive it
        from a lazy `episode_starts` lookup.
        """
        ep = batch["episode_index"]
        if ep.ndim == 2:
            ep = ep[:, -1]
        ep_list = ep.detach().cpu().long().tolist()

        if "frame_index" in batch:
            fr = batch["frame_index"]
            if fr.ndim == 2:
                fr = fr[:, -1]
            return ep_list, fr.detach().cpu().long().tolist()

        # Fallback: index - episode_start.
        if not hasattr(self, "_episode_starts"):
            self._build_episode_starts()
        idx = batch["index"]
        if idx.ndim == 2:
            idx = idx[:, -1]
        idx_list = idx.detach().cpu().long().tolist()
        fr_list = [g - self._episode_starts[e] for e, g in zip(ep_list, idx_list)]
        return ep_list, fr_list

    def _build_episode_starts(self):
        import os
        import datasets as _ds
        repo = os.environ.get("M2_DATASET_REPO_ID")
        if not repo:
            raise RuntimeError("M2_DATASET_REPO_ID env var required")
        root = Path(os.environ.get("HF_LEROBOT_HOME")
                    or Path.home() / ".cache/huggingface/lerobot") / repo
        files = sorted(str(p) for p in (root / "meta/episodes").rglob("*.parquet"))
        ds = _ds.load_dataset("parquet", data_files=files, split="train")
        self._episode_starts = {int(r["episode_index"]): int(r["dataset_from_index"])
                                 for r in ds}
        print(f"[m2 pi05 wrapper] built episode_starts ({len(self._episode_starts)} entries)",
              flush=True)

    def _ensure_klal_hookset(self):
        if self._klal_hookset is not None or self.klal_cfg is None:
            return
        from m2_pi05_hook import _resolve_text_model
        text_model = _resolve_text_model(self.policy)
        attn0 = text_model.layers[0].self_attn
        n_heads = attn0.config.num_attention_heads
        n_kv_heads = attn0.config.num_key_value_heads
        head_dim = attn0.head_dim
        self._klal_hookset = KLALHookSet(text_model, self.klal_cfg.capture_layers,
                                         n_heads, n_kv_heads, head_dim)
        print(f"[m2 pi05 wrapper] attached KLAL hookset on layers "
              f"{list(self.klal_cfg.capture_layers)}", flush=True)

    def forward(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        if self.klal_cfg is not None:
            self._ensure_klal_hookset()
            if self._klal_hookset is not None:
                self._klal_hookset.reset()

        # 1. Inner forward → action_loss + the hook fires.
        result = self.policy.forward(batch, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            action_loss, output = result
        else:
            raise TypeError(f"inner forward returned {type(result)}; expected tuple")

        # 2. M2 loss (mean_cos at face patches → ArcFace centroids).
        captured = self.hook.captured
        self.hook.captured = None
        if captured is None:
            print("[WARN] M2Pi05Wrapped: hook did not fire; skipping M2 + KLAL this step",
                  flush=True)
            output = dict(output)
            output["m2_loss"] = 0.0
            output["m2_n_valid"] = 0
            output["m2_mean_cos"] = float("nan")
            output["klal_loss"] = 0.0
            return action_loss, output

        ep_idxs, frame_idxs = self._extract_indices(batch)
        sup = self.builder.build_batch_from_episode_indices(
            episode_indices=ep_idxs,
            frame_idxs=frame_idxs,
            device=captured.device,
            patch_grid=PI05_PATCH_GRID,
            resize_with_pad_box=resize_with_pad_box_pi05,
            bbox_to_patch_mask=bbox_to_patch_mask_pi05,
        )

        # captured has shape (B, prefix_len, 2048). The first n_cams × 256
        # positions are image-patch hidden states. camera1 is first.
        # We slice the camera1 patches and pass to m2_align_loss as the
        # "hidden_state" — same interface as SmolVLA.
        cam1 = captured[:, :NUM_PI05_PATCHES, :]  # (B, 256, 2048)
        m2 = m2_align_loss(
            hidden_state=cam1,
            bbox_masks=sup["bbox_masks"],
            bbox_valid=sup["bbox_valid"],
            target_centroids=sup["target_centroids"],
            projector=self.projector,
        )

        # 3. KLAL loss (force name-token → face-patch attention).
        klal_v = torch.tensor(0.0, device=captured.device)
        if self.klal_cfg is not None and self._klal_hookset is not None:
            # Build target masks: union of all 3 slot masks per sample = the
            # full face region in image space (centroid will be on the
            # prompted celeb's face since builder put it at the correct slot).
            B = sup["bbox_masks"].shape[0]
            # Per-sample target patch mask: the slot for the PROMPTED celeb.
            # For simplicity, use slot 1 (middle) unless builder marks a
            # specific "target_slot" key. The builder already routes by
            # target_celeb_short → so sup["target_slot_idx"] (if present)
            # tells us which slot index to pick.
            tgt_slot = sup.get("target_slot_idx", None)
            if tgt_slot is None:
                # Fall back: union across slots (less precise — but better
                # than no signal).
                target_masks_2d = sup["bbox_masks"].any(dim=1)  # (B, P)
            else:
                target_masks_2d = torch.stack(
                    [sup["bbox_masks"][b, tgt_slot[b]] for b in range(B)], dim=0
                )

            # Name-token positions in the prefix per sample. Builder must
            # provide these (offsetted to skip image-patch positions + any
            # left-pad on language tokens). If absent, KLAL is a no-op.
            name_pos = sup.get("name_token_positions", None)
            if name_pos is not None:
                klal_v = klal_loss(
                    self._klal_hookset,
                    image_patch_slice=slice(0, NUM_PI05_PATCHES),
                    name_token_positions=name_pos.to(captured.device),
                    target_masks=target_masks_2d.to(captured.device),
                    cfg=self.klal_cfg,
                )

        # 4. Combine. action_loss dtype is bf16 typically; cast aux losses.
        m2_typed = m2.loss.to(dtype=action_loss.dtype)
        klal_typed = klal_v.to(dtype=action_loss.dtype)
        total = action_loss + self.lam_m2 * m2_typed + klal_typed

        with torch.no_grad():
            valid_cos = m2.per_slot_cos[~torch.isnan(m2.per_slot_cos)]
            mean_cos = float(valid_cos.mean().item()) if valid_cos.numel() > 0 else float("nan")

        output = dict(output)
        output["m2_loss"] = float(m2.loss.item())
        output["m2_n_valid"] = int(m2.n_valid)
        output["m2_mean_cos"] = mean_cos
        output["klal_loss"] = float(klal_v.item()) if torch.is_tensor(klal_v) else float(klal_v)

        if self.log_every > 0 and self._step % self.log_every == 0:
            print(f"[m2-pi05] step={self._step:>6d}  m2={m2.loss.item():+.4f}  "
                  f"klal={float(klal_v):+.4f}  "
                  f"mean_cos={mean_cos:+.4f}  n_valid={m2.n_valid}/{3*len(ep_idxs)}",
                  flush=True)
        self._step += 1
        return total, output

    # ─── delegate everything else to the inner policy ──────────────

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)
