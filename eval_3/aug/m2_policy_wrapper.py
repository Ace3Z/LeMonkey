"""M2WrappedPolicy — drop-in `SmolVLAPolicy` that adds the M2 alignment loss.

Wraps a `SmolVLAPolicy` so that calling `wrapped.forward(batch)` returns
`(action_loss + λ * m2_loss, output_dict_with_m2_metrics)`. Everything else
(state_dict, named_parameters, eval, train, etc.) delegates to the inner
policy via `__getattr__`, so LeRobot's training script needs **no code
changes** — just call `policy = M2WrappedPolicy(policy, ...)` after
`make_policy(...)` and before `accelerator.prepare(...)`.

Construction:

    builder = M2SupervisionBuilder(
        face_labels_dir=..., manifest_path=..., aug_root=...,
        episode_mapping_path=...,
    )
    projector = FrozenProjector()
    policy = make_policy(cfg)
    policy = M2WrappedPolicy(policy, builder, projector,
                              capture_layer=9, lam=0.2)
    policy.apply_partial_freeze()  # call after wrapping, before train

What it does at every `forward(batch)`:
1. Call inner policy.forward(batch) → (action_loss, output_dict).
   The hook (registered in __init__) fires during this call.
2. Build M2 supervision tensors from batch["episode_index"] + batch["frame_index"].
3. Compute m2_align_loss(captured_hidden_state, masks, valid, targets, projector).
4. total_loss = action_loss + λ * m2_loss.
5. Append M2 metrics (m2_loss, m2_n_valid, m2_mean_cos) to output_dict.
6. Return (total_loss, output_dict).

The wrapper handles `reduction="none"` (RA-BC) by adding the M2 loss as a
scalar to every per-sample loss — M2 contributes equally across samples
since it's already mean-pooled inside m2_align_loss.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from m2_alignment import FrozenProjector, m2_align_loss
from m2_dataloader import M2SupervisionBuilder
from m2_smolvla_hook import (
    DEFAULT_CAPTURE_LAYER,
    apply_m2_partial_freeze,
    attach_m2_hook,
)


class M2WrappedPolicy(nn.Module):
    """Transparent wrapper that adds the M2 alignment loss."""

    # We're a thin wrapper, not a real nn.Module subclass that owns submodules.
    # All trainable params are in self.policy. The projector is frozen by
    # construction. We register the projector as a child for state_dict
    # completeness but it won't show in named_parameters() since all its
    # params have requires_grad=False (after FrozenProjector's __init__).
    def __init__(
        self,
        policy: nn.Module,
        builder: M2SupervisionBuilder,
        projector: FrozenProjector | None = None,
        capture_layer: int = DEFAULT_CAPTURE_LAYER,
        lam: float = 0.2,
        log_every: int = 100,
    ):
        super().__init__()
        self.policy = policy
        self.builder = builder
        self.projector = projector or FrozenProjector()
        self.capture_layer = capture_layer
        self.lam = lam
        self.log_every = log_every
        self._step = 0

        # Attach hook. Config inspection happens inside attach_m2_hook.
        self.hook = attach_m2_hook(policy, capture_layer=capture_layer)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def apply_partial_freeze(self, freeze_below: int | None = None) -> tuple[int, int]:
        """Freeze SmolLM2 layers below `freeze_below` (default = capture_layer).

        Must be called AFTER the inner policy's __init__ (which sets
        requires_grad via SmolVLMWithExpertModel.set_requires_grad). Returns
        (n_frozen_params, n_trainable_params) for sanity logging.
        """
        freeze_below = freeze_below if freeze_below is not None else self.capture_layer
        return apply_m2_partial_freeze(self.policy, freeze_below=freeze_below)

    # ------------------------------------------------------------------
    # Forward + supervision build
    # ------------------------------------------------------------------

    def _extract_indices(self, batch: dict) -> tuple[list[int], list[int]]:
        """Pull episode_index + frame_index per sample from a LeRobot batch.

        LeRobot's LeRobotDataset emits these as tensors of shape (B,) or
        (B, n_obs_steps). We take the LAST timestep when the dim has time.
        """
        for key_ep in ("episode_index", "episode_idx"):
            if key_ep in batch:
                ep = batch[key_ep]
                break
        else:
            raise KeyError(
                "M2WrappedPolicy: batch missing 'episode_index'. Keys: "
                f"{sorted(k for k in batch if isinstance(k, str))}"
            )
        for key_fr in ("frame_index", "frame_idx"):
            if key_fr in batch:
                fr = batch[key_fr]
                break
        else:
            raise KeyError(
                "M2WrappedPolicy: batch missing 'frame_index'. Keys: "
                f"{sorted(k for k in batch if isinstance(k, str))}"
            )

        # Handle (B,) or (B, T) — take last timestep.
        if ep.ndim == 2:
            ep = ep[:, -1]
        if fr.ndim == 2:
            fr = fr[:, -1]
        return ep.detach().cpu().long().tolist(), fr.detach().cpu().long().tolist()

    def forward(self, batch: dict, **kwargs) -> tuple[torch.Tensor, dict]:
        # 1. Run inner policy forward; hook captures layer-9 output during this.
        result = self.policy.forward(batch, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            action_loss, output = result
        else:
            raise TypeError(f"Inner policy.forward returned {type(result)}; "
                             "expected (loss, output_dict) tuple.")

        if self.hook.captured is None:
            print("[WARN] M2WrappedPolicy: hook did not fire during policy.forward "
                  "— expected captured tensor, got None, fallback=skip M2 this step",
                  flush=True)
            output["m2_loss"] = 0.0
            output["m2_n_valid"] = 0
            output["m2_mean_cos"] = float("nan")
            return action_loss, output

        captured = self.hook.captured
        # Free the reference so subsequent forwards don't accidentally reuse stale data.
        self.hook.captured = None

        # 2. Build M2 supervision from batch episode + frame indices.
        episode_indices, frame_idxs = self._extract_indices(batch)
        sup = self.builder.build_batch_from_episode_indices(
            episode_indices=episode_indices,
            frame_idxs=frame_idxs,
            device=captured.device,
        )

        # 3. M2 loss (fp32 internally; cast back to the action-loss dtype).
        m2 = m2_align_loss(
            hidden_state=captured,
            bbox_masks=sup["bbox_masks"],
            bbox_valid=sup["bbox_valid"],
            target_centroids=sup["target_centroids"],
            projector=self.projector,
        )

        # 4. Combine.
        m2_loss_typed = m2.loss.to(dtype=action_loss.dtype)
        total = action_loss + self.lam * m2_loss_typed

        # 5. Metrics for the loss_dict (only non-NaN cosines).
        with torch.no_grad():
            valid_cos = m2.per_slot_cos[~torch.isnan(m2.per_slot_cos)]
            mean_cos = float(valid_cos.mean().item()) if valid_cos.numel() > 0 else float("nan")
        output = dict(output)
        output["m2_loss"] = float(m2.loss.item())
        output["m2_n_valid"] = int(m2.n_valid)
        output["m2_mean_cos"] = mean_cos
        output["m2_n_base_samples"] = int(sup.get("n_base_samples", 0))

        if self.log_every > 0 and self._step % self.log_every == 0:
            print(f"[m2] step={self._step:>6d}  m2_loss={m2.loss.item():+.4f}  "
                  f"n_valid={m2.n_valid}/{3 * len(episode_indices)}  "
                  f"mean_cos={mean_cos:+.4f}  "
                  f"base={sup.get('n_base_samples', 0)}", flush=True)
        self._step += 1
        return total, output

    # ------------------------------------------------------------------
    # Delegation to the inner policy
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # nn.Module.__getattr__ is only called when normal attribute lookup
        # fails. Forward unknown attrs to the inner policy so LeRobot's
        # training script can call policy.config, policy.normalize_inputs,
        # policy.update, etc., without seeing a wrapper-induced AttributeError.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)
