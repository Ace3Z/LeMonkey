"""Residual-augmented SmolVLA wrapper.

Inference-only. Loads the frozen base SmolVLA + the trained residual head and
combines them per-frame:

    base_action = base.select_action(obs)
    image_feat  = base.vision_encoder(obs.image)         # mean-pooled
    residual    = residual_head(image_feat, state, base_action)
    final       = base_action + clip(residual, ±BOUND)

Per CLAUDE.md §5: log a [WARN] when the residual is clipped on >50% of joints
in a single step (a signal the residual is producing OOD outputs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_head import ResidualHead

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action


# Per-joint residual clip (degrees for arm, range_0_100 for gripper).
# Borrowed from Policy Decorator (Mu et al. 2024) — small bounds prevent
# runaway corrections on OOD inputs.
DEFAULT_CLIP = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 10.0], dtype=np.float32)


class ResidualWrapper:
    """Wraps a frozen base SmolVLA + trained residual head.

    Only exposes the inference path. Training uses train_residual.py.
    """
    def __init__(self,
                 base_policy_path: str,
                 residual_ckpt_path: str,
                 device: str = "cuda",
                 clip: np.ndarray = DEFAULT_CLIP):
        self.device = torch.device(device)
        # 1. Frozen base
        self.base = SmolVLAPolicy.from_pretrained(base_policy_path).eval().to(self.device)
        for p in self.base.parameters():
            p.requires_grad = False
        # 2. Trained residual head
        self.residual = ResidualHead.load(residual_ckpt_path).eval().to(self.device)
        for p in self.residual.parameters():
            p.requires_grad = False
        # 3. Preprocessor / postprocessor (re-uses base's saved processors)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.base.config,
            pretrained_path=base_policy_path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                "rename_observations_processor": {
                    "rename_map": {"observation.images.front": "observation.images.camera1"}
                },
            },
        )
        # 4. Clipping bounds
        self.clip = torch.from_numpy(clip).to(self.device)

        # Tracking for [WARN] logging
        self._n_steps = 0
        self._n_clipped_steps = 0

    def reset(self) -> None:
        """Clear base policy's internal action queue."""
        self.base.reset()
        self._n_steps = 0
        self._n_clipped_steps = 0

    @torch.inference_mode()
    def _extract_image_features(self, image_uint8_HWC_np: np.ndarray) -> torch.Tensor:
        """Match SmolVLA's resize_with_pad (modeling_smolvla.py:134-153) EXACTLY.
        Kept identical to train_residual.py's `_pad_to_512` to avoid train/inference drift."""
        img = torch.from_numpy(image_uint8_HWC_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img = img.to(self.device)
        B, C, H, W = img.shape
        target = 512
        ratio = max(W / target, H / target)
        new_h = int(H / ratio)
        new_w = int(W / ratio)
        img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h = max(0, target - new_h)
        pad_w = max(0, target - new_w)
        # SmolVLA pads RIGHT and BOTTOM only — not centered. (left, right, top, bottom).
        img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
        img = img * 2.0 - 1.0  # SigLIP normalization
        feats = self.base.model.vlm_with_expert.embed_image(img)  # (1, num_patches, 960)
        return feats.mean(dim=1)  # (1, 960)

    def select_action(self, image_uint8_HWC_np: np.ndarray, state_np: np.ndarray,
                      task_str: str) -> np.ndarray:
        """Compute the next action: base + clipped residual.

        CRITICAL: we call self.base.reset() before every predict_action so the
        base always returns the FIRST action of a fresh chunk. This matches
        what train_residual.py sees (which also resets per-frame).
        Without this reset, train sees chunk_idx=0 base actions, inference
        sees chunk_idx=0..49 base actions — silent generalization failure.
        Tradeoff: ~50ms/frame extra inference cost on 1660-class GPUs.
        """
        # 1. Reset base + run fresh chunk's first action (matches training)
        self.base.reset()
        obs = {
            "observation.images.front": image_uint8_HWC_np,
            "observation.state": state_np.astype(np.float32),
        }
        base_a_t = predict_action(
            obs, self.base, self.device, self.preprocessor, self.postprocessor,
            use_amp=False, task=task_str, robot_type="so101_follower",
        )
        base_a = base_a_t.detach().squeeze().to(self.device)  # (6,)

        # 2. Image features (mean-pooled patches from frozen vision encoder)
        img_feat = self._extract_image_features(image_uint8_HWC_np)  # (1, 960)

        # 3. Residual prediction
        state_t = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        base_t  = base_a.unsqueeze(0)
        with torch.inference_mode():
            r = self.residual(img_feat, state_t, base_t).squeeze(0)   # (6,)

        # 4. Clip per-joint
        clipped = torch.clamp(r, min=-self.clip, max=self.clip)
        n_clipped_joints = (clipped != r).sum().item()
        self._n_steps += 1
        if n_clipped_joints >= 4:  # >half of 6 joints clipped
            self._n_clipped_steps += 1
            if self._n_clipped_steps == 1 or self._n_clipped_steps % 10 == 0:
                print(f"[WARN] residual clipping triggered on {n_clipped_joints}/6 joints "
                      f"this step (total clipped steps so far: {self._n_clipped_steps}/{self._n_steps}). "
                      f"Raw residual abs-max: {r.abs().max().item():.2f}", flush=True)

        return (base_a + clipped).detach().cpu().numpy().astype(np.float32)

    def episode_summary(self) -> dict:
        return {
            "n_steps": self._n_steps,
            "n_clipped_steps": self._n_clipped_steps,
            "clip_pct": 100 * self._n_clipped_steps / max(self._n_steps, 1),
        }
