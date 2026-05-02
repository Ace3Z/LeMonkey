"""ResidualHead — small MLP that learns a per-step joint correction on top of
a frozen base SmolVLA policy.

Inputs:
    image_features: (B, 960)   mean-pooled SmolVLM2 vision-encoder features
    state:          (B, 6)     current joint positions of the SO-101 follower
    base_action:    (B, 6)     SmolVLA's predicted next joint target

Output:
    residual:       (B, 6)     joint-position delta added to base_action

Per-joint clipping (Policy Decorator, Mu et al. 2024) bounds the residual to
prevent OOD blow-ups. This module *does not* clip — clipping is applied at
inference time by the wrapper, which can also log when clipping triggers.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


IMAGE_DIM   = 960  # SmolVLM2 connector output dim (verified empirically)
STATE_DIM   = 6
ACTION_DIM  = 6


class ResidualHead(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        in_dim = IMAGE_DIM + STATE_DIM + ACTION_DIM   # 972
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
            last = hidden
        layers += [nn.Linear(last, ACTION_DIM)]
        # Initialize the final layer to zeros so the residual starts as the
        # identity (zero correction). This gives the wrapper a safe init that
        # exactly matches the base policy until training kicks in.
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)
        self.config = {"hidden": hidden, "depth": depth, "dropout": dropout}

    def forward(self, image_features: torch.Tensor,
                state: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_features, state, base_action], dim=-1)
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(path / "residual.safetensors"))
        import json
        (path / "config.json").write_text(json.dumps(self.config, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ResidualHead":
        path = Path(path)
        import json
        cfg = json.loads((path / "config.json").read_text())
        m = cls(**cfg)
        m.load_state_dict(load_file(str(path / "residual.safetensors")))
        return m
