"""Adaptive Group Normalization (AdaGN) for diffusion conditioning.

AdaGN conditions feature maps on an external embedding vector (the combined
timestep + physics conditioning signal) by predicting per-channel scale and
shift parameters — identical in spirit to AdaIN / FiLM but applied after
GroupNorm.

Reference:
    Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS
    2021.  Equation (3).

Public API:
    AdaGN   — applies GroupNorm then modulates with scale/shift from cond.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdaGN(nn.Module):
    """Adaptive Group Normalization.

    Normalises ``x`` with GroupNorm, then applies affine modulation::

        y = scale * GroupNorm(x) + shift

    where ``scale`` and ``shift`` are predicted from a conditioning vector
    ``cond`` by a single linear layer.

    Args:
        num_channels: Number of channels in the feature map (C dimension).
        cond_dim:     Dimensionality of the conditioning embedding.
        num_groups:   Number of groups for GroupNorm.  Must divide
                      ``num_channels``.  Default: 32, falls back to
                      ``num_channels`` if C < 32.
        eps:          GroupNorm epsilon.
    """

    def __init__(
        self,
        num_channels: int,
        cond_dim: int,
        num_groups: int = 32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        # Ensure num_groups divides num_channels gracefully
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups //= 2

        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=False
        )

        # Projects cond → (scale, shift) pair, initialised to identity
        self.cond_proj = nn.Linear(cond_dim, num_channels * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x:    Feature map.  Shape: (B, C, H, W) *or* (B, L, C) for
                  sequence-shaped Swin features.
            cond: Conditioning vector.  Shape: (B, cond_dim).

        Returns:
            Modulated tensor with same shape as ``x``.
        """
        # Predict scale and shift from conditioning
        scale_shift = self.cond_proj(cond)  # (B, 2*C)

        # Handle both spatial (B,C,H,W) and sequence (B,L,C) layouts
        if x.ndim == 4:
            # Spatial layout
            B, C, H, W = x.shape
            scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, C)
            scale = scale[:, :, None, None]  # (B, C, 1, 1)
            shift = shift[:, :, None, None]
            return self.norm(x) * (1.0 + scale) + shift
        elif x.ndim == 3:
            # Sequence layout  (B, L, C)  — used inside Swin blocks
            B, L, C = x.shape
            # GroupNorm expects (B, C, *) so we reshape
            x_4d = x.permute(0, 2, 1).unsqueeze(-1)  # (B, C, L, 1)
            normed = self.norm(x_4d)  # (B, C, L, 1)
            normed = normed.squeeze(-1).permute(0, 2, 1)  # (B, L, C)

            scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, C)
            scale = scale.unsqueeze(1)  # (B, 1, C)
            shift = shift.unsqueeze(1)
            return normed * (1.0 + scale) + shift
        else:
            raise ValueError(f"AdaGN: expected 3-D or 4-D input, got {x.ndim}-D")
