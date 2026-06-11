"""Embeddings for the SwinUNet denoiser.

Provides:
  - SinusoidalTimestepEmbedding : classic DDPM sinusoidal positional encoding
  - TimestepMLP                 : two-layer MLP that projects the sinusoidal
                                   embedding to a conditioning vector
  - ConditioningProjection      : projects physics conditioning signals
                                   (ambient, transmission stats, degradation)
                                   to the same conditioning dimension
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    """Deterministic sinusoidal embedding for diffusion timesteps.

    Follows Ho et al. (DDPM, 2020).  For a timestep t the embedding vector
    has alternating sin / cos channels at geometrically spaced frequencies.

    Args:
        dim: Output dimensionality.  Must be even.
        max_period: Controls the minimum frequency (default 10 000).
    """

    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) integer or float timestep indices.
        Returns:
            emb: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )  # (half,)
        args = t[:, None].float() * freqs[None]  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        return emb


class TimestepMLP(nn.Module):
    """Two-layer MLP that refines a sinusoidal timestep embedding.

    The output is used as the primary conditioning signal for AdaGN layers
    throughout the denoising backbone.

    Args:
        sinusoidal_dim: Dimensionality of the raw sinusoidal embedding.
        hidden_dim:     Width of the hidden layer (default 4× sinusoidal_dim).
        out_dim:        Output conditioning dimension.
    """

    def __init__(
        self,
        sinusoidal_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or sinusoidal_dim * 4
        self.net = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, sinusoidal_emb: Tensor) -> Tensor:
        """
        Args:
            sinusoidal_emb: (B, sinusoidal_dim)
        Returns:
            cond: (B, out_dim)
        """
        return self.net(sinusoidal_emb)


# ---------------------------------------------------------------------------
# Physics conditioning projection
# ---------------------------------------------------------------------------


class ConditioningProjection(nn.Module):
    """Fuses all physics prior signals into one conditioning vector.

    Input signals (all per-batch):
      - ambient      : (B, 3)       – global ambient light estimate
      - t_stats      : (B, 2)       – [mean, std] of per-pixel transmission map
      - degradation  : (B, 6)       – 6-D degradation feature vector
      - severity     : (B, 1)       – scalar severity score

    The four signals are concatenated → 12-D → projected to ``out_dim``.

    Args:
        out_dim: Projection output dimension; should equal the main conditioning
                 dimension so it can be summed with the timestep embedding.
    """

    _IN_DIM: int = 12  # 3 + 2 + 6 + 1

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self._IN_DIM, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        ambient: Tensor,  # (B, 3)
        transmission: Tensor,  # (B, 1, H, W)  → spatially pooled inside
        degradation: Tensor,  # (B, 6)
        severity: Tensor,  # (B, 1)
    ) -> Tensor:
        """
        Returns:
            cond: (B, out_dim)
        """
        # Reduce spatial transmission map to [mean, std] statistics
        t_flat = transmission.flatten(2)  # (B, 1, H*W)
        t_mean = t_flat.mean(dim=-1)  # (B, 1)
        t_std = t_flat.std(dim=-1).clamp(min=1e-6)  # (B, 1)
        t_stats = torch.cat([t_mean, t_std], dim=-1)  # (B, 2)

        physics = torch.cat(
            [ambient, t_stats, degradation, severity], dim=-1
        )  # (B, 12)
        return self.proj(physics)  # (B, out_dim)
