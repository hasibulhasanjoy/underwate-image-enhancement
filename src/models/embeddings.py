"""
PATCH for src/models/embeddings.py
===================================
Only ConditioningProjection needs to change for Phase 3.
Replace your existing ConditioningProjection class with the one below.
Everything else in embeddings.py (SinusoidalTimestepEmbedding, TimestepMLP)
is unchanged.

What changed and why
---------------------
OLD signature:
    ConditioningProjection(out_dim)
    forward(ambient (B,3), transmission (B,1,H,W), degradation (B,6), severity (B,1))

    Internally it projected:
        ambient      (B,3)       → Linear(3,   64)  → (B, 64)
        trans mean+std (B,2)     → Linear(2,   64)  → (B, 64)
        degradation  (B,6)       → Linear(6,   64)  → (B, 64)
        severity     (B,1)       → Linear(1,   64)  → (B, 64)
        concat → (B, 256) → MLP → (B, out_dim)

NEW signature:
    ConditioningProjection(out_dim, cond_embed_dim=128)
    forward(a_embedding (B,128), t_embedding (B,128), degradation (B,6), severity (B,1))

    Projects:
        a_embedding  (B, 128)    → Linear(128, 128) → (B, 128)
        t_embedding  (B, 128)    → Linear(128, 128) → (B, 128)
        degradation  (B, 6)      → Linear(6,   64)  → (B, 64)   unchanged
        severity     (B, 1)      → Linear(1,   64)  → (B, 64)   unchanged
        concat → (B, 384) → MLP → (B, out_dim)

The total out_dim (512) and the downstream cond_fuse MLP are unchanged.
The degradation and severity branches are unchanged.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ConditioningProjection(nn.Module):
    """Projects physics conditioning signals into a unified vector.

    Accepts learned embeddings from A-Net and T-Net (Phase 3+), plus raw
    degradation features and severity scalar which have no dedicated network.

    Parameters
    ----------
    out_dim : int
        Output dimension.  Must match SwinUNetConfig.cond_dim (default 512).
    cond_embed_dim : int
        Embedding dimension produced by A-Net and T-Net.
        Must match ANet(embed_dim=X) / TNet(embed_dim=X).  Default 128.
    """

    def __init__(self, out_dim: int = 512, cond_embed_dim: int = 128) -> None:
        super().__init__()
        self.cond_embed_dim = cond_embed_dim

        # A-Net embedding branch
        self.ambient_proj = nn.Sequential(
            nn.Linear(cond_embed_dim, cond_embed_dim),
            nn.SiLU(),
        )

        # T-Net embedding branch
        self.trans_proj = nn.Sequential(
            nn.Linear(cond_embed_dim, cond_embed_dim),
            nn.SiLU(),
        )

        # Degradation feature branch (unchanged)
        self.deg_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
        )

        # Severity branch (unchanged)
        self.sev_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
        )

        # Fusion MLP: 128 + 128 + 64 + 64 = 384 → out_dim
        fused_dim = cond_embed_dim * 2 + 64 + 64  # 384 at defaults
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        a_embedding: Tensor,  # (B, cond_embed_dim)  from A-Net
        t_embedding: Tensor,  # (B, cond_embed_dim)  from T-Net
        degradation: Tensor,  # (B, 6)
        severity: Tensor,  # (B, 1)
    ) -> Tensor:
        """
        Returns
        -------
        (B, out_dim) unified physics conditioning vector.
        """
        a = self.ambient_proj(a_embedding)  # (B, 128)
        t = self.trans_proj(t_embedding)  # (B, 128)
        deg = self.deg_proj(degradation)  # (B, 64)
        sev = self.sev_proj(severity)  # (B, 64)

        fused = torch.cat([a, t, deg, sev], dim=-1)  # (B, 384)
        return self.fusion(fused)  # (B, out_dim)
