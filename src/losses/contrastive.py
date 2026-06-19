"""
Contrastive Loss (NT-Xent)
==========================
Image-level contrastive loss for underwater image enhancement.

Intuition:
  - POSITIVE pair : (enhanced, reference)  → should be close in feature space
  - NEGATIVE pairs: (enhanced, raw)        → should be far apart

This encourages the model to move enhanced images towards the clean
reference manifold and away from the degraded input manifold.

Implementation: NT-Xent (Normalized Temperature-scaled Cross-Entropy)
over a projection head embedding, applied at the image (not patch) level.

    z_e = proj(pool(enhanced))    ∈ ℝ^d
    z_r = proj(pool(reference))   ∈ ℝ^d
    z_x = proj(pool(raw))         ∈ ℝ^d

    sim(a,b)  = a·b / (‖a‖·‖b‖·τ)

    L_con = −log[ exp(sim(z_e, z_r)) / (exp(sim(z_e, z_r)) + Σ_j exp(sim(z_e, z_xj))) ]

The projection head is a 2-layer MLP: GAP → Linear(C,256) → ReLU → Linear(256,128) → L2-norm.

Args:
    temperature:    NT-Xent temperature τ (default 0.07).
    proj_in_channels: Input channels to the projection head (default 3,
                      operates directly on pixel space via GAP).
    proj_hidden_dim: Hidden dim of projection MLP (default 256).
    proj_out_dim:   Output dim after L2-norm (default 128).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProjectionHead(nn.Module):
    """
    2-layer MLP projection head: GlobalAvgPool → Linear → ReLU → Linear → L2-norm.
    """

    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            z: (B, out_dim)  L2-normalized embedding
        """
        h = self.pool(x).flatten(1)  # (B, C)
        z = self.mlp(h)  # (B, out_dim)
        return F.normalize(z, dim=1)  # L2-norm


class ContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss with a learnable projection head.

    The projection head is trained jointly with the rest of the network
    and discarded at inference time.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        proj_in_channels: int = 3,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.proj = _ProjectionHead(proj_in_channels, proj_hidden_dim, proj_out_dim)

    def forward(
        self,
        enhanced: torch.Tensor,
        reference: torch.Tensor,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            enhanced:  (B, 3, H, W)  model output (denoised)
            reference: (B, 3, H, W)  clean reference image
            raw:       (B, 3, H, W)  original degraded image

        Returns:
            Scalar NT-Xent contrastive loss.
        """
        z_e = self.proj(enhanced)  # (B, D)  — anchor
        z_r = self.proj(reference)  # (B, D)  — positive
        z_x = self.proj(raw)  # (B, D)  — negative

        B = z_e.size(0)

        # Cosine similarities scaled by temperature
        sim_pos = (z_e * z_r).sum(dim=1) / self.temperature  # (B,)
        sim_neg = (z_e * z_x).sum(dim=1) / self.temperature  # (B,)

        # For each anchor, also treat other samples in the batch as negatives
        # Compute full (B, B) similarity matrix between enhanced and reference
        sim_matrix = torch.mm(z_e, z_r.T) / self.temperature  # (B, B)

        # Diagonal = positive pairs; off-diagonal = in-batch negatives
        # Include raw negatives in the denominator
        sim_neg_full = torch.mm(z_e, z_x.T) / self.temperature  # (B, B)

        # Numerator: positive similarity (B,)
        numerator = torch.exp(sim_matrix.diag())

        # Denominator: sum of all similarities (positives + all negatives)
        # Remove self from in-batch raw negatives to avoid double-counting
        denom = (
            torch.exp(sim_matrix).sum(dim=1)  # all ref in batch
            + torch.exp(sim_neg_full).sum(dim=1)  # all raw in batch
            - torch.exp(sim_matrix.diag())  # subtract positive from denom
        )

        loss = -torch.log(numerator / denom.clamp(min=1e-8))
        return loss.mean()
