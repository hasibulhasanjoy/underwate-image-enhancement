"""
Histogram Loss
==============
Differentiable per-channel colour distribution alignment via soft histograms.

Underwater images suffer from strong colour casts (blue/green dominance,
red attenuation). The histogram loss penalises distributional mismatch
between the enhanced image and the reference across all three RGB channels.

Soft histogram approach (Ace & Slesareva 2004 style):
    h_k = Σ_i  K_σ(p_i − b_k)         K_σ: Gaussian kernel, σ=bandwidth
    L_hist = (1/C) Σ_c  ‖CDF_enh_c − CDF_ref_c‖₁

Using the CDF (cumulative histogram) instead of the PDF makes gradients
smoother and avoids bin-boundary artefacts.

Args:
    n_bins:    Number of histogram bins (default 256).
    bandwidth: Gaussian kernel bandwidth for soft binning (default 0.02,
               roughly 5 intensity steps out of [0,1]).
    value_range: Expected pixel value range (default (0.0, 1.0)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistogramLoss(nn.Module):
    """
    Soft per-channel CDF matching loss.

    Inputs are expected in [0, 1].  If images are in a different range,
    set value_range accordingly.
    """

    def __init__(
        self,
        n_bins: int = 256,
        bandwidth: float = 0.02,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.bandwidth = bandwidth
        lo, hi = value_range

        # Fixed bin centres — register as buffer so .to(device) works
        bins = torch.linspace(lo, hi, n_bins)  # (n_bins,)
        self.register_buffer("bins", bins)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable soft histogram for a (B, H*W) tensor.

        Returns:
            hist: (B, n_bins)  — normalised to sum to 1 per sample.
        """
        # x: (B, N)  bins: (n_bins,)
        # Broadcast: (B, N, 1) vs (1, 1, n_bins) → (B, N, n_bins)
        diff = x.unsqueeze(-1) - self.bins.view(1, 1, -1)  # (B, N, n_bins)
        weights = torch.exp(-0.5 * (diff / self.bandwidth) ** 2)  # Gaussian
        hist = weights.sum(dim=1)  # (B, n_bins)
        hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-8)  # normalise
        return hist

    @staticmethod
    def _cdf(hist: torch.Tensor) -> torch.Tensor:
        """Cumulative distribution function: (B, n_bins) → (B, n_bins)."""
        return torch.cumsum(hist, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        enhanced: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            enhanced:  (B, 3, H, W) in [0, 1]
            reference: (B, 3, H, W) in [0, 1]

        Returns:
            Scalar histogram CDF-matching loss.
        """
        B, C, H, W = enhanced.shape
        assert C == 3, "Expected 3-channel (RGB) input."

        loss = torch.tensor(0.0, device=enhanced.device, dtype=enhanced.dtype)
        for c in range(C):
            enh_c = enhanced[:, c, :, :].reshape(B, -1)  # (B, H*W)
            ref_c = reference[:, c, :, :].reshape(B, -1)

            hist_enh = self._soft_histogram(enh_c)  # (B, n_bins)
            hist_ref = self._soft_histogram(ref_c)

            cdf_enh = self._cdf(hist_enh)  # (B, n_bins)
            cdf_ref = self._cdf(hist_ref)

            loss = loss + F.l1_loss(cdf_enh, cdf_ref)

        return loss / C
