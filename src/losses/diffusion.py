"""
Diffusion Loss
==============
Core DDPM/DDIM denoising objective. Predicts noise ε added at timestep t
and computes a SNR-weighted MSE between predicted and actual noise.

  L_diff = E_{t,ε}[ w(t) · ‖ε_θ(x_t, t, c) − ε‖² ]

where w(t) = SNR(t) / (SNR(t) + 1)  (min-SNR-gamma weighting, γ=5)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    SNR-weighted noise-prediction MSE.

    Args:
        snr_gamma: Min-SNR clipping gamma. Set to None to use uniform weights.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, snr_gamma: float | None = 5.0, reduction: str = "mean") -> None:
        super().__init__()
        self.snr_gamma = snr_gamma
        self.reduction = reduction

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snr_weights(
        alphas_cumprod: torch.Tensor,
        timesteps: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Compute per-sample Min-SNR-γ loss weights.

        SNR(t) = ᾱ_t / (1 − ᾱ_t)
        weight  = min(SNR(t), γ) / SNR(t)
        """
        acp = alphas_cumprod[timesteps]  # (B,)
        snr = acp / (1.0 - acp).clamp(min=1e-8)  # (B,)
        weights = torch.clamp(snr, max=gamma) / snr  # (B,)
        return weights  # (B,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            noise_pred:      (B, C, H, W)  model noise prediction ε_θ
            noise_target:    (B, C, H, W)  ground-truth noise ε
            timesteps:       (B,)           integer diffusion timesteps
            alphas_cumprod:  (T,)           scheduler ᾱ values; if None → uniform weights

        Returns:
            Scalar loss tensor.
        """
        assert (
            noise_pred.shape == noise_target.shape
        ), f"Shape mismatch: pred {noise_pred.shape} vs target {noise_target.shape}"

        # Element-wise squared error: (B, C, H, W)
        mse = F.mse_loss(noise_pred, noise_target, reduction="none")

        # Reduce spatial+channel dims → (B,)
        mse_per_sample = mse.mean(dim=(1, 2, 3))

        if alphas_cumprod is not None and self.snr_gamma is not None:
            weights = self._snr_weights(alphas_cumprod, timesteps, self.snr_gamma)
            mse_per_sample = mse_per_sample * weights

        if self.reduction == "mean":
            return mse_per_sample.mean()
        elif self.reduction == "sum":
            return mse_per_sample.sum()
        else:
            return mse_per_sample
