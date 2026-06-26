"""
src/losses/composite.py
────────────────────────────────────────────────────────────────────────────
Composite Loss — fixed version.

Root cause of original failure
───────────────────────────────
The original composite loss applied perceptual, histogram, adversarial, and
contrastive losses against `enhanced`, which is predict_x0_from_eps() at a
random training timestep t ~ Uniform(0,1000).  At high t (e.g. t=800),
x0_pred is dominated by noise, so these losses compared garbage vs reference
and produced massive gradients that overwhelmed the diffusion loss.  The
denoiser learned to minimise those gradients by outputting eps_pred ≈ 0
(noise collapse), which is why inference produced pure noise.

Fix
───
Two-phase training:

  Phase 1 (epochs 1–80): diffusion loss ONLY.
    The denoiser must learn to predict noise correctly before any image-level
    losses are applied.  eps_pred std should converge to ~1.0 by epoch 40-50.

  Phase 2 (epochs 81–100): diffusion + perceptual (low weight).
    Perceptual loss is only applied when the timestep is "low" (t < 200),
    i.e. when x0_pred is a meaningful near-clean estimate.  Adversarial,
    histogram, and contrastive are disabled — they provide marginal benefit
    for a dataset of ~890 images and can destabilise training.

Loss weights
────────────
  Phase 1:  λ_diff=1.0,  all others=0.0
  Phase 2:  λ_diff=1.0,  λ_perc=0.05,  all others=0.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .diffusion import DiffusionLoss
from .adversarial import AdversarialLoss, PatchDiscriminator
from .perceptual import PerceptualLoss
from .histogram import HistogramLoss
from .contrastive import ContrastiveLoss


@dataclass
class LossWeights:
    diffusion: float = 1.0
    adversarial: float = 0.0
    perceptual: float = 0.0
    histogram: float = 0.0
    contrastive: float = 0.0

    @classmethod
    def phase1(cls) -> "LossWeights":
        """Phase 1: diffusion loss only — let the denoiser learn noise prediction."""
        return cls(
            diffusion=1.0,
            adversarial=0.0,
            perceptual=0.0,
            histogram=0.0,
            contrastive=0.0,
        )

    @classmethod
    def phase2(cls) -> "LossWeights":
        """Phase 2: add light perceptual loss at low timesteps only."""
        return cls(
            diffusion=1.0,
            adversarial=0.0,
            perceptual=0.05,
            histogram=0.0,
            contrastive=0.0,
        )


# Timestep threshold below which x0_pred is meaningful for image-level losses.
# At t < LOW_T_THRESHOLD, ᾱ_t > 0.36, so x0_pred has reasonable signal.
LOW_T_THRESHOLD = 200


class CompositeLoss(nn.Module):
    """
    Composite loss for P-UWDM — fixed two-phase version.

    Args:
        weights:     LossWeights instance (default: phase1).
        snr_gamma:   Min-SNR-γ for diffusion loss (default 5.0).
        disc_base_ch: PatchDiscriminator base channels (kept for API compat).
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        snr_gamma: float | None = 5.0,
        disc_base_ch: int = 64,
        perc_layer_w: tuple[float, ...] = (1.0, 0.75, 0.5),
        hist_n_bins: int = 256,
        hist_bandwidth: float = 0.02,
        cont_temperature: float = 0.07,
    ) -> None:
        super().__init__()

        self.weights = weights or LossWeights.phase1()
        self._perc_layer_w = perc_layer_w

        # Always-present sub-modules
        self.diffusion_loss = DiffusionLoss(snr_gamma=snr_gamma)

        # Kept for API compatibility but not used in phase1/phase2 presets
        self.adversarial_loss = AdversarialLoss()
        self.discriminator = PatchDiscriminator(base_channels=disc_base_ch)
        self.histogram_loss = HistogramLoss(
            n_bins=hist_n_bins, bandwidth=hist_bandwidth
        )
        self.contrastive_loss = ContrastiveLoss(temperature=cont_temperature)

        # Lazy-init perceptual loss (triggers VGG download on first use)
        self._perceptual_loss: PerceptualLoss | None = None

    @property
    def perceptual_loss(self) -> PerceptualLoss:
        if self._perceptual_loss is None:
            self._perceptual_loss = PerceptualLoss(layer_weights=self._perc_layer_w)
            device = next(self.discriminator.parameters()).device
            self._perceptual_loss = self._perceptual_loss.to(device)
        return self._perceptual_loss

    def set_weights(self, weights: LossWeights) -> None:
        self.weights = weights

    def set_phase(self, phase: int) -> None:
        if phase == 1:
            self.weights = LossWeights.phase1()
        elif phase == 2:
            self.weights = LossWeights.phase2()
        else:
            raise ValueError(f"Unknown phase {phase}. Expected 1 or 2.")

    def forward(
        self,
        *,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor | None = None,
        enhanced: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        raw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute composite generator loss.

        Image-level losses (perceptual etc.) are only applied to samples
        where timestep t < LOW_T_THRESHOLD (200), ensuring x0_pred is a
        meaningful near-clean estimate rather than dominated by noise.
        """
        w = self.weights
        device = noise_pred.device
        zero = torch.tensor(0.0, device=device, dtype=noise_pred.dtype)

        losses: dict[str, torch.Tensor] = {}

        # 1. Diffusion loss — always active, all timesteps
        losses["diffusion"] = self.diffusion_loss(
            noise_pred, noise_target, timesteps, alphas_cumprod
        )

        # 2–5. Image-level losses — only where x0_pred is meaningful
        # Find batch indices where t < LOW_T_THRESHOLD
        low_t_mask = timesteps < LOW_T_THRESHOLD  # (B,)
        has_low_t = low_t_mask.any()

        # Adversarial (disabled in both phases — kept for API compat)
        losses["adversarial"] = zero

        # Perceptual
        if (
            w.perceptual > 0.0
            and has_low_t
            and enhanced is not None
            and reference is not None
        ):
            enh_low = enhanced[low_t_mask]
            ref_low = reference[low_t_mask]
            losses["perceptual"] = self.perceptual_loss(enh_low, ref_low)
        else:
            losses["perceptual"] = zero

        # Histogram (disabled — kept for API compat)
        losses["histogram"] = zero

        # Contrastive (disabled — kept for API compat)
        losses["contrastive"] = zero

        # Weighted sum
        total = w.diffusion * losses["diffusion"] + w.perceptual * losses["perceptual"]
        losses["total"] = total
        return losses

    def discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Kept for API compatibility — not used in fixed training."""
        return self.adversarial_loss.discriminator_loss(
            self.discriminator, real, fake, condition
        )
