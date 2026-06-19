"""
Composite Loss
==============
Orchestrates all five P-UWDM loss components with phase-aware weighting.

Two training phases (aligned with the two-phase training schedule):

  Phase 1 (warm-up, epochs 1–30):
    Focus on diffusion + perceptual + histogram.
    Adversarial and contrastive are disabled (weight = 0) to stabilise
    early training before the discriminator is meaningful.

  Phase 2 (full training, epochs 31–100):
    All five components active.

Default weights (tunable via LossWeights):
    λ_diff   = 1.0
    λ_adv    = 0.01   (low — adversarial destabilises if too large)
    λ_perc   = 0.1
    λ_hist   = 0.05
    λ_con    = 0.05

Total:  L = λ_diff·L_diff + λ_adv·L_adv + λ_perc·L_perc + λ_hist·L_hist + λ_con·L_con
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
    """
    Scalar multipliers for each loss component.

    Set a weight to 0.0 to disable that component entirely (no forward pass).
    """

    diffusion: float = 1.0
    adversarial: float = 0.01
    perceptual: float = 0.1
    histogram: float = 0.05
    contrastive: float = 0.05

    @classmethod
    def phase1(cls) -> "LossWeights":
        """Phase-1 preset: diffusion + perceptual + histogram only."""
        return cls(
            diffusion=1.0,
            adversarial=0.0,
            perceptual=0.1,
            histogram=0.05,
            contrastive=0.0,
        )

    @classmethod
    def phase2(cls) -> "LossWeights":
        """Phase-2 preset: all five components active."""
        return cls(
            diffusion=1.0,
            adversarial=0.01,
            perceptual=0.1,
            histogram=0.05,
            contrastive=0.05,
        )


class CompositeLoss(nn.Module):
    """
    Five-component composite loss for P-UWDM.

    PerceptualLoss (VGG-16) is lazy-initialised on first use so that
    importing this class does not trigger a network download.  On your
    server the weights are already cached in ~/.cache/torch after the
    first run.

    Args:
        weights:          LossWeights instance; defaults to phase-2 weights.
        snr_gamma:        Min-SNR-γ clipping for diffusion loss (None = uniform).
        disc_base_ch:     Base channels for PatchDiscriminator (default 64).
        perc_layer_w:     Per-layer weights for VGG perceptual loss.
        hist_n_bins:      Histogram bins (default 256).
        hist_bandwidth:   Histogram soft-binning bandwidth (default 0.02).
        cont_temperature: NT-Xent temperature (default 0.07).
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

        self.weights = weights or LossWeights.phase2()
        self._perc_layer_w = perc_layer_w

        # Core sub-modules (always instantiated — no network access needed)
        self.diffusion_loss = DiffusionLoss(snr_gamma=snr_gamma)
        self.adversarial_loss = AdversarialLoss()
        self.discriminator = PatchDiscriminator(base_channels=disc_base_ch)
        self.histogram_loss = HistogramLoss(
            n_bins=hist_n_bins, bandwidth=hist_bandwidth
        )
        self.contrastive_loss = ContrastiveLoss(temperature=cont_temperature)

        # PerceptualLoss is lazy-initialised on first use (triggers VGG-16 download
        # on first call if weights not already cached in ~/.cache/torch/).
        self._perceptual_loss: PerceptualLoss | None = None

    # ------------------------------------------------------------------
    # Lazy perceptual loss property
    # ------------------------------------------------------------------

    @property
    def perceptual_loss(self) -> PerceptualLoss:
        """Initialise VGG-16 perceptual loss on first access."""
        if self._perceptual_loss is None:
            self._perceptual_loss = PerceptualLoss(layer_weights=self._perc_layer_w)
            # Move to same device as discriminator
            device = next(self.discriminator.parameters()).device
            self._perceptual_loss = self._perceptual_loss.to(device)
        return self._perceptual_loss

    # ------------------------------------------------------------------
    # Weight / phase management
    # ------------------------------------------------------------------

    def set_weights(self, weights: LossWeights) -> None:
        """Replace loss weights (e.g. when transitioning between phases)."""
        self.weights = weights

    def set_phase(self, phase: int) -> None:
        """
        Convenience method: set weights for phase 1 or 2.

        Args:
            phase: 1 → phase-1 preset (no adversarial/contrastive),
                   2 → phase-2 preset (all five components).
        """
        if phase == 1:
            self.weights = LossWeights.phase1()
        elif phase == 2:
            self.weights = LossWeights.phase2()
        else:
            raise ValueError(f"Unknown phase {phase}. Expected 1 or 2.")

    # ------------------------------------------------------------------
    # Generator / denoiser loss
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        # Diffusion inputs
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor | None = None,
        # Image inputs (denoised/enhanced vs references) — all in [0, 1]
        enhanced: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        raw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute composite generator loss.

        All image tensors must be in [0, 1].

        Required always:
            noise_pred, noise_target, timesteps

        Required for perceptual / histogram / adversarial / contrastive:
            enhanced, reference, raw

        Returns:
            dict with keys:
                'total', 'diffusion', 'adversarial',
                'perceptual', 'histogram', 'contrastive'
        """
        w = self.weights
        device = noise_pred.device
        zero = torch.tensor(0.0, device=device, dtype=noise_pred.dtype)

        losses: dict[str, torch.Tensor] = {}

        # 1. Diffusion loss (always active)
        losses["diffusion"] = self.diffusion_loss(
            noise_pred, noise_target, timesteps, alphas_cumprod
        )

        # 2. Adversarial (generator side)
        if w.adversarial > 0.0 and enhanced is not None and raw is not None:
            losses["adversarial"] = self.adversarial_loss(
                self.discriminator, enhanced, raw
            )
        else:
            losses["adversarial"] = zero

        # 3. Perceptual (lazy VGG init on first call)
        if w.perceptual > 0.0 and enhanced is not None and reference is not None:
            losses["perceptual"] = self.perceptual_loss(enhanced, reference)
        else:
            losses["perceptual"] = zero

        # 4. Histogram
        if w.histogram > 0.0 and enhanced is not None and reference is not None:
            losses["histogram"] = self.histogram_loss(enhanced, reference)
        else:
            losses["histogram"] = zero

        # 5. Contrastive
        if (
            w.contrastive > 0.0
            and enhanced is not None
            and reference is not None
            and raw is not None
        ):
            losses["contrastive"] = self.contrastive_loss(enhanced, reference, raw)
        else:
            losses["contrastive"] = zero

        # Weighted sum
        total = (
            w.diffusion * losses["diffusion"]
            + w.adversarial * losses["adversarial"]
            + w.perceptual * losses["perceptual"]
            + w.histogram * losses["histogram"]
            + w.contrastive * losses["contrastive"]
        )
        losses["total"] = total

        return losses

    # ------------------------------------------------------------------
    # Discriminator loss (separate update step)
    # ------------------------------------------------------------------

    def discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator update loss.

        Args:
            real:      (B, 3, H, W)  reference (clean) images in [0, 1]
            fake:      (B, 3, H, W)  enhanced images  — caller must .detach()
            condition: (B, 3, H, W)  raw degraded images in [0, 1]

        Returns:
            Scalar discriminator loss.
        """
        return self.adversarial_loss.discriminator_loss(
            self.discriminator, real, fake, condition
        )
