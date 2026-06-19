"""
Adversarial Loss (LSGAN)
========================
Least-squares GAN objective for perceptual realism.  Uses a lightweight
70×70 PatchGAN discriminator conditioned on the raw (degraded) input.

Generator  loss: L_G = E[(D(x_enhanced, x_raw) − 1)²]
Discriminator:   L_D = E[(D(x_real, x_raw) − 1)²] + E[(D(x_fake, x_raw))²]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# PatchDiscriminator
# ---------------------------------------------------------------------------


class _ConvLNLeaky(nn.Module):
    """Conv → LayerNorm → LeakyReLU building block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 2,
        norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not norm
            )
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator conditioned on the degraded input.

    Accepts concatenated [enhanced/real, raw] → 6-channel input.

    Architecture (receptive field ≈ 70 pixels at 256×256 input):
        C64 → C128 → C256 → C512 → Conv(1)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        C = base_channels
        self.net = nn.Sequential(
            _ConvLNLeaky(in_channels * 2, C, stride=2, norm=False),  # no norm on first
            _ConvLNLeaky(C, C * 2, stride=2),
            _ConvLNLeaky(C * 2, C * 4, stride=2),
            _ConvLNLeaky(C * 4, C * 8, stride=1),
            nn.Conv2d(C * 8, 1, kernel_size=4, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:     (B, 3, H, W)  enhanced or reference image
            condition: (B, 3, H, W)  raw degraded image (conditioning signal)

        Returns:
            patch_logits: (B, 1, H', W')
        """
        x = torch.cat([image, condition], dim=1)  # (B, 6, H, W)
        return self.net(x)


# ---------------------------------------------------------------------------
# AdversarialLoss
# ---------------------------------------------------------------------------


class AdversarialLoss(nn.Module):
    """
    LSGAN adversarial loss for both generator and discriminator.

    Usage pattern during training:
        # Discriminator update
        d_loss = adv_loss.discriminator_loss(disc, x_real, x_fake, x_raw)
        d_loss.backward()

        # Generator update
        g_loss = adv_loss.generator_loss(disc, x_fake, x_raw)
        g_loss.backward()
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _lsgan(logits: torch.Tensor, target: float) -> torch.Tensor:
        """LSGAN: E[(D(x) − target)²]"""
        return F.mse_loss(logits, torch.full_like(logits, target))

    def generator_loss(
        self,
        discriminator: PatchDiscriminator,
        fake: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generator wants discriminator to output 1 for fakes.

        Args:
            discriminator: PatchDiscriminator instance (eval mode during G step)
            fake:          (B, 3, H, W)  enhanced image (detached noise not needed here)
            condition:     (B, 3, H, W)  raw degraded image

        Returns:
            Scalar generator adversarial loss.
        """
        logits_fake = discriminator(fake, condition)
        return self._lsgan(logits_fake, 1.0)

    def discriminator_loss(
        self,
        discriminator: PatchDiscriminator,
        real: torch.Tensor,
        fake: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator: real → 1, fake → 0.

        Args:
            real:      (B, 3, H, W)  reference (clean) image
            fake:      (B, 3, H, W)  enhanced image  ← should be .detach()ed
            condition: (B, 3, H, W)  raw degraded image

        Returns:
            Scalar discriminator loss.
        """
        logits_real = discriminator(real, condition)
        logits_fake = discriminator(fake.detach(), condition)
        return 0.5 * (self._lsgan(logits_real, 1.0) + self._lsgan(logits_fake, 0.0))

    def forward(
        self,
        discriminator: PatchDiscriminator,
        fake: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: generator loss only (use during composite loss computation)."""
        return self.generator_loss(discriminator, fake, condition)
