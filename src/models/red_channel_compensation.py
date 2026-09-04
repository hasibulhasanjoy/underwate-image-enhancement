"""
src/models/red_channel_compensation.py
────────────────────────────────────────────────────────────────────────────
Red Channel Compensation (RCC) — physics-guided restoration of the most
attenuated colour channel in underwater imagery.

Architecture context
─────────────────────
Per the P-UWDM system diagram (Fig. 1, thesis proposal):

    Decoder → Red Channel Compensation → DDIM Sampling → Enhanced Output

RCC sits between the SwinUNet decoder's per-step clean-image estimate
(x̂_0 — the denoiser's implicit prediction of the noise-free image at the
current DDIM step) and the DDIM update rule. It is applied at every reverse
step, so the correction compounds as the sample is progressively denoised,
and it is also the final operation on the last step (t_prev = 0), where
DDIM returns x̂_0 directly as the "Enhanced Output".

Physics background
───────────────────
Underwater light attenuates wavelength-selectively: red (~620–750 nm) is
absorbed within the first few metres, long before green or blue. This is
the dominant cause of the blue/green colour cast in raw underwater images,
and it is exactly why src/physics/transmission.py deliberately builds its
dark channel from the RED channel alone rather than the standard
min-over-channels haze prior (see TransmissionConfig.use_red_channel). That
red-channel transmission estimate — refined into a learned spatial map by
T-Net (`refined_map`) — is precisely t_r(x) in the classical underwater
image formation model (Jaffe-McGlamery / Beer-Lambert scattering; see also
Chiang & Chen, "Underwater Image Enhancement by Wavelength Compensation and
Dehazing", 2012, and Galdran et al., "Automatic Red-Channel Underwater
Image Restoration", 2015):

    I_r(x) = J_r(x) · t_r(x) + A_r · (1 − t_r(x))

Inverting for the scene radiance J_r — the lost red signal we want back:

    J_r_phys(x) = (I_r(x) − A_r · (1 − t_r(x))) / max(t_r(x), t_min)

This closed-form inversion is exact under the physical model but numerically
unstable wherever t_r(x) → 0 (division blow-up, noise amplification): the
deeper / hazier a pixel, the less trustworthy the physics estimate. RCC
therefore never overwrites the network's own red-channel prediction
outright. Instead it learns *where* to trust the physics prior:

    1. Compute J_r_phys(x) from (raw, ambient A, transmission t_r) — a
       closed-form physics computation, no learnable parameters.
    2. Compare it against the denoiser's own predicted red channel R_pred.
    3. A small spatial gating CNN, conditioned on the raw image, R_pred,
       J_r_phys, t_r(x), and their pointwise disagreement, predicts a
       per-pixel trust mask α(x) ∈ [0, 1].
    4. Blend:  R_out(x) = α(x) · J_r_phys(x) + (1 − α(x)) · R_pred(x)

Green and blue channels pass through the denoiser's prediction unchanged —
consistent with the module's name and its placement in the architecture,
and with the physical observation that red is the channel that actually
needs closed-form restoration underwater; G/B degrade far more gently and
are already handled by the diffusion denoiser + AdaGN conditioning.

This mirrors the "physics prior as a learnable correction, not a
replacement" design principle already used by A-Net / T-Net
(see conditioning.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _make_norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with automatic group-count fallback for small channel counts."""
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RedChannelCompensationConfig:
    """
    Hyper-parameters for the Red Channel Compensation module.

    Attributes
    ----------
    base_ch : int
        Channel width of the gating CNN. Default 16 — this is a
        deliberately lightweight module (a few thousand parameters); it
        gates one channel, it does not re-decode the whole image.
    t_min : float
        Minimum transmission clamp used before physics inversion, to avoid
        division blow-up. Matches
        src.physics.transmission.TransmissionConfig.t_min (default 0.10).
    num_gate_layers : int
        Depth of the gating CNN body (default 2 conv blocks + a 1x1 head).
    init_gate_bias : float
        Bias added before the sigmoid on the final gate layer. A negative
        value biases the module toward trusting the network's own
        prediction (α ≈ 0) at initialisation, so RCC starts as a
        near-identity operation and only learns to lean on the physics
        prior where the training signal supports it.
        Default −2.0 (sigmoid(−2.0) ≈ 0.12).
    """

    base_ch: int = 16
    t_min: float = 0.10
    num_gate_layers: int = 2
    init_gate_bias: float = -2.0


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────


class RedChannelCompensation(nn.Module):
    """
    Physics-guided red-channel restoration module.

    Parameters
    ----------
    cfg : RedChannelCompensationConfig, optional

    Usage
    -----
        rcc = RedChannelCompensation()
        out, alpha = rcc(
            pred=x0_pred,              # (B, 3, H, W) denoiser's clean-image estimate
            raw=raw,                   # (B, 3, H, W) degraded input, [0, 1]
            ambient=physics_A,         # (B, 3)       ambient light estimate, [0, 1]
            transmission=refined_map,  # (B, 1, H, W) T-Net refined transmission map
        )
        # out   : (B, 3, H, W) — pred with the red channel physics-gated
        # alpha : (B, 1, H, W) — learned trust mask in [0, 1], for diagnostics
    """

    def __init__(self, cfg: Optional[RedChannelCompensationConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or RedChannelCompensationConfig()
        c = self.cfg.base_ch

        # Gate input channels:
        #   raw(3) + pred(3) + phys_red(1) + transmission(1) + |pred_r-phys_r|(1) = 9
        in_ch = 9
        layers: List[nn.Module] = []
        prev = in_ch
        for _ in range(self.cfg.num_gate_layers):
            layers += [
                nn.Conv2d(prev, c, kernel_size=3, padding=1, bias=False),
                _make_norm(c),
                nn.SiLU(inplace=True),
            ]
            prev = c
        self.gate_body = nn.Sequential(*layers)
        self.gate_head = nn.Conv2d(prev, 1, kernel_size=1)

        # Bias the gate toward "trust the network prediction" at init, so
        # RCC begins as a near-identity op and the training signal decides
        # how much (and where) to lean on the physics prior.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.cfg.init_gate_bias)

    # ------------------------------------------------------------------
    # Physics term (no learnable parameters)
    # ------------------------------------------------------------------

    def physics_restore_red(
        self, raw: Tensor, ambient: Tensor, transmission: Tensor
    ) -> Tensor:
        """
        Closed-form Beer-Lambert inversion of the red channel:

            J_r(x) = (I_r(x) − A_r · (1 − t_r(x))) / max(t_r(x), t_min)

        Parameters
        ----------
        raw          : (B, 3, H, W)  raw degraded image, [0, 1]
        ambient      : (B, 3)        ambient light estimate, [0, 1]
        transmission : (B, 1, H, W)  transmission map t_r(x), [0, 1]

        Returns
        -------
        (B, 1, H, W)  physics-restored red channel, clamped to [0, 1].
        """
        I_r = raw[:, 0:1]  # (B, 1, H, W)
        A_r = ambient[:, 0].clamp(0.0, 1.0).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        t = transmission.clamp(min=self.cfg.t_min, max=1.0)
        J_r = (I_r - A_r * (1.0 - t)) / t
        return J_r.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: Tensor,
        raw: Tensor,
        ambient: Tensor,
        transmission: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        pred         : (B, 3, H, W)  network's clean-image estimate (x̂_0)
        raw          : (B, 3, H, W)  raw degraded input image, [0, 1]
        ambient      : (B, 3)        ambient light estimate, [0, 1]
        transmission : (B, 1, H, W)  transmission map (T-Net refined_map)

        Returns
        -------
        out   : (B, 3, H, W)  `pred` with the red channel replaced by the
                physics-gated blend; G/B channels unchanged.
        alpha : (B, 1, H, W)  learned trust mask in [0, 1].
        """
        if raw.shape[-2:] != pred.shape[-2:]:
            raw = F.interpolate(
                raw, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )
        if transmission.shape[-2:] != pred.shape[-2:]:
            transmission = F.interpolate(
                transmission,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        phys_red = self.physics_restore_red(raw, ambient, transmission)  # (B,1,H,W)
        pred_red = pred[:, 0:1]
        disagreement = (pred_red - phys_red).abs()

        gate_in = torch.cat(
            [raw, pred, phys_red, transmission, disagreement], dim=1
        )  # (B, 9, H, W)
        feat = self.gate_body(gate_in)
        alpha = torch.sigmoid(self.gate_head(feat))  # (B, 1, H, W)

        red_out = alpha * phys_red + (1.0 - alpha) * pred_red
        out = torch.cat([red_out, pred[:, 1:2], pred[:, 2:3]], dim=1)
        return out, alpha

    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        return (
            f"RedChannelCompensation(base_ch={self.cfg.base_ch}, "
            f"t_min={self.cfg.t_min}, params={self.num_parameters():,})"
        )
