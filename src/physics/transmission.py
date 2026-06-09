"""
src/physics/transmission.py
────────────────────────────────────────────────────────────────────────────
Transmission Map Estimation — feeds T-Net conditioning in P-UWDM.

Physics background
──────────────────
The underwater image formation model (Beer-Lambert scattering):

    I(x) = J(x) · t(x) + A · (1 − t(x))

where t(x) = exp(−β·d(x)) is the transmission at pixel x,
β is the attenuation coefficient, and d(x) is scene depth.

A high transmission (≈ 1.0) means the pixel is clear and close.
A low transmission (≈ 0.0) means heavy scattering / deep region.

Estimation pipeline (Dark Channel Prior, adapted for underwater)
────────────────────────────────────────────────────────────────
1. **Normalised radiance** — divide I by estimated ambient A to suppress
   the global illuminant.
2. **Dark channel** — min-filter over a local patch × min over channels.
   For underwater, we use the **red channel** dark map (He et al., adapted):
   underwater scenes lack red light at depth, so the red channel encodes
   attenuation better than the haze-based min(channels) approach.
3. **Raw transmission** — t̃(x) = 1 − ω · dark(x), ω ∈ (0,1] controls
   how aggressively to restore (ω=0.95 typical for haze; 0.85 for water).
4. **Soft-matting refinement** — guided filter with the input image as
   guide.  Preserves edges without expensive Laplacian matting.
5. Clamp to [t_min, 1.0] to avoid division-by-zero in restoration.

Output
──────
Returns a (1, H, W) float32 map in [t_min, 1.0], ready for T-Net
conditioning and the physics-based restoration path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TransmissionConfig:
    """Hyper-parameters for transmission map estimation."""

    # Dark channel patch size (should be odd)
    patch_size: int = 15

    # Attenuation weight ω  (how much dark channel drives transmission)
    omega: float = 0.85

    # Minimum allowable transmission (prevents blow-up in restoration)
    t_min: float = 0.10

    # Guided filter radius
    # 20 is appropriate for 256px input; increase to 40 for 512px+
    guided_radius: int = 20

    # Guided filter regularisation ε
    guided_eps: float = 1e-3

    # If True, use red-channel dark map (better for underwater).
    # If False, use min-channel dark map (standard He et al. haze).
    use_red_channel: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Guided filter (edge-preserving smoother)
# ──────────────────────────────────────────────────────────────────────────────


class GuidedFilter:
    """
    Fast O(N) guided filter (He et al., 2013).

    Parameters
    ----------
    radius : int
        Box-filter radius.
    eps : float
        Regularisation parameter.
    """

    def __init__(self, radius: int = 40, eps: float = 1e-3) -> None:
        self.r = radius
        self.eps = eps
        self._k = 2 * radius + 1

    def _box(self, x: Tensor) -> Tensor:
        """Box filter via separable average pooling (no padding artefacts)."""
        # x: (1, 1, H, W)
        k = self._k
        return F.avg_pool2d(
            F.pad(x, [self.r] * 4, mode="reflect"),
            kernel_size=k,
            stride=1,
            padding=0,
        )

    @torch.no_grad()
    def filter(self, guide: Tensor, src: Tensor) -> Tensor:
        """
        Parameters
        ----------
        guide : Tensor  (1, 1, H, W)  — the guiding image (greyscale)
        src   : Tensor  (1, 1, H, W)  — the image to be filtered

        Returns
        -------
        Tensor  (1, 1, H, W)
        """
        mean_I = self._box(guide)
        mean_p = self._box(src)
        mean_Ip = self._box(guide * src)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self._box(guide * guide)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = self._box(a)
        mean_b = self._box(b)

        return mean_a * guide + mean_b


# ──────────────────────────────────────────────────────────────────────────────
# Transmission estimator
# ──────────────────────────────────────────────────────────────────────────────


class TransmissionEstimator:
    """
    Estimates a per-pixel transmission map t(x) ∈ [t_min, 1] from a
    raw underwater image, given a pre-computed ambient light estimate.

    Parameters
    ----------
    cfg : TransmissionConfig

    Usage
    -----
    >>> est = TransmissionEstimator()
    >>> img = torch.rand(3, 256, 256)           # CHW float32 [0,1]
    >>> A   = torch.tensor([0.8, 0.85, 0.9])   # (3,) from AmbientEstimator
    >>> t   = est(img, A)                        # (1, H, W)
    """

    def __init__(self, cfg: TransmissionConfig | None = None) -> None:
        self.cfg = cfg or TransmissionConfig()
        self._gf = GuidedFilter(
            radius=cfg.guided_radius if cfg else 40, eps=cfg.guided_eps if cfg else 1e-3
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, image: Tensor, ambient: Tensor) -> Tensor:
        """
        Parameters
        ----------
        image   : Tensor  (3, H, W) float32 [0,1]
        ambient : Tensor  (3,) float32 [0,1]   — from AmbientLightEstimator

        Returns
        -------
        Tensor  (1, H, W) float32 in [t_min, 1.0]
        """
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected CHW (C=3), got {tuple(image.shape)}")

        img = image.float().cpu()
        A = ambient.float().cpu().clamp(1e-6, 1.0)

        C, H, W = img.shape

        # Step 1: normalise by ambient
        norm = img / A[:, None, None]  # (3, H, W)

        # Step 2: dark channel
        dark = self._dark_channel(norm)  # (1, H, W)

        # Step 3: raw transmission
        t_raw = 1.0 - self.cfg.omega * dark  # (1, H, W)

        # Step 4: guided filter refinement
        guide = img.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
        t_in = t_raw.unsqueeze(0)  # (1,1,H,W)
        t_refined = self._gf.filter(guide, t_in).squeeze(0)  # (1,H,W)

        # Step 5: clamp
        t_refined = torch.clamp(t_refined, self.cfg.t_min, 1.0)
        return t_refined  # (1, H, W)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dark_channel(self, norm: Tensor) -> Tensor:
        """
        Compute the dark channel of the normalised image.

        For underwater images: use the red channel (norm[0]) instead of
        the standard min-over-channels approach, because the red channel
        is most strongly attenuated and carries the clearest depth signal.

        Returns (1, H, W) dark channel in [0, 1].
        """
        if self.cfg.use_red_channel:
            # Red channel dark map — dominant underwater attenuation axis
            channel = norm[0:1]  # (1, H, W)
        else:
            # Standard haze dark channel — min over RGB
            channel = norm.min(dim=0, keepdim=True).values  # (1, H, W)

        # Min-pool over local patch (approximates pixel-level dark channel)
        p = self.cfg.patch_size
        # negate → max-pool → negate  ≡  min-pool (F.max_pool2d is fast)
        padded = F.pad(-channel.unsqueeze(0), [p // 2] * 4, mode="reflect")
        dark = -F.max_pool2d(padded, kernel_size=p, stride=1, padding=0)
        dark = dark.squeeze(0)  # (1, H, W)
        return dark.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Batch utility
# ──────────────────────────────────────────────────────────────────────────────


def estimate_transmission_batch(
    images: Tensor,
    ambients: Tensor,
    cfg: TransmissionConfig | None = None,
) -> Tensor:
    """
    Parameters
    ----------
    images   : Tensor  (B, 3, H, W)
    ambients : Tensor  (B, 3)

    Returns
    -------
    Tensor  (B, 1, H, W)
    """
    est = TransmissionEstimator(cfg)
    return torch.stack([est(img, A) for img, A in zip(images, ambients)])
