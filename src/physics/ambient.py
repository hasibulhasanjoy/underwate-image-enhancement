"""
src/physics/ambient.py
────────────────────────────────────────────────────────────────────────────
Ambient Light Estimation — feeds A-Net conditioning in P-UWDM.

Physics background
──────────────────
In the underwater image formation model:

    I(x) = J(x) · t(x) + A · (1 − t(x))

A is the global ambient / background light vector (R, G, B).  It is the
colour the water converges to at infinite depth — practically estimated as
the brightest (highest-intensity) patch in the image, which most
dark-channel or quadtree-based methods agree on.

Implementation strategy (no neural net; pure signal processing)
───────────────────────────────────────────────────────────────
1. **Quad-tree bright-pixel selection** — recursively find the brightest
   sub-region by average intensity; stop when region < min_patch_px.
2. **Median colour of that patch** — robust to single-pixel hot spots.
3. **Depth-channel correction** — optionally weight by the blue channel
   (underwater scattering biases ambient toward blue).
4. Returns a (3,) float32 tensor in [0, 1] suitable for conditioning.

The A-Net in the paper *refines* this estimate with learned corrections;
this module provides the physics-based prior that A-Net conditions on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AmbientConfig:
    """Hyper-parameters for ambient light estimation."""

    # Quadtree: stop recursing when region area < this many pixels
    min_patch_px: int = 200

    # Fraction of brightest pixels used for final colour estimation
    # (after quadtree gives us the bright region)
    bright_frac: float = 0.001  # top-0.1 % of pixels

    # Blue-channel bias correction — underwater ambient is blue-shifted;
    # a small correction pulls the estimate back toward true scene ambient.
    # Set to 0.0 to disable.
    blue_correction: float = 0.05

    # Clamp output to this max value (some very bright scenes clip to >0.95)
    clamp_max: float = 0.98


# ──────────────────────────────────────────────────────────────────────────────
# Core estimator
# ──────────────────────────────────────────────────────────────────────────────


class AmbientLightEstimator:
    """
    Estimates the global ambient light vector A ∈ [0,1]^3 from a raw
    underwater image.

    Parameters
    ----------
    cfg : AmbientConfig
        Estimation hyper-parameters.

    Usage
    -----
    >>> est = AmbientLightEstimator()
    >>> img = torch.rand(3, 256, 256)           # CHW, float32, [0,1]
    >>> A   = est(img)                           # shape (3,)
    """

    def __init__(self, cfg: AmbientConfig | None = None) -> None:
        self.cfg = cfg or AmbientConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, image: Tensor) -> Tensor:
        """
        Parameters
        ----------
        image : Tensor
            CHW float32 image in [0, 1].  C must be 3 (RGB).

        Returns
        -------
        Tensor
            Shape (3,) — estimated ambient light per channel.
        """
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected CHW image with C=3, got shape {tuple(image.shape)}"
            )

        # Work on CPU (called per-sample in DataLoader workers)
        img = image.float().cpu()

        # Step 1: quadtree to find the brightest region
        region = self._quadtree_brightest(img)

        # Step 2: pick the top-bright_frac pixels by luminance in that region
        A = self._robust_colour(region)

        # Step 3: blue correction (blue channel inflated by scattering)
        if self.cfg.blue_correction > 0.0:
            A = A.clone()
            A[2] = torch.clamp(A[2] - self.cfg.blue_correction, 0.0, 1.0)

        A = torch.clamp(A, 0.0, self.cfg.clamp_max)
        return A  # shape (3,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quadtree_brightest(self, img: Tensor) -> Tensor:
        """
        Recursively halve the image region, keeping the brighter half.
        Returns the CHW sub-tensor of the brightest final region.
        """
        C, H, W = img.shape
        # luminance = 0.299R + 0.587G + 0.114B
        lum = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]  # HW

        region = img
        region_lum = lum

        while region.shape[1] * region.shape[2] > self.cfg.min_patch_px:
            C, h, w = region.shape
            # Split into four quadrants; keep the one with highest mean lum
            h2, w2 = h // 2, w // 2
            if h2 == 0 or w2 == 0:
                break

            quads = [
                (region[:, :h2, :w2], region_lum[:h2, :w2]),
                (region[:, :h2, w2:], region_lum[:h2, w2:]),
                (region[:, h2:, :w2], region_lum[h2:, :w2]),
                (region[:, h2:, w2:], region_lum[h2:, w2:]),
            ]
            means = [lq.mean().item() for _, lq in quads]
            best = int(torch.tensor(means).argmax().item())
            region, region_lum = quads[best]

        return region  # CHW

    def _robust_colour(self, region: Tensor) -> Tensor:
        """
        Compute the ambient colour as the mean colour of the top
        `bright_frac` pixels (by luminance) inside `region`.
        """
        C, H, W = region.shape
        lum = 0.299 * region[0] + 0.587 * region[1] + 0.114 * region[2]
        flat_lum = lum.reshape(-1)
        flat_img = region.reshape(C, -1)

        n = max(1, int(flat_lum.numel() * self.cfg.bright_frac))
        _, top_idx = flat_lum.topk(n)
        bright_pixels = flat_img[:, top_idx]  # (3, n)

        return bright_pixels.mean(dim=1)  # (3,)


# ──────────────────────────────────────────────────────────────────────────────
# Batch-level utility (used in DataModule's collation or offline caching)
# ──────────────────────────────────────────────────────────────────────────────


def estimate_ambient_batch(images: Tensor, cfg: AmbientConfig | None = None) -> Tensor:
    """
    Parameters
    ----------
    images : Tensor
        Shape (B, 3, H, W), float32 [0, 1].

    Returns
    -------
    Tensor
        Shape (B, 3) — ambient light per image.
    """
    est = AmbientLightEstimator(cfg)
    return torch.stack([est(img) for img in images])  # (B, 3)
