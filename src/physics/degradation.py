"""
src/physics/degradation.py
────────────────────────────────────────────────────────────────────────────
Degradation Score Computation — feeds the Dual-Stream Degradation Estimator
in P-UWDM.

What does "degradation score" mean here?
─────────────────────────────────────────
The thesis uses a dual-stream estimator to condition the diffusion backbone
on the *type and severity* of underwater degradation.  Common degradation
modes in underwater imagery are:

  1. **Colour cast** — severe channel imbalance (bluish, greenish, or
     yellowish tint depending on water depth/turbidity).
  2. **Hazing / low contrast** — global contrast compression; low dynamic
     range, washed-out colours.
  3. **Blur / scattering** — frequency-domain energy concentrated in low
     frequencies; lack of sharp edges.
  4. **Noise** — high-frequency energy in flat regions; typically shot or
     read noise amplified by digital gain correction.

Output
──────
This module returns a feature dict (and optionally a single scalar "severity"
score) that the DataLoader exposes as a conditioning signal:

  DegradationFeatures (dataclass):
    colour_cast   : Tensor (3,)   — per-channel mean deviation from grey
    contrast      : Tensor (1,)   — RMS contrast (Michelson-like)
    blur          : Tensor (1,)   — Laplacian energy proxy for sharpness
    noise         : Tensor (1,)   — high-freq noise power estimate
    severity      : Tensor (1,)   — scalar in [0,1]; weighted combination
    feature_vec   : Tensor (6,)   — concat of all sub-scores for network input

Architecture note
─────────────────
The dual-stream degradation estimator in P-UWDM takes this feature_vec as
input to its first linear projection (stream 1: physics scores) alongside a
learned CNN encoder (stream 2: patch statistics), which is implemented in
the model, not here.  This module computes the *stream-1* signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DegradationConfig:
    """Hyper-parameters for degradation score computation."""

    # Colour cast: deviation from neutral grey measured in L*a*b* proxy
    # We use a simpler RGB metric: std-dev of per-channel means
    colour_cast_ref: float = 0.5  # reference (mid-grey) per channel

    # Contrast: patch size for local contrast estimation
    contrast_patch: int = 32

    # Blur: Laplacian kernel size
    blur_ksize: int = 3

    # Noise estimation window (should be ≥ 16)
    noise_window: int = 16

    # Severity weights (must sum to ~1.0; tuned on UIEB validation set)
    w_colour: float = 0.30
    w_contrast: float = 0.25
    w_blur: float = 0.25
    w_noise: float = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DegradationFeatures:
    """
    Container for per-image degradation measurements.

    All tensors are CPU float32.
    """

    colour_cast: Tensor  # (3,)  per-channel deviation
    contrast: Tensor  # (1,)  normalised contrast score
    blur: Tensor  # (1,)  1 − sharpness; 1 = fully blurred
    noise: Tensor  # (1,)  normalised noise power
    severity: Tensor  # (1,)  weighted scalar ∈ [0,1]
    feature_vec: Tensor  # (6,)  [mean_cast, r_cast, g_cast, b_cast — but only 3
    #        channels — contrast, blur, noise]
    #  actual layout: [r_dev, g_dev, b_dev, contrast, blur, noise]

    def to(self, device: torch.device | str) -> "DegradationFeatures":
        return DegradationFeatures(
            colour_cast=self.colour_cast.to(device),
            contrast=self.contrast.to(device),
            blur=self.blur.to(device),
            noise=self.noise.to(device),
            severity=self.severity.to(device),
            feature_vec=self.feature_vec.to(device),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Estimator
# ──────────────────────────────────────────────────────────────────────────────


class DegradationEstimator:
    """
    Computes physics-based degradation features for a single underwater image.

    Parameters
    ----------
    cfg : DegradationConfig

    Usage
    -----
    >>> est = DegradationEstimator()
    >>> img = torch.rand(3, 256, 256)   # CHW float32 [0,1]
    >>> deg = est(img)                   # DegradationFeatures
    >>> deg.feature_vec                  # (6,)
    >>> deg.severity                     # (1,)
    """

    def __init__(self, cfg: DegradationConfig | None = None) -> None:
        self.cfg = cfg or DegradationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, image: Tensor) -> DegradationFeatures:
        """
        Parameters
        ----------
        image : Tensor  (3, H, W) float32 [0,1]

        Returns
        -------
        DegradationFeatures
        """
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected CHW (C=3), got {tuple(image.shape)}")

        img = image.float().cpu().clamp(0.0, 1.0)

        colour_cast = self._colour_cast(img)  # (3,)
        contrast = self._contrast(img)  # (1,)
        blur = self._blur(img)  # (1,)
        noise = self._noise(img)  # (1,)

        cfg = self.cfg
        severity = (
            (
                cfg.w_colour * colour_cast.mean()
                + cfg.w_contrast * (1.0 - contrast[0])  # low contrast = high severity
                + cfg.w_blur * blur[0]
                + cfg.w_noise * noise[0]
            )
            .unsqueeze(0)
            .clamp(0.0, 1.0)
        )

        # Layout: [r_dev, g_dev, b_dev, contrast, blur, noise]
        feature_vec = torch.cat(
            [
                colour_cast,  # (3,)
                contrast,  # (1,)
                blur,  # (1,)
                noise,  # (1,)
            ]
        )  # (6,)

        return DegradationFeatures(
            colour_cast=colour_cast,
            contrast=contrast,
            blur=blur,
            noise=noise,
            severity=severity,
            feature_vec=feature_vec,
        )

    # ------------------------------------------------------------------
    # Sub-score methods
    # ------------------------------------------------------------------

    def _colour_cast(self, img: Tensor) -> Tensor:
        """
        Per-channel absolute deviation from the expected neutral-grey mean.

        For a perfectly grey scene, each channel mean ≈ 0.5.  In underwater
        images the blue/green channels are elevated, red is suppressed.

        Returns (3,) in [0, 1] — higher = stronger cast in that channel.
        """
        channel_means = img.mean(dim=[1, 2])  # (3,) global mean
        deviation = (channel_means - self.cfg.colour_cast_ref).abs()
        return deviation.clamp(0.0, 1.0)

    def _contrast(self, img: Tensor) -> Tensor:
        """
        Patch-based RMS contrast.

        Divide image into non-overlapping patches of size contrast_patch×p.
        Compute std-dev of luminance in each patch, average across patches.
        Normalise to [0, 1] by dividing by 0.5 (max possible std for binary).

        Returns (1,) — higher = more contrast (less degraded).
        """
        lum = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]  # (H, W)
        p = self.cfg.contrast_patch
        H, W = lum.shape
        # Trim to multiple of p
        H_, W_ = (H // p) * p, (W // p) * p
        if H_ == 0 or W_ == 0:
            return lum.std().unsqueeze(0) / 0.5

        patches = lum[:H_, :W_].reshape(H_ // p, p, W_ // p, p)
        patch_std = patches.std(dim=(1, 3))  # (H_//p, W_//p)
        contrast = patch_std.mean() / 0.5  # normalise
        return contrast.unsqueeze(0).clamp(0.0, 1.0)

    def _blur(self, img: Tensor) -> Tensor:
        """
        Laplacian energy proxy for blur.

        A sharp image has high Laplacian variance.
        We invert and normalise so that: 1 = maximally blurred, 0 = sharp.

        Returns (1,).
        """
        # Convert to greyscale
        grey = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        grey = grey.unsqueeze(0)  # (1, 1, H, W)

        # Laplacian kernel
        lap_kernel = (
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,3,3)

        lap = F.conv2d(grey, lap_kernel, padding=1)  # (1,1,H,W)
        sharpness = lap.pow(2).mean().sqrt()  # RMS Laplacian energy

        # Normalise: typical sharp images have sharpness ~0.03–0.15;
        # cap at 0.20 for normalisation.
        sharpness_norm = (sharpness / 0.20).clamp(0.0, 1.0)
        blur = (1.0 - sharpness_norm).unsqueeze(0)
        return blur.clamp(0.0, 1.0)

    def _noise(self, img: Tensor) -> Tensor:
        """
        High-frequency noise power estimate.

        Uses Immerkær's method: compute residuals from a local mean, estimate
        noise std from a smooth central patch.

        Returns (1,) in [0, 1].
        """
        grey = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        grey = grey.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        w = self.cfg.noise_window
        # Ensure even kernel size for symmetric padding
        pad = w // 2
        padded = F.pad(grey, [pad, w - 1 - pad, pad, w - 1 - pad], mode="reflect")
        # Compute local mean via average pooling
        local_mean = F.avg_pool2d(padded, kernel_size=w, stride=1, padding=0)
        residual = grey - local_mean
        noise_std = residual.std()

        # Typical clean images: noise_std ~0.001–0.005;
        # noisy underwater images: ~0.02–0.05.
        # Normalise with cap at 0.05.
        noise_norm = (noise_std / 0.05).clamp(0.0, 1.0)
        return noise_norm.unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Batch utility
# ──────────────────────────────────────────────────────────────────────────────


def estimate_degradation_batch(
    images: Tensor,
    cfg: DegradationConfig | None = None,
) -> Dict[str, Tensor]:
    """
    Parameters
    ----------
    images : Tensor  (B, 3, H, W)

    Returns
    -------
    dict with keys:
      'feature_vec' : (B, 6)
      'severity'    : (B, 1)
      'colour_cast' : (B, 3)
      'contrast'    : (B, 1)
      'blur'        : (B, 1)
      'noise'       : (B, 1)
    """
    est = DegradationEstimator(cfg)
    results = [est(img) for img in images]
    return {
        "feature_vec": torch.stack([r.feature_vec for r in results]),
        "severity": torch.stack([r.severity for r in results]),
        "colour_cast": torch.stack([r.colour_cast for r in results]),
        "contrast": torch.stack([r.contrast for r in results]),
        "blur": torch.stack([r.blur for r in results]),
        "noise": torch.stack([r.noise for r in results]),
    }
