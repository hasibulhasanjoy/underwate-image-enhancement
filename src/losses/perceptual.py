"""
Perceptual Loss
===============
Multi-scale VGG-16 feature matching loss, computed at three depth levels:

    relu2_2  (block2, conv2)  — texture/edge features
    relu3_3  (block3, conv3)  — mid-level structural features
    relu4_3  (block4, conv3)  — high-level semantic features

L_perc = Σ_l  λ_l · ‖φ_l(x̂) − φ_l(x_ref)‖₁

Default layer weights: [1.0, 0.75, 0.5] (shallower → finer, higher weight).

VGG features are frozen (no gradient) — they serve as a fixed perceptual
distance metric, not trainable components.

Input normalization: VGG was trained on ImageNet with μ=(0.485,0.456,0.406),
σ=(0.229,0.224,0.225).  This module normalizes internally, so callers can
pass images in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights


class PerceptualLoss(nn.Module):
    """
    VGG-16 perceptual feature-matching loss.

    Args:
        layer_weights: Weights for [relu2_2, relu3_3, relu4_3].
        normalize_input: If True, apply ImageNet normalization inside forward.
                         Set False if images are already normalized.
    """

    # ImageNet statistics
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # VGG-16 feature indices (0-based within features Sequential):
    # relu2_2 = index 9, relu3_3 = index 16, relu4_3 = index 23
    _SLICE_ENDS = (9, 16, 23)

    def __init__(
        self,
        layer_weights: tuple[float, ...] = (1.0, 0.75, 0.5),
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        assert len(layer_weights) == 3, "Exactly 3 layer weights required."
        self.layer_weights = layer_weights
        self.normalize_input = normalize_input

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Build three slices of the VGG feature extractor
        prev = 0
        slices = []
        for end in self._SLICE_ENDS:
            slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end
        self.slices = nn.ModuleList(slices)

        # Freeze VGG parameters
        for p in self.parameters():
            p.requires_grad_(False)

        # Register buffers for normalization constants
        self.register_buffer("mean", self._MEAN)
        self.register_buffer("std", self._STD)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [0,1] to ImageNet-standardized range."""
        return (x - self.mean) / self.std

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        h = x
        for s in self.slices:
            h = s(h)
            features.append(h)
        return features

    def forward(
        self,
        enhanced: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            enhanced:  (B, 3, H, W) in [0, 1]  enhanced / predicted image
            reference: (B, 3, H, W) in [0, 1]  clean reference image

        Returns:
            Scalar perceptual loss.
        """
        if self.normalize_input:
            enhanced = self._normalize(enhanced)
            reference = self._normalize(reference)

        feats_enh = self._extract_features(enhanced)
        with torch.no_grad():
            feats_ref = self._extract_features(reference)

        loss = torch.tensor(0.0, device=enhanced.device, dtype=enhanced.dtype)
        for w, f_e, f_r in zip(self.layer_weights, feats_enh, feats_ref):
            loss = loss + w * F.l1_loss(f_e, f_r)

        return loss
