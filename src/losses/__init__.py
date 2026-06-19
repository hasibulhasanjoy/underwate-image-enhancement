"""
P-UWDM Loss Module
==================
Five-component composite loss for physics-guided underwater diffusion model:
  1. Diffusion loss       – weighted denoising MSE (core DDPM objective)
  2. Adversarial loss     – LSGAN (least-squares GAN) for perceptual realism
  3. Perceptual loss      – VGG-16 multi-scale feature matching
  4. Histogram loss       – per-channel colour distribution alignment
  5. Contrastive loss     – NT-Xent pulling enhanced↔reference, pushing enhanced↔raw
"""

from .composite import CompositeLoss, LossWeights
from .diffusion import DiffusionLoss
from .adversarial import AdversarialLoss, PatchDiscriminator
from .perceptual import PerceptualLoss
from .histogram import HistogramLoss
from .contrastive import ContrastiveLoss

__all__ = [
    "CompositeLoss",
    "LossWeights",
    "DiffusionLoss",
    "AdversarialLoss",
    "PatchDiscriminator",
    "PerceptualLoss",
    "HistogramLoss",
    "ContrastiveLoss",
]
