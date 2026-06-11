"""
src/data/transforms.py
----------------------
Image transformation pipelines for each dataset split.

Key design decisions
--------------------
* Both the *raw* (degraded) and *reference* (clean) images receive the
  **same** spatial transforms (flip, rotate, crop) so pixel alignment is
  preserved.  Photometric augmentations are applied to the raw image
  **only** — the reference stays pristine.
* Transforms are built from the typed ``DataConfig`` so there is a single
  source of truth for hyper-parameters.
* Returning plain ``torchvision.transforms.Compose`` objects keeps the
  transforms framework-agnostic and easy to swap for Albumentations.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from ..utils.config import DataConfig, AugmentationSplitConfig, PreprocessingConfig

# ---------------------------------------------------------------------------
# Paired transform wrapper
# ---------------------------------------------------------------------------


class PairedTransform:
    """
    Apply *spatial* transforms identically to both images in a pair,
    and apply *photometric* transforms to the raw image only.

    Parameters
    ----------
    spatial_transform:
        A callable that accepts a PIL Image and returns a PIL Image.
        The **same random state** is used for both images.
    raw_photometric_transform:
        Additional colour-level augmentations applied only to the raw
        (degraded) image.  Pass ``None`` to skip.
    to_tensor:
        Final to-tensor + normalize pipeline applied to both images.
    """

    def __init__(
        self,
        spatial_transform: Callable,
        raw_photometric_transform: Callable | None,
        to_tensor: Callable,
    ) -> None:
        self.spatial = spatial_transform
        self.raw_photo = raw_photometric_transform
        self.to_tensor = to_tensor

    def __call__(
        self,
        raw: Image.Image,
        reference: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Deterministic spatial transform using manual seed ----------
        seed = torch.randint(0, 2**31, (1,)).item()

        torch.manual_seed(seed)
        raw = self.spatial(raw)

        torch.manual_seed(seed)
        reference = self.spatial(reference)

        # --- Photometric augmentation (raw only) -------------------------
        if self.raw_photo is not None:
            raw = self.raw_photo(raw)

        # --- To tensor + normalize ---------------------------------------
        raw_t: torch.Tensor = self.to_tensor(raw)
        ref_t: torch.Tensor = self.to_tensor(reference)

        return raw_t, ref_t


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_spatial_train(
    aug: AugmentationSplitConfig, size: Tuple[int, int]
) -> Callable:
    ops = []

    if aug.random_crop.enabled:
        ops.append(T.RandomCrop(aug.random_crop.size))
    else:
        ops.append(T.Resize(size, interpolation=T.InterpolationMode.BICUBIC))

    if aug.random_horizontal_flip:
        ops.append(T.RandomHorizontalFlip())
    if aug.random_vertical_flip:
        ops.append(T.RandomVerticalFlip())
    if aug.random_rotation.enabled:
        ops.append(T.RandomRotation(degrees=aug.random_rotation.degrees))

    return T.Compose(ops)


def _build_photometric_train(aug: AugmentationSplitConfig) -> Callable | None:
    ops = []

    if aug.color_jitter.enabled:
        ops.append(
            T.ColorJitter(
                brightness=aug.color_jitter.brightness,
                contrast=aug.color_jitter.contrast,
                saturation=aug.color_jitter.saturation,
                hue=aug.color_jitter.hue,
            )
        )
    if aug.gaussian_blur.enabled:
        ops.append(
            T.GaussianBlur(
                kernel_size=aug.gaussian_blur.kernel_size,
                sigma=aug.gaussian_blur.sigma,
            )
        )

    return T.Compose(ops) if ops else None


def _build_to_tensor(prep: PreprocessingConfig) -> Callable:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=list(prep.normalize.mean), std=list(prep.normalize.std)),
        ]
    )


def _build_spatial_eval(size: Tuple[int, int]) -> Callable:
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_train_transforms(config: DataConfig) -> PairedTransform:
    """
    Return the paired transform used during training.
    Includes spatial and photometric augmentations.
    """
    size = tuple(config.preprocessing.image_size)  # type: ignore[arg-type]
    aug = config.augmentation.train
    return PairedTransform(
        spatial_transform=_build_spatial_train(aug, size),
        raw_photometric_transform=_build_photometric_train(aug),
        to_tensor=_build_to_tensor(config.preprocessing),
    )


def get_val_transforms(config: DataConfig) -> PairedTransform:
    """
    Return the deterministic paired transform used during validation.
    Only resize + normalize — no stochastic augmentations.
    """
    size = tuple(config.preprocessing.image_size)  # type: ignore[arg-type]
    return PairedTransform(
        spatial_transform=_build_spatial_eval(size),
        raw_photometric_transform=None,
        to_tensor=_build_to_tensor(config.preprocessing),
    )


def get_test_transforms(config: DataConfig) -> PairedTransform:
    """
    Identical to val transforms.  Kept as a separate symbol so callers
    can treat test-time inference independently in future.
    """
    return get_val_transforms(config)
