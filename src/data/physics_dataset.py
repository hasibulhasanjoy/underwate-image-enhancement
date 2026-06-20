"""
src/data/physics_dataset.py
────────────────────────────────────────────────────────────────────────────
Physics-aware UIEB dataset with integrated prior estimation.

Each sample returned by __getitem__ contains:

    raw          : Tensor (3, H, W) — raw underwater image (transform output)
    reference    : Tensor (3, H, W) — clean reference image (transform output)
    ambient      : Tensor (3,)       — estimated ambient light A
    transmission : Tensor (1, H, W)  — estimated transmission map t(x)
    degradation  : Tensor (6,)       — degradation feature vector
    severity     : Tensor (1,)       — scalar degradation severity

Integration with the diffusion model
──────────────────────────────────────
At model input time:
  • A-Net receives `raw` + `ambient` broadcast to (3+3, H, W) or via
    cross-attention on the (3,) vector.
  • T-Net receives `raw` + `transmission` → (3+1, H, W).
  • Dual-stream degradation estimator receives `degradation` as stream-1
    input; its stream-2 CNN encoder processes raw image patches.

This module computes all three priors on-the-fly in DataLoader workers
(no offline pre-computation needed for UIEB's 900 images; each sample
takes ~10ms on CPU workers).  For large datasets (LSUI 4k, EUVP 12k)
consider offline caching via `PhysicsCacheBuilder`.

Performance on RTX 4090 setup
──────────────────────────────
With 16 DataLoader workers and pin_memory=True, physics estimation
adds ~0 wall-clock overhead because workers are CPU-bound and run in
parallel while the GPU processes the previous batch.

Bug fixes applied
─────────────────
BUG-1 (PhysicsUIEBDataModule.setup — manifest key mismatch):
    The splitter persists integer index lists under the keys
    ``train_indices`` / ``val_indices`` / ``test_indices``.
    The old code incorrectly read ``manifest["train"]`` / ``"val"`` /
    ``"test"]``, which do not exist, causing a ``KeyError`` at setup time.
    Fixed: read the correct ``*_indices`` keys, then use those integer
    indices to select paths from a sorted listing of the raw/ref directories.

BUG-2 (PhysicsUIEBDataset.__getitem__ — physics estimator input range):
    Physics estimators (AmbientLightEstimator, TransmissionEstimator,
    DegradationEstimator) expect pixel values in [0, 1].  When
    ``physics_on_augmented=True`` the tensor passed in is the transform
    output, which may be ImageNet-normalised (range ≈ −2.1 … +2.6).
    Fixed: introduce a ``_denorm_for_physics()`` helper that reverses
    ImageNet normalisation before calling the estimators.  The model
    still receives the fully-normalised ``raw_t`` / ``ref_t`` tensors —
    only the physics branch sees the [0, 1] version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.physics import (
    AmbientConfig,
    AmbientLightEstimator,
    DegradationConfig,
    DegradationEstimator,
    DegradationFeatures,
    TransmissionConfig,
    TransmissionEstimator,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation constants (mirrors torchvision defaults)
# ──────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm_for_physics(t: Tensor) -> Tensor:
    """
    Reverse ImageNet normalisation so physics estimators receive [0, 1] input.

    If the tensor is already in [0, 1] (i.e. no normalisation was applied by
    the transform), the result is still valid because:
        t_01 * std + mean  stays ≈ in [0, 1] for typical underwater images.

    To guarantee correctness the caller should pass ``physics_on_augmented``
    consistently with the transform used (see PhysicsDatasetConfig).

    Parameters
    ----------
    t : Tensor shape (3, H, W), ImageNet-normalised float32

    Returns
    -------
    Tensor shape (3, H, W), values clamped to [0, 1]
    """
    mean = _IMAGENET_MEAN.to(t.device)
    std = _IMAGENET_STD.to(t.device)
    return (t * std + mean).clamp_(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Sample type
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PhysicsSample:
    """
    A single P-UWDM training/validation/test sample.

    All tensors are float32.  Spatial tensors have shape (C, H, W).
    ``raw`` and ``reference`` are in whatever space the transform
    outputs (e.g. ImageNet-normalised); physics tensors are always
    derived from the [0, 1] representation.
    """

    raw: Tensor  # (3, H, W) — transform output (may be normalised)
    reference: Tensor  # (3, H, W) — transform output
    ambient: Tensor  # (3,)       — estimated A in [0, 1]
    transmission: Tensor  # (1, H, W)  — estimated t(x) in [0, 1]
    degradation: Tensor  # (6,)       — degradation feature vector
    severity: Tensor  # (1,)       — scalar severity score

    # Optional metadata (not stacked in collation)
    raw_path: str = ""
    ref_path: str = ""


def physics_collate_fn(
    batch: List[PhysicsSample],
) -> Dict[str, Tensor]:
    """
    Custom collate function for PhysicsSample lists → batched dict.

    Returns
    -------
    dict with keys:
        'raw'          : (B, 3, H, W)
        'reference'    : (B, 3, H, W)
        'ambient'      : (B, 3)
        'transmission' : (B, 1, H, W)
        'degradation'  : (B, 6)
        'severity'     : (B, 1)
    """
    return {
        "raw": torch.stack([s.raw for s in batch]),
        "reference": torch.stack([s.reference for s in batch]),
        "ambient": torch.stack([s.ambient for s in batch]),
        "transmission": torch.stack([s.transmission for s in batch]),
        "degradation": torch.stack([s.degradation for s in batch]),
        "severity": torch.stack([s.severity for s in batch]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PhysicsDatasetConfig:
    """Configuration for PhysicsUIEBDataset."""

    # Physics prior configs (defaults used if None)
    ambient_cfg: Optional[AmbientConfig] = None
    transmission_cfg: Optional[TransmissionConfig] = None
    degradation_cfg: Optional[DegradationConfig] = None

    # Image loading
    load_size: Tuple[int, int] = (256, 256)  # (H, W) — resize on load

    # If True, physics is computed on the *augmented* image (post-transform).
    # If False, physics is computed on the raw PIL → tensor *before* augment.
    # Recommendation: True — priors should reflect the augmented spatial layout.
    physics_on_augmented: bool = True

    # Whether the transform applies ImageNet normalisation.
    # Set to False if the transform keeps pixel values in [0, 1].
    # When True, _denorm_for_physics() is called before physics estimation.
    imagenet_normalised: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class PhysicsUIEBDataset(Dataset):
    """
    Physics-aware UIEB dataset.

    Loads paired (raw, reference) images, applies optional transforms,
    then computes ambient light, transmission map, and degradation features
    on the raw image — all in the DataLoader worker process.

    Parameters
    ----------
    raw_paths : list of Path
        Paths to raw underwater images.
    ref_paths : list of Path
        Paths to corresponding reference images (same order).
    transform : callable, optional
        PairedTransform or similar that takes (raw_pil, ref_pil) and
        returns (raw_tensor, ref_tensor) in CHW float32.
        If ImageNet normalisation is included, set
        ``cfg.imagenet_normalised = True`` (the default).
    cfg : PhysicsDatasetConfig

    Notes
    ─────
    Physics estimators are instantiated *once per worker* via a lazy
    property.  This avoids pickling overhead (estimators hold no state
    that changes per-sample).

    BUG-2 FIX — input range contract
    ─────────────────────────────────
    Physics estimators expect [0, 1] input.  When ``physics_on_augmented``
    is True AND ``imagenet_normalised`` is True, ``_denorm_for_physics()``
    is called on ``raw_t`` before passing it to the estimators.  The model
    still receives the original (normalised) ``raw_t``.
    """

    def __init__(
        self,
        raw_paths: List[Path],
        ref_paths: List[Path],
        transform: Optional[Callable] = None,
        cfg: Optional[PhysicsDatasetConfig] = None,
    ) -> None:
        assert len(raw_paths) == len(
            ref_paths
        ), f"Mismatch: {len(raw_paths)} raw vs {len(ref_paths)} ref paths"

        self.raw_paths = raw_paths
        self.ref_paths = ref_paths
        self.transform = transform
        self.cfg = cfg or PhysicsDatasetConfig()

        # Lazy-initialised per-worker estimators (set in _init_estimators)
        self._ambient_est: Optional[AmbientLightEstimator] = None
        self._transmission_est: Optional[TransmissionEstimator] = None
        self._degradation_est: Optional[DegradationEstimator] = None

        logger.info(
            "PhysicsUIEBDataset: %d samples, load_size=%s, "
            "physics_on_augmented=%s, imagenet_normalised=%s",
            len(self),
            self.cfg.load_size,
            self.cfg.physics_on_augmented,
            self.cfg.imagenet_normalised,
        )

    def __len__(self) -> int:
        return len(self.raw_paths)

    def __getitem__(self, idx: int) -> PhysicsSample:
        # ── Load images ──────────────────────────────────────────────
        raw_path = self.raw_paths[idx]
        ref_path = self.ref_paths[idx]

        raw_pil = Image.open(raw_path).convert("RGB")
        ref_pil = Image.open(ref_path).convert("RGB")

        # ── Resize (before transform to ensure consistent spatial dims) ──
        H, W = self.cfg.load_size
        raw_pil = raw_pil.resize((W, H), Image.BICUBIC)
        ref_pil = ref_pil.resize((W, H), Image.BICUBIC)

        # ── PIL → Tensor (with optional augmentation) ─────────────
        if self.transform is not None:
            raw_t, ref_t = self.transform(raw_pil, ref_pil)
        else:
            raw_t = _pil_to_tensor(raw_pil)
            ref_t = _pil_to_tensor(ref_pil)

        # ── Physics priors ───────────────────────────────────────────
        self._init_estimators()

        # Determine the source tensor for physics estimation.
        if self.cfg.physics_on_augmented:
            physics_src = raw_t  # may be ImageNet-normalised
        else:
            physics_src = _pil_to_tensor(raw_pil)  # always [0, 1]

        # BUG-2 FIX: denormalise if the tensor is ImageNet-normalised so
        # the physics estimators always receive values in [0, 1].
        if self.cfg.physics_on_augmented and self.cfg.imagenet_normalised:
            physics_src = _denorm_for_physics(physics_src)

        ambient = self._ambient_est(physics_src)
        transmission = self._transmission_est(physics_src, ambient)
        deg_feat = self._degradation_est(physics_src)

        return PhysicsSample(
            raw=raw_t,
            reference=ref_t,
            ambient=ambient,
            transmission=transmission,
            degradation=deg_feat.feature_vec,
            severity=deg_feat.severity,
            raw_path=str(raw_path),
            ref_path=str(ref_path),
        )

    # ------------------------------------------------------------------
    # Lazy estimator init (called once per DataLoader worker)
    # ------------------------------------------------------------------

    def _init_estimators(self) -> None:
        if self._ambient_est is not None:
            return  # already initialised in this worker

        cfg = self.cfg
        self._ambient_est = AmbientLightEstimator(cfg.ambient_cfg)
        self._transmission_est = TransmissionEstimator(cfg.transmission_cfg)
        self._degradation_est = DegradationEstimator(cfg.degradation_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# DataModule (Lightning-style, manually managed)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PhysicsDataModuleConfig:
    """Configuration for PhysicsUIEBDataModule."""

    # Dataset paths
    raw_dir: str = "dataset/UIEB/raw"
    ref_dir: str = "dataset/UIEB/reference"

    # Split manifest JSON produced by src.data.splitter
    # Must contain keys: train_indices, val_indices, test_indices, n_samples
    split_manifest: str = "dataset/UIEB/split_manifest.json"

    # DataLoader settings — tuned for RTX 4090 / Ryzen 9 7950X
    batch_size: int = 32
    num_workers: int = 16  # matches physical core count
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Dataset config
    dataset_cfg: Optional[PhysicsDatasetConfig] = None

    # Image size
    load_size: Tuple[int, int] = (256, 256)

    # Supported image file extensions
    image_extensions: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )


class PhysicsUIEBDataModule:
    """
    DataModule that exposes train/val/test DataLoaders with full physics
    priors for P-UWDM conditioning.

    BUG-1 FIX — manifest key mismatch
    ───────────────────────────────────
    The splitter (src.data.splitter) saves integer index lists under the
    keys ``train_indices``, ``val_indices``, and ``test_indices``.  The
    old code read ``manifest["train"]`` / ``"val"`` / ``"test"]``, which
    do not exist in the manifest, causing a ``KeyError``.

    The fixed ``setup()`` method:
      1. Reads ``manifest["train_indices"]`` etc. (integer lists).
      2. Builds a sorted list of all raw/ref image paths from the
         directories (same ordering the splitter saw).
      3. Uses the integer indices to select the correct path subsets.

    Usage
    ──────
    >>> dm = PhysicsUIEBDataModule(cfg)
    >>> dm.setup()
    >>> for batch in dm.train_dataloader():
    ...     raw = batch["raw"]          # (B, 3, 256, 256)
    ...     A   = batch["ambient"]      # (B, 3)
    ...     t   = batch["transmission"] # (B, 1, 256, 256)
    ...     deg = batch["degradation"]  # (B, 6)
    ...     sev = batch["severity"]     # (B, 1)
    """

    def __init__(
        self,
        cfg: Optional[PhysicsDataModuleConfig] = None,
        transform_train: Optional[Callable] = None,
        transform_val: Optional[Callable] = None,
    ) -> None:
        self.cfg = cfg or PhysicsDataModuleConfig()
        self.transform_train = transform_train
        self.transform_val = transform_val

        self._train_ds: Optional[PhysicsUIEBDataset] = None
        self._val_ds: Optional[PhysicsUIEBDataset] = None
        self._test_ds: Optional[PhysicsUIEBDataset] = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Load split manifest and initialise datasets.

        BUG-1 FIX: reads ``train_indices`` / ``val_indices`` /
        ``test_indices`` from the manifest (integer index lists produced
        by src.data.splitter), then selects the corresponding paths from
        a sorted directory listing — matching exactly the ordering the
        splitter used when it created the manifest.
        """
        import json

        manifest_path = Path(self.cfg.split_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {manifest_path}\n"
                "Run `src.data.splitter.create_splits` first to generate it."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        # ── Validate manifest schema ───────────────────────────────
        required_keys = {"train_indices", "val_indices", "test_indices", "n_samples"}
        missing = required_keys - manifest.keys()
        if missing:
            raise KeyError(
                f"Split manifest is missing required keys: {missing}.\n"
                f"Keys present: {list(manifest.keys())}.\n"
                "Regenerate the manifest with src.data.splitter.create_splits()."
            )

        train_indices: List[int] = manifest["train_indices"]
        val_indices: List[int] = manifest["val_indices"]
        test_indices: List[int] = manifest["test_indices"]
        n_manifest: int = manifest["n_samples"]

        # ── Discover all paths (must match the ordering used at split time) ──
        raw_dir = Path(self.cfg.raw_dir)
        ref_dir = Path(self.cfg.ref_dir)

        exts = set(self.cfg.image_extensions)
        all_raw_paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in exts)
        all_ref_paths = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in exts)

        if len(all_raw_paths) != n_manifest:
            raise RuntimeError(
                f"Manifest was created with {n_manifest} samples but the "
                f"raw directory now contains {len(all_raw_paths)} images. "
                "Delete the manifest and regenerate it."
            )
        if len(all_raw_paths) != len(all_ref_paths):
            raise RuntimeError(
                f"raw_dir has {len(all_raw_paths)} images but ref_dir has "
                f"{len(all_ref_paths)}.  Directories must be aligned."
            )

        # ── Select path subsets via integer indices ────────────────
        def _select(indices: List[int], paths: List[Path]) -> List[Path]:
            return [paths[i] for i in indices]

        train_raw = _select(train_indices, all_raw_paths)
        train_ref = _select(train_indices, all_ref_paths)
        val_raw = _select(val_indices, all_raw_paths)
        val_ref = _select(val_indices, all_ref_paths)
        test_raw = _select(test_indices, all_raw_paths)
        test_ref = _select(test_indices, all_ref_paths)

        # ── Build dataset config ───────────────────────────────────
        ds_cfg = self.cfg.dataset_cfg or PhysicsDatasetConfig(
            load_size=self.cfg.load_size
        )

        # ── Instantiate datasets ───────────────────────────────────
        self._train_ds = PhysicsUIEBDataset(
            train_raw, train_ref, transform=self.transform_train, cfg=ds_cfg
        )
        self._val_ds = PhysicsUIEBDataset(
            val_raw, val_ref, transform=self.transform_val, cfg=ds_cfg
        )
        self._test_ds = PhysicsUIEBDataset(
            test_raw, test_ref, transform=self.transform_val, cfg=ds_cfg  # no augment
        )

        logger.info(
            "PhysicsUIEBDataModule ready | train=%d | val=%d | test=%d",
            len(self._train_ds),
            len(self._val_ds),
            len(self._test_ds),
        )

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def _loader(self, dataset: PhysicsUIEBDataset, shuffle: bool) -> DataLoader:
        cfg = self.cfg
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            collate_fn=physics_collate_fn,
            persistent_workers=cfg.num_workers > 0,
            drop_last=shuffle,  # drop last incomplete batch during training
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup() first"
        return self._loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "Call setup() first"
        return self._loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None, "Call setup() first"
        return self._loader(self._test_ds, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _pil_to_tensor(img: Image.Image) -> Tensor:
    """PIL RGB → (3, H, W) float32 [0, 1]."""
    import numpy as np

    arr = np.array(img, dtype=np.float32) / 255.0  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW
