"""
src/data/physics_dataset.py
────────────────────────────────────────────────────────────────────────────
Physics-aware UIEB dataset with integrated prior estimation.

Each sample returned by __getitem__ contains:
  raw         : Tensor (3,  H, W)  — raw underwater image
  reference   : Tensor (3,  H, W)  — clean reference image
  ambient     : Tensor (3,)        — estimated ambient light A
  transmission: Tensor (1,  H, W)  — estimated transmission map t(x)
  degradation : Tensor (6,)        — degradation feature vector
  severity    : Tensor (1,)        — scalar degradation severity

Integration with the diffusion model
──────────────────────────────────────
At model input time:
  • A-Net  receives `raw` + `ambient` broadcast to (3+3, H, W) or via
    cross-attention on the (3,) vector.
  • T-Net  receives `raw` + `transmission` → (3+1, H, W).
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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
# Sample type
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PhysicsSample:
    """
    A single P-UWDM training/validation/test sample.

    All tensors are float32. Spatial tensors have shape (C, H, W).
    """

    raw: Tensor  # (3, H, W)  — raw underwater image
    reference: Tensor  # (3, H, W)  — ground-truth clean image
    ambient: Tensor  # (3,)       — ambient light A
    transmission: Tensor  # (1, H, W)  — transmission map t(x)
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
    # Recommendation: True — priors should match the augmented image.
    physics_on_augmented: bool = True


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
        returns (raw_tensor, ref_tensor) in CHW float32 [0,1].
    cfg : PhysicsDatasetConfig

    Notes
    ─────
    Physics estimators are instantiated *once per worker* via a lazy
    property.  This avoids pickling overhead (estimators hold no state
    that changes per-sample).
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
            "PhysicsUIEBDataset: %d samples, load_size=%s, " "physics_on_augmented=%s",
            len(self),
            self.cfg.load_size,
            self.cfg.physics_on_augmented,
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

        # ── PIL → Tensor (before optional augmentation) ─────────────
        if self.transform is not None:
            raw_t, ref_t = self.transform(raw_pil, ref_pil)
        else:
            raw_t = _pil_to_tensor(raw_pil)
            ref_t = _pil_to_tensor(ref_pil)

        # ── Physics priors ───────────────────────────────────────────
        self._init_estimators()

        # Use augmented tensor if physics_on_augmented=True
        physics_src = (
            raw_t if self.cfg.physics_on_augmented else _pil_to_tensor(raw_pil)
        )

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
    raw_dir: str = "data/UIEB/raw"
    ref_dir: str = "data/UIEB/reference"

    # Split manifest JSON produced by uie.data.splitter
    split_manifest: str = "data/UIEB/split_manifest.json"

    # DataLoader settings — tuned for RTX 4090 / Ryzen 9 7950X
    batch_size: int = 32
    num_workers: int = 16  # matches physical core count
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Dataset config
    dataset_cfg: Optional[PhysicsDatasetConfig] = None

    # Image size
    load_size: Tuple[int, int] = (256, 256)


class PhysicsUIEBDataModule:
    """
    DataModule that exposes train/val/test DataLoaders with full physics
    priors for P-UWDM conditioning.

    Usage
    ──────
    >>> dm = PhysicsUIEBDataModule(cfg)
    >>> dm.setup()
    >>> for batch in dm.train_dataloader():
    ...     raw  = batch["raw"]           # (B, 3, 256, 256)
    ...     A    = batch["ambient"]       # (B, 3)
    ...     t    = batch["transmission"]  # (B, 1, 256, 256)
    ...     deg  = batch["degradation"]   # (B, 6)
    ...     sev  = batch["severity"]      # (B, 1)
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

    def setup(self) -> None:
        """Load split manifest and initialise datasets."""
        import json

        manifest_path = Path(self.cfg.split_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {manifest_path}\n"
                "Run `uie.data.splitter` first to generate it."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        raw_dir = Path(self.cfg.raw_dir)
        ref_dir = Path(self.cfg.ref_dir)

        ds_cfg = self.cfg.dataset_cfg or PhysicsDatasetConfig(
            load_size=self.cfg.load_size
        )

        def _make_paths(names: List[str]):
            raw_paths = [raw_dir / n for n in names]
            ref_paths = [ref_dir / n for n in names]
            return raw_paths, ref_paths

        train_raw, train_ref = _make_paths(manifest["train"])
        val_raw, val_ref = _make_paths(manifest["val"])
        test_raw, test_ref = _make_paths(manifest["test"])

        self._train_ds = PhysicsUIEBDataset(
            train_raw,
            train_ref,
            transform=self.transform_train,
            cfg=ds_cfg,
        )
        self._val_ds = PhysicsUIEBDataset(
            val_raw,
            val_ref,
            transform=self.transform_val,
            cfg=ds_cfg,
        )
        self._test_ds = PhysicsUIEBDataset(
            test_raw,
            test_ref,
            transform=self.transform_val,  # same as val (no augment)
            cfg=ds_cfg,
        )

        logger.info(
            "PhysicsUIEBDataModule: train=%d, val=%d, test=%d",
            len(self._train_ds),
            len(self._val_ds),
            len(self._test_ds),
        )

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
