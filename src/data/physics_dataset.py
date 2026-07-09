"""
src/data/physics_dataset.py
────────────────────────────────────────────────────────────────────────────
Physics-aware UIEB(+LSUI) dataset with integrated prior estimation.

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

This module computes all three priors on-the-fly in DataLoader workers.
On UIEB (~900 images) each sample takes ~10ms on CPU workers and this
overhead is fully hidden behind GPU compute with 16 workers. For LSUI-scale
data (~4.3k pairs) the same per-sample cost applies — see
scripts/benchmark_physics_throughput.py to confirm wall-clock behaviour on
your actual hardware before committing to a long run. If it turns out to
be a bottleneck, an offline PhysicsCacheBuilder (precompute once to disk)
is the natural next step — not yet implemented, since live computation is
expected to remain hidden behind GPU compute.

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
    DegradationEstimator) expect pixel values in [0, 1]. ``_denorm_for_physics()``
    reverses ImageNet normalisation before calling the estimators, gated by
    ``cfg.imagenet_normalised``.

BUG-3 (PhysicsDatasetConfig.imagenet_normalised default — FIXED THIS ROUND):
    The dataclass default was ``imagenet_normalised=True``, but the actual
    training pipeline (trainer.py._build_data) has NEVER applied real
    ImageNet normalisation — either no transform was passed at all (plain
    [0,1] tensors), or (as of this round) an explicit identity-normalize
    augmentation pipeline is used (flip/rotation/color-jitter, but
    mean=[0,0,0]/std=[1,1,1] so pixel values stay in [0,1] — required for
    compatibility with the diffusion model's [0,1]/clip_denoised=True
    assumptions). With the old default, ``_denorm_for_physics()`` was being
    called on already-[0,1] pixels, compressing/shifting them into roughly
    [0.485,0.714] (R) / [0.456,0.68] (G) / [0.406,0.631] (B) before the
    physics estimators ever saw them — corrupting every ambient/transmission
    /degradation prior used for conditioning in every run to date (phase1
    through phase3_hist).
    Fixed: default changed to ``imagenet_normalised=False``. Callers that
    genuinely do apply real ImageNet normalisation in their transform must
    now explicitly pass ``imagenet_normalised=True``.
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

    Only call this when the tensor passed in was ACTUALLY ImageNet-normalised
    by the transform (cfg.imagenet_normalised=True). Calling it on plain
    [0,1] data silently corrupts the physics priors — see BUG-3 above.

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

    # Whether the transform applies REAL ImageNet normalisation (mean/std
    # that actually shift data out of [0,1]). Set True ONLY if your
    # transform truly normalises with non-identity mean/std.
    #
    # FIXED THIS ROUND (BUG-3): default changed from True → False. The
    # training pipeline has never applied real ImageNet normalisation (see
    # module docstring). The old True default caused _denorm_for_physics()
    # to be wrongly applied to already-[0,1] pixels, corrupting every
    # physics prior computed in phases 1 through 3_hist.
    imagenet_normalised: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class PhysicsUIEBDataset(Dataset):
    """
    Physics-aware paired-image dataset. Despite the name (kept for backward
    compatibility), this class is dataset-agnostic — it just takes lists of
    (raw_path, ref_path) pairs, so the same class serves UIEB, LSUI, or a
    combined list of both (see PhysicsUIEBDataModule.setup below).

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
        If the transform applies REAL ImageNet normalisation, set
        ``cfg.imagenet_normalised = True``. Default is False (identity
        normalize / plain [0,1] — see BUG-3 in module docstring).
    cfg : PhysicsDatasetConfig

    Notes
    ─────
    Physics estimators are instantiated *once per worker* via a lazy
    property.  This avoids pickling overhead (estimators hold no state
    that changes per-sample).

    Images of any native resolution are accepted — they are resized to
    ``cfg.load_size`` with BICUBIC interpolation before any transform is
    applied, so variable-resolution sources (e.g. LSUI) work unmodified.
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
        # Works for any native resolution — LSUI images (variable, e.g.
        # 640x320, 720x405, 1280x1024) are resized identically to UIEB's
        # already-256x256 images.
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

        # Denormalise ONLY if the transform genuinely applied real ImageNet
        # normalisation (cfg.imagenet_normalised=True). Default is False —
        # see BUG-3 in module docstring for why this must not be applied
        # to already-[0,1] data.
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
# LSUI pair discovery (NEW — for the combined UIEB+LSUI training phase)
# ──────────────────────────────────────────────────────────────────────────────


def _discover_lsui_pairs(
    lsui_raw_dir: Path,
    lsui_ref_dir: Path,
    exts: set,
) -> Tuple[List[Path], List[Path]]:
    """
    Discover paired (input, GT) images in an LSUI-style directory layout,
    where both directories contain files with IDENTICAL filenames
    (e.g. "0.jpg" in both input/ and GT/).

    Unlike UIEB's manifest+sorted-index approach, LSUI has no split
    manifest — pairing is done by filename intersection, sorted for
    determinism. Any filename present in only one of the two directories
    is dropped and logged as a warning (defensive; the known LSUI dataset
    has been verified to match 1:1, but this guards against a partial
    download or future dataset variant).

    Returns
    -------
    (raw_paths, ref_paths) — same length, same order, sorted by filename.
    """
    raw_files = {p.name: p for p in lsui_raw_dir.iterdir() if p.suffix.lower() in exts}
    ref_files = {p.name: p for p in lsui_ref_dir.iterdir() if p.suffix.lower() in exts}

    common_names = sorted(set(raw_files) & set(ref_files))
    missing_ref = set(raw_files) - set(ref_files)
    missing_raw = set(ref_files) - set(raw_files)

    if missing_ref or missing_raw:
        logger.warning(
            "LSUI pairing mismatch: %d input files with no GT match, "
            "%d GT files with no input match. These are DROPPED. "
            "First few missing (input-only): %s | (GT-only): %s",
            len(missing_ref),
            len(missing_raw),
            sorted(missing_ref)[:5],
            sorted(missing_raw)[:5],
        )

    raw_paths = [raw_files[name] for name in common_names]
    ref_paths = [ref_files[name] for name in common_names]
    return raw_paths, ref_paths


# ──────────────────────────────────────────────────────────────────────────────
# DataModule (Lightning-style, manually managed)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PhysicsDataModuleConfig:
    """Configuration for PhysicsUIEBDataModule."""

    # Dataset paths (UIEB — the fixed thesis benchmark)
    raw_dir: str = "dataset/UIEB/raw"
    ref_dir: str = "dataset/UIEB/reference"

    # Split manifest JSON produced by src.data.splitter
    # Must contain keys: train_indices, val_indices, test_indices, n_samples
    split_manifest: str = "dataset/UIEB/split_manifest.json"

    # ── LSUI augmentation dataset (NEW, optional) ──────────────────────
    # When enabled, LSUI pairs are appended to the TRAIN split ONLY.
    # UIEB val/test splits (the fixed 134-image thesis benchmark) are
    # NEVER touched by this — LSUI never enters val/test evaluation.
    use_lsui: bool = False
    lsui_raw_dir: Optional[str] = None  # e.g. "dataset/LSUI/input"
    lsui_ref_dir: Optional[str] = None  # e.g. "dataset/LSUI/GT"

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

    LSUI integration (NEW)
    ───────────────────────
    When ``cfg.use_lsui=True``, ``setup()`` additionally discovers all
    (input, GT) pairs under ``cfg.lsui_raw_dir`` / ``cfg.lsui_ref_dir``
    (paired by identical filename — see ``_discover_lsui_pairs``) and
    appends them to the TRAIN path lists only, after the UIEB train
    subset. Val and test datasets remain pure UIEB, built exactly as
    before from the split manifest — this preserves the fixed 134-image
    thesis benchmark untouched by the new data.

    BUG-1 FIX — manifest key mismatch
    ───────────────────────────────────
    The splitter (src.data.splitter) saves integer index lists under the
    keys ``train_indices``, ``val_indices``, and ``test_indices``. The
    fixed ``setup()`` method:
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
        Load split manifest and initialise datasets. If ``cfg.use_lsui``
        is set, LSUI pairs are appended to the TRAIN dataset only.
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

        # ── Discover all UIEB paths (must match ordering used at split time) ──
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

        n_uieb_train = len(train_raw)

        # ── NEW: append LSUI pairs to TRAIN only ────────────────────
        if self.cfg.use_lsui:
            if not self.cfg.lsui_raw_dir or not self.cfg.lsui_ref_dir:
                raise ValueError(
                    "cfg.use_lsui=True requires both lsui_raw_dir and "
                    "lsui_ref_dir to be set."
                )
            lsui_raw, lsui_ref = _discover_lsui_pairs(
                Path(self.cfg.lsui_raw_dir), Path(self.cfg.lsui_ref_dir), exts
            )
            train_raw = train_raw + lsui_raw
            train_ref = train_ref + lsui_ref
            logger.info(
                "LSUI enabled: +%d paired images appended to TRAIN only "
                "(UIEB train=%d -> combined train=%d; UIEB val=%d / test=%d unaffected)",
                len(lsui_raw),
                n_uieb_train,
                len(train_raw),
                len(val_indices),
                len(test_indices),
            )

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
