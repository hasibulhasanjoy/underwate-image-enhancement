"""
src/data/datamodule.py
----------------------
High-level orchestrator that wires together dataset discovery,
train/val/test splitting, transforms, and DataLoader creation.

Design pattern
--------------
Inspired by PyTorch Lightning's LightningDataModule interface so the
class can be dropped into a PL training loop without changes, while
remaining usable standalone (no PL dependency required).

Responsibilities
----------------
* Accept a ``DataConfig`` as the single configuration source.
* Build three ``UIEBDataset`` instances (train / val / test) with the
  appropriate transforms and index subsets.
* Expose ``train_dataloader()``, ``val_dataloader()``,
  ``test_dataloader()`` methods returning configured DataLoaders.
* Provide a ``setup()`` hook so heavy work (file discovery, split
  computation) happens exactly once and is logged clearly.
* Persist the split manifest next to the dataset for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from ..utils.config import DataConfig
from ..utils.logging import get_logger
from .dataset import UIEBDataset
from .splitter import create_splits, splits_from_manifest
from .transforms import get_test_transforms, get_train_transforms, get_val_transforms

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_manifest_safe(manifest_path: Path) -> dict | None:
    """Load split manifest JSON; return None on any error."""
    import json

    try:
        with manifest_path.open("r") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read split manifest (%s): %s", manifest_path, exc)
        return None


class UIEBDataModule:
    """
    Self-contained data module for the UIEB dataset.

    Parameters
    ----------
    config:
        Fully populated ``DataConfig`` (loaded from YAML or constructed
        programmatically).
    project_root:
        Absolute path to the project root.  All relative paths in
        *config* are resolved against this directory.
    manifest_path:
        Where to save / load the split manifest JSON.  Defaults to
        ``<project_root>/logs/split_manifest.json``.
    resume_split:
        If ``True`` and *manifest_path* exists, load the split from the
        manifest instead of re-generating it.  Guarantees the exact same
        train/val/test assignment across resumed runs.

    Usage
    -----
    >>> dm = UIEBDataModule(config, project_root=".")
    >>> dm.setup()
    >>> for raw, ref, meta in dm.train_dataloader():
    ...     ...
    """

    def __init__(
        self,
        config: DataConfig,
        project_root: str | Path = ".",
        manifest_path: Optional[str | Path] = None,
        resume_split: bool = True,
    ) -> None:
        self.config = config
        self.root = Path(project_root).resolve()
        self.manifest_path = Path(
            manifest_path or self.root / "logs" / "split_manifest.json"
        )
        self.resume_split = resume_split

        # Built by setup()
        self._train_dataset: Optional[UIEBDataset] = None
        self._val_dataset: Optional[UIEBDataset] = None
        self._test_dataset: Optional[UIEBDataset] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Discover images, compute splits, build datasets.
        Safe to call multiple times (idempotent).
        """
        if self._train_dataset is not None:
            logger.debug("DataModule already set up — skipping.")
            return

        cfg = self.config
        ds_cfg = cfg.dataset

        raw_dir = self.root / ds_cfg.root_dir / ds_cfg.raw_subdir
        ref_dir = self.root / ds_cfg.root_dir / ds_cfg.reference_subdir

        logger.info("Setting up UIEBDataModule …")
        logger.info("  raw_dir       : %s", raw_dir)
        logger.info("  reference_dir : %s", ref_dir)

        # ── Discover ALL pairs (no transforms yet) ─────────────────────
        full_dataset = UIEBDataset(
            raw_dir=raw_dir,
            reference_dir=ref_dir,
            image_extensions=ds_cfg.image_extensions,
            cache_images=False,  # discovery only
        )
        n = len(full_dataset)
        fingerprint = full_dataset.dataset_fingerprint()
        logger.info("Dataset fingerprint : %s  (%d pairs)", fingerprint, n)

        # ── Compute or reload split ────────────────────────────────────
        if self.resume_split and self.manifest_path.is_file():
            manifest = _load_manifest_safe(self.manifest_path)
            if manifest is not None and manifest.get("n_samples") == n:
                logger.info("Resuming split from manifest: %s", self.manifest_path)
                train_idx = manifest["train_indices"]
                val_idx = manifest["val_indices"]
                test_idx = manifest["test_indices"]
            else:
                logger.warning(
                    "Manifest n_samples=%s does not match current dataset n=%d "
                    "-- recomputing split.",
                    manifest.get("n_samples") if manifest else "?",
                    n,
                )
                train_idx, val_idx, test_idx = create_splits(
                    n_samples=n,
                    config=cfg.splits,
                    manifest_path=self.manifest_path,
                )
        else:
            train_idx, val_idx, test_idx = create_splits(
                n_samples=n,
                config=cfg.splits,
                manifest_path=self.manifest_path,
            )

        # ── Build typed sub-datasets ───────────────────────────────────
        cache = cfg.cache.enabled

        self._train_dataset = UIEBDataset(
            raw_dir=raw_dir,
            reference_dir=ref_dir,
            transform=get_train_transforms(cfg),
            image_extensions=ds_cfg.image_extensions,
            cache_images=cache,
            indices=train_idx,
        )
        self._val_dataset = UIEBDataset(
            raw_dir=raw_dir,
            reference_dir=ref_dir,
            transform=get_val_transforms(cfg),
            image_extensions=ds_cfg.image_extensions,
            cache_images=cache,
            indices=val_idx,
        )
        self._test_dataset = UIEBDataset(
            raw_dir=raw_dir,
            reference_dir=ref_dir,
            transform=get_test_transforms(cfg),
            image_extensions=ds_cfg.image_extensions,
            cache_images=cache,
            indices=test_idx,
        )

        logger.info(
            "DataModule ready | train=%d | val=%d | test=%d",
            len(self._train_dataset),
            len(self._val_dataset),
            len(self._test_dataset),
        )

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def _check_setup(self) -> None:
        if self._train_dataset is None:
            raise RuntimeError(
                "Call UIEBDataModule.setup() before requesting DataLoaders."
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader (shuffled)."""
        self._check_setup()
        ldr = self.config.loader
        return DataLoader(
            self._train_dataset,  # type: ignore[arg-type]
            batch_size=ldr.batch_size,
            shuffle=True,
            num_workers=ldr.num_workers,
            pin_memory=ldr.pin_memory,
            persistent_workers=ldr.persistent_workers and ldr.num_workers > 0,
            prefetch_factor=ldr.prefetch_factor if ldr.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (deterministic order)."""
        self._check_setup()
        ldr = self.config.loader
        return DataLoader(
            self._val_dataset,  # type: ignore[arg-type]
            batch_size=ldr.batch_size,
            shuffle=False,
            num_workers=ldr.num_workers,
            pin_memory=ldr.pin_memory,
            persistent_workers=ldr.persistent_workers and ldr.num_workers > 0,
            prefetch_factor=ldr.prefetch_factor if ldr.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader (deterministic order)."""
        self._check_setup()
        ldr = self.config.loader
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=ldr.batch_size,
            shuffle=False,
            num_workers=ldr.num_workers,
            pin_memory=ldr.pin_memory,
            persistent_workers=ldr.persistent_workers and ldr.num_workers > 0,
            prefetch_factor=ldr.prefetch_factor if ldr.num_workers > 0 else None,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def train_dataset(self) -> UIEBDataset:
        self._check_setup()
        return self._train_dataset  # type: ignore[return-value]

    @property
    def val_dataset(self) -> UIEBDataset:
        self._check_setup()
        return self._val_dataset  # type: ignore[return-value]

    @property
    def test_dataset(self) -> UIEBDataset:
        self._check_setup()
        return self._test_dataset  # type: ignore[return-value]

    def __repr__(self) -> str:
        if self._train_dataset is None:
            return "UIEBDataModule(not set up)"
        return (
            f"UIEBDataModule("
            f"train={len(self._train_dataset)}, "
            f"val={len(self._val_dataset)}, "  # type: ignore[arg-type]
            f"test={len(self._test_dataset)})"  # type: ignore[arg-type]
        )
