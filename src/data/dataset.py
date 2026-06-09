"""
src/data/dataset.py
-------------------
Core PyTorch Dataset for the UIEB underwater image enhancement benchmark.

UIEB structure assumed on disk
-------------------------------
  dataset/UIEB/
    Raw/        ← degraded underwater images  (e.g., ``T001.png``)
    Reference/  ← clean ground-truth images   (e.g., ``T001.png``)

Both directories must contain images with **matching filenames**.

Responsibilities
----------------
* Discover and validate paired (raw, reference) image paths.
* Optionally cache decoded PIL images in RAM to avoid repeated disk I/O.
* Apply a ``PairedTransform`` and return (raw_tensor, reference_tensor, meta).
* Expose statistics helpers (dataset size, per-channel mean/std estimation).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from ..utils.logging import get_logger
from .transforms import PairedTransform

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImagePair = Tuple[Path, Path]  # (raw_path, reference_path)
Sample = Tuple[torch.Tensor, torch.Tensor, Dict]


class UIEBDataset(Dataset):
    """
    PyTorch Dataset for the UIEB paired image enhancement benchmark.

    Parameters
    ----------
    raw_dir:
        Path to the folder containing degraded underwater images.
    reference_dir:
        Path to the folder containing clean reference images.
    transform:
        A ``PairedTransform`` that accepts two PIL images and returns
        two tensors.  If ``None`` the raw PIL images are returned as-is
        (useful for quick inspection).
    image_extensions:
        Accepted file extensions (lower-case, dot-prefixed).
    cache_images:
        If ``True`` the dataset will load every image into RAM on first
        access and return the cached copy on subsequent calls.
        Useful when the dataset fits in memory and CPU I/O is a bottleneck.
    indices:
        Optional integer list to select a subset of the discovered pairs.
        Intended for use by ``UIEBDataModule`` when constructing train /
        val / test splits.
    """

    def __init__(
        self,
        raw_dir: str | Path,
        reference_dir: str | Path,
        transform: Optional[PairedTransform] = None,
        image_extensions: Optional[List[str]] = None,
        cache_images: bool = False,
        indices: Optional[List[int]] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.reference_dir = Path(reference_dir)
        self.transform = transform
        self.cache_images = cache_images

        self.image_extensions = set(
            image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        )

        self._pairs: List[ImagePair] = []
        self._cache: Dict[int, Tuple[Image.Image, Image.Image]] = {}

        self._discover_pairs()

        if indices is not None:
            self._pairs = [self._pairs[i] for i in indices]

        logger.info(
            "UIEBDataset ready | pairs=%d | cache=%s | transform=%s",
            len(self._pairs),
            cache_images,
            type(transform).__name__ if transform else "None",
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_pairs(self) -> None:
        """
        Scan *raw_dir* for valid images, then verify that a matching
        file exists in *reference_dir*.  Logs warnings for unmatched files.
        """
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(
                f"Raw directory not found: {self.raw_dir.resolve()}"
            )
        if not self.reference_dir.is_dir():
            raise FileNotFoundError(
                f"Reference directory not found: {self.reference_dir.resolve()}"
            )

        raw_files: Dict[str, Path] = {
            f.name: f
            for f in sorted(self.raw_dir.iterdir())
            if f.suffix.lower() in self.image_extensions
        }

        ref_files: Dict[str, Path] = {
            f.name: f
            for f in sorted(self.reference_dir.iterdir())
            if f.suffix.lower() in self.image_extensions
        }

        matched = sorted(set(raw_files) & set(ref_files))
        unmatched_raw = sorted(set(raw_files) - set(ref_files))
        unmatched_ref = sorted(set(ref_files) - set(raw_files))

        if unmatched_raw:
            logger.warning(
                "%d raw images have no matching reference: %s …",
                len(unmatched_raw),
                unmatched_raw[:5],
            )
        if unmatched_ref:
            logger.warning(
                "%d reference images have no matching raw: %s …",
                len(unmatched_ref),
                unmatched_ref[:5],
            )
        if not matched:
            raise RuntimeError(
                f"No matched image pairs found between "
                f"'{self.raw_dir}' and '{self.reference_dir}'."
            )

        self._pairs = [(raw_files[n], ref_files[n]) for n in matched]
        logger.debug("Discovered %d valid image pairs.", len(self._pairs))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Sample:
        raw_path, ref_path = self._pairs[index]

        # ── Load (with optional cache) ─────────────────────────────────
        if self.cache_images and index in self._cache:
            raw_img, ref_img = self._cache[index]
        else:
            raw_img = self._load_image(raw_path)
            ref_img = self._load_image(ref_path)
            if self.cache_images:
                self._cache[index] = (raw_img, ref_img)

        # ── Apply transforms ───────────────────────────────────────────
        if self.transform is not None:
            raw_t, ref_t = self.transform(raw_img, ref_img)
        else:
            # Return PIL images wrapped in a dummy "tensor" slot
            raw_t, ref_t = raw_img, ref_img  # type: ignore[assignment]

        # ── Build metadata dict ────────────────────────────────────────
        meta: Dict = {
            "index": index,
            "raw_path": str(raw_path),
            "reference_path": str(ref_path),
            "stem": raw_path.stem,
        }

        return raw_t, ref_t, meta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        """Load a single image as an RGB PIL Image (robust version)."""
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.load()  # force decoding now (not lazy)
                return img

        except Exception as exc:
            raise OSError(f"Corrupted or unreadable image: {path}") from exc

    def get_pair_paths(self, index: int) -> Tuple[Path, Path]:
        """Return the (raw, reference) Path tuple for *index*."""
        return self._pairs[index]

    def dataset_fingerprint(self) -> str:
        """
        Return a short MD5 hash of all file stems — useful for verifying
        that two runs used the same underlying data.
        """
        stems = "|".join(p.stem for p, _ in self._pairs)
        return hashlib.md5(stems.encode()).hexdigest()[:8]

    def __repr__(self) -> str:
        return (
            f"UIEBDataset("
            f"pairs={len(self._pairs)}, "
            f"raw_dir='{self.raw_dir.name}', "
            f"ref_dir='{self.reference_dir.name}', "
            f"cache={self.cache_images})"
        )
