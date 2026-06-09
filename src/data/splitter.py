"""
src/data/splitter.py
--------------------
Deterministic train / val / test index generation.

Responsibilities
----------------
* Accept a total dataset size and split ratios.
* Return three non-overlapping index lists that together cover every
  sample exactly once.
* Guarantee reproducibility via a fixed seed.
* Optionally persist the split manifest (JSON) so experiments can be
  traced and reproduced without re-running the splitter.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config import SplitsConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

SplitIndices = Tuple[List[int], List[int], List[int]]


def create_splits(
    n_samples: int,
    config: SplitsConfig,
    manifest_path: Optional[str | Path] = None,
) -> SplitIndices:
    """
    Randomly shuffle ``range(n_samples)`` and partition into
    ``(train_indices, val_indices, test_indices)``.

    Parameters
    ----------
    n_samples:
        Total number of samples in the dataset.
    config:
        ``SplitsConfig`` containing ratios and seed.
    manifest_path:
        If provided the split is saved as a JSON file at this path,
        enabling future reproducibility checks.

    Returns
    -------
    (train_indices, val_indices, test_indices)
    """
    if n_samples < 3:
        raise ValueError(f"Dataset too small for a 3-way split: {n_samples} samples.")

    indices = list(range(n_samples))
    rng = random.Random(config.seed)
    rng.shuffle(indices)

    n_train = int(n_samples * config.train)
    n_val = int(n_samples * config.val)
    # test gets the remainder to avoid off-by-one from float rounding
    n_test = n_samples - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    logger.info(
        "Split | total=%d | train=%d (%.1f%%) | val=%d (%.1f%%) | test=%d (%.1f%%)",
        n_samples,
        len(train_idx),
        100 * len(train_idx) / n_samples,
        len(val_idx),
        100 * len(val_idx) / n_samples,
        len(test_idx),
        100 * len(test_idx) / n_samples,
    )

    if manifest_path is not None:
        _save_manifest(
            manifest_path,
            n_samples=n_samples,
            seed=config.seed,
            ratios={"train": config.train, "val": config.val, "test": config.test},
            counts={"train": n_train, "val": n_val, "test": n_test},
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )

    return train_idx, val_idx, test_idx


def load_split_manifest(manifest_path: str | Path) -> Dict:
    """Load a previously saved split manifest from disk."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with path.open("r") as fh:
        return json.load(fh)


def splits_from_manifest(
    manifest_path: str | Path,
) -> SplitIndices:
    """
    Reconstruct split index lists from a saved manifest file.
    Useful for resuming training without re-shuffling.
    """
    data = load_split_manifest(manifest_path)
    return (
        data["train_indices"],
        data["val_indices"],
        data["test_indices"],
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_manifest(path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(kwargs, fh, indent=2)
    logger.info("Split manifest saved → %s", path)
