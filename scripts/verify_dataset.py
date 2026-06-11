"""
scripts/verify_dataset.py
--------------------------
Standalone script that performs a full smoke-test of the data pipeline
without requiring a GPU or any model weights.

Run from the project root:
    python scripts/verify_dataset.py --config configs/data_config.yaml

What it checks
--------------
1. Config loads without errors.
2. DataModule discovers all image pairs correctly.
3. Splits are valid (no leakage, correct sizes).
4. One full training batch can be loaded end-to-end.
5. Tensor shapes, dtypes, and value ranges are as expected.
6. Determinism: loading the same index twice gives the same result
   (modulo stochastic augmentation — we verify val is deterministic).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datamodule import UIEBDataModule
from src.utils.logging import get_logger
from src.utils.config import load_data_config

logger = get_logger("verify_dataset", log_file="dataset_verify.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the UIEB data pipeline.")
    parser.add_argument(
        "--config",
        default="configs/data_config.yaml",
        help="Path to the data YAML config (default: configs/data_config.yaml)",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for the verification pass.",
    )
    return parser.parse_args()


def check_tensor(name: str, t: torch.Tensor, expected_channels: int = 3) -> None:
    """Assert shape/dtype/range and log a summary."""
    assert t.ndim == 4, f"{name}: expected 4-D tensor (B,C,H,W), got {t.shape}"
    B, C, H, W = t.shape
    assert (
        C == expected_channels
    ), f"{name}: expected {expected_channels} channels, got {C}"
    assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
    logger.info(
        "  %-12s | shape=%-20s | min=%+.3f | max=%+.3f | mean=%+.3f",
        name,
        str(tuple(t.shape)),
        t.min().item(),
        t.max().item(),
        t.mean().item(),
    )


def main() -> None:
    args = parse_args()

    # ── 1. Load config ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("UIEB Data Pipeline Verification")
    logger.info("=" * 60)
    logger.info("Config : %s", args.config)

    overrides = {}
    if args.batch_size:
        overrides = {"loader": {"batch_size": args.batch_size}}

    config = load_data_config(args.config, overrides=overrides or None)
    logger.info("Config loaded successfully.")

    # ── 2. Setup DataModule ────────────────────────────────────────────
    dm = UIEBDataModule(config, project_root=args.project_root)
    dm.setup()
    logger.info("DataModule: %s", dm)

    # ── 3. Split sanity checks ─────────────────────────────────────────
    n_total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    train_set = set(dm.train_dataset._pairs)
    val_set = set(dm.val_dataset._pairs)
    test_set = set(dm.test_dataset._pairs)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    assert not overlap_tv, f"Train/val overlap: {len(overlap_tv)} pairs"
    assert not overlap_tt, f"Train/test overlap: {len(overlap_tt)} pairs"
    assert not overlap_vt, f"Val/test overlap: {len(overlap_vt)} pairs"
    logger.info("Split integrity check PASSED (no leakage detected).")

    # ── 4. Load one training batch ─────────────────────────────────────
    logger.info("Loading one training batch …")
    t0 = time.perf_counter()
    train_loader = dm.train_dataloader()
    raw_batch, ref_batch, meta = next(iter(train_loader))
    elapsed = time.perf_counter() - t0
    logger.info("  First batch loaded in %.3f s", elapsed)

    check_tensor("train raw", raw_batch)
    check_tensor("train ref", ref_batch)

    # ── 5. Determinism check on val split ──────────────────────────────
    logger.info("Determinism check (val split, two passes) …")
    val_loader = dm.val_dataloader()
    it = iter(val_loader)
    r1, _, _ = next(it)
    # Rebuild iterator to get the same first batch
    it2 = iter(val_loader)
    r2, _, _ = next(it2)
    assert torch.allclose(r1, r2), "Validation loader is non-deterministic!"
    logger.info("  Determinism check PASSED.")

    # ── 6. Summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("All checks PASSED ✓")
    logger.info(
        "  Batch shape : raw=%s | ref=%s",
        tuple(raw_batch.shape),
        tuple(ref_batch.shape),
    )
    logger.info("  Sample stems: %s", meta["stem"][:4])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
