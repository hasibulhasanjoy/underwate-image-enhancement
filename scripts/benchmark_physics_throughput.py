#!/usr/bin/env python3
"""
scripts/benchmark_physics_throughput.py
────────────────────────────────────────────────────────────────────────────
Measures real wall-clock cost of the physics-prior pipeline (ambient light +
transmission map + degradation features) on actual LSUI images, so you can
decide — with real numbers from YOUR server — whether live per-sample
computation (as used today for UIEB) is fine at LSUI's ~7x scale, or whether
it's worth building an offline PhysicsCacheBuilder before starting the long
phase1b run.

This does NOT touch the DataLoader/Dataset machinery at all — it calls the
three estimators directly on real images, so the numbers reflect the actual
CPU cost per sample regardless of how many workers you eventually use.

Usage
─────
    cd ~/underwater_image_enhancement && source venv/bin/activate
    python scripts/benchmark_physics_throughput.py \\
        --lsui_raw_dir dataset/LSUI/input \\
        --lsui_ref_dir dataset/LSUI/GT \\
        --n_samples 200 \\
        --num_workers 16 \\
        --uieb_train_count 623

What to look at
────────────────
The script prints:
  1. Load+resize time per sample (PIL I/O + BICUBIC resize to load_size).
  2. Physics computation time per sample (ambient + transmission + degradation).
  3. Projected single-worker wall-clock for one full epoch over the COMBINED
     UIEB+LSUI train set, divided by --num_workers to estimate the real
     per-epoch CPU-side cost with your actual worker count.
  4. A simple decision heuristic: if the estimated worker-side cost per
     epoch is small relative to typical GPU epoch time (~tens of minutes,
     per your phase3_hist logs), live computation is very likely fine and
     fully hidden behind GPU compute, matching the same finding already
     confirmed for UIEB in physics_dataset.py's own docstring. If it's
     large, an offline PhysicsCacheBuilder becomes worth building.

This script makes no changes to your training pipeline — it's purely a
measurement tool to inform the decision.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import torch

from src.physics import (
    AmbientLightEstimator,
    TransmissionEstimator,
    DegradationEstimator,
)


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark physics-prior throughput on LSUI"
    )
    p.add_argument("--lsui_raw_dir", default="dataset/LSUI/input")
    p.add_argument("--lsui_ref_dir", default="dataset/LSUI/GT")
    p.add_argument(
        "--n_samples", type=int, default=200, help="Random sample count to time"
    )
    p.add_argument(
        "--load_size", type=int, default=256, help="Must match training image_size"
    )
    p.add_argument(
        "--num_workers", type=int, default=16, help="Planned DataLoader num_workers"
    )
    p.add_argument(
        "--uieb_train_count",
        type=int,
        default=623,
        help="UIEB train-split size, for the combined-epoch projection",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    raw_dir = Path(args.lsui_raw_dir)
    ref_dir = Path(args.lsui_ref_dir)
    if not raw_dir.is_dir() or not ref_dir.is_dir():
        raise FileNotFoundError(
            f"LSUI dirs not found: {raw_dir} / {ref_dir}. "
            "Pass --lsui_raw_dir / --lsui_ref_dir explicitly."
        )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    raw_files = {p.name: p for p in raw_dir.iterdir() if p.suffix.lower() in exts}
    ref_files = {p.name: p for p in ref_dir.iterdir() if p.suffix.lower() in exts}
    common = sorted(set(raw_files) & set(ref_files))
    n_lsui_total = len(common)
    if n_lsui_total == 0:
        raise RuntimeError("No matching (input, GT) pairs found — check the paths.")

    sample_names = random.sample(common, min(args.n_samples, n_lsui_total))
    print(f"LSUI pairs found: {n_lsui_total}")
    print(
        f"Benchmarking on {len(sample_names)} random samples "
        f"(load_size={args.load_size}x{args.load_size}) ...\n"
    )

    ambient_est = AmbientLightEstimator()
    transmission_est = TransmissionEstimator()
    degradation_est = DegradationEstimator()

    load_times, physics_times = [], []
    ambient_times, transmission_times, degradation_times = [], [], []

    H = W = args.load_size

    for name in sample_names:
        t0 = time.perf_counter()
        raw_pil = (
            Image.open(raw_files[name]).convert("RGB").resize((W, H), Image.BICUBIC)
        )
        img = _pil_to_tensor01(raw_pil)
        t1 = time.perf_counter()
        load_times.append(t1 - t0)

        ta0 = time.perf_counter()
        ambient = ambient_est(img)
        ta1 = time.perf_counter()
        transmission = transmission_est(img, ambient)
        ta2 = time.perf_counter()
        _ = degradation_est(img)
        ta3 = time.perf_counter()

        ambient_times.append(ta1 - ta0)
        transmission_times.append(ta2 - ta1)
        degradation_times.append(ta3 - ta2)
        physics_times.append(ta3 - ta0)

    def _ms(x: list[float]) -> float:
        return statistics.mean(x) * 1000

    avg_load_ms = _ms(load_times)
    avg_physics_ms = _ms(physics_times)
    avg_total_ms = avg_load_ms + avg_physics_ms

    print("── Per-sample timing (single CPU worker) ──────────────────────")
    print(f"  Load + resize      : {avg_load_ms:8.2f} ms")
    print(f"  Ambient estimation  : {_ms(ambient_times):8.2f} ms")
    print(f"  Transmission map    : {_ms(transmission_times):8.2f} ms")
    print(f"  Degradation features: {_ms(degradation_times):8.2f} ms")
    print(f"  Physics total       : {avg_physics_ms:8.2f} ms")
    print(f"  TOTAL per sample    : {avg_total_ms:8.2f} ms")

    n_combined_train = args.uieb_train_count + n_lsui_total
    single_worker_epoch_s = (avg_total_ms / 1000) * n_combined_train
    projected_epoch_s = single_worker_epoch_s / max(args.num_workers, 1)

    print("\n── Projected combined-epoch CPU cost ───────────────────────────")
    print(
        f"  Combined train set size (UIEB {args.uieb_train_count} + LSUI {n_lsui_total}) "
        f"= {n_combined_train}"
    )
    print(
        f"  Single-worker time for 1 epoch of this pipeline: {single_worker_epoch_s:6.1f} s"
    )
    print(
        f"  With num_workers={args.num_workers}: ~{projected_epoch_s:6.1f} s "
        f"of CPU-worker time per epoch"
    )

    print("\n── Decision guidance ────────────────────────────────────────────")
    if projected_epoch_s < 60:
        print(
            "  Projected worker-side cost is well under a minute per epoch — "
            "live computation should stay fully hidden behind GPU compute "
            "(same finding as UIEB). No caching needed."
        )
    elif projected_epoch_s < 5 * 60:
        print(
            "  Projected worker-side cost is a few minutes per epoch. Likely "
            "still hidden if your GPU epoch time is tens of minutes (as with "
            "phase3_hist), but worth watching actual epoch wall-clock in "
            "training.log during the first few epochs of phase1b."
        )
    else:
        print(
            "  Projected worker-side cost is significant (>5 min/epoch of "
            "CPU-worker time). Given GPU sharing on this server can also "
            "reduce effective throughput, consider building an offline "
            "PhysicsCacheBuilder (precompute ambient/transmission/degradation "
            "once to disk, load from cache in __getitem__) before committing "
            "to the full phase1b run."
        )


if __name__ == "__main__":
    main()
