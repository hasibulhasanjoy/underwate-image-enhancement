"""
scripts/verify_physics.py
────────────────────────────────────────────────────────────────────────────
Smoke-test for the physics-prior pipeline (no real dataset required).

Run from project root:
    python scripts/verify_physics.py

What this tests:
  1. AmbientLightEstimator on a synthetic underwater-like image
  2. TransmissionEstimator on the same image
  3. DegradationEstimator feature shape and value ranges
  4. PhysicsUIEBDataset.__getitem__ via synthetic in-memory paths
  5. DataLoader iteration with physics_collate_fn
  6. Output tensor shapes for all P-UWDM conditioning signals

Expected output: all checks PASS with no assertions raised.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make sure uie is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics import (
    AmbientConfig,
    AmbientLightEstimator,
    DegradationConfig,
    DegradationEstimator,
    TransmissionConfig,
    TransmissionEstimator,
)
from src.data.physics_dataset import (
    PhysicsDatasetConfig,
    PhysicsUIEBDataset,
    physics_collate_fn,
)
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")
    sys.exit(1)


def section(title: str) -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetic test image
# ──────────────────────────────────────────────────────────────────────────────


def make_synthetic_underwater(H: int = 256, W: int = 256) -> torch.Tensor:
    """
    Simulate an underwater-degraded image:
    - Blue-green colour cast
    - Reduced contrast
    - Slight blur (simulate scattering)
    """
    rng = torch.Generator().manual_seed(42)

    # Base scene: textured gradient
    x = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    y = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    base = 0.5 * x + 0.3 * y + 0.1 * torch.rand(H, W, generator=rng)

    # Channels: R strongly attenuated, G moderate, B elevated
    R = base * 0.4
    G = base * 0.65
    B = base * 0.85 + 0.10

    img = torch.stack([R, G, B]).clamp(0, 1)  # (3, H, W)

    # Mild Gaussian-like blur via depthwise avg_pool
    import torch.nn.functional as F

    img = F.avg_pool2d(img.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    return img.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_ambient() -> torch.Tensor:
    section("1. Ambient Light Estimator")
    img = make_synthetic_underwater()

    t0 = time.perf_counter()
    est = AmbientLightEstimator(AmbientConfig(blue_correction=0.05))
    A = est(img)
    dt = (time.perf_counter() - t0) * 1000

    if A.shape != (3,):
        fail(f"Expected shape (3,), got {tuple(A.shape)}")
    ok(f"Shape: {tuple(A.shape)}")

    if not (0.0 <= A.min() and A.max() <= 1.0):
        fail(f"Values out of [0,1]: min={A.min():.4f} max={A.max():.4f}")
    ok(f"Value range: [{A.min():.4f}, {A.max():.4f}]")

    # For underwater: Blue > Green > Red (expected cast)
    if not (A[2] > A[0]):
        fail(f"Expected Blue > Red for underwater; got R={A[0]:.3f} B={A[2]:.3f}")
    ok(f"Colour order (B>R expected): R={A[0]:.4f} G={A[1]:.4f} B={A[2]:.4f}")
    ok(f"Computation time: {dt:.1f}ms")
    return A


def test_transmission(A: torch.Tensor) -> None:
    section("2. Transmission Map Estimator")
    img = make_synthetic_underwater()

    t0 = time.perf_counter()
    est = TransmissionEstimator(TransmissionConfig(t_min=0.10))
    t_map = est(img, A)
    dt = (time.perf_counter() - t0) * 1000

    if t_map.shape != (1, 256, 256):
        fail(f"Expected (1,256,256), got {tuple(t_map.shape)}")
    ok(f"Shape: {tuple(t_map.shape)}")

    lo, hi = t_map.min().item(), t_map.max().item()
    if not (0.10 <= lo and hi <= 1.0):
        fail(f"Values outside [t_min=0.1, 1.0]: min={lo:.4f} max={hi:.4f}")
    ok(f"Value range: [{lo:.4f}, {hi:.4f}]")
    ok(f"Computation time: {dt:.1f}ms")


def test_degradation() -> None:
    section("3. Degradation Estimator")
    img = make_synthetic_underwater()

    t0 = time.perf_counter()
    est = DegradationEstimator(DegradationConfig())
    deg = est(img)
    dt = (time.perf_counter() - t0) * 1000

    # feature_vec shape
    if deg.feature_vec.shape != (6,):
        fail(f"feature_vec: expected (6,), got {tuple(deg.feature_vec.shape)}")
    ok(f"feature_vec shape: {tuple(deg.feature_vec.shape)}")

    # severity shape & range
    if deg.severity.shape != (1,):
        fail(f"severity: expected (1,), got {tuple(deg.severity.shape)}")
    if not (0.0 <= deg.severity[0] <= 1.0):
        fail(f"severity out of [0,1]: {deg.severity[0]:.4f}")
    ok(f"severity: {deg.severity[0]:.4f} ∈ [0,1] ✓")

    # Sub-scores
    ok(f"colour_cast: {deg.colour_cast.tolist()} (R↓ G- B↑ expected)")
    ok(f"contrast:    {deg.contrast[0]:.4f}")
    ok(f"blur:        {deg.blur[0]:.4f}")
    ok(f"noise:       {deg.noise[0]:.4f}")
    ok(f"Computation time: {dt:.1f}ms")


def test_dataloader_integration() -> None:
    section("4. DataLoader Integration (synthetic images on disk)")

    # Create temp directory with fake PNG images
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_dir = tmp / "raw"
        ref_dir = tmp / "reference"
        raw_dir.mkdir()
        ref_dir.mkdir()

        # Save 8 synthetic image pairs
        n_samples = 8
        for i in range(n_samples):
            raw_t = make_synthetic_underwater(128, 128)
            ref_t = torch.rand(3, 128, 128)

            def to_pil(t: torch.Tensor) -> Image.Image:
                arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(arr)

            to_pil(raw_t).save(raw_dir / f"{i:04d}.png")
            to_pil(ref_t).save(ref_dir / f"{i:04d}.png")

        raw_paths = sorted(raw_dir.glob("*.png"))
        ref_paths = sorted(ref_dir.glob("*.png"))

        ds_cfg = PhysicsDatasetConfig(load_size=(128, 128))
        dataset = PhysicsUIEBDataset(raw_paths, ref_paths, cfg=ds_cfg)

        if len(dataset) != n_samples:
            fail(f"Dataset length: expected {n_samples}, got {len(dataset)}")
        ok(f"Dataset length: {len(dataset)}")

        # Single sample
        sample = dataset[0]
        checks = {
            "raw": ((3, 128, 128), sample.raw.shape),
            "reference": ((3, 128, 128), sample.reference.shape),
            "ambient": ((3,), sample.ambient.shape),
            "transmission": ((1, 128, 128), sample.transmission.shape),
            "degradation": ((6,), sample.degradation.shape),
            "severity": ((1,), sample.severity.shape),
        }
        for name, (expected, actual) in checks.items():
            if tuple(actual) != expected:
                fail(f"{name}: expected {expected}, got {tuple(actual)}")
            ok(f"sample['{name}']: shape={tuple(actual)} ✓")

        # DataLoader with collate
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # 0 for test (no forking in temp dir)
            collate_fn=physics_collate_fn,
        )

        t0 = time.perf_counter()
        batch = next(iter(loader))
        dt = (time.perf_counter() - t0) * 1000

        B = 4
        batch_checks = {
            "raw": (B, 3, 128, 128),
            "reference": (B, 3, 128, 128),
            "ambient": (B, 3),
            "transmission": (B, 1, 128, 128),
            "degradation": (B, 6),
            "severity": (B, 1),
        }
        for key, expected in batch_checks.items():
            actual = tuple(batch[key].shape)
            if actual != expected:
                fail(f"batch['{key}']: expected {expected}, got {actual}")
            ok(f"batch['{key}']: shape={actual} ✓")

        ok(f"DataLoader iteration time (batch=4): {dt:.1f}ms")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{BOLD}P-UWDM Physics Prior Verification{RESET}")
    print(f"torch {torch.__version__} | device: CPU (workers)")

    A = test_ambient()
    test_transmission(A)
    test_degradation()
    test_dataloader_integration()

    print(f"\n{BOLD}{GREEN}All checks passed.{RESET}\n")
