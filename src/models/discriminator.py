"""
src/models/discriminator.py
────────────────────────────────────────────────────────────────────────────
PatchGAN discriminator for P-UWDM adversarial training.

This module re-exports ``PatchDiscriminator`` from ``src.losses.adversarial``
so that the models package can import it cleanly:

    from src.models.discriminator import PatchDiscriminator

Rationale
─────────
The discriminator is architecturally part of the model family (it runs a
forward pass over image tensors), but it participates in the loss computation
and receives its own optimizer.  To avoid a circular import between
``src.models`` and ``src.losses``, the canonical implementation lives in
``src.losses.adversarial`` and this shim re-exports it.

If you need a standalone discriminator with a custom architecture, subclass
``PatchDiscriminator`` here and override ``__init__`` / ``forward``.

Public API
──────────
PatchDiscriminator   — 70×70 PatchGAN conditioned on the degraded input.
build_discriminator  — factory that respects the SwinUNetConfig channel budget.
"""

from __future__ import annotations

# Re-export the canonical implementation from the loss package
from src.losses.adversarial import PatchDiscriminator  # noqa: F401


def build_discriminator(
    in_channels: int = 3,
    base_channels: int = 64,
) -> PatchDiscriminator:
    """
    Factory for constructing the P-UWDM PatchGAN discriminator.

    Parameters
    ----------
    in_channels : int
        Number of channels in each input image (raw + enhanced are
        concatenated, so the discriminator actually receives 2 * in_channels).
        Default 3 (RGB).
    base_channels : int
        Width of the first discriminator layer.  The network scales as
        [base, 2×base, 4×base, 8×base, 1].  Default 64 ≈ 2.8 M params.

    Returns
    -------
    PatchDiscriminator
    """
    return PatchDiscriminator(in_channels=in_channels, base_channels=base_channels)
