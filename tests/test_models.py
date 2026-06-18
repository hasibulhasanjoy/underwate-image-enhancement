"""Tests for src/models — SwinUNetDenoiser + component modules.

Run from project root:
    pytest tests/test_models.py -v

Phase 3 changes vs original
----------------------------
- _physics_batch now returns raw image + physics priors (for ConditioningNetworks)
  plus degradation/severity (for the denoiser directly).
- TestEmbeddings.test_conditioning_projection_shape updated: ConditioningProjection
  now accepts (a_embedding, t_embedding, degradation, severity) not raw physics.
- TestSwinUNetDenoiser: ConditioningNetworks is run first to produce a_embedding
  and t_embedding; these are passed to the denoiser. The old **phys unpacking is
  replaced with an explicit two-step forward pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _physics_batch(B: int, H: int = 256, W: int = 256) -> dict:
    """Return all tensors needed for one forward pass through the full model.

    Keys
    ----
    raw        : (B, 3, H, W)   raw degraded image — input to ConditioningNetworks
    physics_A  : (B, 3)         ambient estimate from AmbientEstimator
    physics_t  : (B, 1, H, W)   transmission map from TransmissionEstimator
    degradation: (B, 6)         degradation features — passed directly to denoiser
    severity   : (B, 1)         severity scalar    — passed directly to denoiser
    """
    return dict(
        raw=torch.rand(B, 3, H, W, device=DEVICE),
        physics_A=torch.rand(B, 3, device=DEVICE),
        physics_t=torch.rand(B, 1, H, W, device=DEVICE).clamp(1e-3, 1.0),
        degradation=torch.rand(B, 6, device=DEVICE),
        severity=torch.rand(B, 1, device=DEVICE),
    )


def _run_full_forward(model, cond_nets, x_t, t, phys, *, no_grad=True):
    """Run ConditioningNetworks then SwinUNetDenoiser and return noise prediction.

    This is the canonical two-step forward used by all denoiser tests.
    """

    def _forward():
        cond_out = cond_nets(phys["raw"], phys["physics_A"], phys["physics_t"])
        return model(
            x_t=x_t,
            t=t,
            a_embedding=cond_out["a_embedding"],
            t_embedding=cond_out["t_embedding"],
            degradation=phys["degradation"],
            severity=phys["severity"],
        )

    if no_grad:
        with torch.no_grad():
            return _forward()
    return _forward()


# ---------------------------------------------------------------------------
# AdaGN
# ---------------------------------------------------------------------------


class TestAdaGN:
    def test_spatial_output_shape(self):
        from src.models.adagn import AdaGN

        m = AdaGN(64, cond_dim=128).to(DEVICE)
        x = torch.randn(2, 64, 16, 16, device=DEVICE)
        cond = torch.randn(2, 128, device=DEVICE)
        assert m(x, cond).shape == x.shape

    def test_sequence_output_shape(self):
        from src.models.adagn import AdaGN

        m = AdaGN(64, cond_dim=128).to(DEVICE)
        x = torch.randn(2, 256, 64, device=DEVICE)
        cond = torch.randn(2, 128, device=DEVICE)
        assert m(x, cond).shape == x.shape

    def test_identity_init(self):
        """scale=0, shift=0 at init → output ≈ GroupNorm(x)."""
        from src.models.adagn import AdaGN

        m = AdaGN(32, cond_dim=64).to(DEVICE)
        x = torch.randn(2, 32, 8, 8, device=DEVICE)
        cond = torch.zeros(2, 64, device=DEVICE)
        out = m(x, cond)
        gn = nn.GroupNorm(32, 32, eps=1e-6, affine=False).to(DEVICE)
        assert torch.allclose(out, gn(x), atol=1e-5)


# ---------------------------------------------------------------------------
# MDWA
# ---------------------------------------------------------------------------


class TestMDWAttention:
    def test_output_shape(self):
        from src.models.attention import MDWAttention

        m = MDWAttention(96, window_size=8, num_heads=3).to(DEVICE)
        x = torch.randn(2, 64 * 64, 96, device=DEVICE)
        out = m(x, 64, 64)
        assert out.shape == x.shape

    def test_shifted_output_shape(self):
        from src.models.attention import MDWAttention

        m = MDWAttention(96, window_size=8, num_heads=3, shift=True).to(DEVICE)
        x = torch.randn(2, 64 * 64, 96, device=DEVICE)
        out = m(x, 64, 64)
        assert out.shape == x.shape

    def test_single_scale_fallback(self):
        """window_size=2 → small_ws=1 < 2 → single-scale fallback."""
        from src.models.attention import MDWAttention

        m = MDWAttention(64, window_size=2, num_heads=4).to(DEVICE)
        assert not m.use_multiscale
        x = torch.randn(2, 16 * 16, 64, device=DEVICE)
        out = m(x, 16, 16)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_sinusoidal_shape(self):
        from src.models.embeddings import SinusoidalTimestepEmbedding

        m = SinusoidalTimestepEmbedding(256).to(DEVICE)
        t = torch.randint(0, 1000, (4,), device=DEVICE)
        out = m(t)
        assert out.shape == (4, 256)

    def test_timestep_mlp_shape(self):
        from src.models.embeddings import TimestepMLP, SinusoidalTimestepEmbedding

        sin = SinusoidalTimestepEmbedding(256).to(DEVICE)
        mlp = TimestepMLP(256, out_dim=512).to(DEVICE)
        t = torch.randint(0, 1000, (4,), device=DEVICE)
        out = mlp(sin(t))
        assert out.shape == (4, 512)

    def test_conditioning_projection_shape(self):
        """ConditioningProjection now takes learned embeddings, not raw physics."""
        from src.models.embeddings import ConditioningProjection

        # cond_embed_dim=128 matches default ANet/TNet embed_dim
        m = ConditioningProjection(out_dim=512, cond_embed_dim=128).to(DEVICE)
        a_embedding = torch.rand(4, 128, device=DEVICE)
        t_embedding = torch.rand(4, 128, device=DEVICE)
        degradation = torch.rand(4, 6, device=DEVICE)
        severity = torch.rand(4, 1, device=DEVICE)
        out = m(a_embedding, t_embedding, degradation, severity)
        assert out.shape == (4, 512)

    def test_conditioning_projection_custom_embed_dim(self):
        """cond_embed_dim is configurable and must match ANet/TNet settings."""
        from src.models.embeddings import ConditioningProjection

        m = ConditioningProjection(out_dim=512, cond_embed_dim=64).to(DEVICE)
        a_embedding = torch.rand(2, 64, device=DEVICE)
        t_embedding = torch.rand(2, 64, device=DEVICE)
        degradation = torch.rand(2, 6, device=DEVICE)
        severity = torch.rand(2, 1, device=DEVICE)
        out = m(a_embedding, t_embedding, degradation, severity)
        assert out.shape == (2, 512)


# ---------------------------------------------------------------------------
# ConditioningNetworks (A-Net + T-Net)
# ---------------------------------------------------------------------------


class TestConditioningNetworks:
    """Smoke tests for the Phase 3 conditioning networks in isolation."""

    @pytest.fixture(scope="class")
    def cond_nets(self):
        from src.models.conditioning import ConditioningNetworks

        return ConditioningNetworks(embed_dim=128).to(DEVICE).eval()

    def test_output_keys(self, cond_nets):
        phys = _physics_batch(2)
        with torch.no_grad():
            out = cond_nets(phys["raw"], phys["physics_A"], phys["physics_t"])
        assert set(out.keys()) == {"a_embedding", "refined_map", "t_embedding"}

    def test_output_shapes(self, cond_nets):
        B, H, W = 2, 256, 256
        phys = _physics_batch(B, H, W)
        with torch.no_grad():
            out = cond_nets(phys["raw"], phys["physics_A"], phys["physics_t"])
        assert out["a_embedding"].shape == (B, 128)
        assert out["refined_map"].shape == (B, 1, H, W)
        assert out["t_embedding"].shape == (B, 128)

    def test_refined_map_range(self, cond_nets):
        phys = _physics_batch(2)
        with torch.no_grad():
            out = cond_nets(phys["raw"], phys["physics_A"], phys["physics_t"])
        assert out["refined_map"].min() >= 0.0 - 1e-6
        assert out["refined_map"].max() <= 1.0 + 1e-6

    def test_outputs_finite(self, cond_nets):
        phys = _physics_batch(2)
        with torch.no_grad():
            out = cond_nets(phys["raw"], phys["physics_A"], phys["physics_t"])
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k} contains NaN/Inf"


# ---------------------------------------------------------------------------
# SwinUNetDenoiser
# ---------------------------------------------------------------------------


class TestSwinUNetDenoiser:

    @pytest.fixture(scope="class")
    def model_and_batch(self):
        from src.models.swin_unet import SwinUNetDenoiser, SwinUNetConfig
        from src.models.conditioning import ConditioningNetworks

        cfg = SwinUNetConfig(
            embed_dim=96,
            depths=[1, 1, 2, 1, 1, 2, 1],
            num_heads=[3, 6, 12, 24, 12, 6, 3],
            window_size=8,
            cond_embed_dim=128,  # must match ConditioningNetworks embed_dim
        )
        model = SwinUNetDenoiser(cfg).to(DEVICE).eval()
        cond_nets = ConditioningNetworks(embed_dim=cfg.cond_embed_dim).to(DEVICE).eval()

        B = 2
        x_t = torch.randn(B, 3, 256, 256, device=DEVICE)
        t = torch.randint(0, 1000, (B,), device=DEVICE)
        phys = _physics_batch(B)
        return model, cond_nets, x_t, t, phys

    def test_output_shape(self, model_and_batch):
        model, cond_nets, x_t, t, phys = model_and_batch
        out = _run_full_forward(model, cond_nets, x_t, t, phys)
        assert out.shape == x_t.shape

    def test_output_dtype(self, model_and_batch):
        model, cond_nets, x_t, t, phys = model_and_batch
        out = _run_full_forward(model, cond_nets, x_t, t, phys)
        assert out.dtype == torch.float32

    def test_output_finite(self, model_and_batch):
        model, cond_nets, x_t, t, phys = model_and_batch
        out = _run_full_forward(model, cond_nets, x_t, t, phys)
        assert torch.isfinite(out).all(), "Denoiser output contains NaN/Inf"

    def test_gradient_flow(self, model_and_batch):
        model, cond_nets, x_t, t, phys = model_and_batch
        model.train()
        cond_nets.train()
        out = _run_full_forward(model, cond_nets, x_t, t, phys, no_grad=False)
        loss = out.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients reached denoiser parameters"
        # Also verify gradients flow back through cond_nets
        cond_grads = [p.grad for p in cond_nets.parameters() if p.grad is not None]
        assert (
            len(cond_grads) > 0
        ), "No gradients reached conditioning network parameters"
        model.eval()
        cond_nets.eval()

    def test_batch_size_1(self, model_and_batch):
        model, cond_nets, x_t, t, phys = model_and_batch
        phys1 = {k: v[:1] for k, v in phys.items()}
        out = _run_full_forward(model, cond_nets, x_t[:1], t[:1], phys1)
        assert out.shape == (1, 3, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bfloat16_amp(self, model_and_batch):
        model, cond_nets, x_t, t, phys = model_and_batch
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            cond_out = cond_nets(phys["raw"], phys["physics_A"], phys["physics_t"])
            out = model(
                x_t=x_t,
                t=t,
                a_embedding=cond_out["a_embedding"],
                t_embedding=cond_out["t_embedding"],
                degradation=phys["degradation"],
                severity=phys["severity"],
            )
        assert out.dtype == torch.bfloat16

    def test_different_timesteps_give_different_outputs(self, model_and_batch):
        model, cond_nets, x_t, t, phys = model_and_batch
        t0 = torch.zeros_like(t)
        t999 = torch.full_like(t, 999)
        out0 = _run_full_forward(model, cond_nets, x_t, t0, phys)
        out999 = _run_full_forward(model, cond_nets, x_t, t999, phys)
        assert not torch.allclose(
            out0, out999, atol=1e-4
        ), "Denoiser outputs identical for t=0 and t=999"

    def test_128x128_resolution(self, model_and_batch):
        model, cond_nets, _, t, _ = model_and_batch
        B = 2
        x_sm = torch.randn(B, 3, 128, 128, device=DEVICE)
        phys_sm = _physics_batch(B, 128, 128)
        out = _run_full_forward(model, cond_nets, x_sm, t, phys_sm)
        assert out.shape == x_sm.shape

    def test_num_parameters(self, model_and_batch):
        model, cond_nets, _, _, _ = model_and_batch
        n_denoiser = model.num_parameters()
        n_cond = sum(p.numel() for p in cond_nets.parameters())
        assert (
            n_denoiser > 1_000_000
        ), f"Denoiser has suspiciously few parameters: {n_denoiser:,}"
        assert (
            n_cond > 100_000
        ), f"ConditioningNetworks has suspiciously few parameters: {n_cond:,}"

    def test_cond_embed_dim_mismatch_raises(self):
        """Mismatched cond_embed_dim between config and ConditioningNetworks
        should fail fast with a shape error, not silently produce wrong results."""
        from src.models.swin_unet import SwinUNetDenoiser, SwinUNetConfig
        from src.models.conditioning import ConditioningNetworks

        cfg = SwinUNetConfig(
            embed_dim=96,
            depths=[1, 1, 2, 1, 1, 2, 1],
            num_heads=[3, 6, 12, 24, 12, 6, 3],
            cond_embed_dim=128,
        )
        model = SwinUNetDenoiser(cfg).to(DEVICE).eval()
        # ConditioningNetworks with wrong embed_dim (64 ≠ 128)
        bad_cond = ConditioningNetworks(embed_dim=64).to(DEVICE).eval()
        phys = _physics_batch(1)
        x_t = torch.randn(1, 3, 256, 256, device=DEVICE)
        t = torch.randint(0, 1000, (1,), device=DEVICE)

        with pytest.raises((RuntimeError, AssertionError)):
            with torch.no_grad():
                cond_out = bad_cond(phys["raw"], phys["physics_A"], phys["physics_t"])
                model(
                    x_t=x_t,
                    t=t,
                    a_embedding=cond_out["a_embedding"],  # (B, 64) ← wrong dim
                    t_embedding=cond_out["t_embedding"],  # (B, 64) ← wrong dim
                    degradation=phys["degradation"],
                    severity=phys["severity"],
                )
