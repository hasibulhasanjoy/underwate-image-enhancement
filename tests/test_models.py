"""Tests for src/models — SwinUNetDenoiser + component modules.

Run from project root:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest  # type: ignore[import-not-found]
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def _physics_batch(B: int, H: int = 256, W: int = 256):
    """Return a dict of physics conditioning tensors."""
    return dict(
        ambient=torch.rand(B, 3, device=DEVICE),
        transmission=torch.rand(B, 1, H, W, device=DEVICE),
        degradation=torch.rand(B, 6, device=DEVICE),
        severity=torch.rand(B, 1, device=DEVICE),
    )


# ---------------------------------------------------------------------------
# AdaGN
# ---------------------------------------------------------------------------


class TestAdaGN:
    def test_spatial_output_shape(self):
        from src.models.adagn import AdaGN

        m = AdaGN(64, cond_dim=128).to(DEVICE)
        x = torch.randn(2, 64, 16, 16, device=DEVICE)
        cond = torch.randn(2, 128, device=DEVICE)
        out = m(x, cond)
        assert out.shape == x.shape

    def test_sequence_output_shape(self):
        from src.models.adagn import AdaGN

        m = AdaGN(64, cond_dim=128).to(DEVICE)
        x = torch.randn(2, 256, 64, device=DEVICE)
        cond = torch.randn(2, 128, device=DEVICE)
        out = m(x, cond)
        assert out.shape == x.shape

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
        from src.models.embeddings import ConditioningProjection

        m = ConditioningProjection(out_dim=512).to(DEVICE)
        p = _physics_batch(4)
        out = m(p["ambient"], p["transmission"], p["degradation"], p["severity"])
        assert out.shape == (4, 512)


# ---------------------------------------------------------------------------
# SwinUNetDenoiser
# ---------------------------------------------------------------------------


class TestSwinUNetDenoiser:
    @pytest.fixture(scope="class")
    def model_and_batch(self):
        from src.models.swin_unet import SwinUNetDenoiser, SwinUNetConfig

        cfg = SwinUNetConfig(
            embed_dim=96,
            depths=[1, 1, 2, 1, 1, 2, 1],
            num_heads=[3, 6, 12, 24, 12, 6, 3],
            window_size=8,
        )
        model = SwinUNetDenoiser(cfg).to(DEVICE)
        model.eval()
        B = 2
        x_t = torch.randn(B, 3, 256, 256, device=DEVICE)
        t = torch.randint(0, 1000, (B,), device=DEVICE)
        phys = _physics_batch(B)
        return model, x_t, t, phys

    def test_output_shape(self, model_and_batch):
        model, x_t, t, phys = model_and_batch
        with torch.no_grad():
            out = model(x_t, t, **phys)
        assert out.shape == x_t.shape

    def test_output_dtype(self, model_and_batch):
        model, x_t, t, phys = model_and_batch
        with torch.no_grad():
            out = model(x_t, t, **phys)
        assert out.dtype == torch.float32

    def test_gradient_flow(self, model_and_batch):
        model, x_t, t, phys = model_and_batch
        model.train()
        out = model(x_t, t, **phys)
        loss = out.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        model.eval()

    def test_batch_size_1(self, model_and_batch):
        model, x_t, t, phys = model_and_batch
        p1 = {k: v[:1] for k, v in phys.items()}
        with torch.no_grad():
            out = model(x_t[:1], t[:1], **p1)
        assert out.shape == (1, 3, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bfloat16_amp(self, model_and_batch):
        model, x_t, t, phys = model_and_batch
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            out = model(x_t, t, **phys)
        assert out.dtype == torch.bfloat16

    def test_different_timesteps_give_different_outputs(self, model_and_batch):
        model, x_t, t, phys = model_and_batch
        t2 = torch.zeros_like(t)
        t3 = torch.full_like(t, 999)
        with torch.no_grad():
            out2 = model(x_t, t2, **phys)
            out3 = model(x_t, t3, **phys)
        assert not torch.allclose(
            out2, out3, atol=1e-4
        ), "Model outputs identical for t=0 and t=999"

    def test_128x128_resolution(self, model_and_batch):
        model, _, t, _ = model_and_batch
        B = 2
        x_sm = torch.randn(B, 3, 128, 128, device=DEVICE)
        phys_sm = _physics_batch(B, 128, 128)
        with torch.no_grad():
            out = model(x_sm, t, **phys_sm)
        assert out.shape == x_sm.shape

    def test_num_parameters(self, model_and_batch):
        model, _, _, _ = model_and_batch
        n = model.num_parameters()
        assert n > 1_000_000, f"Suspiciously few parameters: {n}"
