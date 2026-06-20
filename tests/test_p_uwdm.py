"""
tests/test_p_uwdm.py
====================
Unit and integration tests for:
  - src/models/diffusion.py   (DDIMScheduler)
  - src/models/discriminator.py (build_discriminator / PatchDiscriminator re-export)
  - src/models/p_uwdm.py      (PUWDM, PUWDMConfig, EMAModel)

Run from project root:
    pytest tests/test_p_uwdm.py -v

All tests pass on CPU; GPU tests are auto-skipped when CUDA is unavailable.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, C, H, W = 2, 3, 64, 64  # small spatial size for fast CPU tests
T_FULL = 1000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_batch(B: int = B, H: int = H, W: int = W, device=DEVICE) -> dict:
    """
    Minimal batch that matches the DataLoader contract:
        raw          (B, 3, H, W)   in [0, 1]
        reference    (B, 3, H, W)   in [0, 1]
        ambient      (B, 3)
        transmission (B, 1, H, W)
        degradation  (B, 6)
        severity     (B, 1)
    """
    return {
        "raw": torch.rand(B, C, H, W, device=device),
        "reference": torch.rand(B, C, H, W, device=device),
        "ambient": torch.rand(B, 3, device=device),
        "transmission": torch.rand(B, 1, H, W, device=device).clamp(0.1, 1.0),
        "degradation": torch.rand(B, 6, device=device),
        "severity": torch.rand(B, 1, device=device),
    }


# ============================================================================
# 1.  DDIMScheduler
# ============================================================================


class TestDDIMScheduler:

    @pytest.fixture(scope="class")
    def sched(self):
        from src.models.diffusion import DDIMScheduler

        return DDIMScheduler(T=T_FULL)

    # --- schedule tables ---

    def test_betas_in_range(self, sched):
        assert sched.betas.min() > 0.0
        assert sched.betas.max() < 1.0

    def test_alphas_cumprod_decreasing(self, sched):
        acp = sched.alphas_cumprod
        assert (acp[1:] < acp[:-1]).all(), "ᾱ_t must be strictly decreasing"

    def test_alphas_cumprod_boundary(self, sched):
        acp = sched.alphas_cumprod
        # Cosine schedule: ᾱ_0 ≈ 1, ᾱ_{T-1} ≈ 0
        assert acp[0] > 0.99, f"ᾱ_0 should be ≈ 1, got {acp[0]:.4f}"
        assert acp[-1] < 0.05, f"ᾱ_T should be ≈ 0, got {acp[-1]:.4f}"

    def test_schedule_length(self, sched):
        assert len(sched.betas) == T_FULL
        assert len(sched.alphas_cumprod) == T_FULL

    # --- sample_timesteps ---

    def test_sample_timesteps_shape(self, sched):
        t = sched.sample_timesteps(B, device=DEVICE)
        assert t.shape == (B,)
        assert t.dtype == torch.long

    def test_sample_timesteps_range(self, sched):
        t = sched.sample_timesteps(256, device=DEVICE)
        assert t.min() >= 0
        assert t.max() < T_FULL

    # --- add_noise ---

    def test_add_noise_shapes(self, sched):
        x0 = torch.rand(B, C, H, W, device=DEVICE)
        t = sched.sample_timesteps(B, device=DEVICE)
        x_t, eps = sched.add_noise(x0, t)
        assert x_t.shape == x0.shape
        assert eps.shape == x0.shape

    def test_add_noise_at_t0_close_to_x0(self, sched):
        """At t=0, ᾱ_0 ≈ 1 → x_t ≈ x_0."""
        x0 = torch.rand(B, C, H, W)
        t = torch.zeros(B, dtype=torch.long)
        x_t, _ = sched.add_noise(x0, t)
        # sqrt(ᾱ_0) ≈ 1, sqrt(1-ᾱ_0) ≈ 0 → x_t very close to x0
        assert torch.allclose(
            x_t, x0, atol=0.05
        ), "At t=0 noisy image should be very close to x_0"

    def test_add_noise_at_tmax_high_variance(self, sched):
        """At t=T-1, ᾱ_T ≈ 0 → x_t dominated by noise."""
        x0 = torch.zeros(B, C, H, W)  # all-zero signal
        t = torch.full((B,), T_FULL - 1, dtype=torch.long)
        x_t, _ = sched.add_noise(x0, t)
        # Noise should dominate; std ≈ 1
        assert x_t.std().item() > 0.5, "At t=T-1 signal should be mostly noise"

    def test_add_noise_output_finite(self, sched):
        x0 = torch.rand(B, C, H, W)
        t = sched.sample_timesteps(B)
        x_t, eps = sched.add_noise(x0, t)
        assert torch.isfinite(x_t).all()
        assert torch.isfinite(eps).all()

    # --- predict_x0_from_eps ---

    def test_predict_x0_recovers_signal(self, sched):
        """When we know the true noise, x̂_0 should equal x_0."""
        x0 = torch.rand(B, C, H, W)
        t = sched.sample_timesteps(B)
        x_t, eps = sched.add_noise(x0, t)
        x0_pred = sched.predict_x0_from_eps(x_t, t, eps)
        # Allow some floating-point error; clip_denoised may truncate at boundary
        assert torch.allclose(
            x0_pred, x0.clamp(-1, 1), atol=1e-4
        ), f"x̂_0 should recover x_0; max err={( x0_pred - x0.clamp(-1,1) ).abs().max():.6f}"

    # --- ddim_step ---

    def test_ddim_step_shapes(self, sched):
        x_t = torch.randn(B, C, H, W)
        t = torch.full((B,), 500, dtype=torch.long)
        t_prev = torch.full((B,), 490, dtype=torch.long)
        eps = torch.randn_like(x_t)
        x_prev = sched.ddim_step(x_t, t, t_prev, eps, eta=0.0)
        assert x_prev.shape == x_t.shape

    def test_ddim_step_finite(self, sched):
        x_t = torch.randn(B, C, H, W)
        t = torch.full((B,), 500, dtype=torch.long)
        t_prev = torch.full((B,), 490, dtype=torch.long)
        eps = torch.randn_like(x_t)
        x_prev = sched.ddim_step(x_t, t, t_prev, eps, eta=0.0)
        assert torch.isfinite(x_prev).all()

    def test_ddim_step_deterministic_at_eta0(self, sched):
        """η=0 should give identical output for identical inputs."""
        x_t = torch.randn(B, C, H, W)
        t = torch.full((B,), 500, dtype=torch.long)
        t_prev = torch.full((B,), 490, dtype=torch.long)
        eps = torch.randn_like(x_t)
        out1 = sched.ddim_step(x_t, t, t_prev, eps, eta=0.0)
        out2 = sched.ddim_step(x_t, t, t_prev, eps, eta=0.0)
        assert torch.allclose(out1, out2)

    def test_ddim_step_stochastic_at_eta1(self, sched):
        """η=1 should add noise → outputs differ across calls."""
        x_t = torch.randn(B, C, H, W)
        t = torch.full((B,), 500, dtype=torch.long)
        t_prev = torch.full((B,), 490, dtype=torch.long)
        eps = torch.randn_like(x_t)
        out1 = sched.ddim_step(x_t, t, t_prev, eps, eta=1.0)
        out2 = sched.ddim_step(x_t, t, t_prev, eps, eta=1.0)
        assert not torch.allclose(
            out1, out2
        ), "Stochastic DDIM (η=1) should produce different outputs across calls"

    # --- ddim_sample_loop (stub model) ---

    def test_ddim_sample_loop_shape(self, sched):
        shape = (B, C, H, W)
        # Identity model: always returns zeros (fast)
        model_fn = lambda x, t: torch.zeros_like(x)
        out = sched.ddim_sample_loop(model_fn, shape, device=DEVICE, num_steps=5)
        assert out.shape == shape

    def test_ddim_sample_loop_finite(self, sched):
        shape = (B, C, H, W)
        model_fn = lambda x, t: torch.randn_like(x) * 0.01
        out = sched.ddim_sample_loop(model_fn, shape, device=DEVICE, num_steps=5)
        assert torch.isfinite(out).all()

    def test_ddim_sample_loop_uses_x_T(self, sched):
        """Seeded x_T must produce deterministic output with deterministic model."""
        shape = (B, C, H, W)
        x_T = torch.randn(shape, device=DEVICE)
        model_fn = lambda x, t: torch.zeros_like(x)
        out1 = sched.ddim_sample_loop(
            model_fn, shape, device=DEVICE, num_steps=5, x_T=x_T.clone()
        )
        out2 = sched.ddim_sample_loop(
            model_fn, shape, device=DEVICE, num_steps=5, x_T=x_T.clone()
        )
        assert torch.allclose(out1, out2)

    # --- snr utility ---

    def test_snr_decreasing(self, sched):
        t = torch.arange(0, T_FULL, 100)
        snr = sched.snr(t)
        assert (snr[1:] < snr[:-1]).all(), "SNR should decrease with timestep"

    def test_snr_positive(self, sched):
        t = torch.arange(0, T_FULL, 50)
        snr = sched.snr(t)
        assert (snr > 0).all()

    def test_repr(self, sched):
        r = repr(sched)
        assert "DDIMScheduler" in r
        assert "T=1000" in r


# ============================================================================
# 2.  discriminator.py
# ============================================================================


class TestDiscriminator:

    def test_build_discriminator_default(self):
        from src.models.discriminator import build_discriminator

        disc = build_discriminator()
        assert isinstance(disc, nn.Module)

    def test_build_discriminator_output_shape(self):
        from src.models.discriminator import build_discriminator

        disc = build_discriminator().to(DEVICE)
        fake = torch.rand(B, C, H, W, device=DEVICE)
        cond = torch.rand(B, C, H, W, device=DEVICE)
        out = disc(fake, cond)
        assert out.shape[0] == B
        assert out.shape[1] == 1
        assert out.ndim == 4

    def test_patch_discriminator_import(self):
        """PatchDiscriminator must be importable from src.models.discriminator."""
        from src.models.discriminator import PatchDiscriminator

        disc = PatchDiscriminator().to(DEVICE)
        fake = torch.rand(B, C, H, W, device=DEVICE)
        cond = torch.rand(B, C, H, W, device=DEVICE)
        out = disc(fake, cond)
        assert out.ndim == 4

    def test_discriminator_gradient_flows(self):
        from src.models.discriminator import PatchDiscriminator

        disc = PatchDiscriminator().to(DEVICE)
        fake = torch.rand(B, C, H, W, device=DEVICE, requires_grad=True)
        cond = torch.rand(B, C, H, W, device=DEVICE)
        out = disc(fake, cond)
        out.mean().backward()
        assert fake.grad is not None
        assert fake.grad.abs().sum() > 0

    def test_discriminator_real_vs_fake_differ(self):
        from src.models.discriminator import PatchDiscriminator

        disc = PatchDiscriminator().to(DEVICE).eval()
        real = torch.rand(B, C, H, W, device=DEVICE)
        fake = torch.rand(B, C, H, W, device=DEVICE)
        cond = torch.rand(B, C, H, W, device=DEVICE)
        with torch.no_grad():
            out_r = disc(real, cond)
            out_f = disc(fake, cond)
        assert not torch.allclose(out_r, out_f, atol=1e-4)

    def test_discriminator_param_count(self):
        from src.models.discriminator import PatchDiscriminator

        disc = PatchDiscriminator()
        n = sum(p.numel() for p in disc.parameters())
        # Typical 70×70 PatchGAN: ~2–4 M
        assert 500_000 < n < 10_000_000, f"Param count {n:,} looks wrong"


# ============================================================================
# 3.  EMAModel
# ============================================================================


class TestEMAModel:

    def _simple_model(self):
        return nn.Linear(16, 16)

    def test_shadow_initialised_correctly(self):
        from src.models.p_uwdm import EMAModel

        m = self._simple_model()
        ema = EMAModel(m, decay=0.99)
        for name, param in m.named_parameters():
            if param.requires_grad:
                assert torch.allclose(ema.shadow[name], param.data.float())

    def test_update_moves_toward_new_weights(self):
        from src.models.p_uwdm import EMAModel

        m = self._simple_model()
        ema = EMAModel(m, decay=0.5)  # large update for testing
        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Zero out model weights
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(0.0)
        ema.update(m)

        for name in ema.shadow:
            # shadow should have moved toward 0
            assert (ema.shadow[name].abs() < original_shadow[name].abs() + 1e-6).all()

    def test_copy_to_replaces_weights(self):
        from src.models.p_uwdm import EMAModel

        m = self._simple_model()
        ema = EMAModel(m, decay=0.99)
        # Manually set EMA shadow to all-ones
        for k in ema.shadow:
            ema.shadow[k] = torch.ones_like(ema.shadow[k])
        ema.copy_to(m)
        for p in m.parameters():
            assert torch.allclose(p.data, torch.ones_like(p.data))

    def test_apply_context_restores_weights(self):
        from src.models.p_uwdm import EMAModel

        m = self._simple_model()
        original = {n: p.data.clone() for n, p in m.named_parameters()}
        ema = EMAModel(m, decay=0.99)
        # Override shadow with zeros
        for k in ema.shadow:
            ema.shadow[k] = torch.zeros_like(ema.shadow[k])

        with ema.apply(m):
            # Inside context: weights should be EMA (zeros)
            for p in m.parameters():
                assert p.data.abs().max() < 1e-6

        # After context: weights should be restored
        for name, param in m.named_parameters():
            assert torch.allclose(param.data, original[name])


# ============================================================================
# 4.  PUWDMConfig
# ============================================================================


class TestPUWDMConfig:

    def test_default_cond_embed_dim_propagates(self):
        from src.models.p_uwdm import PUWDMConfig

        cfg = PUWDMConfig(cond_embed_dim=64)
        assert cfg.denoiser_cfg.cond_embed_dim == 64

    def test_custom_denoiser_cfg(self):
        from src.models.p_uwdm import PUWDMConfig
        from src.models.swin_unet import SwinUNetConfig

        d_cfg = SwinUNetConfig(
            embed_dim=48, depths=[1, 1, 1, 1, 1, 1, 1], num_heads=[3, 3, 3, 3, 3, 3, 3]
        )
        cfg = PUWDMConfig(denoiser_cfg=d_cfg, cond_embed_dim=64)
        assert cfg.denoiser_cfg.embed_dim == 48
        assert cfg.denoiser_cfg.cond_embed_dim == 64


# ============================================================================
# 5.  PUWDM — component integration
# ============================================================================


@pytest.fixture(scope="module")
def small_model():
    """
    A tiny P-UWDM variant (small depths/heads) that runs fast on CPU.
    Full 49M model would be too slow for unit tests.
    """
    from src.models.p_uwdm import PUWDM, PUWDMConfig
    from src.models.swin_unet import SwinUNetConfig

    d_cfg = SwinUNetConfig(
        image_size=64,
        embed_dim=48,
        depths=[1, 1, 1, 1, 1, 1, 1],
        num_heads=[3, 3, 3, 3, 3, 3, 3],
        window_size=8,
        cond_embed_dim=64,
    )
    cfg = PUWDMConfig(
        denoiser_cfg=d_cfg,
        cond_embed_dim=64,
        diffusion_T=1000,
        use_ema=True,
        ema_decay=0.999,
    )
    return PUWDM(cfg).to(DEVICE).eval()


class TestPUWDM:

    # --- basic smoke tests ---

    def test_repr_has_param_counts(self, small_model):
        r = repr(small_model)
        assert "cond_nets" in r
        assert "denoiser" in r
        assert "total" in r

    def test_num_parameters(self, small_model):
        counts = small_model.num_parameters()
        assert counts["total"] > 500_000
        assert counts["cond_nets"] > 0
        assert counts["denoiser"] > 0
        assert counts["cond_nets"] + counts["denoiser"] == counts["total"]

    # --- training_step ---

    def test_training_step_output_keys(self, small_model):
        batch = _make_batch()
        small_model.train()
        out = small_model.training_step(batch)
        expected_keys = {
            "noise_pred",
            "noise_target",
            "timesteps",
            "alphas_cumprod",
            "enhanced",
            "refined_map",
            "a_embedding",
            "t_embedding",
        }
        assert set(out.keys()) == expected_keys
        small_model.eval()

    def test_training_step_shapes(self, small_model):
        batch = _make_batch()
        small_model.train()
        out = small_model.training_step(batch)
        assert out["noise_pred"].shape == (B, C, H, W)
        assert out["noise_target"].shape == (B, C, H, W)
        assert out["timesteps"].shape == (B,)
        assert out["alphas_cumprod"].shape == (1000,)
        assert out["enhanced"].shape == (B, C, H, W)
        assert out["refined_map"].shape == (B, 1, H, W)
        assert out["a_embedding"].shape == (B, 64)
        assert out["t_embedding"].shape == (B, 64)
        small_model.eval()

    def test_training_step_all_finite(self, small_model):
        batch = _make_batch()
        small_model.train()
        out = small_model.training_step(batch)
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), f"training_step['{key}'] has NaN/Inf"
        small_model.eval()

    def test_training_step_gradients_flow(self, small_model):
        """noise_pred must carry gradients to both denoiser and cond_nets."""
        from src.models.p_uwdm import PUWDM, PUWDMConfig
        from src.models.swin_unet import SwinUNetConfig

        # Fresh model in train mode
        d_cfg = SwinUNetConfig(
            image_size=64,
            embed_dim=48,
            depths=[1, 1, 1, 1, 1, 1, 1],
            num_heads=[3, 3, 3, 3, 3, 3, 3],
            window_size=8,
            cond_embed_dim=64,
        )
        m = (
            PUWDM(
                PUWDMConfig(
                    denoiser_cfg=d_cfg,
                    cond_embed_dim=64,
                    diffusion_T=1000,
                    use_ema=False,
                )
            )
            .to(DEVICE)
            .train()
        )

        batch = _make_batch()
        out = m.training_step(batch)
        loss = out["noise_pred"].mean()
        loss.backward()

        denoiser_grads = [p.grad for p in m.denoiser.parameters() if p.grad is not None]
        cond_grads = [p.grad for p in m.cond_nets.parameters() if p.grad is not None]
        assert len(denoiser_grads) > 0, "No gradients reached denoiser"
        assert len(cond_grads) > 0, "No gradients reached conditioning networks"

    def test_training_step_timesteps_in_range(self, small_model):
        batch = _make_batch()
        small_model.train()
        out = small_model.training_step(batch)
        t = out["timesteps"]
        assert (t >= 0).all()
        assert (t < 1000).all()
        small_model.eval()

    def test_training_step_different_noise_per_call(self, small_model):
        """Two calls should produce different noise targets (different timestep samples)."""
        batch = _make_batch()
        small_model.train()
        out1 = small_model.training_step(batch)
        out2 = small_model.training_step(batch)
        # Very unlikely to be identical (different random noise / timestep)
        assert not torch.allclose(
            out1["noise_target"], out2["noise_target"]
        ), "Noise target identical across two training steps — RNG broken?"
        small_model.eval()

    def test_training_step_enhanced_in_clip_range(self, small_model):
        """With clip_denoised=True, enhanced should be in [-1, 1]."""
        batch = _make_batch()
        small_model.train()
        out = small_model.training_step(batch)
        enhanced = out["enhanced"]
        assert enhanced.min() >= -1.0 - 1e-5
        assert enhanced.max() <= 1.0 + 1e-5
        small_model.eval()

    # --- sample (inference) ---

    def test_sample_shape(self, small_model):
        batch = _make_batch()
        out = small_model.sample(
            raw=batch["raw"],
            physics_A=batch["ambient"],
            physics_t=batch["transmission"],
            degradation=batch["degradation"],
            severity=batch["severity"],
            num_steps=3,  # very fast for testing
        )
        assert out.shape == (B, C, H, W)

    def test_sample_finite(self, small_model):
        batch = _make_batch()
        out = small_model.sample(
            raw=batch["raw"],
            physics_A=batch["ambient"],
            physics_t=batch["transmission"],
            degradation=batch["degradation"],
            severity=batch["severity"],
            num_steps=3,
        )
        assert torch.isfinite(out).all()

    def test_sample_deterministic_eta0(self, small_model):
        """Deterministic DDIM (η=0) + same x_T → identical outputs."""
        batch = _make_batch()
        sched = small_model.scheduler
        x_T = torch.randn(B, C, H, W, device=DEVICE)

        # We can't pass x_T directly through .sample() in the public API,
        # so we call ddim_sample_loop directly with a fixed model_fn.
        cond_out = small_model.cond_nets(
            batch["raw"], batch["ambient"], batch["transmission"]
        )

        def _fn(x, t):
            return small_model.denoiser(
                x_t=x,
                t=t,
                a_embedding=cond_out["a_embedding"],
                t_embedding=cond_out["t_embedding"],
                degradation=batch["degradation"],
                severity=batch["severity"],
            )

        with torch.no_grad():
            out1 = sched.ddim_sample_loop(
                _fn, (B, C, H, W), DEVICE, num_steps=3, eta=0.0, x_T=x_T.clone()
            )
            out2 = sched.ddim_sample_loop(
                _fn, (B, C, H, W), DEVICE, num_steps=3, eta=0.0, x_T=x_T.clone()
            )
        assert torch.allclose(
            out1, out2, atol=1e-5
        ), "Deterministic DDIM should give identical outputs given same x_T"

    def test_sample_no_grad_leak(self, small_model):
        """Sampling should not accumulate gradients."""
        batch = _make_batch()
        out = small_model.sample(
            raw=batch["raw"],
            physics_A=batch["ambient"],
            physics_t=batch["transmission"],
            degradation=batch["degradation"],
            severity=batch["severity"],
            num_steps=3,
        )
        assert not out.requires_grad

    def test_sample_single_image(self, small_model):
        """Batch size 1 must work."""
        batch = _make_batch(B=1)
        out = small_model.sample(
            raw=batch["raw"],
            physics_A=batch["ambient"],
            physics_t=batch["transmission"],
            degradation=batch["degradation"],
            severity=batch["severity"],
            num_steps=3,
        )
        assert out.shape == (1, C, H, W)

    # --- EMA ---

    def test_ema_update_runs(self, small_model):
        """update_ema should not raise."""
        small_model.train()
        small_model.update_ema()
        small_model.eval()

    def test_ema_context_runs(self, small_model):
        """ema_context should not raise."""
        batch = _make_batch()
        with small_model.ema_context():
            _ = small_model.sample(
                raw=batch["raw"],
                physics_A=batch["ambient"],
                physics_t=batch["transmission"],
                degradation=batch["degradation"],
                severity=batch["severity"],
                num_steps=2,
            )

    def test_ema_weights_differ_from_model_after_updates(self):
        """After parameter updates + EMA update, EMA shadow ≠ model params."""
        from src.models.p_uwdm import PUWDM, PUWDMConfig, EMAModel
        from src.models.swin_unet import SwinUNetConfig

        d_cfg = SwinUNetConfig(
            image_size=64,
            embed_dim=48,
            depths=[1, 1, 1, 1, 1, 1, 1],
            num_heads=[3, 3, 3, 3, 3, 3, 3],
            window_size=8,
            cond_embed_dim=64,
        )
        m = PUWDM(
            PUWDMConfig(
                denoiser_cfg=d_cfg, cond_embed_dim=64, use_ema=True, ema_decay=0.9
            )
        ).to(DEVICE)

        # Snapshot initial EMA shadow
        initial = {k: v.clone() for k, v in m._ema.shadow.items()}

        # Make a big parameter update
        opt = torch.optim.SGD(m.denoiser.parameters(), lr=10.0)
        batch = _make_batch()
        m.train()
        out = m.training_step(batch)
        out["noise_pred"].mean().backward()
        opt.step()
        m.update_ema()

        # EMA shadow must have moved (but less than model weights due to decay)
        changed = any(not torch.allclose(m._ema.shadow[k], initial[k]) for k in initial)
        assert changed, "EMA shadow did not update after model parameter change"

    # --- full composite loss integration ---

    def test_compatible_with_composite_loss(self, small_model):
        """
        Verify training_step output can be fed directly into CompositeLoss.forward().
        Only diffusion + perceptual + histogram are tested (no adv/contrastive).
        """
        from src.losses import CompositeLoss, LossWeights

        loss_fn = CompositeLoss(
            weights=LossWeights(
                diffusion=1.0,
                adversarial=0.0,
                perceptual=0.1,
                histogram=0.05,
                contrastive=0.0,
            )
        ).to(DEVICE)

        batch = _make_batch()
        small_model.train()
        out = small_model.training_step(batch)

        loss_dict = loss_fn(
            noise_pred=out["noise_pred"],
            noise_target=out["noise_target"],
            timesteps=out["timesteps"],
            alphas_cumprod=out["alphas_cumprod"].to(DEVICE),
            enhanced=out["enhanced"],
            reference=batch["reference"],
            raw=batch["raw"],
        )

        assert "total" in loss_dict
        assert torch.isfinite(
            loss_dict["total"]
        ), f"Composite loss NaN/Inf: {loss_dict}"
        assert loss_dict["total"].item() >= 0.0

        # Verify total.backward() works
        loss_dict["total"].backward()
        small_model.eval()

    def test_training_step_then_discriminator_loss(self, small_model):
        """Discriminator loss from CompositeLoss on enhanced outputs."""
        from src.losses import CompositeLoss

        loss_fn = CompositeLoss().to(DEVICE)
        batch = _make_batch()

        small_model.train()
        out = small_model.training_step(batch)
        enhanced = out["enhanced"].detach()

        d_loss = loss_fn.discriminator_loss(
            real=batch["reference"],
            fake=enhanced,
            condition=batch["raw"],
        )
        assert d_loss.shape == ()
        assert torch.isfinite(d_loss)
        assert d_loss.item() >= 0.0
        small_model.eval()


# ============================================================================
# 6.  GPU tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:

    def test_ddim_scheduler_on_gpu(self):
        from src.models.diffusion import DDIMScheduler

        sched = DDIMScheduler(T=1000)
        x0 = torch.rand(2, 3, 64, 64, device="cuda")
        t = sched.sample_timesteps(2, device="cuda")
        x_t, eps = sched.add_noise(x0, t)
        assert x_t.device.type == "cuda"

    def test_puwdm_training_step_on_gpu(self):
        from src.models.p_uwdm import PUWDM, PUWDMConfig
        from src.models.swin_unet import SwinUNetConfig

        d_cfg = SwinUNetConfig(
            image_size=64,
            embed_dim=48,
            depths=[1, 1, 1, 1, 1, 1, 1],
            num_heads=[3, 3, 3, 3, 3, 3, 3],
            window_size=8,
            cond_embed_dim=64,
        )
        m = (
            PUWDM(PUWDMConfig(denoiser_cfg=d_cfg, cond_embed_dim=64, diffusion_T=1000))
            .cuda()
            .train()
        )

        batch = _make_batch(device=torch.device("cuda"))
        out = m.training_step(batch)
        assert out["noise_pred"].device.type == "cuda"
        assert torch.isfinite(out["noise_pred"]).all()

    def test_puwdm_bfloat16_autocast(self):
        from src.models.p_uwdm import PUWDM, PUWDMConfig
        from src.models.swin_unet import SwinUNetConfig

        d_cfg = SwinUNetConfig(
            image_size=64,
            embed_dim=48,
            depths=[1, 1, 1, 1, 1, 1, 1],
            num_heads=[3, 3, 3, 3, 3, 3, 3],
            window_size=8,
            cond_embed_dim=64,
        )
        m = (
            PUWDM(PUWDMConfig(denoiser_cfg=d_cfg, cond_embed_dim=64, diffusion_T=1000))
            .cuda()
            .train()
        )

        batch = _make_batch(device=torch.device("cuda"))

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = m.training_step(batch)

        assert torch.isfinite(out["noise_pred"]).all()

    def test_puwdm_sample_on_gpu(self):
        from src.models.p_uwdm import PUWDM, PUWDMConfig
        from src.models.swin_unet import SwinUNetConfig

        d_cfg = SwinUNetConfig(
            image_size=64,
            embed_dim=48,
            depths=[1, 1, 1, 1, 1, 1, 1],
            num_heads=[3, 3, 3, 3, 3, 3, 3],
            window_size=8,
            cond_embed_dim=64,
        )
        m = (
            PUWDM(
                PUWDMConfig(
                    denoiser_cfg=d_cfg,
                    cond_embed_dim=64,
                    diffusion_T=1000,
                    use_ema=False,
                )
            )
            .cuda()
            .eval()
        )

        batch = _make_batch(device=torch.device("cuda"))
        out = m.sample(
            raw=batch["raw"],
            physics_A=batch["ambient"],
            physics_t=batch["transmission"],
            degradation=batch["degradation"],
            severity=batch["severity"],
            num_steps=3,
        )
        assert out.device.type == "cuda"
        assert torch.isfinite(out).all()


# ============================================================================
# Entry-point for running without pytest
# ============================================================================

if __name__ == "__main__":
    import time

    print("=" * 65)
    print("P-UWDM Smoke Test (no pytest)")
    print("=" * 65)

    from src.models.diffusion import DDIMScheduler
    from src.models.p_uwdm import PUWDM, PUWDMConfig
    from src.models.swin_unet import SwinUNetConfig

    d_cfg = SwinUNetConfig(
        image_size=64,
        embed_dim=48,
        depths=[1, 1, 1, 1, 1, 1, 1],
        num_heads=[3, 3, 3, 3, 3, 3, 3],
        window_size=8,
        cond_embed_dim=64,
    )
    cfg = PUWDMConfig(
        denoiser_cfg=d_cfg, cond_embed_dim=64, diffusion_T=1000, use_ema=True
    )
    model = PUWDM(cfg).to(DEVICE).train()
    print(model)

    batch = _make_batch()

    t0 = time.perf_counter()
    out = model.training_step(batch)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  training_step ({dt:.0f} ms)")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            ok = "✓" if torch.isfinite(v).all() else "✗ NaN/Inf"
            print(f"    {ok}  {k:<20s} {tuple(v.shape)}")

    t0 = time.perf_counter()
    enhanced = model.eval()
    enhanced = model.sample(
        raw=batch["raw"],
        physics_A=batch["ambient"],
        physics_t=batch["transmission"],
        degradation=batch["degradation"],
        severity=batch["severity"],
        num_steps=5,
        progress=True,
    )
    dt = (time.perf_counter() - t0) * 1000
    ok = "✓" if torch.isfinite(enhanced).all() else "✗ NaN/Inf"
    print(f"\n  sample (5 steps, {dt:.0f} ms)  {ok}  shape={tuple(enhanced.shape)}")

    print("\n✅ Smoke test complete.\n")
