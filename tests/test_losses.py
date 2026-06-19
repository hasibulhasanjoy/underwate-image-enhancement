"""
tests/test_losses.py
====================
Unit and integration tests for the P-UWDM five-component composite loss.

Run from project root:
    pytest tests/test_losses.py -v

All tests run on CPU by default; GPU tests are auto-skipped if CUDA is unavailable.
"""

from __future__ import annotations

import sys
import os

# Allow running directly: python tests/test_losses.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from src.losses import (
    CompositeLoss,
    LossWeights,
    DiffusionLoss,
    AdversarialLoss,
    PatchDiscriminator,
    PerceptualLoss,
    HistogramLoss,
    ContrastiveLoss,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
B, C, H, W = 2, 3, 64, 64  # small spatial size for fast tests
T = 1000  # total diffusion timesteps


@pytest.fixture(scope="module")
def imgs():
    """Return a dict of canonical image tensors in [0, 1]."""
    return {
        "enhanced": torch.rand(B, C, H, W, device=DEVICE),
        "reference": torch.rand(B, C, H, W, device=DEVICE),
        "raw": torch.rand(B, C, H, W, device=DEVICE),
    }


@pytest.fixture(scope="module")
def diff_inputs():
    """Noise-prediction tensors and scheduler variables."""
    noise_pred = torch.randn(B, C, H, W, device=DEVICE)
    noise_target = torch.randn(B, C, H, W, device=DEVICE)
    timesteps = torch.randint(0, T, (B,), device=DEVICE)
    alphas_cumprod = torch.linspace(0.9999, 0.0001, T, device=DEVICE)
    return noise_pred, noise_target, timesteps, alphas_cumprod


# ---------------------------------------------------------------------------
# 1. DiffusionLoss
# ---------------------------------------------------------------------------


class TestDiffusionLoss:

    def test_output_is_scalar(self, diff_inputs):
        loss_fn = DiffusionLoss()
        pred, tgt, ts, acp = diff_inputs
        loss = loss_fn(pred, tgt, ts, acp)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_loss_non_negative(self, diff_inputs):
        loss_fn = DiffusionLoss()
        pred, tgt, ts, acp = diff_inputs
        loss = loss_fn(pred, tgt, ts, acp)
        assert loss.item() >= 0.0, "Diffusion loss must be non-negative"

    def test_perfect_prediction_near_zero(self):
        """When noise_pred == noise_target, loss should be ≈ 0."""
        loss_fn = DiffusionLoss()
        x = torch.randn(B, C, H, W)
        ts = torch.zeros(B, dtype=torch.long)
        acp = torch.ones(T) * 0.9  # constant ᾱ
        loss = loss_fn(x, x, ts, acp)
        assert loss.item() < 1e-6, f"Expected ≈0, got {loss.item()}"

    def test_without_snr_weighting(self, diff_inputs):
        """snr_gamma=None should return plain MSE mean."""
        loss_fn = DiffusionLoss(snr_gamma=None)
        pred, tgt, ts, _ = diff_inputs
        loss = loss_fn(pred, tgt, ts, alphas_cumprod=None)
        expected = nn.functional.mse_loss(pred, tgt)
        assert (
            abs(loss.item() - expected.item()) < 1e-5
        ), f"Plain MSE mismatch: {loss.item()} vs {expected.item()}"

    def test_snr_weighting_different_from_plain(self, diff_inputs):
        """SNR weighting should produce a different value than plain MSE."""
        loss_snr = DiffusionLoss(snr_gamma=5.0)
        loss_plain = DiffusionLoss(snr_gamma=None)
        pred, tgt, ts, acp = diff_inputs
        v_snr = loss_snr(pred, tgt, ts, acp).item()
        v_plain = loss_plain(pred, tgt, ts).item()
        assert abs(v_snr - v_plain) > 1e-6, "SNR weighting had no effect"

    def test_shape_mismatch_raises(self):
        loss_fn = DiffusionLoss()
        with pytest.raises(AssertionError):
            loss_fn(
                torch.randn(2, 3, 64, 64),
                torch.randn(2, 3, 32, 32),
                torch.zeros(2, dtype=torch.long),
            )

    def test_gradients_flow(self, diff_inputs):
        loss_fn = DiffusionLoss()
        pred, tgt, ts, acp = diff_inputs
        pred = pred.detach().requires_grad_(True)
        loss = loss_fn(pred, tgt, ts, acp)
        loss.backward()
        assert pred.grad is not None, "No gradient on noise_pred"
        assert pred.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 2. PatchDiscriminator
# ---------------------------------------------------------------------------


class TestPatchDiscriminator:

    def test_output_shape(self, imgs):
        disc = PatchDiscriminator()
        out = disc(imgs["enhanced"], imgs["raw"])
        # With default 64-px base and 4 strides/downsamples, output is (B,1,H',W')
        assert out.shape[0] == B
        assert out.shape[1] == 1
        assert out.ndim == 4, f"Expected 4D output, got shape {out.shape}"

    def test_conditioned_differently(self, imgs):
        """Discriminator output should differ for real vs fake given same condition."""
        disc = PatchDiscriminator()
        out_real = disc(imgs["reference"], imgs["raw"])
        out_fake = disc(imgs["enhanced"], imgs["raw"])
        assert not torch.allclose(
            out_real, out_fake, atol=1e-4
        ), "Discriminator produced identical outputs for real and fake"

    def test_gradients_flow(self, imgs):
        disc = PatchDiscriminator()
        x = imgs["enhanced"].detach().requires_grad_(True)
        out = disc(x, imgs["raw"])
        out.mean().backward()
        assert x.grad is not None and x.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 3. AdversarialLoss
# ---------------------------------------------------------------------------


class TestAdversarialLoss:

    def test_generator_loss_scalar(self, imgs):
        disc = PatchDiscriminator()
        adv = AdversarialLoss()
        g_loss = adv.generator_loss(disc, imgs["enhanced"], imgs["raw"])
        assert g_loss.shape == ()

    def test_discriminator_loss_scalar(self, imgs):
        disc = PatchDiscriminator()
        adv = AdversarialLoss()
        d_loss = adv.discriminator_loss(
            disc, imgs["reference"], imgs["enhanced"].detach(), imgs["raw"]
        )
        assert d_loss.shape == ()

    def test_lsgan_real_targets(self, imgs):
        """Real targets → LSGAN D-loss should be lower than random-output baseline."""
        disc = PatchDiscriminator()
        adv = AdversarialLoss()
        # D-loss for real = E[(D(real) - 1)^2]. With random init, not zero, but finite.
        d_loss = adv.discriminator_loss(
            disc, imgs["reference"], imgs["enhanced"].detach(), imgs["raw"]
        )
        assert d_loss.item() < 10.0, f"D-loss suspiciously high: {d_loss.item()}"

    def test_generator_loss_non_negative(self, imgs):
        disc = PatchDiscriminator()
        adv = AdversarialLoss()
        g_loss = adv.generator_loss(disc, imgs["enhanced"], imgs["raw"])
        assert g_loss.item() >= 0.0

    def test_forward_equals_generator_loss(self, imgs):
        disc = PatchDiscriminator()
        adv = AdversarialLoss()
        g1 = adv.generator_loss(disc, imgs["enhanced"], imgs["raw"])
        g2 = adv(disc, imgs["enhanced"], imgs["raw"])
        assert torch.allclose(g1, g2)


# ---------------------------------------------------------------------------
# 4. PerceptualLoss
# ---------------------------------------------------------------------------


class TestPerceptualLoss:

    def test_output_scalar(self, imgs):
        perc = PerceptualLoss()
        loss = perc(imgs["enhanced"], imgs["reference"])
        assert loss.shape == ()

    def test_same_image_near_zero(self, imgs):
        perc = PerceptualLoss()
        x = imgs["reference"]
        loss = perc(x, x)
        assert (
            loss.item() < 1e-4
        ), f"Same-image perceptual loss should be ~0, got {loss.item()}"

    def test_different_images_positive(self, imgs):
        perc = PerceptualLoss()
        loss = perc(imgs["enhanced"], imgs["reference"])
        assert loss.item() > 0.0

    def test_vgg_frozen(self):
        """VGG feature extractor weights must not require gradients."""
        perc = PerceptualLoss()
        for name, p in perc.named_parameters():
            if "slices" in name:
                assert not p.requires_grad, f"VGG param {name} should be frozen"

    def test_gradient_flows_to_enhanced(self, imgs):
        perc = PerceptualLoss()
        x = imgs["enhanced"].detach().requires_grad_(True)
        loss = perc(x, imgs["reference"])
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum().item() > 0

    def test_no_gradient_to_reference(self, imgs):
        """Reference image should not accumulate gradients."""
        perc = PerceptualLoss()
        ref = imgs["reference"].detach().requires_grad_(True)
        x = imgs["enhanced"].detach()
        loss = perc(x, ref)
        loss.backward()
        # ref.grad should be None (torch.no_grad used internally)
        assert (
            ref.grad is None or ref.grad.abs().sum().item() == 0.0
        ), "Reference image accumulated gradients through perceptual loss"

    def test_layer_weight_effect(self, imgs):
        """Different layer weights should give different loss values."""
        perc1 = PerceptualLoss(layer_weights=(1.0, 0.75, 0.5))
        perc2 = PerceptualLoss(layer_weights=(0.5, 0.5, 1.0))
        l1 = perc1(imgs["enhanced"], imgs["reference"]).item()
        l2 = perc2(imgs["enhanced"], imgs["reference"]).item()
        assert (
            abs(l1 - l2) > 1e-6
        ), "Different layer weights should produce different losses"


# ---------------------------------------------------------------------------
# 5. HistogramLoss
# ---------------------------------------------------------------------------


class TestHistogramLoss:

    def test_output_scalar(self, imgs):
        hist = HistogramLoss()
        loss = hist(imgs["enhanced"], imgs["reference"])
        assert loss.shape == ()

    def test_same_image_near_zero(self):
        """Identical images → identical histograms → loss ≈ 0."""
        hist = HistogramLoss()
        x = torch.rand(B, C, H, W) * 0.5 + 0.25  # avoid edge bins
        loss = hist(x, x)
        assert (
            loss.item() < 1e-3
        ), f"Same-image histogram loss should be ~0, got {loss.item()}"

    def test_different_images_positive(self, imgs):
        hist = HistogramLoss()
        loss = hist(imgs["enhanced"], imgs["reference"])
        assert loss.item() > 0.0

    def test_gradient_flows(self, imgs):
        hist = HistogramLoss()
        x = imgs["enhanced"].detach().requires_grad_(True)
        loss = hist(x, imgs["reference"])
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum().item() > 0

    def test_range_0_1(self):
        """Loss should stay finite for values in [0, 1]."""
        hist = HistogramLoss()
        x = torch.zeros(B, C, H, W)
        y = torch.ones(B, C, H, W)
        loss = hist(x, y)
        assert torch.isfinite(loss), "Histogram loss is not finite at extremes"

    def test_non_negative(self, imgs):
        hist = HistogramLoss()
        loss = hist(imgs["enhanced"], imgs["reference"])
        assert loss.item() >= 0.0

    def test_custom_bins(self, imgs):
        """n_bins parameter should work without errors."""
        hist = HistogramLoss(n_bins=64, bandwidth=0.05)
        loss = hist(imgs["enhanced"], imgs["reference"])
        assert loss.shape == () and torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 6. ContrastiveLoss
# ---------------------------------------------------------------------------


class TestContrastiveLoss:

    def test_output_scalar(self, imgs):
        cont = ContrastiveLoss()
        loss = cont(imgs["enhanced"], imgs["reference"], imgs["raw"])
        assert loss.shape == ()

    def test_non_negative(self, imgs):
        cont = ContrastiveLoss()
        loss = cont(imgs["enhanced"], imgs["reference"], imgs["raw"])
        assert loss.item() >= 0.0

    def test_perfect_enhanced_lower_loss(self):
        """
        When enhanced == reference, the positive pair is identical.
        This should produce a lower loss than a random enhanced image.
        """
        cont = ContrastiveLoss()
        ref = torch.rand(B, C, H, W)
        raw = torch.rand(B, C, H, W)

        loss_perfect = cont(ref.clone(), ref, raw)  # enhanced = reference
        loss_random = cont(torch.rand(B, C, H, W), ref, raw)

        assert loss_perfect.item() <= loss_random.item() + 1e-2, (
            f"Perfect enhanced should have lower contrastive loss: "
            f"{loss_perfect.item():.4f} vs {loss_random.item():.4f}"
        )

    def test_gradient_flows(self, imgs):
        cont = ContrastiveLoss()
        x = imgs["enhanced"].detach().requires_grad_(True)
        loss = cont(x, imgs["reference"], imgs["raw"])
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum().item() > 0

    def test_projection_head_is_trainable(self):
        cont = ContrastiveLoss()
        proj_params = list(cont.proj.parameters())
        assert len(proj_params) > 0
        assert all(p.requires_grad for p in proj_params)

    def test_temperature_effect(self, imgs):
        """Different temperatures should give different losses."""
        c1 = ContrastiveLoss(temperature=0.07)
        c2 = ContrastiveLoss(temperature=0.5)
        l1 = c1(imgs["enhanced"], imgs["reference"], imgs["raw"]).item()
        l2 = c2(imgs["enhanced"], imgs["reference"], imgs["raw"]).item()
        assert (
            abs(l1 - l2) > 1e-4
        ), "Temperature change had no effect on contrastive loss"


# ---------------------------------------------------------------------------
# 7. LossWeights
# ---------------------------------------------------------------------------


class TestLossWeights:

    def test_phase1_preset(self):
        w = LossWeights.phase1()
        assert w.adversarial == 0.0
        assert w.contrastive == 0.0
        assert w.diffusion == 1.0

    def test_phase2_preset(self):
        w = LossWeights.phase2()
        assert w.adversarial > 0.0
        assert w.contrastive > 0.0

    def test_custom_weights(self):
        w = LossWeights(
            diffusion=2.0,
            adversarial=0.5,
            perceptual=0.3,
            histogram=0.1,
            contrastive=0.2,
        )
        assert w.diffusion == 2.0
        assert w.histogram == 0.1


# ---------------------------------------------------------------------------
# 8. CompositeLoss — integration tests
# ---------------------------------------------------------------------------


class TestCompositeLoss:

    @pytest.fixture
    def composite(self):
        return CompositeLoss(weights=LossWeights.phase2())

    @pytest.fixture
    def composite_p1(self):
        return CompositeLoss(weights=LossWeights.phase1())

    def test_returns_dict_with_all_keys(self, composite, imgs, diff_inputs):
        pred, tgt, ts, acp = diff_inputs
        out = composite(
            noise_pred=pred,
            noise_target=tgt,
            timesteps=ts,
            alphas_cumprod=acp,
            enhanced=imgs["enhanced"],
            reference=imgs["reference"],
            raw=imgs["raw"],
        )
        expected_keys = {
            "total",
            "diffusion",
            "adversarial",
            "perceptual",
            "histogram",
            "contrastive",
        }
        assert set(out.keys()) == expected_keys

    def test_all_losses_scalar(self, composite, imgs, diff_inputs):
        pred, tgt, ts, acp = diff_inputs
        out = composite(
            noise_pred=pred,
            noise_target=tgt,
            timesteps=ts,
            alphas_cumprod=acp,
            enhanced=imgs["enhanced"],
            reference=imgs["reference"],
            raw=imgs["raw"],
        )
        for k, v in out.items():
            assert v.shape == (), f"Loss '{k}' is not scalar: {v.shape}"

    def test_total_is_weighted_sum(self, composite, imgs, diff_inputs):
        pred, tgt, ts, acp = diff_inputs
        w = composite.weights
        out = composite(
            noise_pred=pred,
            noise_target=tgt,
            timesteps=ts,
            alphas_cumprod=acp,
            enhanced=imgs["enhanced"],
            reference=imgs["reference"],
            raw=imgs["raw"],
        )
        expected_total = (
            w.diffusion * out["diffusion"]
            + w.adversarial * out["adversarial"]
            + w.perceptual * out["perceptual"]
            + w.histogram * out["histogram"]
            + w.contrastive * out["contrastive"]
        )
        assert torch.allclose(
            out["total"], expected_total, atol=1e-5
        ), f"total={out['total'].item():.6f} expected={expected_total.item():.6f}"

    def test_phase1_disables_adv_and_contrastive(self, composite_p1, imgs, diff_inputs):
        pred, tgt, ts, acp = diff_inputs
        out = composite_p1(
            noise_pred=pred,
            noise_target=tgt,
            timesteps=ts,
            alphas_cumprod=acp,
            enhanced=imgs["enhanced"],
            reference=imgs["reference"],
            raw=imgs["raw"],
        )
        assert out["adversarial"].item() == 0.0, "Phase-1: adversarial should be zero"
        assert out["contrastive"].item() == 0.0, "Phase-1: contrastive should be zero"
        assert out["diffusion"].item() > 0.0
        assert out["perceptual"].item() > 0.0

    def test_diffusion_only_mode(self, composite, diff_inputs):
        """Without image tensors, only diffusion loss is computed."""
        pred, tgt, ts, acp = diff_inputs
        out = composite(
            noise_pred=pred, noise_target=tgt, timesteps=ts, alphas_cumprod=acp
        )
        assert out["adversarial"].item() == 0.0
        assert out["perceptual"].item() == 0.0
        assert out["histogram"].item() == 0.0
        assert out["contrastive"].item() == 0.0
        assert out["diffusion"].item() > 0.0

    def test_set_phase_changes_weights(self, composite):
        composite.set_phase(1)
        assert composite.weights.adversarial == 0.0
        composite.set_phase(2)
        assert composite.weights.adversarial > 0.0

    def test_set_weights_custom(self, composite):
        custom = LossWeights(
            diffusion=2.0,
            adversarial=0.0,
            perceptual=0.2,
            histogram=0.0,
            contrastive=0.0,
        )
        composite.set_weights(custom)
        assert composite.weights.diffusion == 2.0
        # reset
        composite.set_phase(2)

    def test_total_loss_gradients_flow(self, composite, imgs, diff_inputs):
        pred, tgt, ts, acp = diff_inputs
        pred_g = pred.detach().requires_grad_(True)
        out = composite(
            noise_pred=pred_g,
            noise_target=tgt,
            timesteps=ts,
            alphas_cumprod=acp,
            enhanced=imgs["enhanced"],
            reference=imgs["reference"],
            raw=imgs["raw"],
        )
        out["total"].backward()
        assert (
            pred_g.grad is not None
        ), "No gradient to noise_pred through composite loss"
        assert pred_g.grad.abs().sum().item() > 0

    def test_discriminator_loss(self, composite, imgs):
        d_loss = composite.discriminator_loss(
            imgs["reference"],
            imgs["enhanced"].detach(),
            imgs["raw"],
        )
        assert d_loss.shape == ()
        assert d_loss.item() >= 0.0

    def test_discriminator_loss_gradients(self, composite, imgs):
        for p in composite.discriminator.parameters():
            p.grad = None
        d_loss = composite.discriminator_loss(
            imgs["reference"],
            imgs["enhanced"].detach(),
            imgs["raw"],
        )
        d_loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in composite.discriminator.parameters()
        )
        assert has_grad, "No gradients to discriminator parameters"

    def test_finite_all_losses(self, composite, imgs, diff_inputs):
        pred, tgt, ts, acp = diff_inputs
        out = composite(
            noise_pred=pred,
            noise_target=tgt,
            timesteps=ts,
            alphas_cumprod=acp,
            enhanced=imgs["enhanced"],
            reference=imgs["reference"],
            raw=imgs["raw"],
        )
        for k, v in out.items():
            assert torch.isfinite(v), f"Loss '{k}' is not finite: {v.item()}"

    def test_total_non_negative(self, composite, imgs, diff_inputs):
        pred, tgt, ts, acp = diff_inputs
        out = composite(
            noise_pred=pred,
            noise_target=tgt,
            timesteps=ts,
            alphas_cumprod=acp,
            enhanced=imgs["enhanced"],
            reference=imgs["reference"],
            raw=imgs["raw"],
        )
        assert out["total"].item() >= 0.0


# ---------------------------------------------------------------------------
# 9. GPU tests (auto-skipped on CPU-only machines)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:

    def test_composite_loss_on_gpu(self):
        device = torch.device("cuda")
        composite = CompositeLoss().to(device)

        noise_pred = torch.randn(2, 3, 64, 64, device=device)
        noise_target = torch.randn(2, 3, 64, 64, device=device)
        timesteps = torch.randint(0, 1000, (2,), device=device)
        acp = torch.linspace(0.9999, 0.0001, 1000, device=device)
        enhanced = torch.rand(2, 3, 64, 64, device=device)
        reference = torch.rand(2, 3, 64, 64, device=device)
        raw = torch.rand(2, 3, 64, 64, device=device)

        out = composite(
            noise_pred=noise_pred,
            noise_target=noise_target,
            timesteps=timesteps,
            alphas_cumprod=acp,
            enhanced=enhanced,
            reference=reference,
            raw=raw,
        )
        assert out["total"].device.type == "cuda"
        assert torch.isfinite(out["total"])

    def test_bfloat16_composite(self):
        """Full composite loss should run in bfloat16 on GPU."""
        device = torch.device("cuda")
        composite = CompositeLoss().to(device).to(torch.bfloat16)

        noise_pred = torch.randn(2, 3, 64, 64, dtype=torch.bfloat16, device=device)
        noise_target = torch.randn(2, 3, 64, 64, dtype=torch.bfloat16, device=device)
        timesteps = torch.randint(0, 1000, (2,), device=device)
        acp = torch.linspace(0.9999, 0.0001, 1000, dtype=torch.bfloat16, device=device)

        out = composite(
            noise_pred=noise_pred,
            noise_target=noise_target,
            timesteps=timesteps,
            alphas_cumprod=acp,
        )
        assert torch.isfinite(out["diffusion"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test without pytest
    print("=" * 60)
    print("P-UWDM Loss Module — Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)
    B, C, H, W, T = 2, 3, 64, 64, 1000

    imgs = {
        "enhanced": torch.rand(B, C, H, W),
        "reference": torch.rand(B, C, H, W),
        "raw": torch.rand(B, C, H, W),
    }
    noise_pred = torch.randn(B, C, H, W)
    noise_target = torch.randn(B, C, H, W)
    timesteps = torch.randint(0, T, (B,))
    acp = torch.linspace(0.9999, 0.0001, T)

    composite = CompositeLoss(weights=LossWeights.phase2())

    out = composite(
        noise_pred=noise_pred,
        noise_target=noise_target,
        timesteps=timesteps,
        alphas_cumprod=acp,
        **imgs,
    )

    print("\n[Phase 2 — all components]")
    for k, v in out.items():
        flag = "✓" if torch.isfinite(v) else "✗ NaN/Inf!"
        print(f"  {flag}  {k:<15s} = {v.item():.6f}")

    composite.set_phase(1)
    out1 = composite(
        noise_pred=noise_pred,
        noise_target=noise_target,
        timesteps=timesteps,
        alphas_cumprod=acp,
        **imgs,
    )
    print("\n[Phase 1 — adv + contrastive disabled]")
    for k, v in out1.items():
        flag = "✓" if torch.isfinite(v) else "✗ NaN/Inf!"
        print(f"  {flag}  {k:<15s} = {v.item():.6f}")

    d_loss = composite.discriminator_loss(
        imgs["reference"], imgs["enhanced"].detach(), imgs["raw"]
    )
    print(f"\n  ✓  discriminator  = {d_loss.item():.6f}")

    print(
        "\n✅ Smoke test passed."
        if all(torch.isfinite(v) for v in out.values())
        else "\n❌ Some losses are not finite!"
    )
