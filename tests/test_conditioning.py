"""
Tests for src/models/conditioning.py

Mirrors the style of your existing test_models.py:
  - shape contracts on every public output
  - gradient flow through both networks
  - device transfer (CPU + CUDA if available)
  - parameter budget checks
  - ConditioningNetworks wrapper round-trip
  - edge cases: single-item batch, non-square input, large batch
"""

from __future__ import annotations

import sys
import os

# Allow running from tests/ directly or from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from src.models.conditioning import ANet, TNet, ConditioningNetworks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch():
    """Standard test batch: B=2, 256×256."""
    B, H, W = 2, 256, 256
    return {
        "raw": torch.rand(B, 3, H, W, device=DEVICE),
        "physics_A": torch.rand(B, 3, device=DEVICE),
        "physics_t": torch.rand(B, 1, H, W, device=DEVICE).clamp(1e-3, 1.0),
    }


@pytest.fixture
def a_net():
    return ANet(embed_dim=128, base_ch=32).to(DEVICE)


@pytest.fixture
def t_net():
    return TNet(embed_dim=128, base_ch=32).to(DEVICE)


@pytest.fixture
def cond_nets():
    return ConditioningNetworks(embed_dim=128, base_ch=32).to(DEVICE)


# ---------------------------------------------------------------------------
# A-Net tests
# ---------------------------------------------------------------------------


class TestANet:

    def test_output_shape(self, a_net, batch):
        emb = a_net(batch["raw"], batch["physics_A"])
        assert emb.shape == (2, 128), f"Expected (2,128), got {emb.shape}"

    def test_output_is_finite(self, a_net, batch):
        emb = a_net(batch["raw"], batch["physics_A"])
        assert torch.isfinite(emb).all(), "ANet output contains NaN/Inf"

    def test_gradient_flows(self, a_net, batch):
        raw = batch["raw"].requires_grad_(True)
        emb = a_net(raw, batch["physics_A"])
        emb.sum().backward()
        assert raw.grad is not None, "No gradient reached raw input"
        assert torch.isfinite(raw.grad).all(), "Gradient through ANet is NaN/Inf"

    def test_physics_hint_matters(self, a_net, batch):
        """Different physics_A vectors should produce different embeddings."""
        emb1 = a_net(batch["raw"], batch["physics_A"])
        alt_A = torch.rand_like(batch["physics_A"])
        emb2 = a_net(batch["raw"], alt_A)
        assert not torch.allclose(
            emb1, emb2
        ), "ANet ignores physics_A hint (outputs identical for different hints)"

    def test_batch_independence(self, a_net, batch):
        """Output for item 0 must not change when item 1's input changes."""
        emb_full = a_net(batch["raw"], batch["physics_A"])
        single_raw = batch["raw"][[0]]
        single_A = batch["physics_A"][[0]]
        emb_single = a_net(single_raw, single_A)
        assert torch.allclose(
            emb_full[[0]], emb_single, atol=1e-5
        ), "ANet has cross-batch contamination"

    def test_single_item_batch(self, a_net):
        raw = torch.rand(1, 3, 256, 256, device=DEVICE)
        A = torch.rand(1, 3, device=DEVICE)
        emb = a_net(raw, A)
        assert emb.shape == (1, 128)

    def test_non_square_input(self, a_net):
        raw = torch.rand(2, 3, 192, 256, device=DEVICE)
        A = torch.rand(2, 3, device=DEVICE)
        emb = a_net(raw, A)
        assert emb.shape == (2, 128), f"ANet fails on non-square input, got {emb.shape}"

    def test_large_batch(self, a_net):
        raw = torch.rand(8, 3, 128, 128, device=DEVICE)
        A = torch.rand(8, 3, device=DEVICE)
        emb = a_net(raw, A)
        assert emb.shape == (8, 128)

    def test_parameter_budget(self, a_net):
        n_params = sum(p.numel() for p in a_net.parameters())
        assert n_params < 1_000_000, f"ANet param count {n_params:,} exceeds 1M budget"

    def test_custom_embed_dim(self):
        net = ANet(embed_dim=64).to(DEVICE)
        raw = torch.rand(2, 3, 256, 256, device=DEVICE)
        A = torch.rand(2, 3, device=DEVICE)
        assert net(raw, A).shape == (2, 64)

    def test_eval_train_consistency(self, a_net, batch):
        """eval() should not change output shape (only affects dropout/BN)."""
        a_net.eval()
        with torch.no_grad():
            emb = a_net(batch["raw"], batch["physics_A"])
        assert emb.shape == (2, 128)
        a_net.train()


# ---------------------------------------------------------------------------
# T-Net tests
# ---------------------------------------------------------------------------


class TestTNet:

    def test_output_shapes(self, t_net, batch):
        B, H, W = 2, 256, 256
        refined_map, t_emb = t_net(batch["raw"], batch["physics_t"])
        assert refined_map.shape == (
            B,
            1,
            H,
            W,
        ), f"refined_map shape mismatch: {refined_map.shape}"
        assert t_emb.shape == (B, 128), f"t_embedding shape mismatch: {t_emb.shape}"

    def test_map_range(self, t_net, batch):
        refined_map, _ = t_net(batch["raw"], batch["physics_t"])
        assert refined_map.min() >= 0.0 - 1e-6, "refined_map below 0"
        assert refined_map.max() <= 1.0 + 1e-6, "refined_map above 1 (sigmoid broken)"

    def test_output_is_finite(self, t_net, batch):
        refined_map, t_emb = t_net(batch["raw"], batch["physics_t"])
        assert torch.isfinite(refined_map).all(), "refined_map contains NaN/Inf"
        assert torch.isfinite(t_emb).all(), "t_embedding contains NaN/Inf"

    def test_gradient_flows_map(self, t_net, batch):
        raw = batch["raw"].requires_grad_(True)
        refined_map, _ = t_net(raw, batch["physics_t"])
        refined_map.sum().backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()

    def test_gradient_flows_embedding(self, t_net, batch):
        raw = batch["raw"].requires_grad_(True)
        _, t_emb = t_net(raw, batch["physics_t"])
        t_emb.sum().backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()

    def test_physics_map_input_matters(self, t_net, batch):
        """Different physics_t inputs must produce different outputs."""
        _, emb1 = t_net(batch["raw"], batch["physics_t"])
        alt_t = torch.rand_like(batch["physics_t"])
        _, emb2 = t_net(batch["raw"], alt_t)
        assert not torch.allclose(emb1, emb2), "TNet ignores physics_t input"

    def test_spatial_resolution_preserved(self, t_net, batch):
        """refined_map must have same H,W as input."""
        H, W = batch["raw"].shape[2], batch["raw"].shape[3]
        refined_map, _ = t_net(batch["raw"], batch["physics_t"])
        assert (
            refined_map.shape[2] == H and refined_map.shape[3] == W
        ), f"TNet spatial resolution changed: input {H}×{W}, output {refined_map.shape[2]}×{refined_map.shape[3]}"

    def test_single_item_batch(self, t_net):
        raw = torch.rand(1, 3, 256, 256, device=DEVICE)
        phyt = torch.rand(1, 1, 256, 256, device=DEVICE)
        refined_map, t_emb = t_net(raw, phyt)
        assert refined_map.shape == (1, 1, 256, 256)
        assert t_emb.shape == (1, 128)

    def test_non_square_input(self, t_net):
        raw = torch.rand(2, 3, 192, 256, device=DEVICE)
        phyt = torch.rand(2, 1, 192, 256, device=DEVICE)
        refined_map, t_emb = t_net(raw, phyt)
        assert refined_map.shape == (2, 1, 192, 256)
        assert t_emb.shape == (2, 128)

    def test_batch_independence(self, t_net, batch):
        _, emb_full = t_net(batch["raw"], batch["physics_t"])
        _, emb_single = t_net(batch["raw"][[0]], batch["physics_t"][[0]])
        assert torch.allclose(
            emb_full[[0]], emb_single, atol=1e-5
        ), "TNet has cross-batch contamination"

    def test_parameter_budget(self, t_net):
        n_params = sum(p.numel() for p in t_net.parameters())
        assert n_params < 4_000_000, f"TNet param count {n_params:,} exceeds 4M budget"

    def test_custom_embed_dim(self):
        net = TNet(embed_dim=64).to(DEVICE)
        raw = torch.rand(2, 3, 256, 256, device=DEVICE)
        phyt = torch.rand(2, 1, 256, 256, device=DEVICE)
        refined_map, t_emb = net(raw, phyt)
        assert t_emb.shape == (2, 64)
        assert refined_map.shape == (2, 1, 256, 256)

    def test_eval_train_consistency(self, t_net, batch):
        t_net.eval()
        with torch.no_grad():
            refined_map, t_emb = t_net(batch["raw"], batch["physics_t"])
        assert refined_map.shape[1:] == (1, 256, 256)
        assert t_emb.shape == (2, 128)
        t_net.train()


# ---------------------------------------------------------------------------
# ConditioningNetworks wrapper tests
# ---------------------------------------------------------------------------


class TestConditioningNetworks:

    def test_output_keys(self, cond_nets, batch):
        out = cond_nets(batch["raw"], batch["physics_A"], batch["physics_t"])
        assert set(out.keys()) == {
            "a_embedding",
            "refined_map",
            "t_embedding",
        }, f"Unexpected output keys: {out.keys()}"

    def test_output_shapes(self, cond_nets, batch):
        out = cond_nets(batch["raw"], batch["physics_A"], batch["physics_t"])
        assert out["a_embedding"].shape == (2, 128)
        assert out["refined_map"].shape == (2, 1, 256, 256)
        assert out["t_embedding"].shape == (2, 128)

    def test_all_outputs_finite(self, cond_nets, batch):
        out = cond_nets(batch["raw"], batch["physics_A"], batch["physics_t"])
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k} contains NaN/Inf"

    def test_end_to_end_gradient(self, cond_nets, batch):
        """Loss on all outputs must propagate gradients to raw input."""
        raw = batch["raw"].requires_grad_(True)
        out = cond_nets(raw, batch["physics_A"], batch["physics_t"])
        loss = (
            out["a_embedding"].sum()
            + out["refined_map"].sum()
            + out["t_embedding"].sum()
        )
        loss.backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()

    def test_parameter_count_total(self, cond_nets):
        n = sum(p.numel() for p in cond_nets.parameters())
        assert n < 5_000_000, f"ConditioningNetworks total params {n:,} exceeds 5M"
        print(f"\n  ConditioningNetworks total params: {n:,}")

    def test_a_net_and_t_net_independent(self, cond_nets, batch):
        """Verify A-Net and T-Net params don't overlap (no weight sharing)."""
        a_ids = {id(p) for p in cond_nets.a_net.parameters()}
        t_ids = {id(p) for p in cond_nets.t_net.parameters()}
        assert a_ids.isdisjoint(t_ids), "A-Net and T-Net share parameters unexpectedly"

    def test_device_cpu(self):
        net = ConditioningNetworks().cpu()
        raw = torch.rand(1, 3, 64, 64)
        A = torch.rand(1, 3)
        phyt = torch.rand(1, 1, 64, 64)
        out = net(raw, A, phyt)
        assert out["a_embedding"].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self):
        net = ConditioningNetworks().cuda()
        raw = torch.rand(2, 3, 256, 256, device="cuda")
        A = torch.rand(2, 3, device="cuda")
        phyt = torch.rand(2, 1, 256, 256, device="cuda")
        out = net(raw, A, phyt)
        assert out["a_embedding"].device.type == "cuda"
        assert out["refined_map"].device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bfloat16_amp(self, cond_nets, batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = cond_nets(batch["raw"], batch["physics_A"], batch["physics_t"])
        # outputs may be bfloat16 inside autocast; shapes must still hold
        assert out["a_embedding"].shape == (2, 128)
        assert out["refined_map"].shape == (2, 1, 256, 256)
