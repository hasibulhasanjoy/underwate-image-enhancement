"""
Conditioning networks for P-UWDM.

A-Net: Learns a refined ambient-light embedding from the raw degraded image,
       using the physics-estimated A ∈ ℝ³ as an input hint.
       Output: (B, ambient_embed_dim)  [default 128]

T-Net: Learns a refined spatial transmission map, using the raw image and the
       physics DCP map as inputs.  Output has two heads:
         - refined_map  : (B, 1, H, W)  — used in physics/histogram loss
         - t_embedding  : (B, trans_embed_dim)  — used for denoiser conditioning
                          derived from global mean+std pooling → linear projection

Both networks treat the physics priors as *learnable corrections*, not
replacements — the physics estimates are concatenated as input features so the
network can choose how much to trust them.

Design constraints
------------------
- A-Net: no spatial output needed (A is a global scene property)
- T-Net: spatial output is mandatory (t(x) is pixel-wise)
- Parameter budgets: A-Net ~500 K, T-Net ~2–3 M
- No external attention libs required; pure PyTorch + einops
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adagn import AdaGN

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def _make_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with automatic group-count fallback for small channel counts."""
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, num_channels)


class _ConvBnAct(nn.Sequential):
    """Conv2d → GroupNorm → SiLU building block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False
            ),
            _make_norm(out_ch),
            nn.SiLU(inplace=True),
        )


class _ResBlock(nn.Module):
    """Residual block with optional channel projection on the skip path."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            _ConvBnAct(in_ch, out_ch),
            _ConvBnAct(out_ch, out_ch),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


# ---------------------------------------------------------------------------
# A-Net
# ---------------------------------------------------------------------------


class ANet(nn.Module):
    """
    Ambient-light conditioning network.

    Architecture
    ------------
    raw image (B,3,H,W)
      → 4-layer strided-conv stem  →  (B, 128, H/8, W/8)
      → global average pool        →  (B, 128)
      → concat physics A (B,3)     →  (B, 131)
      → 2-layer MLP                →  (B, embed_dim)

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension. Must match what SwinUNetDenoiser expects
        for the ambient branch (default 128).
    base_ch : int
        Base channel width for the CNN stem (default 32).
    """

    def __init__(self, embed_dim: int = 128, base_ch: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        # CNN stem: 3 strided convolutions (stride 2 each → 1/8 spatial)
        self.stem = nn.Sequential(
            _ConvBnAct(3, base_ch, stride=2),  # /2
            _ResBlock(base_ch, base_ch * 2),
            _ConvBnAct(base_ch * 2, base_ch * 2, stride=2),  # /4
            _ResBlock(base_ch * 2, base_ch * 4),
            _ConvBnAct(base_ch * 4, base_ch * 4, stride=2),  # /8
            _ResBlock(base_ch * 4, base_ch * 4),
        )  # output channels = base_ch * 4 = 128

        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 128, 1, 1)

        stem_out = base_ch * 4  # 128

        # MLP: fuse pooled features with physics A hint
        self.mlp = nn.Sequential(
            nn.Linear(stem_out + 3, stem_out),
            nn.SiLU(),
            nn.Linear(stem_out, embed_dim),
        )

    def forward(self, raw: torch.Tensor, physics_A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        raw      : (B, 3, H, W)   raw degraded image, values in [0, 1]
        physics_A: (B, 3)         ambient estimate from AmbientEstimator

        Returns
        -------
        embedding: (B, embed_dim)
        """
        feat = self.pool(self.stem(raw)).flatten(1)  # (B, 128)
        fused = torch.cat([feat, physics_A], dim=1)  # (B, 131)
        return self.mlp(fused)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# T-Net
# ---------------------------------------------------------------------------


class _TNetEncoder(nn.Module):
    """Three-scale encoder shared between T-Net's two heads."""

    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        c = base_ch
        self.enc1 = nn.Sequential(_ConvBnAct(in_ch, c), _ResBlock(c, c))
        self.enc2 = nn.Sequential(
            _ConvBnAct(c, c * 2, stride=2), _ResBlock(c * 2, c * 2)
        )
        self.enc3 = nn.Sequential(
            _ConvBnAct(c * 2, c * 4, stride=2), _ResBlock(c * 4, c * 4)
        )
        self.bottleneck = _ResBlock(c * 4, c * 4)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)  # (B, c,   H,   W)
        e2 = self.enc2(e1)  # (B, 2c,  H/2, W/2)
        e3 = self.enc3(e2)  # (B, 4c,  H/4, W/4)
        b = self.bottleneck(e3)  # (B, 4c,  H/4, W/4)
        return e1, e2, e3, b


class TNet(nn.Module):
    """
    Transmission-map conditioning network.

    Architecture
    ------------
    Inputs concatenated: raw image (B,3,H,W) + physics DCP map (B,1,H,W)
      → (B, 4, H, W)
      → 3-scale encoder with skip connections
      → decoder  → sigmoid  → refined_map (B, 1, H, W)
      → global mean+std pool → (B, 2) → linear → t_embedding (B, embed_dim)

    Two output heads
    ----------------
    refined_map  : spatial (B,1,H,W) — plug into physics / histogram loss
    t_embedding  : vector  (B, embed_dim) — plug into denoiser conditioning

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension for the conditioning vector (default 128).
    base_ch : int
        Base channel width (default 32).  Total params ~2.5 M at default.
    """

    def __init__(self, embed_dim: int = 128, base_ch: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        c = base_ch

        # Encoder (raw + DCP map → 4 input channels)
        self.encoder = _TNetEncoder(in_ch=4, base_ch=c)

        # Decoder with skip connections.
        # Spatial upsampling is done in the forward pass (F.interpolate) before
        # concatenating the skip feature, so no Upsample layer needed here.
        # up2 input: cat(b_up=4c, e2=2c) → 6c channels
        # up1 input: cat(d2_up=2c, e1=c) → 3c channels
        self.up2 = nn.Sequential(
            _ConvBnAct(c * 6, c * 2),
            _ResBlock(c * 2, c * 2),
        )
        self.up1 = nn.Sequential(
            _ConvBnAct(c * 3, c),
            _ResBlock(c, c),
        )

        # Spatial head → refined transmission map
        self.map_head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Embedding head → conditioning vector
        # input: 2 (mean + std of refined map)
        self.embed_head = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(
        self, raw: torch.Tensor, physics_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        raw      : (B, 3, H, W)   raw degraded image, values in [0, 1]
        physics_t: (B, 1, H, W)   transmission map from TransmissionEstimator

        Returns
        -------
        refined_map : (B, 1, H, W)  refined transmission map (sigmoid-bounded)
        t_embedding : (B, embed_dim) conditioning embedding for denoiser
        """
        x = torch.cat([raw, physics_t], dim=1)  # (B, 4, H, W)

        e1, e2, _, b = self.encoder(x)

        # Decode with skip connections.
        # Upsample b (H/4) to match e2 (H/2) before concatenating, then
        # upsample d2 (H/2) to match e1 (H) before the second concat.
        b_up = F.interpolate(b, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([b_up, e2], dim=1))  # (B, 2c, H/2, W/2)

        d2_up = F.interpolate(
            d2, size=e1.shape[2:], mode="bilinear", align_corners=False
        )
        d1 = self.up1(torch.cat([d2_up, e1], dim=1))  # (B, c,  H,   W)

        refined_map = self.map_head(d1)  # (B, 1, H, W)

        # Global statistics for conditioning embedding
        mean = refined_map.mean(dim=[2, 3])  # (B, 1)
        std = refined_map.std(dim=[2, 3])  # (B, 1)
        stats = torch.cat([mean, std], dim=1)  # (B, 2)
        t_embedding = self.embed_head(stats)  # (B, embed_dim)

        return refined_map, t_embedding


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class ConditioningNetworks(nn.Module):
    """
    Wraps A-Net and T-Net into a single module for clean training-loop usage.

    Forward returns a dict so callers can access each output by name without
    relying on tuple position.

    Usage
    -----
        cond = ConditioningNetworks()
        out  = cond(raw, physics_A, physics_t)
        # out["a_embedding"]  : (B, 128)
        # out["refined_map"]  : (B, 1, H, W)
        # out["t_embedding"]  : (B, 128)
    """

    def __init__(self, embed_dim: int = 128, base_ch: int = 32):
        super().__init__()
        self.a_net = ANet(embed_dim=embed_dim, base_ch=base_ch)
        self.t_net = TNet(embed_dim=embed_dim, base_ch=base_ch)

    def forward(
        self,
        raw: torch.Tensor,
        physics_A: torch.Tensor,
        physics_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        raw      : (B, 3, H, W)
        physics_A: (B, 3)         from AmbientEstimator
        physics_t: (B, 1, H, W)   from TransmissionEstimator

        Returns
        -------
        dict with keys: a_embedding, refined_map, t_embedding
        """
        a_embedding = self.a_net(raw, physics_A)
        refined_map, t_embedding = self.t_net(raw, physics_t)
        return {
            "a_embedding": a_embedding,  # (B, 128)
            "refined_map": refined_map,  # (B, 1, H, W)
            "t_embedding": t_embedding,  # (B, 128)
        }
