"""Building blocks for the SwinUNet denoiser.

Provides:
  SwinBlock       — one Swin transformer block: MDWA → MLP, both conditioned
                    via AdaGN.  Alternates W-MSA / SW-MSA by index parity.
  SwinStage       — sequence of SwinBlocks at a fixed resolution.
  PatchEmbed      — image → patch-token sequence (stem).
  PatchExpand     — upsample patch tokens ×2 (decoder).
  PatchMerge      — downsample patch tokens ×2 (encoder).
  ConvResBlock    — lightweight residual conv block for skip-connection fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.adagn import AdaGN
from src.models.attention import MDWAttention

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PreNormResidual(nn.Module):
    """Applies f(AdaGN(x, cond)) + x — pre-norm residual with conditioning."""

    def __init__(self, fn: nn.Module, adagn: AdaGN) -> None:
        super().__init__()
        self.fn = fn
        self.adagn = adagn

    def forward(self, x: Tensor, cond: Tensor, H: int, W: int) -> Tensor:
        normed = self.adagn(x, cond)
        return self.fn(normed, H, W) + x


# ---------------------------------------------------------------------------
# Feed-forward MLP (inside each Swin block)
# ---------------------------------------------------------------------------


class SwinMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self, x: Tensor, H: int, W: int
    ) -> Tensor:  # H, W unused but for uniform sig
        return self.net(x)


# ---------------------------------------------------------------------------
# Single Swin block
# ---------------------------------------------------------------------------


class SwinBlock(nn.Module):
    """One Swin Transformer block with AdaGN conditioning.

    Structure::

        x → AdaGN → MDWA(shift=even/odd) → +x
          → AdaGN → MLP                  → +x

    Args:
        dim:         Channel dimension of the token sequence.
        cond_dim:    Conditioning vector dimension.
        num_heads:   Number of attention heads.
        window_size: Primary MDWA window size.
        shift:       Whether to apply cyclic shift for this block.
        mlp_ratio:   FFN hidden-dim ratio.
        attn_drop:   Attention dropout rate.
        proj_drop:   Projection dropout rate.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        window_size: int,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.attn = MDWAttention(
            embed_dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            shift=shift,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.mlp = SwinMLP(dim, mlp_ratio, proj_drop)

        self.adagn_attn = AdaGN(dim, cond_dim)
        self.adagn_mlp = AdaGN(dim, cond_dim)

    def forward(self, x: Tensor, cond: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x:    (B, H*W, C) token sequence.
            cond: (B, cond_dim) conditioning vector.
            H, W: spatial dimensions.
        Returns:
            (B, H*W, C)
        """
        # Attention sub-block
        x = x + self.attn(self.adagn_attn(x, cond), H, W)
        # MLP sub-block
        x = x + self.mlp(self.adagn_mlp(x, cond), H, W)
        return x


# ---------------------------------------------------------------------------
# Swin stage (sequence of blocks)
# ---------------------------------------------------------------------------


class SwinStage(nn.Module):
    """Sequence of SwinBlocks at a fixed resolution.

    Alternates non-shifted (W-MSA) and shifted (SW-MSA) blocks to allow
    cross-window information flow.

    Args:
        depth:       Number of SwinBlocks.
        dim:         Channel dimension.
        cond_dim:    Conditioning dimension.
        num_heads:   Attention heads per block.
        window_size: Primary window size.
        mlp_ratio:   FFN expansion ratio.
        attn_drop:   Attention dropout.
        proj_drop:   Projection dropout.
    """

    def __init__(
        self,
        depth: int,
        dim: int,
        cond_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=(i % 2 == 1),  # alternate W-MSA / SW-MSA
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor, cond: Tensor, H: int, W: int) -> Tensor:
        for blk in self.blocks:
            x = blk(x, cond, H, W)
        return x


# ---------------------------------------------------------------------------
# Patch embed / merge / expand
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Image → overlapping patch token sequence (stem).

    Uses a strided 4×4 convolution to map (B, 3, H, W) → (B, H/4*W/4, embed_dim).

    Args:
        in_channels: Input image channels (typically 3 for RGB, or 6 for
                     concatenated noisy+reference).
        embed_dim:   Output token dimension.
        patch_size:  Patch side length (default 4, as in original Swin).
    """

    def __init__(
        self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tokens: (B, H'*W', embed_dim)
            H', W': patch-grid spatial dimensions
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        x = self.norm(x)
        return x, H, W


class PatchMerge(nn.Module):
    """Downsample token sequence ×2 by merging 2×2 patches.

    (B, H*W, C) → (B, H/2 * W/2, 2C)

    Args:
        dim: Input channel dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        B, _, C = x.shape
        x = x.view(B, H, W, C)

        # Pad if H or W is odd
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        Hd, Wd = x.shape[1], x.shape[2]
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.proj(x)
        return x, Hd, Wd


class PatchExpand(nn.Module):
    """Upsample token sequence ×2 by splitting channels into 2×2 spatial.

    (B, H*W, 2C) → (B, 2H * 2W, C)

    Uses a linear projection then pixel-shuffle to avoid checkerboard artefacts.

    Args:
        dim: Input channel dimension (will be halved after expansion).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        B, _, C = x.shape
        x = self.expand(x)  # (B, L, 2C)
        x = x.view(B, H, W, 2 * C)

        # Pixel-shuffle: rearrange (2C) → (2, 2, C//2) spatially
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H, 2, W, 2, C//2)
        x = x.view(B, 2 * H, 2 * W, C // 2)

        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x, 2 * H, 2 * W


# ---------------------------------------------------------------------------
# Skip-connection fusion
# ---------------------------------------------------------------------------


class ConvResBlock(nn.Module):
    """Lightweight residual conv block for fusing skip connections.

    Used at each decoder level to process concatenated (skip + upsampled)
    feature maps before passing them through the Swin stage.

    (B, 2C, H, W) → (B, C, H, W)  — halves channels via 1×1 then refines
    with 3×3 depthwise + 1×1 pointwise.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        def _gn(c: int) -> nn.GroupNorm:
            g = min(32, c)
            while c % g != 0 and g > 1:
                g //= 2
            return nn.GroupNorm(g, c)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _gn(out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),  # DW
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),  # PW
            _gn(out_channels),
            nn.SiLU(),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(x) + self.shortcut(x)
