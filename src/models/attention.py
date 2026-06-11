"""Multi-Scale Dynamic-Windowed Attention (MDWA).

MDWA extends standard Swin shifted-window self-attention with two additions:

1. **Multi-scale windows**: attention is computed at two window sizes
   simultaneously (``window_size`` and ``window_size // 2``), each in its own
   head partition.  The outputs are concatenated then projected back to
   ``embed_dim``.  This lets the model capture both fine-grained local texture
   and broader structural context — particularly important for recovering
   colour-degraded underwater regions.

2. **Dynamic window bias**: each head has a learnable scalar gate applied to
   the relative positional bias, letting the network adaptively weight
   positional structure vs. content similarity per layer.

Architecture:
    MDWA(Q, K, V):
        # split heads into two groups
        [Q_large | Q_small], ...  →  attend in window_size and window_size//2
        concat outputs            →  linear projection

The implementation is self-contained and requires only PyTorch + einops.

Public API:
    MDWAttention  — drop-in multi-head self-attention module for Swin blocks.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# ---------------------------------------------------------------------------
# Window partition helpers (no grad, cached shapes)
# ---------------------------------------------------------------------------


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """Partition (B, H, W, C) into non-overlapping windows.

    Returns:
        windows : (num_windows*B, window_size, window_size, C)
        (Hp, Wp): padded spatial dims
    """
    B, H, W, C = x.shape
    # Pad if needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_reverse(
    windows: Tensor, window_size: int, Hp: int, Wp: int, H: int, W: int
) -> Tensor:
    """Reverse of window_partition.

    Args:
        windows : (num_windows*B, window_size, window_size, C)
        Hp, Wp  : padded spatial dims
        H,  W   : original (unpadded) spatial dims
    Returns:
        x : (B, H, W, C)
    """
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# ---------------------------------------------------------------------------
# Relative positional bias table
# ---------------------------------------------------------------------------


class RelativePositionBias(nn.Module):
    """Learnable relative position bias for window attention.

    Args:
        window_size: (height, width) of the attention window.
        num_heads:   Number of attention heads this bias is shared across.
    """

    def __init__(self, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        # Table has (2*ws-1)^2 entries per head
        self.table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        # Pre-compute relative index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # (2, ws, ws)
        coords_flat = coords.flatten(1)  # (2, ws*ws)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative = relative.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        index = relative.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", index)

    def forward(self) -> Tensor:
        """Returns: (num_heads, N, N) bias tensor."""
        N = self.window_size**2
        bias = self.table[self.relative_position_index.view(-1)]  # (N*N, H)
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1)  # (H, N, N)
        return bias.unsqueeze(0)  # (1, H, N, N)


# ---------------------------------------------------------------------------
# Single-scale window attention (internal building block)
# ---------------------------------------------------------------------------


class _WindowAttention(nn.Module):
    """Window-partitioned multi-head self-attention for one scale.

    Args:
        embed_dim:   Total embedding dimension (across all heads).
        window_size: Square window side length.
        num_heads:   Number of attention heads.
        shift:       Whether to apply cyclic shift (SW-MSA).
        attn_drop:   Dropout on attention weights.
        proj_drop:   Dropout after output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rel_pos_bias = RelativePositionBias(window_size, num_heads)
        # Dynamic gate: scalar per head, initialised to 1.0
        self.bias_gate = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x : (B, L, C)  where L = H*W
            H, W: spatial dimensions
        Returns:
            out : (B, L, C)
        """
        B, L, C = x.shape
        ws = self.window_size

        # Reshape to spatial for window partition
        x_2d = x.view(B, H, W, C)

        # Cyclic shift
        shift_size = ws // 2
        if self.shift and shift_size > 0:
            x_2d = torch.roll(x_2d, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # Window partition
        windows, (Hp, Wp) = window_partition(x_2d, ws)  # (nW*B, ws, ws, C)
        nW_B = windows.shape[0]
        windows_seq = windows.view(nW_B, ws * ws, C)  # (nW*B, N, C)

        # QKV
        qkv = self.qkv(windows_seq)  # (nW*B, N, 3C)
        qkv = qkv.reshape(nW_B, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, nW*B, H, N, D)
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (nW*B, H, N, N)

        # Relative position bias with dynamic gate
        bias = self.rel_pos_bias()  # (1, H, N, N)
        gate = self.bias_gate.view(1, self.num_heads, 1, 1)
        attn = attn + bias * gate

        # Attention mask for shifted windows
        if self.shift and shift_size > 0:
            mask = self._compute_mask(Hp, Wp, ws, shift_size, x.device)
            # mask: (nW, N, N); attn: (nW*B, num_heads, N, N)
            nW = mask.shape[0]
            mask = mask.unsqueeze(1)  # (nW, 1, N, N)
            # Tile to match nW*B batch dim
            mask = mask.repeat(B, 1, 1, 1)  # (nW*B, 1, N, N)
            attn = attn + mask  # broadcast over heads

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(nW_B, ws * ws, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reverse windows
        out_2d = out.view(nW_B, ws, ws, C)
        out_2d = window_reverse(out_2d, ws, Hp, Wp, H, W)  # (B, H, W, C)

        # Reverse cyclic shift
        if self.shift and shift_size > 0:
            out_2d = torch.roll(out_2d, shifts=(shift_size, shift_size), dims=(1, 2))

        return out_2d.view(B, H * W, C)

    @staticmethod
    @lru_cache(maxsize=8)
    def _compute_mask(
        Hp: int, Wp: int, ws: int, shift_size: int, device: torch.device
    ) -> Tensor:
        """Compute SW-MSA attention mask.  Cached per (Hp, Wp, ws, shift_size)."""
        img_mask = torch.zeros(1, Hp, Wp, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows, _ = window_partition(img_mask, ws)  # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask


# ---------------------------------------------------------------------------
# MDWA: Multi-Scale Dynamic-Windowed Attention
# ---------------------------------------------------------------------------


class MDWAttention(nn.Module):
    """Multi-Scale Dynamic-Windowed Attention (MDWA).

    Splits the head budget evenly between two window scales:
      - large windows  (window_size)     : capture broader structural context
      - small windows  (window_size // 2): capture fine-grained local texture

    Each scale group runs independent window-partitioned attention with its
    own relative position bias and dynamic gate.  Outputs are concatenated and
    projected to ``embed_dim``.

    When ``window_size // 2 < 1`` (very small patches), the small-scale branch
    is disabled and all heads are allocated to the large-scale branch.

    Args:
        embed_dim:   Total channel dimension.
        window_size: Primary (large) window side length.  Must be even.
        num_heads:   Total number of attention heads (split evenly between
                     the two scales).  Must be even when both scales active.
        shift:       If True, use cyclic shift (SW-MSA variant).
        attn_drop:   Dropout on attention weights.
        proj_drop:   Dropout after the output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        small_ws = window_size // 2
        self.use_multiscale = small_ws >= 2

        if self.use_multiscale:
            # Split heads as evenly as possible; large branch gets the extra head if odd
            heads_large = (num_heads + 1) // 2
            heads_small = num_heads // 2
            # Channel split must keep each branch divisible by its head count
            # Use embed_dim // 2 for each branch; heads adjusted accordingly
            # Ensure divisibility: align dim to head count
            dim_large = (embed_dim // 2 // heads_large) * heads_large
            dim_small = embed_dim - dim_large

            self.attn_large = _WindowAttention(
                dim_large, window_size, heads_large, shift, attn_drop, proj_drop
            )
            self.attn_small = _WindowAttention(
                dim_small, small_ws, heads_small, shift, attn_drop, proj_drop
            )
            self.split_dim = dim_large
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            # Fallback: single-scale (all heads on large window)
            self.attn_large = _WindowAttention(
                embed_dim, window_size, num_heads, shift, attn_drop, proj_drop
            )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x : (B, H*W, C)
        Returns:
            out : (B, H*W, C)
        """
        if not self.use_multiscale:
            return self.attn_large(x, H, W)

        # Split channels across two scales
        x_large, x_small = x.split(self.split_dim, dim=-1)  # each (B, L, C/2)

        out_large = self.attn_large(x_large, H, W)  # (B, L, C/2)
        out_small = self.attn_small(x_small, H, W)  # (B, L, C/2)

        out = torch.cat([out_large, out_small], dim=-1)  # (B, L, C)
        return self.proj_drop(self.proj(out))
