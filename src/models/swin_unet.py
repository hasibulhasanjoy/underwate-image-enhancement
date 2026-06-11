"""SwinUNetDenoiser — P-UWDM denoising backbone.

Architecture overview
---------------------

    Input: x_t (B, 3, H, W) — noisy image at timestep t
           cond_image        — physics-conditioned signal tensor fed via
                               A-Net, T-Net, and degradation estimator outputs
                               (handled externally; the denoiser receives the
                               already-computed physics conditioning vectors).

    Conditioning signals (all (B, *)):
       - t              : diffusion timestep scalar (B,)
       - ambient        : (B, 3)   from A-Net
       - transmission   : (B, 1, H, W) from T-Net
       - degradation    : (B, 6)   from degradation estimator
       - severity       : (B, 1)   from degradation estimator

    Stem
       PatchEmbed (patch_size=4): (B,3,H,W) → (B, H/4·W/4, C0)

    Encoder (3 stages + downsampling between each)
       Stage 0:  C0,  depth d0, num_heads h0   →  skip_0
       PatchMerge → C1
       Stage 1:  C1,  depth d1, num_heads h1   →  skip_1
       PatchMerge → C2
       Stage 2:  C2,  depth d2, num_heads h2   →  skip_2
       PatchMerge → C3

    Bottleneck
       Stage 3:  C3,  depth d3, num_heads h3

    Decoder (3 stages + upsampling between each, with skip fusions)
       PatchExpand  → C2;  fuse skip_2 via ConvResBlock;  Stage 4
       PatchExpand  → C1;  fuse skip_1 via ConvResBlock;  Stage 5
       PatchExpand  → C0;  fuse skip_0 via ConvResBlock;  Stage 6

    Head
       Final PatchExpand ×4 → (B, H, W, C0)
       LayerNorm + Linear   → (B, H, W, 3)  predicted noise ε

Default config (paper-scale, fits 24 GB VRAM at B=16, 256×256):
    embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=8
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.embeddings import (
    SinusoidalTimestepEmbedding,
    TimestepMLP,
    ConditioningProjection,
)
from src.models.blocks import (
    PatchEmbed,
    PatchMerge,
    PatchExpand,
    SwinStage,
    ConvResBlock,
)

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SwinUNetConfig:
    """Hyperparameters for SwinUNetDenoiser.

    Attributes:
        image_size:    Input spatial resolution (must be divisible by
                       patch_size × 2^num_stages = 4×8 = 32 for default).
        in_channels:   Input image channels.
        embed_dim:     Base channel dimension (C0).  Doubles each stage.
        depths:        Number of SwinBlocks per stage [enc0, enc1, enc2,
                       bottleneck, dec0, dec1, dec2].  Length must be 7.
                       Pass None to auto-mirror: [d0,d1,d2,d3,d2,d1,d0].
        num_heads:     Attention heads per stage (same length as depths).
        window_size:   Primary MDWA window side length.
        mlp_ratio:     FFN expansion ratio.
        patch_size:    PatchEmbed stride.
        attn_drop:     Attention dropout.
        proj_drop:     Projection dropout.
        sinusoidal_dim: Raw sinusoidal embedding dimension.
        cond_dim:      Unified conditioning vector dimension.
    """

    image_size: int = 256
    in_channels: int = 3
    embed_dim: int = 96
    depths: list[int] = field(default_factory=lambda: [2, 2, 6, 2, 2, 6, 2])
    num_heads: list[int] = field(default_factory=lambda: [3, 6, 12, 24, 12, 6, 3])
    window_size: int = 8
    mlp_ratio: float = 4.0
    patch_size: int = 4
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    sinusoidal_dim: int = 256
    cond_dim: int = 512

    def __post_init__(self) -> None:
        assert len(self.depths) == 7, "depths must have 7 entries"
        assert len(self.num_heads) == 7, "num_heads must have 7 entries"


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------


class SwinUNetDenoiser(nn.Module):
    """Swin-UNet denoising backbone with MDWA + AdaGN conditioning.

    Predicts the noise ε added to a clean image given the noisy image x_t,
    the timestep t, and physics conditioning signals from A-Net / T-Net /
    degradation estimator.

    Args:
        cfg: SwinUNetConfig instance.  Uses default paper-scale config
             when omitted.

    Example::

        cfg = SwinUNetConfig(image_size=256, embed_dim=96)
        model = SwinUNetDenoiser(cfg).cuda().to(torch.bfloat16)
        noise_pred = model(
            x_t=noisy,          # (B, 3, 256, 256)
            t=timesteps,        # (B,)  long
            ambient=A,          # (B, 3)
            transmission=T,     # (B, 1, 256, 256)
            degradation=D,      # (B, 6)
            severity=S,         # (B, 1)
        )  # → (B, 3, 256, 256)
    """

    def __init__(self, cfg: SwinUNetConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SwinUNetConfig()
        self.cfg = cfg

        C = cfg.embed_dim
        cd = cfg.cond_dim

        # ------------------------------------------------------------------ #
        # Conditioning pipeline                                               #
        # ------------------------------------------------------------------ #
        self.time_embed = SinusoidalTimestepEmbedding(cfg.sinusoidal_dim)
        self.time_mlp = TimestepMLP(cfg.sinusoidal_dim, out_dim=cd)
        self.phys_proj = ConditioningProjection(out_dim=cd)
        # Final fusion: timestep + physics → single cond vector
        self.cond_fuse = nn.Sequential(
            nn.Linear(cd * 2, cd),
            nn.SiLU(),
            nn.Linear(cd, cd),
        )

        # ------------------------------------------------------------------ #
        # Stem                                                                #
        # ------------------------------------------------------------------ #
        self.patch_embed = PatchEmbed(
            in_channels=cfg.in_channels,
            embed_dim=C,
            patch_size=cfg.patch_size,
        )

        # ------------------------------------------------------------------ #
        # Encoder                                                             #
        # ------------------------------------------------------------------ #
        # Channel dimensions: C → 2C → 4C → 8C
        self.enc_stage0 = SwinStage(
            cfg.depths[0],
            C,
            cd,
            cfg.num_heads[0],
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.attn_drop,
            cfg.proj_drop,
        )
        self.down0 = PatchMerge(C)  # → 2C

        self.enc_stage1 = SwinStage(
            cfg.depths[1],
            C * 2,
            cd,
            cfg.num_heads[1],
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.attn_drop,
            cfg.proj_drop,
        )
        self.down1 = PatchMerge(C * 2)  # → 4C

        self.enc_stage2 = SwinStage(
            cfg.depths[2],
            C * 4,
            cd,
            cfg.num_heads[2],
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.attn_drop,
            cfg.proj_drop,
        )
        self.down2 = PatchMerge(C * 4)  # → 8C

        # ------------------------------------------------------------------ #
        # Bottleneck                                                          #
        # ------------------------------------------------------------------ #
        self.bottleneck = SwinStage(
            cfg.depths[3],
            C * 8,
            cd,
            cfg.num_heads[3],
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.attn_drop,
            cfg.proj_drop,
        )

        # ------------------------------------------------------------------ #
        # Decoder                                                             #
        # ------------------------------------------------------------------ #
        # Each level: PatchExpand (halves channels) → ConvResBlock (fuse skip) → SwinStage
        self.up2 = PatchExpand(C * 8)  # 8C → 4C
        self.fuse2 = ConvResBlock(C * 8, C * 4)  # cat(skip C*4, up C*4) → C*4
        self.dec_stage2 = SwinStage(
            cfg.depths[4],
            C * 4,
            cd,
            cfg.num_heads[4],
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.attn_drop,
            cfg.proj_drop,
        )

        self.up1 = PatchExpand(C * 4)  # 4C → 2C
        self.fuse1 = ConvResBlock(C * 4, C * 2)  # cat(skip C*2, up C*2) → C*2
        self.dec_stage1 = SwinStage(
            cfg.depths[5],
            C * 2,
            cd,
            cfg.num_heads[5],
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.attn_drop,
            cfg.proj_drop,
        )

        self.up0 = PatchExpand(C * 2)  # 2C → C
        self.fuse0 = ConvResBlock(C * 2, C)  # cat(skip C, up C) → C
        self.dec_stage0 = SwinStage(
            cfg.depths[6],
            C,
            cd,
            cfg.num_heads[6],
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.attn_drop,
            cfg.proj_drop,
        )

        # ------------------------------------------------------------------ #
        # Head: restore full spatial resolution                               #
        # ------------------------------------------------------------------ #
        # After decoder stage0 we are at H/4, W/4 with dim C.
        # We use two sequential ×2 PatchExpands to reach H, W (×4 total),
        # finishing with C//4 channels.
        self.up_head1 = PatchExpand(C)  # C → C//2  (×2)
        self.up_head2 = PatchExpand(C // 2)  # C//2 → C//4  (×2)

        self.head_norm = nn.LayerNorm(C // 4)
        self.head_proj = nn.Linear(C // 4, cfg.in_channels)

        # Weight initialisation
        self.apply(self._init_weights)

    # ---------------------------------------------------------------------- #
    # Init                                                                    #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------- #
    # Conditioning                                                            #
    # ---------------------------------------------------------------------- #

    def _encode_conditioning(
        self,
        t: Tensor,  # (B,)
        ambient: Tensor,  # (B, 3)
        transmission: Tensor,  # (B, 1, H, W)
        degradation: Tensor,  # (B, 6)
        severity: Tensor,  # (B, 1)
    ) -> Tensor:
        """Build the unified (B, cond_dim) conditioning vector."""
        t_emb = self.time_embed(t)  # (B, sin_dim)
        t_cond = self.time_mlp(t_emb)  # (B, cond_dim)
        p_cond = self.phys_proj(
            ambient, transmission, degradation, severity
        )  # (B, cond_dim)
        cond = self.cond_fuse(torch.cat([t_cond, p_cond], dim=-1))  # (B, cond_dim)
        return cond

    # ---------------------------------------------------------------------- #
    # Skip-connection fusion helper                                           #
    # ---------------------------------------------------------------------- #

    def _fuse_skip(
        self,
        x: Tensor,  # (B, L, C_up)   upsampled
        skip: Tensor,  # (B, L_skip, C_skip)  encoder feature
        H_up: int,
        W_up: int,
        fuse_block: ConvResBlock,
        C_out: int,
    ) -> tuple[Tensor, int, int]:
        """Spatial cat + ConvResBlock fusion.

        Handles minor size mismatches between skip and upsampled tokens by
        cropping the larger tensor (arises from odd-dimension padding in
        PatchMerge / PatchExpand).

        Returns:
            fused: (B, H_up*W_up, C_out)
            H_up, W_up: (unchanged)
        """
        B = x.shape[0]

        # Reshape both to spatial
        x_sp = x.view(B, H_up, W_up, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        H_s = round(skip.shape[1] ** 0.5)  # square assumption
        W_s = skip.shape[1] // H_s
        # More robust: derive from saved H/W (passed as context) – here we
        # assume skip has the same resolution as x (they should after expand)
        skip_sp = skip.view(B, H_s, W_s, -1).permute(0, 3, 1, 2)  # (B, C_skip, H, W)

        # Crop to minimum
        H_min = min(H_up, H_s)
        W_min = min(W_up, W_s)
        x_sp = x_sp[:, :, :H_min, :W_min]
        skip_sp = skip_sp[:, :, :H_min, :W_min]

        cat_sp = torch.cat([skip_sp, x_sp], dim=1)  # (B, C_skip+C_up, H, W)
        fused = fuse_block(cat_sp)  # (B, C_out, H, W)
        fused = fused.flatten(2).transpose(1, 2)  # (B, H*W, C_out)
        return fused, H_min, W_min

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x_t: Tensor,  # (B, in_channels, H, W)  noisy image
        t: Tensor,  # (B,)  integer timestep
        ambient: Tensor,  # (B, 3)
        transmission: Tensor,  # (B, 1, H, W)
        degradation: Tensor,  # (B, 6)
        severity: Tensor,  # (B, 1)
    ) -> Tensor:
        """Predict noise ε from noisy image x_t.

        Returns:
            eps_pred: (B, in_channels, H, W) predicted noise tensor.
        """
        B, _, H_in, W_in = x_t.shape

        # ── Conditioning ─────────────────────────────────────────────────── #
        cond = self._encode_conditioning(
            t, ambient, transmission, degradation, severity
        )

        # ── Stem ─────────────────────────────────────────────────────────── #
        x, H, W = self.patch_embed(x_t)  # (B, H/4·W/4, C)

        # ── Encoder ──────────────────────────────────────────────────────── #
        x = self.enc_stage0(x, cond, H, W)
        skip0 = x  # (B, H/4·W/4, C)
        H0, W0 = H, W

        x, H, W = self.down0(x, H, W)  # (B, H/8·W/8, 2C)
        x = self.enc_stage1(x, cond, H, W)
        skip1 = x
        H1, W1 = H, W

        x, H, W = self.down1(x, H, W)  # (B, H/16·W/16, 4C)
        x = self.enc_stage2(x, cond, H, W)
        skip2 = x
        H2, W2 = H, W

        x, H, W = self.down2(x, H, W)  # (B, H/32·W/32, 8C)

        # ── Bottleneck ───────────────────────────────────────────────────── #
        x = self.bottleneck(x, cond, H, W)

        # ── Decoder ──────────────────────────────────────────────────────── #
        # Level 2
        x, H, W = self.up2(x, H, W)  # → 4C, H/16·W/16
        x, H, W = self._fuse_skip(x, skip2, H, W, self.fuse2, self.cfg.embed_dim * 4)
        x = self.dec_stage2(x, cond, H, W)

        # Level 1
        x, H, W = self.up1(x, H, W)  # → 2C, H/8·W/8
        x, H, W = self._fuse_skip(x, skip1, H, W, self.fuse1, self.cfg.embed_dim * 2)
        x = self.dec_stage1(x, cond, H, W)

        # Level 0
        x, H, W = self.up0(x, H, W)  # → C, H/4·W/4
        x, H, W = self._fuse_skip(x, skip0, H, W, self.fuse0, self.cfg.embed_dim)
        x = self.dec_stage0(x, cond, H, W)

        # ── Head: ×4 upsample back to full resolution ─────────────────────── #
        x, H, W = self.up_head1(x, H, W)  # → C//2, H/2·W/2
        x, H, W = self.up_head2(x, H, W)  # → C//4, H·W

        x = self.head_norm(x)
        x = self.head_proj(x)  # (B, H*W, 3)

        eps_pred = x.view(B, H_in, W_in, self.cfg.in_channels).permute(0, 3, 1, 2)
        return eps_pred

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Returns total (or trainable-only) parameter count."""
        params = (
            self.parameters()
            if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)
