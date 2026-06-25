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
       - a_embedding    : (B, 128)  from A-Net          ← Phase 3 change
       - t_embedding    : (B, 128)  from T-Net           ← Phase 3 change
       - degradation    : (B, 6)    from degradation estimator (unchanged)
       - severity       : (B, 1)    from degradation estimator (unchanged)

    [All other architecture details unchanged from Phase 2]

Phase 3 changes (conditioning networks integration)
----------------------------------------------------
Previously the denoiser accepted raw physics estimates:
    ambient      (B, 3)        → projected via Linear(3, 64)
    transmission (B, 1, H, W)  → reduced to mean+std (B, 2) → Linear(2, 64)

Now it accepts learned embeddings from A-Net and T-Net:
    a_embedding  (B, 128)  → projected via Linear(128, 128)
    t_embedding  (B, 128)  → projected via Linear(128, 128)

ConditioningProjection in embeddings.py is updated accordingly.
The total conditioning vector dimension (cond_dim=512) is unchanged.
degradation and severity inputs are unchanged.
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
        cond_embed_dim: Embedding dimension produced by A-Net and T-Net.
                        Must match ANet(embed_dim=X) / TNet(embed_dim=X).
    """

    image_size: int = 256
    in_channels: int = 3  # output channels (noise prediction)
    in_channels_noisy: int = 6  # input channels: concat(raw, x_t) = 6
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
    cond_embed_dim: int = 128  # ← NEW: must match ANet/TNet embed_dim

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

        cfg   = SwinUNetConfig(image_size=256, embed_dim=96)
        cond  = ConditioningNetworks(embed_dim=cfg.cond_embed_dim).cuda()
        model = SwinUNetDenoiser(cfg).cuda().to(torch.bfloat16)

        # Run conditioning networks first:
        cond_out = cond(raw, physics_A, physics_t)

        # Then run denoiser:
        noise_pred = model(
            x_t          = noisy,                    # (B, 3, 256, 256)
            t            = timesteps,                # (B,)  long
            a_embedding  = cond_out["a_embedding"],  # (B, 128)
            t_embedding  = cond_out["t_embedding"],  # (B, 128)
            degradation  = D,                        # (B, 6)
            severity     = S,                        # (B, 1)
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

        # ← CHANGED: pass cond_embed_dim so ConditioningProjection knows the
        #   input size for a_embedding and t_embedding.
        self.phys_proj = ConditioningProjection(
            out_dim=cd,
            cond_embed_dim=cfg.cond_embed_dim,
        )

        # Final fusion: timestep + physics → single cond vector (unchanged)
        self.cond_fuse = nn.Sequential(
            nn.Linear(cd * 2, cd),
            nn.SiLU(),
            nn.Linear(cd, cd),
        )

        # ------------------------------------------------------------------ #
        # Stem                                                                #
        # ------------------------------------------------------------------ #
        # Input stem: cat(raw, x_t) = 6 channels.
        self.patch_embed = PatchEmbed(
            in_channels=cfg.in_channels_noisy,
            embed_dim=C,
            patch_size=cfg.patch_size,
        )

        # ------------------------------------------------------------------ #
        # Encoder                                                             #
        # ------------------------------------------------------------------ #
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
        self.up2 = PatchExpand(C * 8)
        self.fuse2 = ConvResBlock(C * 8, C * 4)
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

        self.up1 = PatchExpand(C * 4)
        self.fuse1 = ConvResBlock(C * 4, C * 2)
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

        self.up0 = PatchExpand(C * 2)
        self.fuse0 = ConvResBlock(C * 2, C)
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
        self.up_head1 = PatchExpand(C)  # C   → C//2  (×2)
        self.up_head2 = PatchExpand(C // 2)  # C//2 → C//4 (×2)

        self.head_norm = nn.LayerNorm(C // 4)
        self.head_proj = nn.Linear(C // 4, cfg.in_channels)

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
        a_embedding: Tensor,  # (B, cond_embed_dim)  from A-Net   ← CHANGED
        t_embedding: Tensor,  # (B, cond_embed_dim)  from T-Net   ← CHANGED
        degradation: Tensor,  # (B, 6)                            unchanged
        severity: Tensor,  # (B, 1)                            unchanged
    ) -> Tensor:
        """Build the unified (B, cond_dim) conditioning vector.

        Changes from Phase 2
        --------------------
        Previously received raw ``ambient (B,3)`` and ``transmission (B,1,H,W)``
        and reduced them inside ConditioningProjection.  Now receives the
        128-dim learned embeddings from A-Net and T-Net directly.
        """
        t_emb = self.time_embed(t)  # (B, sin_dim)
        t_cond = self.time_mlp(t_emb)  # (B, cond_dim)
        p_cond = self.phys_proj(
            a_embedding, t_embedding, degradation, severity
        )  # (B, cond_dim)
        cond = self.cond_fuse(torch.cat([t_cond, p_cond], dim=-1))  # (B, cond_dim)
        return cond

    # ---------------------------------------------------------------------- #
    # Skip-connection fusion helper (unchanged from Phase 2)                 #
    # ---------------------------------------------------------------------- #

    def _fuse_skip(
        self,
        x: Tensor,
        skip: Tensor,
        H_up: int,
        W_up: int,
        fuse_block: ConvResBlock,
        C_out: int,
    ) -> tuple[Tensor, int, int]:
        B = x.shape[0]

        x_sp = x.view(B, H_up, W_up, -1).permute(0, 3, 1, 2)
        H_s = round(skip.shape[1] ** 0.5)
        W_s = skip.shape[1] // H_s
        skip_sp = skip.view(B, H_s, W_s, -1).permute(0, 3, 1, 2)

        H_min, W_min = min(H_up, H_s), min(W_up, W_s)
        x_sp = x_sp[:, :, :H_min, :W_min]
        skip_sp = skip_sp[:, :, :H_min, :W_min]

        cat_sp = torch.cat([skip_sp, x_sp], dim=1)
        fused = fuse_block(cat_sp)
        fused = fused.flatten(2).transpose(1, 2)
        return fused, H_min, W_min

    # ---------------------------------------------------------------------- #
    # Forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x_t: Tensor,  # (B, 3, H, W)  noisy image at timestep t
        t: Tensor,  # (B,)  integer timestep
        a_embedding: Tensor,  # (B, cond_embed_dim)  from A-Net
        t_embedding: Tensor,  # (B, cond_embed_dim)  from T-Net
        degradation: Tensor,  # (B, 6)
        severity: Tensor,  # (B, 1)
        raw: Tensor | None = None,  # (B, 3, H, W)  degraded input image
    ) -> Tensor:
        """Predict noise ε from noisy image x_t.

        Parameters
        ----------
        x_t         : (B, 3, H, W) noisy image at timestep t
        t           : (B,) integer diffusion timestep
        a_embedding : (B, 128) ambient-light embedding from A-Net
        t_embedding : (B, 128) transmission embedding from T-Net
        degradation : (B, 6)   degradation feature vector
        severity    : (B, 1)   degradation severity scalar
        raw         : (B, 3, H, W) raw degraded image — concatenated with x_t
                      as pixel-level conditioning so the denoiser has direct
                      spatial access to the input.  If None, falls back to
                      zeros (for backward compatibility only).

        Returns
        -------
        eps_pred : (B, 3, H, W) predicted noise tensor
        """
        B, _, H_in, W_in = x_t.shape

        # ── Pixel-level conditioning: cat(raw, x_t) → 6-channel stem input ─ #
        if raw is None:
            raw = torch.zeros_like(x_t)
        x_in = torch.cat([raw, x_t], dim=1)  # (B, 6, H, W)

        # ── Conditioning ─────────────────────────────────────────────────── #
        cond = self._encode_conditioning(
            t, a_embedding, t_embedding, degradation, severity
        )

        # ── Stem ─────────────────────────────────────────────────────────── #
        x, H, W = self.patch_embed(x_in)

        # ── Encoder ──────────────────────────────────────────────────────── #
        x = self.enc_stage0(x, cond, H, W)
        skip0, H0, W0 = x, H, W

        x, H, W = self.down0(x, H, W)
        x = self.enc_stage1(x, cond, H, W)
        skip1, H1, W1 = x, H, W

        x, H, W = self.down1(x, H, W)
        x = self.enc_stage2(x, cond, H, W)
        skip2, H2, W2 = x, H, W

        x, H, W = self.down2(x, H, W)

        # ── Bottleneck ───────────────────────────────────────────────────── #
        x = self.bottleneck(x, cond, H, W)

        # ── Decoder ──────────────────────────────────────────────────────── #
        x, H, W = self.up2(x, H, W)
        x, H, W = self._fuse_skip(x, skip2, H, W, self.fuse2, self.cfg.embed_dim * 4)
        x = self.dec_stage2(x, cond, H, W)

        x, H, W = self.up1(x, H, W)
        x, H, W = self._fuse_skip(x, skip1, H, W, self.fuse1, self.cfg.embed_dim * 2)
        x = self.dec_stage1(x, cond, H, W)

        x, H, W = self.up0(x, H, W)
        x, H, W = self._fuse_skip(x, skip0, H, W, self.fuse0, self.cfg.embed_dim)
        x = self.dec_stage0(x, cond, H, W)

        # ── Head ─────────────────────────────────────────────────────────── #
        x, H, W = self.up_head1(x, H, W)
        x, H, W = self.up_head2(x, H, W)

        x = self.head_norm(x)
        x = self.head_proj(x)
        eps_pred = x.view(B, H_in, W_in, self.cfg.in_channels).permute(0, 3, 1, 2)
        return eps_pred

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (
            self.parameters()
            if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)
