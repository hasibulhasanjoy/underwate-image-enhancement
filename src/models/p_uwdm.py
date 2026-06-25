"""
src/models/p_uwdm.py
────────────────────────────────────────────────────────────────────────────
P-UWDM — Physics-Guided Underwater Diffusion Model.

This module is the single entry-point that wires every sub-component into a
unified ``nn.Module`` suitable for training and inference:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Input: raw underwater image  x_raw  (B, 3, H, W)  in [0, 1]    │
    │         physics priors: ambient A (B,3), transmission t (B,1,H,W)│
    │                          degradation (B,6), severity (B,1)       │
    └──────────────────┬───────────────────────────────────────────────┘
                       │
          ┌────────────▼─────────────┐
          │  ConditioningNetworks     │   A-Net  +  T-Net
          │  (src.models.conditioning)│   refined_map, a_emb, t_emb
          └────────────┬─────────────┘
                       │
          ┌────────────▼─────────────┐
          │  DDIMScheduler            │   forward: add_noise → x_t, ε
          │  (src.models.diffusion)   │   reverse: ddim_sample_loop
          └────────────┬─────────────┘
                       │
          ┌────────────▼─────────────┐
          │  SwinUNetDenoiser         │   predicts ε_θ(x_t, t, cond)
          │  (src.models.swin_unet)   │
          └──────────────────────────┘

Training loop usage
───────────────────
    model = PUWDM(cfg)

    # --- forward (training) ---
    out = model.training_step(batch)
    # out["noise_pred"]    : (B, C, H, W)
    # out["noise_target"]  : (B, C, H, W)
    # out["timesteps"]     : (B,)
    # out["alphas_cumprod"]: (T,)
    # out["enhanced"]      : (B, C, H, W)  — single-step denoised estimate
    # out["refined_map"]   : (B, 1, H, W)  — from T-Net (for histogram loss)
    # out["a_embedding"]   : (B, E)
    # out["t_embedding"]   : (B, E)

    # --- inference (sampling) ---
    enhanced = model.sample(
        raw          = batch["raw"],
        physics_A    = batch["ambient"],
        physics_t    = batch["transmission"],
        degradation  = batch["degradation"],
        severity     = batch["severity"],
        num_steps    = 50,
    )  # (B, 3, H, W)

Parameter counts (default config)
──────────────────────────────────
  SwinUNetDenoiser     : ~49 M
  ConditioningNetworks : ~ 1.5 M  (A-Net ~400 K + T-Net ~1.1 M)
  PatchDiscriminator   : ~ 2.8 M  (not part of this module; in CompositeLoss)
  Total denoiser+cond  : ~50.5 M
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.conditioning import ConditioningNetworks
from src.models.diffusion import DDIMScheduler
from src.models.swin_unet import SwinUNetDenoiser, SwinUNetConfig

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PUWDMConfig:
    """
    Top-level configuration for P-UWDM.

    Attributes
    ----------
    denoiser_cfg : SwinUNetConfig
        Full backbone configuration.  ``cond_embed_dim`` inside this config
        must match ``cond_embed_dim`` below.
    cond_embed_dim : int
        Embedding dimension produced by A-Net and T-Net.  Default 128.
    cond_base_ch : int
        Base channel width for A-Net / T-Net CNNs.  Default 32.
    diffusion_T : int
        Total diffusion timesteps.  Default 1000.
    diffusion_s : float
        Cosine schedule offset.  Default 0.008.
    clip_denoised : bool
        Clamp x_0 estimates during DDIM reverse to [-1, 1].  Default True.
    use_ema : bool
        Whether to maintain an EMA copy of the denoiser weights.
        The EMA model is updated externally (by the training loop) via
        ``model.update_ema()``.  Default True.
    ema_decay : float
        EMA smoothing coefficient.  Default 0.9999.
    """

    denoiser_cfg: SwinUNetConfig = field(default_factory=SwinUNetConfig)
    cond_embed_dim: int = 128
    cond_base_ch: int = 32
    diffusion_T: int = 1000
    diffusion_s: float = 0.008
    clip_denoised: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999

    def __post_init__(self) -> None:
        # Keep cond_embed_dim consistent between the wrapper and the denoiser
        self.denoiser_cfg.cond_embed_dim = self.cond_embed_dim


# ─────────────────────────────────────────────────────────────────────────────
# EMA helper
# ─────────────────────────────────────────────────────────────────────────────


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Usage
    -----
        ema = EMAModel(model, decay=0.9999)
        # After each optimiser step:
        ema.update(model)
        # For inference:
        with ema.apply(model):
            output = model(...)  # uses EMA weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Deep-copy shadow parameters
        self.shadow: Dict[str, Tensor] = {
            name: param.data.clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights after an optimiser step."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.data.float(), alpha=1.0 - self.decay
            )

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights into model parameters (for inference)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.data.dtype))

    def restore(self, model: nn.Module, original: Dict[str, Tensor]) -> None:
        """Restore original (non-EMA) weights into model."""
        for name, param in model.named_parameters():
            if name in original:
                param.data.copy_(original[name])

    class _Context:
        """Context manager that temporarily swaps in EMA weights."""

        def __init__(self, ema: "EMAModel", model: nn.Module) -> None:
            self._ema = ema
            self._model = model
            self._original: Dict[str, Tensor] = {}

        def __enter__(self) -> None:
            for name, param in self._model.named_parameters():
                if name in self._ema.shadow:
                    self._original[name] = param.data.clone()
            self._ema.copy_to(self._model)

        def __exit__(self, *args) -> None:
            self._ema.restore(self._model, self._original)

    def apply(self, model: nn.Module) -> "_Context":
        """Return a context manager that temporarily applies EMA weights."""
        return self._Context(self, model)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────


class PUWDM(nn.Module):
    """
    Physics-guided Underwater Diffusion Model.

    Wraps ConditioningNetworks + SwinUNetDenoiser + DDIMScheduler.

    Parameters
    ----------
    cfg : PUWDMConfig
        Complete model configuration.  Defaults to paper-scale settings.

    Notes
    -----
    - The PatchDiscriminator lives inside ``CompositeLoss``; it is NOT
      part of this module so that generator and discriminator optimisers
      can be kept separate.
    - EMA is maintained here but updated by the training loop via
      ``model.update_ema()`` after each generator optimiser step.
    """

    def __init__(self, cfg: Optional[PUWDMConfig] = None) -> None:
        super().__init__()
        cfg = cfg or PUWDMConfig()
        self.cfg = cfg

        # ── Sub-modules ───────────────────────────────────────────────
        self.cond_nets = ConditioningNetworks(
            embed_dim=cfg.cond_embed_dim,
            base_ch=cfg.cond_base_ch,
        )
        self.denoiser = SwinUNetDenoiser(cfg.denoiser_cfg)
        self.scheduler = DDIMScheduler(
            T=cfg.diffusion_T,
            s=cfg.diffusion_s,
            clip_denoised=cfg.clip_denoised,
        )

        # ── EMA (denoiser only — cond_nets are fast to optimise) ─────
        self._ema: Optional[EMAModel] = None
        if cfg.use_ema:
            self._ema = EMAModel(self.denoiser, decay=cfg.ema_decay)

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def update_ema(self) -> None:
        """Call after each generator optimiser step."""
        if self._ema is not None:
            self._ema.update(self.denoiser)

    def ema_context(self):
        """
        Context manager that temporarily applies EMA weights to the denoiser.

        Usage::

            with model.ema_context():
                enhanced = model.sample(...)
        """
        if self._ema is None:
            return _NullContext()
        return self._ema.apply(self.denoiser)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _encode_conditioning(
        self,
        raw: Tensor,
        physics_A: Tensor,
        physics_t: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Run A-Net and T-Net to get learned conditioning embeddings.

        Parameters
        ----------
        raw      : (B, 3, H, W)  raw degraded image in [0, 1]
        physics_A: (B, 3)        physics ambient estimate
        physics_t: (B, 1, H, W)  physics transmission map

        Returns
        -------
        dict with keys: a_embedding, t_embedding, refined_map
        """
        return self.cond_nets(raw, physics_A, physics_t)

    def _predict_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        cond_out: Dict[str, Tensor],
        degradation: Tensor,
        severity: Tensor,
        raw: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run the SwinUNet denoiser with the conditioning tensors.

        raw is concatenated with x_t at the pixel level inside the denoiser
        so it has direct spatial access to the degraded input image.

        Returns ε_pred : (B, C, H, W).
        """
        return self.denoiser(
            x_t=x_t,
            t=t,
            a_embedding=cond_out["a_embedding"],
            t_embedding=cond_out["t_embedding"],
            degradation=degradation,
            severity=severity,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Single training forward pass.

        Expects batch keys:
            raw          : (B, 3, H, W)  normalised degraded image
            reference    : (B, 3, H, W)  normalised clean reference
            ambient      : (B, 3)
            transmission : (B, 1, H, W)
            degradation  : (B, 6)
            severity     : (B, 1)
            t            : (B,), optional diffusion timesteps

        The ``raw`` and ``reference`` tensors are expected in the training
        normalisation range (e.g. ImageNet-normalised or [-1,1]).  Physics
        tensors (ambient, transmission, degradation, severity) must be in
        their native [0,1] / [0,1]^6 ranges as produced by the DataLoader.

        Returns
        -------
        dict with keys required by CompositeLoss.forward():
            noise_pred    : (B, C, H, W)
            noise_target  : (B, C, H, W)
            timesteps     : (B,)
            alphas_cumprod: (T,)            — for SNR weighting
            enhanced      : (B, C, H, W)   — single-step x̂_0 estimate
            refined_map   : (B, 1, H, W)   — from T-Net
            a_embedding   : (B, E)
            t_embedding   : (B, E)
        """
        x0 = batch["reference"]  # (B, C, H, W)
        raw = batch["raw"]  # (B, C, H, W)
        physics_A = batch["ambient"]  # (B, 3)
        physics_t = batch["transmission"]  # (B, 1, H, W)
        degradation = batch["degradation"]  # (B, 6)
        severity = batch["severity"]  # (B, 1)

        B = x0.shape[0]
        device = x0.device

        # 1. Sample timesteps unless the caller supplies them explicitly.
        t = batch.get("t")
        if t is None:
            t = self.scheduler.sample_timesteps(B, device=device)
        else:
            t = t.to(device)

        # 2. Add noise to clean reference
        x_t, eps = self.scheduler.add_noise(x0, t)

        # 3. Conditioning networks
        cond_out = self._encode_conditioning(raw, physics_A, physics_t)

        # 4. Predict noise (pass raw for pixel-level spatial conditioning)
        eps_pred = self._predict_noise(x_t, t, cond_out, degradation, severity, raw=raw)

        # 5. Single-step denoised estimate (for perceptual/histogram/adv losses)
        #    x̂_0 = (x_t − √(1-ᾱ_t)·ε_pred) / √ᾱ_t
        with torch.no_grad():
            enhanced = self.scheduler.predict_x0_from_eps(x_t, t, eps_pred.detach())

        return {
            "noise_pred": eps_pred,
            "noise_target": eps,
            "timesteps": t,
            "alphas_cumprod": self.scheduler.alphas_cumprod,
            "enhanced": enhanced,
            "refined_map": cond_out["refined_map"],
            "a_embedding": cond_out["a_embedding"],
            "t_embedding": cond_out["t_embedding"],
        }

    # ------------------------------------------------------------------
    # Inference / sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        raw: Tensor,
        physics_A: Tensor,
        physics_t: Tensor,
        degradation: Tensor,
        severity: Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        use_ema: bool = True,
        progress: bool = False,
    ) -> Tensor:
        """
        Generate an enhanced image via DDIM reverse diffusion.

        Parameters
        ----------
        raw          : (B, 3, H, W)  raw degraded image in training range
        physics_A    : (B, 3)
        physics_t    : (B, 1, H, W)
        degradation  : (B, 6)
        severity     : (B, 1)
        num_steps    : int  DDIM steps (default 50)
        eta          : float  stochasticity (0 = deterministic)
        use_ema      : bool  use EMA weights if available
        progress     : bool  print step counter

        Returns
        -------
        Tensor  (B, 3, H, W)  enhanced image in training normalisation range
        """
        device = raw.device
        B, C, H, W = raw.shape

        # Compute conditioning once (shared across all DDIM steps)
        cond_out = self._encode_conditioning(raw, physics_A, physics_t)
        a_emb = cond_out["a_embedding"]
        t_emb = cond_out["t_embedding"]

        # Build a closure that the DDIM loop can call.
        # raw is passed so the denoiser can concatenate it with x_t at
        # the pixel level — same as during training.
        def _model_fn(x_t: Tensor, t_step: Tensor) -> Tensor:
            return self.denoiser(
                x_t=x_t,
                t=t_step,
                a_embedding=a_emb,
                t_embedding=t_emb,
                degradation=degradation,
                severity=severity,
                raw=raw,
            )

        ctx = self.ema_context() if use_ema else _NullContext()
        with ctx:
            enhanced = self.scheduler.ddim_sample_loop(
                model_fn=_model_fn,
                shape=(B, C, H, W),
                device=device,
                num_steps=num_steps,
                eta=eta,
                progress=progress,
            )

        return enhanced

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """Return parameter counts per sub-module."""

        def _count(m: nn.Module) -> int:
            params = (
                (p for p in m.parameters() if p.requires_grad)
                if trainable_only
                else m.parameters()
            )
            return sum(p.numel() for p in params)

        return {
            "cond_nets": _count(self.cond_nets),
            "denoiser": _count(self.denoiser),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        counts = self.num_parameters()
        return (
            f"PUWDM(\n"
            f"  cond_nets  : {counts['cond_nets']:>12,} params\n"
            f"  denoiser   : {counts['denoiser']:>12,} params\n"
            f"  total      : {counts['total']:>12,} params\n"
            f"  scheduler  : {self.scheduler}\n"
            f"  ema        : {'enabled' if self._ema else 'disabled'}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Null context (when EMA is disabled)
# ─────────────────────────────────────────────────────────────────────────────


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
