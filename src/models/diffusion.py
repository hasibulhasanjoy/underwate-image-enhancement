"""
src/models/diffusion.py
────────────────────────────────────────────────────────────────────────────
DDPM / DDIM Diffusion Process for P-UWDM.

Responsibilities
────────────────
1. **Noise schedule** — cosine β-schedule (Nichol & Dhariwal 2021).
2. **Forward process** — q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I).
3. **DDIM reverse / sampling** — deterministic (η=0) reverse step.
   50-step accelerated sampling uses a full subsequence from t=T-1 → 0.
4. **Utility** — SNR / ᾱ look-up helpers used by DiffusionLoss.

Key fix (v2)
────────────
The original ddim_sample_loop capped the starting timestep at T_max
(last t where ᾱ_t >= 0.01, ≈ 934).  This was wrong for two reasons:

  1. DDIM must start from x_T ~ N(0,I) which corresponds to t = T-1 = 999
     (ᾱ_999 ≈ 0.00009 with cosine schedule — essentially pure noise).
     Starting at t=934 with ᾱ=0.01 is mid-chain noise, not the correct
     starting distribution, causing the sampling to be incoherent.

  2. At t=934, sqrt_recip_alphas_cumprod = 1/sqrt(0.01) = 10, so
     predict_x0_from_eps amplifies eps_pred by 10× before clamping,
     producing garbage x0_pred in every early DDIM step.

The fix: remove the T_max cap entirely.  Always start from t=T-1=999.
The cosine schedule at t=999 has ᾱ≈0.00009, so x_T is correctly dominated
by Gaussian noise and the denoiser unrolls cleanly from full noise → image.

API summary
───────────
    sched = DDIMScheduler(T=1000)

    # Training: sample timestep, add noise
    t     = sched.sample_timesteps(B)         # (B,)  long
    x_t, ε = sched.add_noise(x0, t)           # each (B, C, H, W)

    # Inference: 50-step DDIM loop
    x     = sched.ddim_sample_loop(
                model_fn,                     # callable: (x_t, t) → ε_pred
                shape=(B, 3, H, W),
                device=device,
                num_steps=50,
            )
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_betas(T: int, s: float = 0.008) -> Tensor:
    """
    Cosine β schedule (Nichol & Dhariwal, "Improved DDPM", 2021).

    β_t = 1 − ᾱ_t / ᾱ_{t-1}    clipped to [0, 0.999]

    where  ᾱ_t = f(t/T) / f(0)  and  f(t) = cos²(π/2 · (t/T + s)/(1+s))
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0, 0.999)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────


class DDIMScheduler:
    """
    Cosine-scheduled DDPM / DDIM diffusion process.

    Parameters
    ----------
    T : int
        Total diffusion timesteps (default 1000).
    s : float
        Cosine schedule offset (default 0.008, per Nichol & Dhariwal).
    clip_denoised : bool
        If True, clamp the predicted x_0 estimate to [0, 1] during
        DDIM reverse steps.  Training images are in [0, 1] (no ImageNet norm).
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 0.008,
        clip_denoised: bool = True,
    ) -> None:
        self.T = T
        self.clip_denoised = clip_denoised

        # ── Pre-compute schedule tables (CPU) ──────────────────────────
        betas = _cosine_betas(T, s)  # (T,)
        alphas = 1.0 - betas  # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t  (T,)
        alphas_cumprod_prev = F.pad(  # ᾱ_{t-1} (T,)
            alphas_cumprod[:-1], (1, 0), value=1.0
        )

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        # Derived quantities
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log())
        self.register("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register(
            "sqrt_recipm1_alphas_cumprod",
            (1.0 / alphas_cumprod - 1.0).sqrt(),
        )

        # Posterior variance β̃_t
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

        self._device_cache: Dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Internal buffer registry
    # ------------------------------------------------------------------

    def register(self, name: str, tensor: Tensor) -> None:
        setattr(self, name, tensor)

    def _get(self, name: str, t: Tensor, device: torch.device) -> Tensor:
        """
        Gather schedule values at timestep indices t.
        Returns shape (B, 1, 1, 1) for easy broadcasting with (B, C, H, W).
        """
        tbl = getattr(self, name).to(device)
        vals = tbl[t]  # (B,)
        return vals.view(-1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Public API — training
    # ------------------------------------------------------------------

    def sample_timesteps(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> Tensor:
        """
        Uniformly sample diffusion timesteps for a training batch.

        Returns
        -------
        Tensor  (B,)  dtype=torch.long
        """
        return torch.randint(0, self.T, (batch_size,), device=device)

    def add_noise(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward diffusion: q(x_t | x_0).

        x_t = √ᾱ_t · x_0  +  √(1 - ᾱ_t) · ε,   ε ~ N(0, I)

        Parameters
        ----------
        x0 : Tensor  (B, C, H, W)  clean image in [0, 1]
        t  : Tensor  (B,)           integer timesteps

        Returns
        -------
        x_t : Tensor  (B, C, H, W)  noisy image
        eps : Tensor  (B, C, H, W)  the noise actually added (= training target)
        """
        device = x0.device
        eps = torch.randn_like(x0)
        sqrt_acp = self._get("sqrt_alphas_cumprod", t, device)
        sqrt_1m = self._get("sqrt_one_minus_alphas_cumprod", t, device)
        x_t = sqrt_acp * x0 + sqrt_1m * eps
        return x_t, eps

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """
        Recover x_0 estimate from noise prediction ε.

        x̂_0 = (x_t − √(1-ᾱ_t)·ε) / √ᾱ_t
        """
        device = x_t.device
        recip = self._get("sqrt_recip_alphas_cumprod", t, device)
        recip_m1 = self._get("sqrt_recipm1_alphas_cumprod", t, device)
        x0_pred = recip * x_t - recip_m1 * eps
        if self.clip_denoised:
            x0_pred = x0_pred.clamp(0.0, 1.0)
        return x0_pred

    # ------------------------------------------------------------------
    # DDIM reverse step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: Tensor,
        t: Tensor,
        t_prev: Tensor,
        eps_pred: Tensor,
        eta: float = 0.0,
    ) -> Tensor:
        """
        Single DDIM reverse step:  x_t  →  x_{t_prev}.

        When t_prev == 0, returns x0_pred directly (final clean image).

        Parameters
        ----------
        x_t      : (B, C, H, W)
        t        : (B,)  current timestep
        t_prev   : (B,)  previous (lower) timestep; use -1 for the final step
        eps_pred : (B, C, H, W)  noise prediction from the denoiser
        eta      : float  stochasticity (0 = deterministic DDIM)

        Returns
        -------
        x_prev : (B, C, H, W)
        """
        device = x_t.device

        # Predict x_0
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)

        # On the very last step (t_prev == 0), just return the clean estimate
        # rather than re-adding noise direction which would corrupt the output.
        if t_prev[0].item() <= 0:
            return x0_pred

        acp = self._get("alphas_cumprod", t, device)  # (B,1,1,1)
        acp_prev = self._get("alphas_cumprod", t_prev, device)  # (B,1,1,1)

        # DDIM direction coefficients
        sqrt_1m_acp_prev = (1.0 - acp_prev).sqrt()
        sqrt_1m_acp = (1.0 - acp).sqrt()

        sigma = (
            eta
            * (sqrt_1m_acp_prev / sqrt_1m_acp)
            * ((1.0 - acp / acp_prev).clamp(min=0.0)).sqrt()
        )

        dir_coef = (1.0 - acp_prev - sigma**2).clamp(min=0.0).sqrt()

        x_prev = acp_prev.sqrt() * x0_pred + dir_coef * eps_pred

        if eta > 0.0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    # ------------------------------------------------------------------
    # Full DDIM sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model_fn: Callable[..., Tensor],
        shape: tuple,
        device: torch.device | str,
        num_steps: int = 50,
        eta: float = 0.0,
        x_T: Optional[Tensor] = None,
        progress: bool = False,
        **model_kwargs,
    ) -> Tensor:
        """
        DDIM accelerated sampling.

        Always starts from t = T-1 (pure Gaussian noise) and steps down to
        t = 0 over `num_steps` steps.  This is the correct DDIM procedure:
        x_T ~ N(0, I) corresponds to ᾱ_T ≈ 0, i.e. nearly-pure noise.

        The old T_max cap (starting at ᾱ_t = 0.01) was wrong — it caused
        DDIM to start mid-chain where the input is only 99% noise but the
        sampler treats it as if it were the initial pure-noise distribution,
        producing incoherent outputs.

        Parameters
        ----------
        model_fn  : callable  (x_t, t_tensor, **model_kwargs) → eps_pred
        shape     : (B, C, H, W)
        device    : torch.device
        num_steps : int    DDIM steps (default 50)
        eta       : float  stochasticity (0 = deterministic)
        x_T       : Tensor or None  starting noise; sampled if None
        progress  : bool   print step counter

        Returns
        -------
        Tensor  (B, C, H, W)  enhanced image in [0, 1]
        """
        device = torch.device(device) if isinstance(device, str) else device

        # ── Timestep subsequence ──────────────────────────────────────────
        # Always start from T-1 (= 999 for T=1000).
        # t_seq  : [999, 979, ..., 19]   length = num_steps
        # t_prev : [979, ..., 19,   0]   length = num_steps
        #
        # torch.linspace(T-1, 0, num_steps+1) gives exactly num_steps+1
        # evenly-spaced values from T-1 down to 0 inclusive.
        timesteps = torch.linspace(self.T - 1, 0, num_steps + 1).long()
        t_seq = timesteps[:-1]  # [T-1, ..., small_t]  length=num_steps
        t_prev_seq = timesteps[1:]  # [t-1, ...,  0]        length=num_steps

        # ── Starting noise ────────────────────────────────────────────────
        x = x_T if x_T is not None else torch.randn(shape, device=device)

        # ── Reverse loop ──────────────────────────────────────────────────
        for i, (t_val, t_prev_val) in enumerate(zip(t_seq, t_prev_seq)):
            t_tensor = torch.full(
                (shape[0],), t_val.item(), device=device, dtype=torch.long
            )
            t_prev_tensor = torch.full(
                (shape[0],), t_prev_val.item(), device=device, dtype=torch.long
            )

            eps_pred = model_fn(x, t_tensor, **model_kwargs)
            x = self.ddim_step(x, t_tensor, t_prev_tensor, eps_pred, eta=eta)

            if progress:
                print(
                    f"\r  DDIM step {i + 1}/{num_steps} "
                    f"(t={t_val.item()} → {t_prev_val.item()})",
                    end="",
                    flush=True,
                )

        if progress:
            print()

        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def snr(self, t: Tensor) -> Tensor:
        """
        Signal-to-noise ratio at timestep t.

        SNR(t) = ᾱ_t / (1 - ᾱ_t)
        """
        device = t.device
        acp = self.alphas_cumprod.to(device)[t]
        return acp / (1.0 - acp).clamp(min=1e-8)

    @property
    def alphas_cumprod_tensor(self) -> Tensor:
        """Return the full (T,) ᾱ tensor (used by DiffusionLoss)."""
        return self.alphas_cumprod

    def __repr__(self) -> str:
        acp = self.alphas_cumprod
        return (
            f"DDIMScheduler(T={self.T}, "
            f"ᾱ_min={acp.min():.6f}, ᾱ_max={acp.max():.4f}, "
            f"clip_denoised={self.clip_denoised})"
        )
