"""
src/models/diffusion.py
────────────────────────────────────────────────────────────────────────────
DDPM / DDIM Diffusion Process for P-UWDM.

Responsibilities
────────────────
1. **Noise schedule** — cosine β-schedule (Nichol & Dhariwal 2021), which
   avoids the abrupt SNR drop at t→T that the linear schedule suffers from.
2. **Forward process** — q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I).
   Returns the noisy image and the noise sample used to produce it.
3. **DDIM reverse / sampling** — deterministic (η=0) or stochastic (η>0)
   reverse step following Song et al. 2020 (DDIM).  50-step accelerated
   sampling uses a subsequence capped at T_max where ᾱ_t >= 0.01.
4. **Utility** — SNR / ᾱ look-up helpers used by DiffusionLoss.

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

Design notes
────────────
- All schedule tensors live on CPU and are moved to the appropriate device
  on first use via `.to(device)`.  This avoids CUDA initialisation in
  DataLoader workers.
- The scheduler is stateless between calls; it holds only the pre-computed
  schedule tables.
- `ddim_sample_loop` accepts an arbitrary `model_fn` so it can be used with
  the full P-UWDM forward pass or with classifier-free guidance in future.
- T_max is capped at the last t where ᾱ_t >= 0.01 to avoid the near-zero
  signal region (t > 934 with cosine schedule) which produces pure noise.
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

        # Derived quantities used in posterior q(x_{t-1}|x_t, x_0)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log())
        self.register("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register(
            "sqrt_recipm1_alphas_cumprod",
            (1.0 / alphas_cumprod - 1.0).sqrt(),
        )

        # Posterior variance β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
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

        # Pre-compute T_max: last timestep where ᾱ_t >= 0.01
        # (cosine schedule collapses to ~2e-9 at t=999; sampling above T_max
        #  means starting from near-pure noise with no recoverable signal)
        valid = (alphas_cumprod >= 0.01).nonzero(as_tuple=True)[0]
        self._T_max = int(valid.max().item()) if len(valid) > 0 else T - 1

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
            # Training images are in [0, 1] — clamp to keep reverse chain stable.
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

        Song et al. (DDIM, 2020) equation 12:
            x_{t-1} = √ᾱ_{t-1} · x̂_0
                    + √(1 - ᾱ_{t-1} - σ²_t) · ε_θ(x_t, t)
                    + σ_t · ε      (σ_t = 0 for deterministic DDIM)

        Parameters
        ----------
        x_t      : (B, C, H, W)
        t        : (B,)  current timestep
        t_prev   : (B,)  previous (lower) timestep in the subsequence
        eps_pred : (B, C, H, W)  noise prediction from the denoiser
        eta      : float  stochasticity (0 = deterministic DDIM)

        Returns
        -------
        x_prev : (B, C, H, W)
        """
        device = x_t.device

        acp = self._get("alphas_cumprod", t, device)  # (B,1,1,1)
        acp_prev = self._get("alphas_cumprod", t_prev, device)  # (B,1,1,1)

        # Predict x_0
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)

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

        Key fix: timestep subsequence is capped at self._T_max (the last t
        where ᾱ_t >= 0.01).  The original loop started at t=980 where
        ᾱ_980 = 0.00088 — essentially pure noise with no recoverable signal,
        which is why all outputs were pure noise regardless of conditioning.

        Parameters
        ----------
        model_fn  : callable  (x_t, t_tensor) → eps_pred (B, C, H, W)
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
        # Cap at T_max where ᾱ_t >= 0.01 to stay in the recoverable region.
        # With cosine schedule T=1000: T_max ~ 934, ᾱ_934 ~ 0.010.
        T_max = self._T_max
        timesteps = torch.linspace(T_max, 0, num_steps + 1).long()
        t_seq = timesteps[:-1]  # [T_max, ..., small_t]  length=num_steps
        t_prev_seq = timesteps[1:]  # [t-1,   ..., 0]        length=num_steps

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
            f"DDIMScheduler(T={self.T}, T_max={self._T_max}, "
            f"ᾱ_min={acp.min():.4f}, ᾱ_max={acp.max():.4f}, "
            f"clip_denoised={self.clip_denoised})"
        )
