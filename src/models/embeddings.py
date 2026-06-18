import torch
import torch.nn as nn
import math
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard DDPM-style sinusoidal timestep embedding.
    Converts scalar timestep → vector embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        t: (B,) or (B,1)
        returns: (B, dim)
        """
        if t.dim() == 2:
            t = t[:, 0]

        device = t.device
        half_dim = self.dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return emb


class TimestepMLP(nn.Module):
    """
    Small MLP that converts timestep embedding → conditioning vector.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConditioningProjection(nn.Module):
    """
    Your patched conditioning fusion module (unchanged logic).
    """

    def __init__(self, out_dim: int = 512, cond_embed_dim: int = 128) -> None:
        super().__init__()

        self.ambient_proj = nn.Sequential(
            nn.Linear(cond_embed_dim, cond_embed_dim),
            nn.SiLU(),
        )

        self.trans_proj = nn.Sequential(
            nn.Linear(cond_embed_dim, cond_embed_dim),
            nn.SiLU(),
        )

        self.deg_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
        )

        self.sev_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
        )

        fused_dim = cond_embed_dim * 2 + 64 + 64

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, a_embedding, t_embedding, degradation, severity):
        a = self.ambient_proj(a_embedding)
        t = self.trans_proj(t_embedding)
        deg = self.deg_proj(degradation)
        sev = self.sev_proj(severity)

        fused = torch.cat([a, t, deg, sev], dim=-1)
        return self.fusion(fused)
