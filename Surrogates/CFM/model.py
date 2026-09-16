"""
Velocity field network for conditional flow matching.

Architecture: plain MLP with sinusoidal time embedding.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding: t ∈ [0,1] -> R^dim."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Pre-compute log-spaced frequencies
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) or (B, 1) → (B, dim)."""
        if t.ndim == 2:
            t = t.squeeze(1)
        angles = t[:, None] * self.freqs[None, :]  # (B, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B, dim)
        return self.mlp(emb)


class VelocityMLP(nn.Module):
    """
    Plain MLP velocity field for conditional flow matching.

    Forward:  v_θ(x_t, t, c) → R^target_dim

    Architecture (default, ~1.4M params):
        t_embed = SinusoidalTimeEmbedding(t, dim=128)
        h = concat([x_t (4), t_embed (128), c (26)])   → 158
        h → Linear(158, 512) → SiLU
        h → [Linear(512, 512) → SiLU] × (depth - 1)
        h → Linear(512, 4)
    """

    def __init__(
        self,
        cond_dim: int = 26,
        target_dim: int = 4,
        width: int = 512,
        depth: int = 6,
        t_embed_dim: int = 128,
        backbone: str = "plain_mlp",
    ):
        super().__init__()
        if backbone != "plain_mlp":
            raise NotImplementedError(
                f"Backbone '{backbone}' not implemented. "
                "Only 'plain_mlp' is supported. Other architectures can be "
                "added later as an appendix robustness check."
            )

        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.width = width
        self.depth = depth

        self.t_embedder = SinusoidalTimeEmbedding(t_embed_dim)

        in_dim = target_dim + t_embed_dim + cond_dim
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.SiLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(width, target_dim))
        self.net = nn.Sequential(*layers)

        # Zero-initialize the final layer for stable early training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, target_dim)  noisy state
        t : (B, 1) or (B,)  time in [0, 1]
        c : (B, cond_dim)  conditioner

        Returns
        -------
        v : (B, target_dim)  predicted velocity
        """
        t_emb = self.t_embedder(t)  # (B, t_embed_dim)
        h = torch.cat([x_t, t_emb, c], dim=-1)
        return self.net(h)
