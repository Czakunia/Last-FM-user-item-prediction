"""KAN pair decoder on graph pair representation (+ optional side features)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.lastfm_lp.models.kan_linear import KANLinear
from src.lastfm_lp.models.mlp_decoder import pair_representation


class KANDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        side_dim: int = 0,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        in_dim = 4 * embed_dim + side_dim
        self.kan1 = KANLinear(
            in_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order, grid_update=False
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.kan2 = KANLinear(
            hidden_dim, 1, grid_size=grid_size, spline_order=spline_order, grid_update=False
        )
        self.side_dim = side_dim

    def forward(
        self,
        zu: torch.Tensor,
        zi: torch.Tensor,
        side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = pair_representation(zu, zi)
        if self.side_dim > 0:
            if side is None:
                raise ValueError("side features required")
            x = torch.cat([x, side], dim=-1)
        x = self.kan1(x)
        x = self.norm(x)
        x = self.drop(x)
        return self.kan2(x).squeeze(-1)
