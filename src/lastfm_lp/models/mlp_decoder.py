"""MLP pair decoder on [zu, zi, zu⊙zi, |zu−zi|] (+ optional side features)."""

from __future__ import annotations

import torch
import torch.nn as nn


def pair_representation(zu: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
    return torch.cat([zu, zi, zu * zi, (zu - zi).abs()], dim=-1)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        side_dim: int = 0,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        in_dim = 4 * embed_dim + side_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
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
        return self.net(x).squeeze(-1)
