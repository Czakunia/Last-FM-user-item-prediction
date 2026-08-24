"""Minimal relation-aware attention encoder (KGAT-style) for CKG."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.lastfm_lp.models.encoders.base import BaseGraphEncoder


class KGATLayer(nn.Module):
    def __init__(self, embed_dim: int, n_relations: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(n_relations)])
        self.att = nn.ModuleList([nn.Linear(embed_dim * 2, 1, bias=False) for _ in range(n_relations)])
        self.dropout = dropout
        self.n_relations = n_relations

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        row, col = edge_index
        out = torch.zeros_like(x)
        # accumulate per-relation messages with attention
        for r in range(self.n_relations):
            mask = edge_type == r
            if not mask.any():
                continue
            r_row = row[mask]
            r_col = col[mask]
            h_i = self.W[r](x[r_row])
            h_j = self.W[r](x[r_col])
            e = F.leaky_relu(self.att[r](torch.cat([h_i, h_j], dim=-1)).squeeze(-1), 0.2)
            # softmax over neighbors of each destination (row)
            exp_e = torch.exp(e - e.max())
            denom = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            denom.index_add_(0, r_row, exp_e)
            alpha = exp_e / denom[r_row].clamp_min(1e-12)
            msg = alpha.unsqueeze(-1) * h_j
            out.index_add_(0, r_row, msg)
        out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
        return out + x  # residual


class KGATEncoder(BaseGraphEncoder):
    def __init__(
        self,
        n_nodes: int,
        n_relations: int,
        edge_index: Tensor,
        edge_type: Tensor,
        *,
        embed_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)
        self.embed = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        self.layers = nn.ModuleList(
            [KGATLayer(embed_dim, n_relations, dropout=dropout) for _ in range(n_layers)]
        )

    def encode_all(self) -> Tensor:
        x = self.embed.weight
        for layer in self.layers:
            x = layer(x, self.edge_index, self.edge_type)
        return x
