"""LightGCN-style encoder on user–item (+ optional KG) undirected graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.lastfm_lp.models.encoders.base import BaseGraphEncoder


class LightGCNEncoder(BaseGraphEncoder):
    """Normalized neighborhood aggregation without nonlinearities / weights."""

    def __init__(
        self,
        n_nodes: int,
        edge_index: Tensor,
        *,
        embed_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        # precompute normalized adjacency as sparse
        self.register_buffer("edge_index", edge_index)
        row, col = edge_index
        deg = torch.bincount(row, minlength=n_nodes).float().clamp_min(1.0)
        norm = (deg[row] * deg[col]).pow(-0.5)
        self.register_buffer("edge_norm", norm)

    def layer_states(self) -> list[Tensor]:
        """Return [e0, e1, ..., eL] without dropout (diagnostic)."""
        x = self.embed.weight
        states = [x]
        row, col = self.edge_index
        for _ in range(self.n_layers):
            messages = x[col] * self.edge_norm.unsqueeze(-1)
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, messages)
            x = agg
            states.append(x)
        return states

    def encode_all(self) -> Tensor:
        x = self.embed.weight
        out = x
        row, col = self.edge_index
        for _ in range(self.n_layers):
            messages = x[col] * self.edge_norm.unsqueeze(-1)
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, messages)
            x = F.dropout(agg, p=self.dropout, training=self.training)
            out = out + x
        return out / float(self.n_layers + 1)
