"""GraphSAGE encoder on Collaborative Knowledge Graph (users ∪ entities)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        self.convs = nn.ModuleList()
        dims = [embed_dim] + [hidden_dim] * (n_layers - 1) + [embed_dim]
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
        self.dropout = dropout
        self.embed_dim = embed_dim

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embed.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class IDEmbeddingEncoder(nn.Module):
    """C0 / BPR-style: no graph convolution, just ID embeddings."""

    def __init__(self, n_nodes: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        self.embed_dim = embed_dim

    def forward(self, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.embed.weight
