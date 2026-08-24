"""R-GCN wrapper for architecture benchmark."""

from __future__ import annotations

import torch
from torch import Tensor

from src.lastfm_lp.models.encoders.base import BaseGraphEncoder
from src.lastfm_lp.models.rgcn_encoder import RGCNEncoder


class RGCNGraphEncoder(BaseGraphEncoder):
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
        num_bases: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._edge_index = edge_index
        self._edge_type = edge_type
        self.core = RGCNEncoder(
            n_nodes,
            n_relations,
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            n_layers=n_layers,
            dropout=dropout,
            num_bases=num_bases,
        )

    def encode_all(self) -> Tensor:
        return self.core(self._edge_index, self._edge_type)
