"""HGT encoder adapting existing HGTEncoder to BaseGraphEncoder."""

from __future__ import annotations

import torch
from torch import Tensor

from src.lastfm_lp.models.encoders.base import BaseGraphEncoder
from src.lastfm_lp.models.hgt_encoder import HGTEncoder


class HGTGraphEncoder(BaseGraphEncoder):
    def __init__(
        self,
        n_users: int,
        n_entities: int,
        metadata: tuple,
        edge_index_dict: dict,
        *,
        embed_dim: int = 64,
        n_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        entity_role: Tensor | None = None,
        aux_kind: str | None = None,
        deg_hist: Tensor | None = None,
        message_norm: bool = False,
        hetero_ln: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._edge_index_dict = edge_index_dict
        self.core = HGTEncoder(
            n_users,
            n_entities,
            metadata=metadata,
            embed_dim=embed_dim,
            n_layers=n_layers,
            heads=heads,
            dropout=dropout,
            entity_role=entity_role,
            aux_kind=aux_kind,
            deg_hist=deg_hist,
            message_norm=message_norm,
            hetero_ln=hetero_ln,
        )

    def encode_all(self, role_enabled: bool = True, aux_enabled: bool = True) -> Tensor:
        return self.core(self._edge_index_dict, role_enabled=role_enabled, aux_enabled=aux_enabled)
