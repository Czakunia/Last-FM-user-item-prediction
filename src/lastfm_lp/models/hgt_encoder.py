"""HGT encoder on heterogeneous CKG (user / entity)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from src.lastfm_lp.models.hgt_conv_residual import HGTConvAddResidual, HGTConvMessageNorm


class HGTEncoder(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_entities: int,
        metadata: tuple,
        embed_dim: int = 32,
        n_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        entity_role: torch.Tensor | None = None,
        aux_kind: str | None = None,
        deg_hist: torch.Tensor | None = None,
        message_norm: bool = False,
        hetero_ln: bool = False,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.embed_dim = embed_dim
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.entity_embed = nn.Embedding(n_entities, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        self.role_proj: nn.Linear | None
        if entity_role is not None:
            if entity_role.ndim != 2 or int(entity_role.size(0)) != n_entities:
                raise ValueError("entity_role must be (n_entities, role_dim)")
            self.role_proj = nn.Linear(int(entity_role.size(1)), embed_dim, bias=False)
            nn.init.zeros_(self.role_proj.weight)
            self.register_buffer("entity_role", entity_role.detach().float().contiguous())
        else:
            self.role_proj = None
            self.entity_role = None
        self.aux_kind = aux_kind
        self.message_norm = bool(message_norm)
        self.use_hetero_ln = bool(hetero_ln)
        if message_norm and aux_kind is not None:
            raise ValueError("message_norm and aux_kind are mutually exclusive in isolated tests")
        if message_norm:
            convs = [
                HGTConvMessageNorm(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(n_layers)
            ]
        elif aux_kind is None:
            convs = [
                HGTConv(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(n_layers)
            ]
        else:
            convs = [
                HGTConvAddResidual(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    metadata=metadata,
                    heads=heads,
                    aux_kind=aux_kind,
                    deg_hist=deg_hist,
                )
                for _ in range(n_layers)
            ]
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout
        self.node_types = metadata[0]
        if hetero_ln:
            lns = []
            for _ in range(n_layers):
                md = nn.ModuleDict({nt: nn.LayerNorm(embed_dim) for nt in metadata[0]})
                for ln in md.values():
                    nn.init.ones_(ln.weight)
                    nn.init.zeros_(ln.bias)
                lns.append(md)
            self.hetero_lns = nn.ModuleList(lns)
        else:
            self.hetero_lns = None

    def entity_input(self, role_enabled: bool = True) -> torch.Tensor:
        x = self.entity_embed.weight
        if role_enabled and self.role_proj is not None and self.entity_role is not None:
            x = x + self.role_proj(self.entity_role)
        return x

    def role_component(self) -> torch.Tensor | None:
        if self.role_proj is None or self.entity_role is None:
            return None
        return self.role_proj(self.entity_role)

    def forward(
        self,
        edge_index_dict: dict,
        role_enabled: bool = True,
        aux_enabled: bool = True,
    ) -> torch.Tensor:
        x_dict = {
            "user": self.user_embed.weight,
            "entity": self.entity_input(role_enabled),
        }
        for i, conv in enumerate(self.convs):
            if isinstance(conv, HGTConvAddResidual):
                x_dict = conv(x_dict, edge_index_dict, aux_enabled=aux_enabled)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            if self.hetero_lns is not None:
                x_dict = {k: self.hetero_lns[i][k](v) for k, v in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {
                    k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                    for k, v in x_dict.items()
                }
        # pack back to flat [user | entity] layout used by pair indexing
        return torch.cat([x_dict["user"], x_dict["entity"]], dim=0)
