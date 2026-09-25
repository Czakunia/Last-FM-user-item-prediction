"""CompGCN encoder (Vashishth et al., ICLR 2020) on the flat CKG.

PyG 2.6.1 ships no CompGCNConv, so this is a minimal self-contained
implementation of the composition operator phi(h_v, z_r) = h_v - z_r
(subtraction, as in the paper's best variant) with per-relation weight
matrices. The flat homogeneous layout (edge_index + edge_type) matches
build_ckg_graph.build_typed_graph: relation 0/1 are user-item interacts,
2..2+n_rel-1 are KG relations, and their reversed counterparts.

Relation embeddings are updated after every layer (CompGCN paper, sec. 4.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompGCNLayer(nn.Module):
    """Relation-batched CompGCN layer (memory-safe on ~3M-edge graphs).

    A naive per-edge bmm materializes an [E, d, d] intermediate (~10 GB at
    E=3M, d=64). Instead edges are processed one relation at a time, which
    keeps the peak at max-edges-per-relation and lets autograd free each
    relation's intermediate right after its segment is scattered.
    """

    def __init__(self, dim: int, n_relations: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.n_relations = n_relations
        self.w_rel = nn.Parameter(torch.empty(n_relations, dim, dim))
        nn.init.xavier_uniform_(self.w_rel)
        self.w_self = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.w_self)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        rel_emb: torch.Tensor,
        rel_ptr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        for r in range(self.n_relations):
            s = int(rel_ptr[r])
            e = int(rel_ptr[r + 1])
            if e <= s:
                continue
            msg = x[src[s:e]] - rel_emb[r]
            agg.index_add_(0, dst[s:e], msg @ self.w_rel[r])
        deg = torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), 1, dtype=x.dtype, device=x.device))
        agg = agg / deg.clamp_min_(1.0)
        out = agg + x @ self.w_self
        return F.dropout(out, p=self.dropout, training=self.training)


class CompGCNEncoder(nn.Module):
    """Flat CompGCN over [user | entity] nodes; same contract as HGTEncoder."""

    def __init__(
        self,
        n_users: int,
        n_entities: int,
        n_relations: int,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        *,
        embed_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.embed_dim = embed_dim
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.entity_embed = nn.Embedding(n_entities, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        self.rel_embed = nn.Embedding(n_relations, embed_dim)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        self.convs = nn.ModuleList(
            CompGCNLayer(embed_dim, n_relations, dropout=dropout)
            for _ in range(n_layers)
        )
        self.w_rel_out = nn.Parameter(torch.empty(n_relations, embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.w_rel_out)
        self.dropout = dropout
        order = torch.argsort(edge_type)
        self.register_buffer("edge_index", edge_index[:, order].contiguous())
        self.register_buffer("edge_type", edge_type[order].contiguous())
        counts = torch.bincount(edge_type, minlength=n_relations)
        ptr = torch.zeros(n_relations + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, dim=0)
        self.register_buffer("rel_ptr", ptr)

    def encode_all(self, role_enabled: bool = True, aux_enabled: bool = True) -> torch.Tensor:
        x = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        rel = self.rel_embed.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, self.edge_index, self.edge_type, rel, self.rel_ptr)
            rel = torch.bmm(rel.unsqueeze(1), self.w_rel_out).squeeze(1)
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

    def forward(self, edge_index_dict: dict | None = None, **kwargs) -> torch.Tensor:
        return self.encode_all()
