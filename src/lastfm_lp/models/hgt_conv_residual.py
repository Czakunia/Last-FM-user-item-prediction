"""Local HGTConv fork: stock ADD plus a gated auxiliary aggregation.

Does not modify site-packages. With beta=0, AGG = ADD (stock path).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.aggr import (
    DegreeScalerAggregation,
    MaxAggregation,
    MultiAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
)
from torch_geometric.nn.conv.hgt_conv import HGTConv
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import degree
from torch_geometric.utils.hetero import construct_bipartite_edge_index


AUX_KINDS = ("max", "softmax", "powermean", "multi", "pna")


def dest_in_degree_histogram(edge_index_dict: dict, n_users: int, n_entities: int) -> Tensor:
    """In-degree histogram on the typed G_FULL dest space [user | entity]."""
    deg_u = torch.zeros(n_users, dtype=torch.long)
    deg_e = torch.zeros(n_entities, dtype=torch.long)
    for (src, _rel, dst), ei in edge_index_dict.items():
        if ei is None or ei.numel() == 0:
            continue
        tgt = ei[1].detach().cpu().long()
        if dst == "user":
            deg_u.scatter_add_(0, tgt, torch.ones_like(tgt))
        elif dst == "entity":
            deg_e.scatter_add_(0, tgt, torch.ones_like(tgt))
    deg = torch.cat([deg_u, deg_e], dim=0)
    return torch.bincount(deg, minlength=1).to(torch.float)


def _sanitize(aux: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Empty neighborhoods → 0; never keep -inf / NaN."""
    if aux.numel() == 0:
        return aux
    deg = degree(index, num_nodes=dim_size, dtype=torch.long)
    while deg.dim() < aux.dim():
        deg = deg.unsqueeze(-1)
    aux = aux.masked_fill(deg == 0, 0.0)
    return torch.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)


class HGTConvAddResidual(HGTConv):
    """Stock HGTConv ADD plus gamma[t] * AUX, gamma = tanh(beta[t]), beta=0."""

    def __init__(
        self,
        in_channels,
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        aux_kind: str = "max",
        deg_hist: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__(in_channels, out_channels, metadata, heads=heads, **kwargs)
        if aux_kind not in AUX_KINDS:
            raise ValueError(f"unknown aux_kind {aux_kind}")
        self.aux_kind = aux_kind
        self.beta = ParameterDict(
            {nt: Parameter(torch.zeros(1)) for nt in self.node_types}
        )
        self.aux_proj: Optional[Linear] = None
        if aux_kind == "max":
            self.aux_aggr = MaxAggregation()
        elif aux_kind == "softmax":
            self.aux_aggr = SoftmaxAggregation(learn=True)
        elif aux_kind == "powermean":
            self.aux_aggr = PowerMeanAggregation(learn=True)
        elif aux_kind == "multi":
            self.aux_aggr = MultiAggregation(["mean", "max", "std"])
            self.aux_proj = Linear(out_channels * 3, out_channels, bias=False)
        elif aux_kind == "pna":
            if deg_hist is None:
                raise ValueError("pna aux_kind requires deg_hist")
            self.aux_aggr = DegreeScalerAggregation(
                aggr=["mean", "max", "std"],
                scaler=["identity", "amplification", "attenuation"],
                deg=deg_hist.detach().float().cpu(),
                train_norm=False,
            )
            self.aux_proj = Linear(out_channels * 9, out_channels, bias=False)
        self._cached_aux: Optional[Tensor] = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "beta"):
            for p in self.beta.values():
                p.data.zero_()
        if hasattr(self, "aux_aggr") and hasattr(self.aux_aggr, "reset_parameters"):
            self.aux_aggr.reset_parameters()

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        add = self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)
        # Stop-grad the neighborhood reduce. Keeping MAX/SOFTMAX/MULTI in the
        # autograd graph doubles HGT backward memory and OOMs 16GB MPS.
        # beta (and aux_proj, if any) still receive gradients.
        with torch.no_grad():
            aux = self.aux_aggr(inputs.detach(), index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)
        n = int(dim_size if dim_size is not None else add.size(0))
        aux = _sanitize(aux, index, n)
        if self.aux_proj is not None:
            aux = self.aux_proj(aux)
        self._cached_aux = aux
        return add

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        aux_enabled: bool = True,
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(k_dict, v_dict, edge_index_dict)
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel, num_nodes=k.size(0)
        )
        add = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)
        aux = self._cached_aux
        if aux is None:
            aux = torch.zeros_like(add)

        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                chunk = add[start_offset:end_offset]
                if aux_enabled:
                    gamma = torch.tanh(self.beta[node_type])
                    chunk = chunk + gamma * aux[start_offset:end_offset]
                out_dict[node_type] = chunk

        a_dict = self.out_lin(
            {
                k: torch.nn.functional.gelu(v) if v is not None else v
                for k, v in out_dict.items()
            }
        )
        for node_type, out in out_dict.items():
            out = a_dict[node_type]
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out
        self._cached_aux = None
        return out_dict

    def gamma_dict(self) -> dict[str, float]:
        return {nt: float(torch.tanh(self.beta[nt]).detach()) for nt in self.beta}

    def beta_dict(self) -> dict[str, float]:
        return {nt: float(self.beta[nt].detach()) for nt in self.beta}


class HGTConvMessageNorm(HGTConv):
    """Stock HGTConv with MessageNorm on aggregated messages, before out_lin/skip.

    Insertion: after propagate ADD, per destination type:
        M_t ← MessageNorm(x_t, M_t)
    then stock GELU → out_lin → skip.
    """

    def __init__(self, in_channels, out_channels: int, metadata: Metadata, heads: int = 1, **kwargs) -> None:
        super().__init__(in_channels, out_channels, metadata, heads=heads, **kwargs)
        self.msg_norm = MessageNorm(learn_scale=True)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H
        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)
        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(k_dict, v_dict, edge_index_dict)
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel, num_nodes=k.size(0)
        )
        add = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                chunk = add[start_offset:end_offset]
                chunk = self.msg_norm(x_dict[node_type], chunk)
                out_dict[node_type] = chunk
        a_dict = self.out_lin(
            {k: torch.nn.functional.gelu(v) if v is not None else v for k, v in out_dict.items()}
        )
        for node_type, out in out_dict.items():
            out = a_dict[node_type]
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out
        return out_dict
