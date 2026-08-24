"""Shared late-fusion head: graph ⊕ A ⊕ frozen H2 HCR (+ optional Stage-P path blocks)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.lastfm_lp.models.encoders.base import BaseGraphEncoder, EncoderOutput
from src.lastfm_lp.models.kan_linear import KANLinear

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _input_dim(graph_dim: int, base_feature_dim: int, hcr_feature_dim: int, use_hcr: bool) -> int:
    return graph_dim + 1 + base_feature_dim + (hcr_feature_dim if use_hcr else 0)


class LateFusionHead(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        base_feature_dim: int = 5,
        hcr_feature_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_hcr: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.use_hcr = use_hcr
        act_name = activation.lower()
        if act_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation}; choose from {sorted(ACTIVATIONS)}")
        act_cls = ACTIVATIONS[act_name]
        input_dim = _input_dim(graph_dim, base_feature_dim, hcr_feature_dim, use_hcr)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.activation = act_name

    def forward(
        self,
        graph_context: Tensor,
        graph_score: Tensor,
        base_features: Tensor,
        hcr_features: Tensor | None = None,
        path_features: Tensor | None = None,
        inter_features: Tensor | None = None,
    ) -> Tensor:
        del path_features, inter_features
        if graph_score.ndim == 1:
            graph_score = graph_score.unsqueeze(-1)
        parts = [graph_context, graph_score, base_features]
        if self.use_hcr:
            if hcr_features is None:
                raise ValueError("hcr_features required when use_hcr=True")
            parts.append(hcr_features)
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class BlockLateFusionHead(nn.Module):
    """Late fusion = concat(graph, score, A, H2, P, C) → MLP (same recipe as LateFusionHead).

    No pre-concat LN on side blocks: features are StandardScaled on train; the MLP
    LayerNorm is the only normalization (keeps P0 ≡ HGT_H2 when P/C absent).
    """

    def __init__(
        self,
        graph_dim: int,
        base_feature_dim: int = 5,
        hcr_feature_dim: int = 8,
        path_feature_dim: int = 0,
        inter_feature_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_hcr: bool = True,
        use_base: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.use_hcr = use_hcr
        self.use_base = use_base
        self.path_feature_dim = int(path_feature_dim)
        self.inter_feature_dim = int(inter_feature_dim)
        act_name = activation.lower()
        if act_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation}")
        act_cls = ACTIVATIONS[act_name]
        self.activation = act_name

        input_dim = graph_dim + 1
        if use_base:
            input_dim += base_feature_dim
        if use_hcr:
            input_dim += hcr_feature_dim
        input_dim += path_feature_dim + inter_feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        graph_context: Tensor,
        graph_score: Tensor,
        base_features: Tensor,
        hcr_features: Tensor | None = None,
        path_features: Tensor | None = None,
        inter_features: Tensor | None = None,
    ) -> Tensor:
        if graph_score.ndim == 1:
            graph_score = graph_score.unsqueeze(-1)
        parts = [graph_context, graph_score]
        if self.use_base:
            parts.append(base_features)
        if self.use_hcr:
            if hcr_features is None:
                raise ValueError("hcr_features required when use_hcr=True")
            parts.append(hcr_features)
        if self.path_feature_dim > 0:
            if path_features is None:
                raise ValueError("path_features required")
            parts.append(path_features)
        if self.inter_feature_dim > 0:
            if inter_features is None:
                raise ValueError("inter_features required")
            parts.append(inter_features)
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class KANFusionHead(nn.Module):
    """KAN readout on the same concatenated fusion vector as LateFusionHead."""

    def __init__(
        self,
        graph_dim: int,
        base_feature_dim: int = 5,
        hcr_feature_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_hcr: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.use_hcr = use_hcr
        input_dim = _input_dim(graph_dim, base_feature_dim, hcr_feature_dim, use_hcr)
        self.kan1 = KANLinear(
            input_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order, grid_update=False
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.kan2 = KANLinear(
            hidden_dim, 1, grid_size=grid_size, spline_order=spline_order, grid_update=False
        )
        self.activation = "kan"

    def forward(
        self,
        graph_context: Tensor,
        graph_score: Tensor,
        base_features: Tensor,
        hcr_features: Tensor | None = None,
        path_features: Tensor | None = None,
        inter_features: Tensor | None = None,
    ) -> Tensor:
        del path_features, inter_features
        if graph_score.ndim == 1:
            graph_score = graph_score.unsqueeze(-1)
        parts = [graph_context, graph_score, base_features]
        if self.use_hcr:
            if hcr_features is None:
                raise ValueError("hcr_features required when use_hcr=True")
            parts.append(hcr_features)
        x = torch.cat(parts, dim=-1)
        x = self.kan1(x)
        x = self.norm(x)
        x = self.drop(x)
        return self.kan2(x).squeeze(-1)


class RecommendationModel(nn.Module):
    def __init__(
        self,
        encoder: BaseGraphEncoder,
        fusion_head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.fusion_head = fusion_head

    def forward(
        self,
        user_idx: Tensor,
        item_idx: Tensor,
        base_features: Tensor,
        hcr_features: Tensor | None = None,
        path_features: Tensor | None = None,
        inter_features: Tensor | None = None,
        z: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if z is None:
            z = self.encoder.encode_all()
        enc: EncoderOutput = self.encoder.pair_outputs(z, user_idx, item_idx)
        logits = self.fusion_head(
            graph_context=enc.graph_context,
            graph_score=enc.graph_score,
            base_features=base_features,
            hcr_features=hcr_features,
            path_features=path_features,
            inter_features=inter_features,
        )
        return {
            "logits": logits,
            "graph_score": enc.graph_score,
            "graph_context": enc.graph_context,
        }
