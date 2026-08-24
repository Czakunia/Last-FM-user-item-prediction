"""Flat LateFusion (reuse) + separated branch fusion for native-measure screen."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.lastfm_lp.models.encoders.base import BaseGraphEncoder, EncoderOutput
from src.lastfm_lp.native_measure_screen.constants import BRANCH_FUSION_IN


class BranchFusionHead(nn.Module):
    """Separate graph / tabular-A / measure branches → late concat → MLP.

    F = [G^64 | T^16 | M^16 | dot^1] = 97 → 64 → 32 → 1
    """

    def __init__(
        self,
        graph_dim: int = 256,
        base_feature_dim: int = 5,
        measure_dim: int = 8,
        dropout: float = 0.2,
        measure_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.base_feature_dim = int(base_feature_dim)
        self.measure_dim = int(measure_dim)

        self.graph_branch = nn.Sequential(
            nn.Linear(graph_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(base_feature_dim, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.measure_branch = nn.Sequential(
            nn.Linear(measure_dim, 32),
            nn.GELU(),
            nn.Dropout(measure_dropout),
            nn.Linear(32, 16),
        )
        self.head = nn.Sequential(
            nn.Linear(BRANCH_FUSION_IN, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward_parts(
        self,
        graph_context: Tensor,
        graph_score: Tensor,
        base_features: Tensor,
        measure_features: Tensor,
    ) -> dict[str, Tensor]:
        if graph_score.ndim == 1:
            graph_score = graph_score.unsqueeze(-1)
        g = self.graph_branch(graph_context)
        t = self.tabular_branch(base_features)
        m = self.measure_branch(measure_features)
        fused = torch.cat([g, t, m, graph_score], dim=-1)
        if fused.shape[-1] != BRANCH_FUSION_IN:
            raise RuntimeError(f"branch fusion dim {fused.shape[-1]} != {BRANCH_FUSION_IN}")
        logit = self.head(fused).squeeze(-1)
        return {"logits": logit, "G": g, "T": t, "M": m, "fused": fused}

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
        if hcr_features is None:
            raise ValueError("measure features required")
        return self.forward_parts(
            graph_context, graph_score, base_features, hcr_features
        )["logits"]

    def grad_norms(self) -> dict[str, float]:
        def _norm(module: nn.Module) -> float:
            total = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total += float(p.grad.detach().pow(2).sum().sqrt().item())
            return total

        return {
            "grad_graph": _norm(self.graph_branch),
            "grad_tabular": _norm(self.tabular_branch),
            "grad_measure": _norm(self.measure_branch),
        }


# RecommendationModel already accepts any fusion_head with the LateFusion forward signature.
