"""CLEAN V2 fusion concat: 256 + 1 + 5 + 3 = 265."""

from __future__ import annotations

import torch
from torch import Tensor

from src.lastfm_lp.clean_v2.constants import CLEAN_V2_FUSION_DIM


def concat_clean_v2_fusion(
    graph_context: Tensor,
    graph_score: Tensor,
    base_features: Tensor,
    hcr_features: Tensor,
) -> Tensor:
    """Concatenate CLEAN V2 blocks; assert last dim == 265."""

    if graph_score.ndim == 1:
        graph_score = graph_score.unsqueeze(-1)
    fused = torch.cat(
        [graph_context, graph_score, base_features, hcr_features], dim=-1
    )
    assert_clean_v2_fusion_dim(fused)
    return fused


def assert_clean_v2_fusion_dim(fused: Tensor) -> None:
    if fused.shape[-1] != CLEAN_V2_FUSION_DIM:
        raise AssertionError(
            f"CLEAN V2 fusion dim {fused.shape[-1]} != {CLEAN_V2_FUSION_DIM}"
        )
