"""KRAM CLEAN V2 — true Jaccard/Cosine tabular + self-excl Top25 + 3-D A11 pool.

Does NOT modify frozen V1 / Phase-10 behavior. New entrypoints only.
"""

from src.lastfm_lp.clean_v2.constants import (
    CLEAN_V2_FUSION_DIM,
    CLEAN_V2_GRAPH_CONTEXT_DIM,
    CLEAN_V2_GRAPH_SCORE_DIM,
    CLEAN_V2_HCR_DIM,
    CLEAN_V2_TABULAR_DIM,
    CLEAN_V2_TOP_K,
)
from src.lastfm_lp.clean_v2.fusion import assert_clean_v2_fusion_dim, concat_clean_v2_fusion
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d
from src.lastfm_lp.clean_v2.routing import (
    CleanV2RoutingPolicy,
    aggregate_clean_v2_a11_pool,
    routing_history_exclude_self,
    select_top25_ids,
)
from src.lastfm_lp.clean_v2.tabular_true import (
    CLEAN_V2_TABULAR_FEATURE_NAMES,
    build_clean_v2_tabular_A,
    candidate_neighborhood_N_X,
    neighborhood_sizes,
    true_tabular_cosine,
    true_tabular_jaccard,
    vectorized_clean_v2_A_for_user,
)

__all__ = [
    "CLEAN_V2_FUSION_DIM",
    "CLEAN_V2_GRAPH_CONTEXT_DIM",
    "CLEAN_V2_GRAPH_SCORE_DIM",
    "CLEAN_V2_HCR_DIM",
    "CLEAN_V2_TABULAR_DIM",
    "CLEAN_V2_TOP_K",
    "CLEAN_V2_TABULAR_FEATURE_NAMES",
    "CleanV2RoutingPolicy",
    "aggregate_clean_v2_a11_pool",
    "assert_clean_v2_fusion_dim",
    "build_clean_v2_tabular_A",
    "candidate_neighborhood_N_X",
    "concat_clean_v2_fusion",
    "pool_signed_a11_3d",
    "routing_history_exclude_self",
    "select_top25_ids",
    "true_tabular_cosine",
    "true_tabular_jaccard",
]
