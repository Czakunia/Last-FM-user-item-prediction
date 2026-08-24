"""Leakage-safe tabular pair features (train statistics only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.user_hcr_aggregation import (  # noqa: F401
    BINARY_DEPENDENCE_FEATURES,
    aggregate_user_hcr,
)

TABULAR_BASE_FEATURES = [
    "log1p_history_length",
    "log1p_candidate_popularity",
    "log1p_candidate_kg_degree",
    "history_candidate_jaccard",
    "history_candidate_cosine",
]


def _cosine_pop(history: set[int], candidate: int, popularity: np.ndarray) -> float:
    """Cheap proxy similarity: cosine between binary history bag and one-hot candidate,
    weighted by popularity as importance — reduces to normalized pop overlap proxy.
    """
    if not history:
        return 0.0
    # use co-visitation proxy via mean popularity of history vs candidate
    hist_pops = popularity[list(history)].astype(np.float64)
    num = float(popularity[candidate]) * float(hist_pops.mean())
    den = float(np.linalg.norm(hist_pops) * max(popularity[candidate], 1))
    return float(num / den) if den > 0 else 0.0


def build_tabular_pair_features(
    user_id: int,
    candidate_item_id: int,
    user_history: dict[int, set[int]],
    item_statistics: dict[str, Any],
    kg_statistics: dict[str, Any],
    *,
    pairwise_index: PairwiseStatsIndex | None = None,
    include_binary: bool = False,
) -> dict[str, float]:
    """Build leakage-safe features using training data only."""
    hist = user_history.get(user_id, set())
    pop = item_statistics["popularity"]
    kg_deg = kg_statistics["item_kg_degree"]

    # overlap: fraction of history items that co-occur with candidate in train
    overlap = 0.0
    if pairwise_index is not None and hist:
        from src.lastfm_lp.binary.contingency_tables import get_n11

        sample = sorted(hist)[:40]
        n11s = [get_n11(pairwise_index.cooc, j, candidate_item_id) for j in sample]
        overlap = float(np.mean([1.0 if x > 0 else 0.0 for x in n11s]))

    feats = {
        "log1p_history_length": float(np.log1p(len(hist))),
        "log1p_candidate_popularity": float(np.log1p(pop[candidate_item_id])),
        "log1p_candidate_kg_degree": float(np.log1p(kg_deg[candidate_item_id])),
        "history_candidate_jaccard": overlap,  # co-occurrence support rate proxy
        "history_candidate_cosine": _cosine_pop(hist, candidate_item_id, pop),
    }

    if include_binary:
        if pairwise_index is None:
            raise ValueError("pairwise_index required for binary dependence features")
        feats.update(aggregate_user_hcr(hist, candidate_item_id, pairwise_index))
    return feats


def feature_matrix(
    pairs: dict[str, np.ndarray],
    user_history: dict[int, set[int]],
    item_statistics: dict[str, Any],
    kg_statistics: dict[str, Any],
    feature_names: list[str],
    *,
    pairwise_index: PairwiseStatsIndex | None = None,
    include_binary: bool = False,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    users = pairs["user_id"]
    items = pairs["item_id"]
    for u, i in zip(users, items):
        f = build_tabular_pair_features(
            int(u),
            int(i),
            user_history,
            item_statistics,
            kg_statistics,
            pairwise_index=pairwise_index,
            include_binary=include_binary,
        )
        rows.append([f[name] for name in feature_names])
    X = np.asarray(rows, dtype=np.float32)
    return X, feature_names


def resolve_feature_names(stage: str) -> list[str]:
    if stage in {"A0", "A1", "A2"}:
        return list(TABULAR_BASE_FEATURES)
    if stage in {"B1", "B2"}:
        # classical dependence without dedicated HCR aggregates? protocol:
        # B1/B2 = classical binary; B3/B4 = + HCR
        # We include classical max stats + keep HCR for B3/B4 only.
        return list(TABULAR_BASE_FEATURES) + [
            "conditional_probability_max",
            "lift_max",
            "odds_ratio_max",
            "npmi_mean",
            "hcr_support_sum",
        ]
    if stage in {"B3", "B4", "B0"}:
        return list(TABULAR_BASE_FEATURES) + list(BINARY_DEPENDENCE_FEATURES)
    raise KeyError(stage)
