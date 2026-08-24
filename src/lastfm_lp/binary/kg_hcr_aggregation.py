"""User-conditioned KG-HCR: HCR stats gated / interacted with KG path evidence."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.kg.path_index import KGPathIndex

# Aggregate feature names for F1 flat late fusion (mean/max over history)
KG_HCR_AGG_FEATURES = [
    # HCR aggregates
    "hcr_mean",
    "hcr_max",
    "hcr_weighted_mean",
    "cond_max",
    "lift_max",
    "npmi_mean",
    "support_sum",
    "uncertainty_mean",
    # KG aggregates
    "kg_connected_frac",
    "kg_shortest_path_mean",
    "kg_path_count_len2_mean",
    "kg_path_count_len3_mean",
    "kg_shared_entity_mean",
    "kg_relation_overlap_mean",
    # interactions
    "hcr_x_kg_connected_mean",
    "hcr_x_path_count_len2_mean",
    "hcr_x_shared_entity_mean",
    "hcr_x_relation_overlap_mean",
    # coverage / trust
    "path_coverage",
    "hcr_on_path_mean",
    "hcr_off_path_mean",
]


def _pair_vector(
    j: int,
    i: int,
    hcr_index: PairwiseStatsIndex,
    kg_index: KGPathIndex,
    *,
    mask_no_path: bool = False,
) -> dict[str, float]:
    m = hcr_index.pair(j, i)
    kg = kg_index.pair_features(j, i)
    # Stage F used LEGACY_BINARY_ASSOC_V1 under the name "hcr" — keep that
    # semantics for frozen F* results; do not treat it as orthonormal HCR.
    assoc = float(m.legacy_binary_assoc)
    support = float(m.n11)
    uncertainty = float(m.uncertainty) if m.uncertainty > 0 else 1.0 / np.sqrt(support + 1.0)
    connected = kg["kg_connected"]
    if mask_no_path and connected < 0.5:
        assoc = 0.0
        # keep support/uncertainty for gate diagnostics
    return {
        "hcr": assoc,
        "cond": float(m.conditional_y_given_x),
        "lift": float(m.lift),
        "npmi": float(m.npmi),
        "support": support,
        "uncertainty": float(uncertainty),
        **kg,
        "hcr_x_kg_connected": assoc * connected,
        "hcr_x_path_count_len2": assoc * kg["log1p_path_count_len2"],
        "hcr_x_shared_entity": assoc * kg["log1p_path_count_len2"],
        "hcr_x_relation_overlap": assoc * kg["kg_relation_overlap"],
    }


def aggregate_kg_hcr(
    history_items: Iterable[int],
    candidate_item: int,
    hcr_index: PairwiseStatsIndex,
    kg_index: KGPathIndex,
    *,
    max_history: int = 25,
    mask_no_path: bool = False,
) -> dict[str, float]:
    hist = sorted(int(x) for x in history_items)
    if len(hist) > max_history:
        hist = hist[:max_history]
    if not hist:
        return {k: 0.0 for k in KG_HCR_AGG_FEATURES}

    rows = [
        _pair_vector(j, int(candidate_item), hcr_index, kg_index, mask_no_path=mask_no_path)
        for j in hist
    ]
    hcrs = np.array([r["hcr"] for r in rows], dtype=np.float64)
    supports = np.array([r["support"] for r in rows], dtype=np.float64)
    w = np.log1p(supports)
    wsum = float(w.sum())
    connected = np.array([r["kg_connected"] for r in rows], dtype=np.float64)

    on = hcrs[connected > 0.5]
    off = hcrs[connected <= 0.5]

    return {
        "hcr_mean": float(hcrs.mean()),
        "hcr_max": float(hcrs.max()),
        "hcr_weighted_mean": float((w * hcrs).sum() / wsum) if wsum > 0 else float(hcrs.mean()),
        "cond_max": float(max(r["cond"] for r in rows)),
        "lift_max": float(max(r["lift"] for r in rows)),
        "npmi_mean": float(np.mean([r["npmi"] for r in rows])),
        "support_sum": float(supports.sum()),
        "uncertainty_mean": float(np.mean([r["uncertainty"] for r in rows])),
        "kg_connected_frac": float(connected.mean()),
        "kg_shortest_path_mean": float(np.mean([r["kg_shortest_path"] for r in rows])),
        "kg_path_count_len2_mean": float(np.mean([r["kg_path_count_len2"] for r in rows])),
        "kg_path_count_len3_mean": float(np.mean([r["kg_path_count_len3"] for r in rows])),
        "kg_shared_entity_mean": float(np.mean([r["kg_shared_entity_count"] for r in rows])),
        "kg_relation_overlap_mean": float(np.mean([r["kg_relation_overlap"] for r in rows])),
        "hcr_x_kg_connected_mean": float(np.mean([r["hcr_x_kg_connected"] for r in rows])),
        "hcr_x_path_count_len2_mean": float(np.mean([r["hcr_x_path_count_len2"] for r in rows])),
        "hcr_x_shared_entity_mean": float(np.mean([r["hcr_x_shared_entity"] for r in rows])),
        "hcr_x_relation_overlap_mean": float(
            np.mean([r["hcr_x_relation_overlap"] for r in rows])
        ),
        "path_coverage": float(connected.mean()),
        "hcr_on_path_mean": float(on.mean()) if len(on) else 0.0,
        "hcr_off_path_mean": float(off.mean()) if len(off) else 0.0,
    }


def history_pair_matrix(
    history_items: Iterable[int],
    candidate_item: int,
    hcr_index: PairwiseStatsIndex,
    kg_index: KGPathIndex,
    *,
    max_history: int = 25,
    mask_no_path: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (H, F) features, (H,) mask, history item ids (padded to max_history)."""
    hist = sorted(int(x) for x in history_items)[:max_history]
    # feature layout for attention
    names = [
        "hcr",
        "cond",
        "lift",
        "npmi",
        "support",
        "uncertainty",
        "kg_connected",
        "kg_shortest_path",
        "log1p_path_count_len2",
        "log1p_path_count_len3",
        "kg_relation_overlap",
        "hcr_x_kg_connected",
        "hcr_x_path_count_len2",
        "hcr_x_relation_overlap",
    ]
    H = max_history
    F = len(names)
    mat = np.zeros((H, F), dtype=np.float32)
    mask = np.zeros((H,), dtype=np.float32)
    ids = [-1] * H
    for t, j in enumerate(hist):
        row = _pair_vector(j, int(candidate_item), hcr_index, kg_index, mask_no_path=mask_no_path)
        mat[t] = np.array([row[n] for n in names], dtype=np.float32)
        mask[t] = 1.0
        ids[t] = j
    return mat, mask, ids
