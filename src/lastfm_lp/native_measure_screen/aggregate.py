"""Route Top25 by ONE measure, pool that measure's native encoding (not shared A11)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from src.lastfm_lp.binary.binary_measures import (
    a11_energy_from_n11_matrix,
    cosine_from_n11_matrix,
    jaccard_from_n11_matrix,  # noqa: F401 — available for audits
    mi_from_n11_matrix,
)
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices
from src.lastfm_lp.clean_v2.routing import routing_history_exclude_self
from src.lastfm_lp.native_measure_screen.constants import (
    A11_DIM,
    MeasureName,
    NONNEG_DIM,
    TOP_K,
)
from src.lastfm_lp.native_measure_screen.pooling_native import (
    encode_a11_8d_matrix,
    encode_nonneg_7d_matrix,
)

MI_SMOOTHING = 0.0


def measure_score_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    measure: MeasureName,
) -> np.ndarray:
    if measure == "a11":
        a11, _ = a11_energy_from_n11_matrix(n11, pop_j, pop_i, n_users)
        return a11
    if measure == "cosine":
        return cosine_from_n11_matrix(n11, pop_j, pop_i)
    if measure == "mi":
        return mi_from_n11_matrix(
            n11, pop_j, pop_i, n_users, smoothing=MI_SMOOTHING
        )
    raise ValueError(f"unknown measure {measure}")


def encoding_dim(measure: MeasureName) -> int:
    return A11_DIM if measure == "a11" else NONNEG_DIM


def _encode_selected(selected_scores: np.ndarray, measure: MeasureName) -> np.ndarray:
    if measure == "a11":
        return encode_a11_8d_matrix(selected_scores)
    return encode_nonneg_7d_matrix(selected_scores)


def _pool_block(
    hist: np.ndarray,
    cands: np.ndarray,
    index: PairwiseStatsIndex,
    *,
    measure: MeasureName,
    max_history: int,
) -> np.ndarray:
    """Shared hist for all cands → (C, D). Assumes X ∉ hist."""

    c = int(cands.size)
    d = encoding_dim(measure)
    out = np.zeros((c, d), dtype=np.float32)
    if hist.size == 0 or c == 0:
        return out
    n11 = index.cooccurrence_block(hist, cands)
    scores = measure_score_matrix(
        n11,
        index.popularity[hist],
        index.popularity[cands],
        index.n_users,
        measure=measure,
    )
    if hist.size <= max_history:
        return _encode_selected(scores, measure)
    idx = deterministic_topk_indices(scores, hist, k=max_history)
    sel = np.take_along_axis(scores, idx, axis=0)
    return _encode_selected(sel, measure)


def select_top25_ids_native(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    measure: MeasureName,
    max_history: int = TOP_K,
) -> np.ndarray:
    x = int(candidate_item)
    hist = routing_history_exclude_self(history_items, x)
    if hist.size == 0:
        return hist
    if int(x) in set(int(t) for t in hist.tolist()):
        raise AssertionError("self candidate leaked into routing history")
    if hist.size <= max_history:
        return hist.copy()
    n11 = index.cooccurrence_block(hist, np.asarray([x], dtype=np.int64))
    scores = measure_score_matrix(
        n11,
        index.popularity[hist],
        index.popularity[np.asarray([x])],
        index.n_users,
        measure=measure,
    )
    idx = deterministic_topk_indices(scores, hist, k=max_history)
    return hist[idx[:, 0]]


def aggregate_native_pool(
    history_items: Iterable[int],
    candidate_items: np.ndarray | Iterable[int],
    index: PairwiseStatsIndex,
    *,
    measure: MeasureName,
    max_history: int = TOP_K,
    cand_block: int = 8192,
) -> np.ndarray:
    """Top25 by ``measure``, then native encode of that measure → (C, D)."""

    cands = (
        np.asarray(candidate_items, dtype=np.int64).reshape(-1)
        if isinstance(candidate_items, np.ndarray)
        else np.asarray(list(candidate_items), dtype=np.int64).reshape(-1)
    )
    n_out = int(cands.size)
    d = encoding_dim(measure)
    out = np.zeros((n_out, d), dtype=np.float32)
    if n_out == 0:
        return out

    base = np.asarray(list(history_items), dtype=np.int64)
    base_set = set(int(x) for x in base.tolist())
    if not any(int(x) in base_set for x in cands.tolist()):
        for start in range(0, n_out, int(cand_block)):
            end = min(start + int(cand_block), n_out)
            out[start:end] = _pool_block(
                base, cands[start:end], index, measure=measure, max_history=max_history
            )
        return out

    normal_mask = np.array([int(x) not in base_set for x in cands.tolist()], dtype=bool)
    if normal_mask.any():
        idx_pos = np.flatnonzero(normal_mask)
        for start in range(0, idx_pos.size, int(cand_block)):
            sl = idx_pos[start : start + int(cand_block)]
            out[sl] = _pool_block(
                base, cands[sl], index, measure=measure, max_history=max_history
            )
    for local_i, x in enumerate(cands.tolist()):
        if int(x) not in base_set:
            continue
        hist = routing_history_exclude_self(base, int(x))
        if int(x) in set(int(t) for t in hist.tolist()):
            raise AssertionError("self in routing_history")
        out[local_i] = _pool_block(
            hist,
            np.asarray([int(x)], dtype=np.int64),
            index,
            measure=measure,
            max_history=max_history,
        )[0]
    return out


def contingency_from_counts(
    n11: int, n_h: int, n_x: int, n: int
) -> tuple[int, int, int, int]:
    n10 = n_h - n11
    n01 = n_x - n11
    n00 = n - n_h - n_x + n11
    if min(n00, n01, n10, n11) < 0:
        raise AssertionError(f"negative cell {(n00, n01, n10, n11)}")
    if n00 + n01 + n10 + n11 != n:
        raise AssertionError("contingency sum != N")
    return n00, n01, n10, n11
