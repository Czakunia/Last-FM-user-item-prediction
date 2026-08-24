"""CLEAN V2 LEVEL B — item-item routing + Top25 + always pool signed A11 (3-D).

LEVEL B: user-incidence Jaccard/Cosine/A11/MI for (h,X).
After selection, ALWAYS pool signed A11 (never route scores).
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from src.lastfm_lp.binary.binary_measures import (
    a11_energy_from_n11_matrix,
    cosine_from_n11_matrix,
    jaccard_from_n11_matrix,
    mi_from_n11_matrix,
)
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_HCR_DIM, CLEAN_V2_TOP_K
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix

CleanV2RoutingPolicy = Literal[
    "a11_top25",
    "jaccard_top25",
    "cosine_top25",
    "mi_top25",
]

CLEAN_V2_ROUTING_POLICIES: tuple[CleanV2RoutingPolicy, ...] = (
    "a11_top25",
    "jaccard_top25",
    "cosine_top25",
    "mi_top25",
)

MI_SMOOTHING = 0.0


def routing_history_exclude_self(
    history_items: Iterable[int],
    candidate_item: int,
) -> np.ndarray:
    """routing_history = [h for h in H_u if h != X]."""

    x = int(candidate_item)
    hist = [int(h) for h in history_items if int(h) != x]
    return np.asarray(hist, dtype=np.int64)


def score_matrix_clean_v2(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    policy: CleanV2RoutingPolicy,
) -> np.ndarray:
    if policy == "a11_top25":
        a11, _ = a11_energy_from_n11_matrix(n11, pop_j, pop_i, n_users)
        return a11
    if policy == "jaccard_top25":
        return jaccard_from_n11_matrix(n11, pop_j, pop_i)
    if policy == "cosine_top25":
        return cosine_from_n11_matrix(n11, pop_j, pop_i)
    if policy == "mi_top25":
        return mi_from_n11_matrix(
            n11, pop_j, pop_i, n_users, smoothing=MI_SMOOTHING
        )
    raise ValueError(f"unknown CLEAN V2 policy {policy}")


def select_top25_ids(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    policy: CleanV2RoutingPolicy,
    max_history: int = CLEAN_V2_TOP_K,
    assert_no_self: bool = True,
) -> np.ndarray:
    x = int(candidate_item)
    hist = routing_history_exclude_self(history_items, x)
    if assert_no_self and x in set(hist.tolist()):
        raise AssertionError("self candidate leaked into routing_history")
    if hist.size == 0:
        return hist
    if hist.size <= max_history:
        sel = hist.copy()
    else:
        n11 = index.cooccurrence_block(hist, np.asarray([x], dtype=np.int64))
        scores = score_matrix_clean_v2(
            n11,
            index.popularity[hist],
            index.popularity[np.asarray([x])],
            index.n_users,
            policy=policy,
        )
        idx = deterministic_topk_indices(scores, hist, k=max_history)
        sel = hist[idx[:, 0]]
    if assert_no_self and int(x) in set(int(t) for t in sel.tolist()):
        raise AssertionError("self candidate leaked into Top25")
    return sel


def _pool_block(
    hist: np.ndarray,
    cands: np.ndarray,
    index: PairwiseStatsIndex,
    *,
    policy: CleanV2RoutingPolicy,
    max_history: int,
) -> np.ndarray:
    """Shared hist for all cands → (C, 3). Assumes X not in hist for all."""

    C = int(cands.size)
    out = np.zeros((C, CLEAN_V2_HCR_DIM), dtype=np.float32)
    if hist.size == 0 or C == 0:
        return out
    n11 = index.cooccurrence_block(hist, cands)
    a11, _ = a11_energy_from_n11_matrix(
        n11, index.popularity[hist], index.popularity[cands], index.n_users
    )
    if hist.size <= max_history:
        return pool_signed_a11_3d_matrix(a11)
    scores = score_matrix_clean_v2(
        n11,
        index.popularity[hist],
        index.popularity[cands],
        index.n_users,
        policy=policy,
    )
    idx = deterministic_topk_indices(scores, hist, k=max_history)
    a_sel = np.take_along_axis(a11, idx, axis=0)
    return pool_signed_a11_3d_matrix(a_sel)


def aggregate_clean_v2_a11_pool(
    history_items: Iterable[int],
    candidate_items: np.ndarray | Iterable[int],
    index: PairwiseStatsIndex,
    *,
    policy: CleanV2RoutingPolicy,
    max_history: int = CLEAN_V2_TOP_K,
    cand_block: int = 8192,
    assert_no_self: bool = True,
) -> np.ndarray:
    """Select Top25 by ``policy``, then pool signed A11 → (C, 3)."""

    cands = (
        np.asarray(candidate_items, dtype=np.int64).reshape(-1)
        if isinstance(candidate_items, np.ndarray)
        else np.asarray(list(candidate_items), dtype=np.int64).reshape(-1)
    )
    n_out = int(cands.size)
    out = np.zeros((n_out, CLEAN_V2_HCR_DIM), dtype=np.float32)
    if n_out == 0:
        return out

    base = np.asarray(list(history_items), dtype=np.int64)
    base_set = set(int(x) for x in base.tolist())
    # Fast path: no candidate in history → shared hist for all
    if not any(int(x) in base_set for x in cands.tolist()):
        hist = base
        if assert_no_self:
            pass
        for start in range(0, n_out, int(cand_block)):
            end = min(start + int(cand_block), n_out)
            out[start:end] = _pool_block(
                hist, cands[start:end], index, policy=policy, max_history=max_history
            )
        return out

    # Mixed: group identical routing histories
    normal_mask = np.array([int(x) not in base_set for x in cands.tolist()], dtype=bool)
    if normal_mask.any():
        hist = base
        idx_pos = np.flatnonzero(normal_mask)
        for start in range(0, idx_pos.size, int(cand_block)):
            sl = idx_pos[start : start + int(cand_block)]
            out[sl] = _pool_block(
                hist, cands[sl], index, policy=policy, max_history=max_history
            )
    for local_i, x in enumerate(cands.tolist()):
        if int(x) not in base_set:
            continue
        hist = routing_history_exclude_self(base, int(x))
        if assert_no_self and int(x) in set(hist.tolist()):
            raise AssertionError("self in routing_history")
        out[local_i] = _pool_block(
            hist,
            np.asarray([int(x)], dtype=np.int64),
            index,
            policy=policy,
            max_history=max_history,
        )[0]
        if assert_no_self and int(x) in set(hist.tolist()[:max_history]):
            # selected set checked via hist exclusion already
            pass
    return out
