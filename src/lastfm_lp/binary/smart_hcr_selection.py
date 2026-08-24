"""Phase 10 — Smart HCR candidate-conditioned history selection.

Only changes WHICH history items enter the frozen F2 A11 pool.
Default policy ``lowest_ids`` delegates to existing aggregate (bit-identical).
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.user_hcr_aggregation import (
    BlockKind,
    _truncate_history,
    aggregate_a11_energy_user_batch,
    pool_a11_energy_matrix,
)

SelectionPolicy = Literal[
    "lowest_ids",
    "raw_a11_top25",
    "support_a11_top25",
    "random25",
    "popular25",
]

MAX_HISTORY_DEFAULT = 25


def select_global_history(
    history_items: Iterable[int],
    *,
    policy: SelectionPolicy,
    max_history: int = MAX_HISTORY_DEFAULT,
    item_popularity: np.ndarray | None = None,
    user_id: int | None = None,
) -> np.ndarray:
    """Non-candidate-conditioned selection (C0 / controls). Returns int64 item ids."""

    hist = np.asarray(list(history_items), dtype=np.int64)
    if hist.size <= max_history:
        if policy == "lowest_ids":
            return np.asarray(_truncate_history(hist.tolist(), max_history), dtype=np.int64)
        return hist

    if policy == "lowest_ids":
        return np.asarray(_truncate_history(hist.tolist(), max_history), dtype=np.int64)

    if policy == "random25":
        # Deterministic user-specific RNG (diagnostic control).
        seed = int(user_id) if user_id is not None else 0
        rng = np.random.default_rng(seed ^ 0xA11C0DE)
        pick = rng.choice(hist, size=max_history, replace=False)
        return np.sort(pick)

    if policy == "popular25":
        if item_popularity is None:
            raise ValueError("popular25 requires item_popularity")
        pops = item_popularity[hist].astype(np.float64)
        # Stable: higher pop first; ties → lower item id
        order = np.lexsort((hist, -pops))
        return hist[order[:max_history]]

    raise ValueError(f"policy {policy} is candidate-conditioned; use aggregate_a11_with_selection")


def _selection_scores(
    a11: np.ndarray,
    n11: np.ndarray,
    *,
    policy: SelectionPolicy,
    support_lambda: float,
) -> np.ndarray:
    if policy == "raw_a11_top25":
        return a11.astype(np.float64, copy=False)
    if policy == "support_a11_top25":
        rel = n11.astype(np.float64) / (n11.astype(np.float64) + float(support_lambda))
        return np.maximum(a11.astype(np.float64, copy=False), 0.0) * rel
    raise ValueError(policy)


def _topk_pool_per_candidate(
    a11: np.ndarray,
    energy: np.ndarray,
    scores: np.ndarray,
    *,
    max_history: int,
    kind: BlockKind,
) -> np.ndarray:
    """Select top-max_history history rows per candidate column, then frozen pool."""

    H, C = a11.shape
    if H == 0:
        d = 1 if kind == "a11" else 2
        return np.zeros((C, 4 * d), dtype=np.float32)
    if H <= max_history:
        return pool_a11_energy_matrix(a11, energy, kind=kind)

    k = int(max_history)
    # argpartition: k largest scores per column (unordered within top-k; pool is invariant)
    idx = np.argpartition(-scores, kth=k - 1, axis=0)[:k, :]
    a_sel = np.take_along_axis(a11, idx, axis=0)
    e_sel = np.take_along_axis(energy, idx, axis=0)
    return pool_a11_energy_matrix(a_sel, e_sel, kind=kind)


def aggregate_a11_with_selection(
    history_items: Iterable[int],
    candidate_items: np.ndarray | Iterable[int],
    index: PairwiseStatsIndex,
    *,
    policy: SelectionPolicy = "lowest_ids",
    max_history: int = MAX_HISTORY_DEFAULT,
    kind: BlockKind = "a11",
    support_lambda: float = 20.0,
    item_popularity: np.ndarray | None = None,
    user_id: int | None = None,
    cand_block: int = 8192,
) -> np.ndarray:
    """One user × many candidates → (C, 4*d) pooled A11 features under ``policy``.

    ``lowest_ids`` / ``random25`` / ``popular25`` select a global history once.
    ``raw_a11_top25`` / ``support_a11_top25`` select per-candidate from full history.
    Candidate blocks avoid unbounded HxC materialization.
    """

    if kind not in {"a11", "a11_energy"}:
        raise ValueError(f"smart selection supports a11/a11_energy, got {kind}")

    cands = (
        np.asarray(candidate_items, dtype=np.int64).reshape(-1)
        if isinstance(candidate_items, np.ndarray)
        else np.asarray(list(candidate_items), dtype=np.int64).reshape(-1)
    )
    n_out = int(cands.size)
    d = 1 if kind == "a11" else 2
    if n_out == 0:
        return np.zeros((0, 4 * d), dtype=np.float32)

    hist_list = list(history_items)
    if not hist_list:
        return np.zeros((n_out, 4 * d), dtype=np.float32)

    # --- global policies (incl. C0 bit-identical path) ---
    if policy in {"lowest_ids", "random25", "popular25"}:
        if policy == "lowest_ids":
            return aggregate_a11_energy_user_batch(
                hist_list, cands, index, max_history=max_history, kind=kind
            )
        sel = select_global_history(
            hist_list,
            policy=policy,
            max_history=max_history,
            item_popularity=item_popularity,
            user_id=user_id,
        )
        return aggregate_a11_energy_user_batch(
            sel.tolist(), cands, index, max_history=max_history, kind=kind
        )

    if policy not in {"raw_a11_top25", "support_a11_top25"}:
        raise ValueError(f"unknown policy {policy}")

    hist = np.asarray(hist_list, dtype=np.int64)
    # |H|<=25 → entire history; pool matches C0 (order-invariant reductions)
    if hist.size <= max_history:
        return aggregate_a11_energy_user_batch(
            hist.tolist(), cands, index, max_history=max_history, kind=kind
        )

    out = np.zeros((n_out, 4 * d), dtype=np.float32)
    block = int(cand_block)
    for start in range(0, n_out, block):
        end = min(start + block, n_out)
        sl = cands[start:end]
        n11 = index.cooccurrence_block(hist, sl)
        a11, energy = index.a11_energy_block(hist, sl)
        scores = _selection_scores(a11, n11, policy=policy, support_lambda=support_lambda)
        out[start:end] = _topk_pool_per_candidate(
            a11, energy, scores, max_history=max_history, kind=kind
        )
    return out
