"""Measure-race Top25 history selection policies (expanded statistical set).

ONLY the per-(history,candidate) score changes. After Top25 selection, the
downstream feature construction remains the frozen pooled signed A11 (4-D).

Main race policies:
  M0 hcr_a11_top25   — frozen orthonormal A11 (Phase-10 C1)
  M1 cooc_top25      — n11
  M2 jaccard_top25
  M3 cosine_top25
  M4 mi_top25        — full binary MI, unsmoothed (not a11²+MI)
  M5 npmi_top25      — NPMI_11
  M6 g2_top25        — unsigned G² / LLR

Controls (separate): random25 / popular25 / shuffled_hcr — not in this tuple.

Tie-break (identical for all): higher score first; ties → lower item id.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from src.lastfm_lp.binary.binary_measures import (
    a11_energy_from_n11_matrix,
    cooc_from_n11_matrix,
    cosine_from_n11_matrix,
    g2_from_n11_matrix,
    jaccard_from_n11_matrix,
    mi_from_n11_matrix,
    npmi11_from_n11_matrix,
)
from src.lastfm_lp.binary.contingency import contingency_batch_from_n11_matrix
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.user_hcr_aggregation import (
    BlockKind,
    aggregate_a11_energy_user_batch,
    pool_a11_energy_matrix,
)

MeasureRacePolicy = Literal[
    "hcr_a11_top25",
    "cooc_top25",
    "jaccard_top25",
    "cosine_top25",
    "mi_top25",
    "npmi_top25",
    "g2_top25",
]

MEASURE_RACE_POLICIES: tuple[MeasureRacePolicy, ...] = (
    "hcr_a11_top25",
    "cooc_top25",
    "jaccard_top25",
    "cosine_top25",
    "mi_top25",
    "npmi_top25",
    "g2_top25",
)

# G2 main-race variant freeze (do not switch after seeing results).
G2_RACE_VARIANT = "unsigned"  # not g2_positive

MAX_HISTORY_DEFAULT = 25
# Pure MI for race: no Jeffreys smoothing (numerical: 0·log = 0).
MI_SMOOTHING = 0.0
# NPMI when n11==0
NPMI_ZERO_N11_SCORE = -1.0


def deterministic_topk_indices(
    scores: np.ndarray,
    hist_ids: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Return (k, C) row indices into hist: best score, tie → lower item id."""

    scores = np.asarray(scores, dtype=np.float64)
    hist_ids = np.asarray(hist_ids, dtype=np.int64).reshape(-1)
    H, C = scores.shape
    if H != hist_ids.size:
        raise ValueError("hist_ids length must match scores rows")
    if H == 0 or C == 0:
        return np.zeros((0, C), dtype=np.int64)
    kk = min(int(k), H)
    # Exact lexsort(hist_id, -score): permute to ascending id, then stable
    # sort by -score. Batch-safe; no per-column Python loop.
    order_id = np.argsort(hist_ids, kind="mergesort")
    scores_by_id = scores[order_id, :]
    order_score = np.argsort(-scores_by_id, axis=0, kind="stable")
    idx = order_id[order_score]
    return idx[:kk, :].astype(np.int64, copy=False)


def score_matrix_for_policy(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
    *,
    policy: MeasureRacePolicy,
) -> np.ndarray:
    """HxC selection scores for one policy (no TopK yet)."""

    if policy == "hcr_a11_top25":
        a11, _energy = a11_energy_from_n11_matrix(n11, pop_j, pop_i, n_users)
        return a11
    if policy == "cooc_top25":
        return cooc_from_n11_matrix(n11, pop_j, pop_i)
    if policy == "jaccard_top25":
        return jaccard_from_n11_matrix(n11, pop_j, pop_i)
    if policy == "cosine_top25":
        return cosine_from_n11_matrix(n11, pop_j, pop_i)
    if policy == "mi_top25":
        return mi_from_n11_matrix(
            n11, pop_j, pop_i, n_users, smoothing=MI_SMOOTHING
        )
    if policy == "npmi_top25":
        return npmi11_from_n11_matrix(
            n11,
            pop_j,
            pop_i,
            n_users,
            zero_n11_score=NPMI_ZERO_N11_SCORE,
        )
    if policy == "g2_top25":
        g2, _direction = g2_from_n11_matrix(n11, pop_j, pop_i, n_users)
        return g2
    raise ValueError(f"unknown measure-race policy {policy}")


def topk_history_ids_for_policy(
    history_items: np.ndarray,
    candidate_items: np.ndarray,
    index: PairwiseStatsIndex,
    *,
    policy: MeasureRacePolicy,
    max_history: int = MAX_HISTORY_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sel_ids[k,C], scores[H,C]) for one user block of candidates."""

    hist = np.asarray(history_items, dtype=np.int64).reshape(-1)
    cands = np.asarray(candidate_items, dtype=np.int64).reshape(-1)
    H, C = int(hist.size), int(cands.size)
    if C == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((H, 0), dtype=np.float64)
    if H == 0:
        return np.zeros((0, C), dtype=np.int64), np.zeros((0, C), dtype=np.float64)
    if H <= max_history:
        return np.repeat(hist.reshape(-1, 1), C, axis=1), np.zeros((H, C), dtype=np.float64)

    n11 = index.cooccurrence_block(hist, cands)
    scores = score_matrix_for_policy(
        n11,
        index.popularity[hist],
        index.popularity[cands],
        index.n_users,
        policy=policy,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(f"non-finite scores for policy {policy}")
    idx = deterministic_topk_indices(scores, hist, k=max_history)
    sel = hist[idx]
    return sel, scores


def aggregate_a11_with_measure_policy(
    history_items: Iterable[int],
    candidate_items: np.ndarray | Iterable[int],
    index: PairwiseStatsIndex,
    *,
    policy: MeasureRacePolicy,
    max_history: int = MAX_HISTORY_DEFAULT,
    kind: BlockKind = "a11",
    cand_block: int = 8192,
) -> np.ndarray:
    """Select Top25 by ``policy``, then pool frozen signed A11 features (C, 4)."""

    if kind not in {"a11", "a11_energy"}:
        raise ValueError(kind)
    if policy not in MEASURE_RACE_POLICIES:
        raise ValueError(policy)

    cands = (
        np.asarray(candidate_items, dtype=np.int64).reshape(-1)
        if isinstance(candidate_items, np.ndarray)
        else np.asarray(list(candidate_items), dtype=np.int64).reshape(-1)
    )
    n_out = int(cands.size)
    d = 1 if kind == "a11" else 2
    if n_out == 0:
        return np.zeros((0, 4 * d), dtype=np.float32)

    hist = np.asarray(list(history_items), dtype=np.int64)
    if hist.size == 0:
        return np.zeros((n_out, 4 * d), dtype=np.float32)
    if hist.size <= max_history:
        return aggregate_a11_energy_user_batch(
            hist.tolist(), cands, index, max_history=max_history, kind=kind
        )

    out = np.zeros((n_out, 4 * d), dtype=np.float32)
    for start in range(0, n_out, int(cand_block)):
        end = min(start + int(cand_block), n_out)
        sl = cands[start:end]
        n11 = index.cooccurrence_block(hist, sl)
        scores = score_matrix_for_policy(
            n11,
            index.popularity[hist],
            index.popularity[sl],
            index.n_users,
            policy=policy,
        )
        idx = deterministic_topk_indices(scores, hist, k=max_history)
        a11, energy = a11_energy_from_n11_matrix(
            n11, index.popularity[hist], index.popularity[sl], index.n_users
        )
        a_sel = np.take_along_axis(a11, idx, axis=0)
        e_sel = np.take_along_axis(energy, idx, axis=0)
        out[start:end] = pool_a11_energy_matrix(a_sel, e_sel, kind=kind)
    return out


def top25_set_for_candidate(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    policy: MeasureRacePolicy,
    max_history: int = MAX_HISTORY_DEFAULT,
) -> set[int]:
    """Convenience: unordered Top25 set for one candidate (audits)."""

    hist = np.asarray(list(history_items), dtype=np.int64)
    cands = np.asarray([int(candidate_item)], dtype=np.int64)
    sel, _ = topk_history_ids_for_policy(
        hist, cands, index, policy=policy, max_history=max_history
    )
    if sel.size == 0:
        return set()
    return {int(x) for x in sel[:, 0].tolist()}


def selection_pair_rows(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    policy: MeasureRacePolicy,
    max_history: int = MAX_HISTORY_DEFAULT,
) -> list[dict]:
    """Diagnostic rows for selected Top25: score, n11, supports, expected_n11."""

    hist = np.asarray(list(history_items), dtype=np.int64)
    cands = np.asarray([int(candidate_item)], dtype=np.int64)
    if hist.size == 0:
        return []
    n11 = index.cooccurrence_block(hist, cands)
    pop_j = index.popularity[hist]
    pop_i = index.popularity[cands]
    scores = score_matrix_for_policy(
        n11, pop_j, pop_i, index.n_users, policy=policy
    )
    batch = contingency_batch_from_n11_matrix(n11, pop_j, pop_i, index.n_users)
    idx = deterministic_topk_indices(scores, hist, k=max_history)[:, 0]
    rows = []
    for r in idx.tolist():
        rows.append(
            {
                "candidate_id": int(candidate_item),
                "history_item_id": int(hist[r]),
                "score": float(scores[r, 0]),
                "n11": float(batch.a[r, 0]),
                "candidate_support": float(batch.candidate_support[0, 0]),
                "history_support": float(batch.history_support[r, 0]),
                "expected_n11": float(batch.expected_n11[r, 0]),
                "policy": policy,
            }
        )
    return rows
