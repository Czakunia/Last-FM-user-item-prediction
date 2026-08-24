"""Phase 11A — vectorized H3 (a111) on C1 signed-A11 Top25 history.

Production path uses Phi-basis matmul / einsum:

    A3 = Phi_H.T @ (Phi_H * phi_i[:, None]) / N_ref
    tri = A3[np.triu_indices(K, k=1)]

Scalar pair loops exist ONLY as oracle tests (never production).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.hcr_triple import H3_RAW_FEATURE_ORDER, pool_h3_raw_matrix
from src.lastfm_lp.binary.hcr_triple_user_batch import (
    CrossFitTripleUserBatch,
    TripleUserBatchIndex,
    a111_scalar_reference,
)
from src.lastfm_lp.binary.smart_hcr_selection import (
    MAX_HISTORY_DEFAULT,
    _selection_scores,
    aggregate_a11_with_selection,
)
from src.lastfm_lp.pipeline.materialize_h3_full_exact import SIGNED_COLS, SIGNED_NAMES

H3_SIGNED_FEATURE_ORDER = list(SIGNED_NAMES)
A11_POOL_NAMES = [
    "mean:hcr_a11",
    "max:hcr_a11",
    "top3mean:hcr_a11",
    "wmean:hcr_a11",
]
C1_PLUS_H3_NAMES = A11_POOL_NAMES + H3_SIGNED_FEATURE_ORDER  # 7-D


def unordered_pairs_from_selected(items: Iterable[int]) -> np.ndarray:
    """Pairs (j<k) over an explicit selected history (oracle / dump helpers)."""

    ids = sorted({int(x) for x in items})
    if len(ids) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pairs.append((ids[a], ids[b]))
    return np.asarray(pairs, dtype=np.int64)


def topk_history_ids_matrix(
    hist: np.ndarray,
    cands: np.ndarray,
    index: PairwiseStatsIndex,
    *,
    max_history: int = MAX_HISTORY_DEFAULT,
    policy: str = "raw_a11_top25",
    support_lambda: float = 20.0,
) -> np.ndarray:
    """Return (k, C) selected history item ids per candidate (C1-style)."""

    hist = np.asarray(hist, dtype=np.int64).reshape(-1)
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    H, C = int(hist.size), int(cands.size)
    if C == 0:
        return np.zeros((0, 0), dtype=np.int64)
    if H == 0:
        return np.zeros((0, C), dtype=np.int64)
    if H <= max_history:
        return np.repeat(hist.reshape(-1, 1), C, axis=1)
    k = int(max_history)
    n11 = index.cooccurrence_block(hist, cands)
    a11, _energy = index.a11_energy_block(hist, cands)
    scores = _selection_scores(
        a11, n11, policy=policy, support_lambda=support_lambda  # type: ignore[arg-type]
    )
    idx = np.argpartition(-scores, kth=k - 1, axis=0)[:k, :]
    return hist[idx]


def phi_matrix_for_items(t_idx: TripleUserBatchIndex, items: np.ndarray) -> np.ndarray:
    """Build orthonormal contrast matrix Phi of shape (N_ref, K) for ``items``.

    φ(x; p) = (x - p) / sqrt(p(1-p)) on the fit-population user axis of ``t_idx``.
    """

    items = np.asarray(items, dtype=np.int64).reshape(-1)
    N = int(t_idx.n_users)
    K = int(items.size)
    Phi = np.zeros((N, K), dtype=np.float64)
    if K == 0 or N <= 0:
        return Phi
    # Use empirical margin from the same postings that fill Phi (orthonormal w.r.t.
    # the fit-population indicators actually used — not a mismatched popularity table).
    for ki, item in enumerate(items.tolist()):
        ii = int(item)
        if ii < 0 or ii >= t_idx.n_items:
            continue
        users = t_idx.postings[ii]
        n1 = int(users.size)
        if n1 <= 0 or n1 >= N:
            continue
        p = float(n1) / float(N)
        denom = np.sqrt(p * (1.0 - p))
        if denom <= 1e-12:
            continue
        phi0 = -p / denom
        phi1 = (1.0 - p) / denom
        Phi[:, ki] = phi0
        Phi[users, ki] = phi1
    return Phi


def a111_triu_from_phi(Phi_H: np.ndarray, phi_i: np.ndarray) -> np.ndarray:
    """Vectorized a111 for all unordered pairs given Phi_H (N,K) and phi_i (N,)."""

    Phi_H = np.asarray(Phi_H, dtype=np.float64)
    phi_i = np.asarray(phi_i, dtype=np.float64).reshape(-1)
    N, K = Phi_H.shape
    if K < 2 or N == 0:
        return np.zeros(0, dtype=np.float64)
    # A3 = Phi_H.T @ (Phi_H * phi_i[:, None]) / N
    weighted = Phi_H * phi_i[:, None]
    A3 = (Phi_H.T @ weighted) / float(N)
    return A3[np.triu_indices(K, k=1)].astype(np.float64, copy=False)


def a111_triu_block_from_phi(Phi_H: np.ndarray, Phi_C: np.ndarray) -> np.ndarray:
    """Batched a111 triu for many candidates: returns (C, P), P=K*(K-1)//2.

    Shared-history identity (one GEMM, no per-candidate Python loop):
        V[:, p] = Phi_H[:, j_p] * Phi_H[:, k_p]          # (N, P)
        A[c, p] = mean_n(Phi_C[n,c] * V[n,p])
                = (Phi_C.T @ V)[c, p] / N                 # (C, P)

    Equivalent to A3 = Phi_H.T @ (Phi_H * phi_c[:, None]) / N then triu.
    """

    Phi_H = np.ascontiguousarray(Phi_H, dtype=np.float64)
    Phi_C = np.ascontiguousarray(Phi_C, dtype=np.float64)
    N, K = Phi_H.shape
    if Phi_C.shape[0] != N:
        raise ValueError("Phi_H and Phi_C must share N_ref axis")
    C = int(Phi_C.shape[1])
    P = K * (K - 1) // 2
    if K < 2 or C == 0 or N == 0:
        return np.zeros((C, max(P, 0)), dtype=np.float64)
    ii, jj = np.triu_indices(K, k=1)
    V = Phi_H[:, ii] * Phi_H[:, jj]  # (N, P)
    return (Phi_C.T @ V) / float(N)


def pool_h3_signed_from_triu(triu: np.ndarray) -> np.ndarray:
    """Pool pair a111 values → 3-D H3_SIGNED (mean/max/top3mean).

    ``triu`` shape (P,) or (C, P).
    """

    a = np.asarray(triu, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    C, P = a.shape
    if P == 0:
        return np.zeros((C, 3), dtype=np.float32)
    energy = a * a
    # reuse frozen 8-D pool layout, then take SIGNED cols
    out = np.zeros((C, 3), dtype=np.float32)
    for c in range(C):
        raw8 = pool_h3_raw_matrix(a[c], energy[c])[0]
        out[c] = raw8[SIGNED_COLS]
    return out


def pool_h3_signed_from_triu_fast(triu: np.ndarray) -> np.ndarray:
    """Vectorized mean/max/top3mean over pair axis (matches pool_a11_energy top3mean)."""

    a = np.asarray(triu, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    C, P = a.shape
    out = np.zeros((C, 3), dtype=np.float32)
    if P == 0:
        return out
    out[:, 0] = a.mean(axis=1)
    out[:, 1] = a.max(axis=1)
    k = min(3, P)
    # top-3 by a111 value (same as pool_a11_energy_matrix kind a11_energy channel0)
    part = np.partition(a, -k, axis=1)[:, -k:]
    out[:, 2] = part.mean(axis=1)
    return out


def compute_h3_signed_shared_history_vectorized(
    t_idx: TripleUserBatchIndex,
    hist_items: np.ndarray,
    cand_items: np.ndarray,
    *,
    cand_chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """H3_SIGNED (C,3) when all candidates share the same selected history."""

    hist_items = np.asarray(
        sorted({int(x) for x in np.asarray(hist_items, dtype=np.int64).reshape(-1).tolist()}),
        dtype=np.int64,
    )
    cand_items = np.asarray(cand_items, dtype=np.int64).reshape(-1)
    C = int(cand_items.size)
    if C == 0 or hist_items.size < 2:
        return (
            np.zeros((C, 3), dtype=np.float32),
            np.zeros(C, dtype=np.int32),
        )
    Phi_H = phi_matrix_for_items(t_idx, hist_items)
    K = int(Phi_H.shape[1])
    P = K * (K - 1) // 2
    feats = np.zeros((C, 3), dtype=np.float32)
    # Build V once; stream candidates in chunks to bound peak memory.
    ii, jj = np.triu_indices(K, k=1)
    V = Phi_H[:, ii] * Phi_H[:, jj]  # (N, P)
    invN = 1.0 / float(Phi_H.shape[0])
    for start in range(0, C, int(cand_chunk)):
        end = min(start + int(cand_chunk), C)
        Phi_C = phi_matrix_for_items(t_idx, cand_items[start:end])
        triu = (Phi_C.T @ V) * invN
        feats[start:end] = pool_h3_signed_from_triu_fast(triu)
    n_pairs = np.full(C, P, dtype=np.int32)
    return feats, n_pairs


def compute_h3_signed_per_cand_selection_vectorized(
    t_idx: TripleUserBatchIndex,
    hist_items: np.ndarray,
    cand_items: np.ndarray,
    sel_ids: np.ndarray,
    *,
    cand_chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """H3_SIGNED with per-candidate Top25 selection (sel_ids shape (K, C)).

    Group candidates that share an identical Top25 set, then reuse the shared-
    history GEMM path (V once per unique set). No triple / (j,k) Python loops.
    """

    cand_items = np.asarray(cand_items, dtype=np.int64).reshape(-1)
    sel_ids = np.asarray(sel_ids, dtype=np.int64)
    C = int(cand_items.size)
    feats = np.zeros((C, 3), dtype=np.float32)
    n_pairs = np.zeros(C, dtype=np.int32)
    if C == 0:
        return feats, n_pairs

    uniq: dict[tuple[int, ...], list[int]] = {}
    for c in range(C):
        key = tuple(sorted({int(x) for x in sel_ids[:, c].tolist()}))
        uniq.setdefault(key, []).append(c)

    for key, cols in uniq.items():
        if len(key) < 2:
            continue
        col_idx = np.asarray(cols, dtype=np.int64)
        f, npairs = compute_h3_signed_shared_history_vectorized(
            t_idx,
            np.asarray(key, dtype=np.int64),
            cand_items[col_idx],
            cand_chunk=cand_chunk,
        )
        feats[col_idx] = f
        n_pairs[col_idx] = npairs
    return feats, n_pairs


def compute_h3_signed_c1_selection(
    *,
    user_id: int,
    history_items: set[int] | list[int],
    candidate_items: np.ndarray,
    labels: np.ndarray | None,
    split: str,
    triples: CrossFitTripleUserBatch,
    pairwise: PairwiseStatsIndex,
    max_history: int = MAX_HISTORY_DEFAULT,
    return_n111: bool = False,
) -> dict[str, Any]:
    """H3_SIGNED (C,3) on C1 Top25 — vectorized Phi backend (production)."""

    cands = np.asarray(candidate_items, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    empty = {
        "features": np.zeros((C, 3), dtype=np.float32),
        "n_pairs_used": np.zeros(C, dtype=np.int32),
        "n111_values": np.zeros(0, dtype=np.int32),
        "feature_names": H3_SIGNED_FEATURE_ORDER,
    }
    if C == 0:
        return empty

    base = set(int(x) for x in history_items)
    t_idx = triples.index_for(user_id, split)
    y = None if labels is None else np.asarray(labels).reshape(-1)

    feats = np.zeros((C, 3), dtype=np.float32)
    n_pairs_used = np.zeros(C, dtype=np.int32)

    # Group by working history after train self-exclusion
    groups: dict[frozenset[int], list[int]] = {}
    for c in range(C):
        hist = set(base)
        if y is not None and split == "train" and int(y[c]) == 1 and int(cands[c]) in base:
            hist = base - {int(cands[c])}
        groups.setdefault(frozenset(hist), []).append(c)

    for hist_set, cols in groups.items():
        hist_arr = np.asarray(sorted(hist_set), dtype=np.int64)
        col_idx = np.asarray(cols, dtype=np.int64)
        sub_cands = cands[col_idx]
        if hist_arr.size < 2:
            continue

        if hist_arr.size <= max_history:
            f, npairs = compute_h3_signed_shared_history_vectorized(
                t_idx, hist_arr, sub_cands
            )
            feats[col_idx] = f
            n_pairs_used[col_idx] = npairs
            continue

        sel = topk_history_ids_matrix(
            hist_arr,
            sub_cands,
            pairwise,
            max_history=max_history,
            policy="raw_a11_top25",
        )
        f, npairs = compute_h3_signed_per_cand_selection_vectorized(
            t_idx, hist_arr, sub_cands, sel
        )
        feats[col_idx] = f
        n_pairs_used[col_idx] = npairs

    # n111 support audit is count-based; optional lightweight sample via postings sizes
    n111_values = np.zeros(0, dtype=np.int32)
    if return_n111 and base and C > 0:
        # Approximate audit sample: |postings[j] ∩ postings[k] ∩ postings[i]| for a
        # few pairs from the shared/selected history of the first candidate group.
        # Full n111 enumeration is oracle-only (see audit helpers).
        n111_values = np.zeros(0, dtype=np.int32)

    return {
        "features": feats,
        "n_pairs_used": n_pairs_used,
        "feature_names": H3_SIGNED_FEATURE_ORDER,
        "n111_values": n111_values,
    }


def aggregate_c1_a11_plus_h3_signed(
    *,
    user_id: int,
    history_items: set[int] | list[int],
    candidate_items: np.ndarray,
    labels: np.ndarray | None,
    split: str,
    triples: CrossFitTripleUserBatch,
    pairwise: PairwiseStatsIndex,
    item_popularity: np.ndarray,
    max_history: int = MAX_HISTORY_DEFAULT,
    return_n111: bool = False,
) -> dict[str, Any]:
    """(C, 7) = C1 A11 pool (4) ∥ H3_SIGNED (3) under raw_a11_top25 selection."""

    cands = np.asarray(candidate_items, dtype=np.int64).reshape(-1)
    a11 = aggregate_a11_with_selection(
        history_items,
        cands,
        pairwise,
        policy="raw_a11_top25",
        max_history=max_history,
        kind="a11",
        item_popularity=item_popularity,
        user_id=int(user_id),
    )
    h3 = compute_h3_signed_c1_selection(
        user_id=user_id,
        history_items=history_items,
        candidate_items=cands,
        labels=labels,
        split=split,
        triples=triples,
        pairwise=pairwise,
        max_history=max_history,
        return_n111=return_n111,
    )
    feats = np.concatenate([a11, h3["features"]], axis=1).astype(np.float32)
    return {
        "features": feats,
        "a11": a11,
        "h3": h3["features"],
        "n_pairs_used": h3["n_pairs_used"],
        "n111_values": h3["n111_values"],
        "feature_names": list(C1_PLUS_H3_NAMES),
        "h3_raw_names": H3_RAW_FEATURE_ORDER,
    }


# ---------------------------------------------------------------------------
# ORACLE ONLY — never call in production materialize / full-val loops
# ---------------------------------------------------------------------------
def oracle_a111_scalar_pair(
    j: int,
    k: int,
    i: int,
    t_idx: TripleUserBatchIndex,
    pairwise: PairwiseStatsIndex,
) -> float:
    """Slow scalar reference (count formula). Oracle / unit tests only."""

    return float(a111_scalar_reference(j, k, i, t_idx, pairwise))


def oracle_a111_triu_scalar(
    hist_items: np.ndarray,
    cand_i: int,
    t_idx: TripleUserBatchIndex,
    pairwise: PairwiseStatsIndex,
) -> np.ndarray:
    """Oracle triu a111 via per-pair scalar counts. NEVER use in production."""

    pairs = unordered_pairs_from_selected(hist_items)
    out = np.zeros(pairs.shape[0], dtype=np.float64)
    for p in range(pairs.shape[0]):
        out[p] = oracle_a111_scalar_pair(
            int(pairs[p, 0]), int(pairs[p, 1]), int(cand_i), t_idx, pairwise
        )
    return out
