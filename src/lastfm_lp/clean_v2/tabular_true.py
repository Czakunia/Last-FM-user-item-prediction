"""CLEAN V2 LEVEL A — true set Jaccard / Cosine between H_u and N_X.

LEVEL A (this module): operands are ITEM ID sets
  H_u = model_train history items of user u
  N_X = population co-occurrence neighborhood of candidate X (item IDs)

LEVEL B (routing): operands are USER ID incidence sets U_h, U_X
  — see clean_v2.routing / binary_measures jaccard_from_n11_matrix.

These levels MUST NOT be mixed.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.lastfm_lp.binary.contingency_tables import cooc_row_arrays
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex

CLEAN_V2_TABULAR_FEATURE_NAMES: list[str] = [
    "log1p_history_length",
    "log1p_candidate_popularity",
    "log1p_candidate_kg_degree",
    "history_candidate_jaccard_true",
    "history_candidate_cosine_true",
]


def candidate_neighborhood_N_X(
    candidate_item: int,
    index: PairwiseStatsIndex,
) -> set[int]:
    """N_X = {j : j != X and n11(j,X) > 0} from fold-excluded cooc.

    Uses the CSR row of X in the fold cooc matrix (symmetric storage).
    X itself is never included.
    """

    x = int(candidate_item)
    cols, data = cooc_row_arrays(index.cooc, x)
    out: set[int] = set()
    for j, n11 in zip(cols.tolist(), data.tolist()):
        jj = int(j)
        if jj == x:
            continue
        if int(n11) > 0:
            out.add(jj)
    return out


def true_tabular_jaccard(H_u: Iterable[int], N_X: Iterable[int]) -> float:
    """J_A(u,X) = |H ∩ N| / |H ∪ N|; empty union → 0."""

    H = set(int(x) for x in H_u)
    N = set(int(x) for x in N_X)
    inter = len(H & N)
    union = len(H) + len(N) - inter
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def true_tabular_cosine(H_u: Iterable[int], N_X: Iterable[int]) -> float:
    """C_A(u,X) = |H ∩ N| / sqrt(|H|·|N|); empty set → 0."""

    H = set(int(x) for x in H_u)
    N = set(int(x) for x in N_X)
    if not H or not N:
        return 0.0
    inter = len(H & N)
    return float(inter) / float(np.sqrt(len(H) * len(N)))


def neighborhood_sizes(index: PairwiseStatsIndex) -> np.ndarray:
    """Precompute |N_X| = #{j != X : n11(j,X)>0} for all items (fold cooc).

    Cooc has no diagonal (j==i never stored), so nnz(row) == |N_X|.
    """

    cooc = index.cooc
    return np.diff(cooc.indptr).astype(np.int32)


def vectorized_clean_v2_A_for_user(
    user_id: int,
    candidates: np.ndarray,
    user_history: dict[int, set[int]],
    popularity: np.ndarray,
    item_kg_degree: np.ndarray,
    index: PairwiseStatsIndex,
    *,
    n_x_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """(C,5) CLEAN Tabular A. |H∩N_X| = #{h in H_u : n11(h,X)>0}."""

    hist = set(int(x) for x in user_history.get(int(user_id), set()))
    cands = np.asarray(candidates, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    X = np.zeros((C, 5), dtype=np.float32)
    X[:, 0] = np.log1p(len(hist))
    X[:, 1] = np.log1p(popularity[cands].astype(np.float64))
    X[:, 2] = np.log1p(item_kg_degree[cands].astype(np.float64))
    if C == 0:
        return X
    if n_x_sizes is None:
        # slow fallback per candidate
        sizes = np.asarray(
            [len(candidate_neighborhood_N_X(int(x), index)) for x in cands.tolist()],
            dtype=np.float64,
        )
    else:
        sizes = n_x_sizes[cands].astype(np.float64)
    if not hist:
        return X
    hist_arr = np.asarray(sorted(hist), dtype=np.int64)
    n11 = index.cooccurrence_block(hist_arr, cands)
    inter = (n11 > 0).sum(axis=0).astype(np.float64)
    h_len = float(len(hist))
    union = h_len + sizes - inter
    j = np.zeros(C, dtype=np.float64)
    ok_u = union > 0
    j[ok_u] = inter[ok_u] / union[ok_u]
    c = np.zeros(C, dtype=np.float64)
    ok_c = (h_len > 0) & (sizes > 0)
    c[ok_c] = inter[ok_c] / np.sqrt(h_len * sizes[ok_c])
    X[:, 3] = j.astype(np.float32)
    X[:, 4] = c.astype(np.float32)
    return X


def build_clean_v2_tabular_A(
    user_id: int,
    candidate_item_id: int,
    user_history: dict[int, set[int]],
    item_statistics: dict[str, Any],
    kg_statistics: dict[str, Any],
    index: PairwiseStatsIndex,
) -> np.ndarray:
    """Return (5,) float32 CLEAN Tabular A for pair (u,X).

    History length uses full model_train H_u (not H_u \\ {X}).
    """

    pop = item_statistics["popularity"]
    kg_deg = kg_statistics["item_kg_degree"]
    row = vectorized_clean_v2_A_for_user(
        int(user_id),
        np.asarray([int(candidate_item_id)], dtype=np.int64),
        user_history,
        pop,
        kg_deg,
        index,
    )[0]
    if row.shape != (5,):
        raise RuntimeError(f"CLEAN V2 A dim {row.shape} != (5,)")
    return row
