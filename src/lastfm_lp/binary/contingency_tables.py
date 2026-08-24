"""Sparse item–item co-occurrence (n11) from training histories only."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy import sparse


def build_cooccurrence(
    train_by_user: dict[int, set[int]],
    n_items: int,
    *,
    max_history_for_pairs: int = 80,
    seed: int = 2026,
) -> sparse.csr_matrix:
    """Count users who interacted with both j and i (upper-tri stored symmetrically).

    For users with very long histories, subsample items before pair expansion
    (frozen by seed) to keep Stage B tractable.
    """
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for items in train_by_user.values():
        arr = np.fromiter(items, dtype=np.int32)
        if arr.size < 2:
            continue
        if arr.size > max_history_for_pairs:
            arr = rng.choice(arr, size=max_history_for_pairs, replace=False)
        arr.sort()
        # all unordered pairs
        for a_idx in range(len(arr)):
            a = int(arr[a_idx])
            for b in arr[a_idx + 1 :]:
                b = int(b)
                rows.append(a)
                cols.append(b)
                data.append(1)
                rows.append(b)
                cols.append(a)
                data.append(1)

    if not data:
        return sparse.csr_matrix((n_items, n_items), dtype=np.int32)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.int32)
    return mat.tocsr()


def contingency_from_pop(
    n11: int,
    pop_j: int,
    pop_i: int,
    n_users: int,
) -> tuple[int, int, int, int]:
    n10 = max(pop_j - n11, 0)
    n01 = max(pop_i - n11, 0)
    n00 = max(n_users - pop_j - pop_i + n11, 0)
    return int(n11), int(n10), int(n01), int(n00)


def get_n11(cooc: sparse.csr_matrix, j: int, i: int) -> int:
    if j == i:
        return 0
    return int(cooc[j, i])


def cooc_row_arrays(cooc: sparse.csr_matrix, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted column indices and data for CSR row ``i`` (views when possible)."""

    i = int(i)
    start = int(cooc.indptr[i])
    end = int(cooc.indptr[i + 1])
    return cooc.indices[start:end], cooc.data[start:end]


def gather_n11_vs_candidate(
    cooc: sparse.csr_matrix,
    js: np.ndarray,
    i: int,
    *,
    row_cols: np.ndarray | None = None,
    row_data: np.ndarray | None = None,
) -> np.ndarray:
    """Batch ``n11(j, i)`` for history items ``js`` vs fixed candidate ``i``.

    Uses symmetry of the co-occurrence matrix (row ``i`` holds the same values
    as column ``i``). Diagonal ``j == i`` is forced to 0, matching ``get_n11``.
    """

    js = np.asarray(js, dtype=np.int64).reshape(-1)
    out = np.zeros(js.shape[0], dtype=np.int32)
    if js.size == 0:
        return out
    if row_cols is None or row_data is None:
        row_cols, row_data = cooc_row_arrays(cooc, int(i))
    if row_cols.size == 0:
        return out
    pos = np.searchsorted(row_cols, js)
    in_range = pos < row_cols.size
    pos_safe = np.minimum(pos, row_cols.size - 1)
    match = in_range & (row_cols[pos_safe] == js)
    out[match] = np.asarray(row_data[pos_safe[match]], dtype=np.int32)
    out[js == int(i)] = 0
    return out
