"""Orthonormal third-order HCR a111 (LASTFM_FULL_SOTA_V2_HIGHER_ORDER_HCR).

Canonical coefficient for three binary variables X_j, X_k, X_i:

    a111 = E[ φ_j(X_j) φ_k(X_k) φ_i(X_i) ]

with orthonormal contrasts

    φ(x; p) = (x - p) / sqrt(p(1-p)).

Energy: E111 = a111².

Pooling for H3_RAW (8-D) mirrors H2 a11_energy with d=2: wmean uses uniform
weights and is therefore identical to mean (documented, not a new weighting).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.lastfm_lp.binary.binary_measures import binary_contrast_phi
from src.lastfm_lp.binary.user_hcr_aggregation import _truncate_history, pool_a11_energy_matrix

H3_RAW_FEATURE_ORDER = [
    "mean:hcr_a111",
    "mean:hcr_energy111",
    "max:hcr_a111",
    "max:hcr_energy111",
    "top3mean:hcr_a111",
    "top3mean:hcr_energy111",
    "wmean:hcr_a111",
    "wmean:hcr_energy111",
]


def a111_from_basis(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Reference a111 = mean(φ(x)φ(y)φ(z)) on empirical margins."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.shape != z.shape or x.size == 0:
        raise ValueError("x,y,z must be non-empty and aligned")
    p_x = float(np.clip(x.mean(), eps, 1.0 - eps))
    p_y = float(np.clip(y.mean(), eps, 1.0 - eps))
    p_z = float(np.clip(z.mean(), eps, 1.0 - eps))
    if (
        p_x <= eps
        or p_x >= 1.0 - eps
        or p_y <= eps
        or p_y >= 1.0 - eps
        or p_z <= eps
        or p_z >= 1.0 - eps
    ):
        return 0.0
    return float(
        np.mean(
            binary_contrast_phi(x, p_x, eps=eps)
            * binary_contrast_phi(y, p_y, eps=eps)
            * binary_contrast_phi(z, p_z, eps=eps)
        )
    )


def a111_from_counts(
    n_jki: int,
    n_jk: int,
    n_ji: int,
    n_ki: int,
    n_j: int,
    n_k: int,
    n_i: int,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """a111 and energy from contingency counts (equivalent to basis form).

    numerator =
        p_jki - p_j*p_ki - p_k*p_ji - p_i*p_jk + 2*p_j*p_k*p_i
    denominator =
        sqrt(p_j(1-p_j) p_k(1-p_k) p_i(1-p_i))
    """

    N = float(n_users)
    if N <= 0:
        return 0.0, 0.0
    p_j = float(n_j) / N
    p_k = float(n_k) / N
    p_i = float(n_i) / N
    p_jk = float(n_jk) / N
    p_ji = float(n_ji) / N
    p_ki = float(n_ki) / N
    p_jki = float(n_jki) / N

    denom = np.sqrt(
        max(p_j * (1.0 - p_j), 0.0)
        * max(p_k * (1.0 - p_k), 0.0)
        * max(p_i * (1.0 - p_i), 0.0)
    )
    if denom <= eps:
        return 0.0, 0.0
    num = p_jki - p_j * p_ki - p_k * p_ji - p_i * p_jk + 2.0 * p_j * p_k * p_i
    a = float(num / denom)
    return a, float(a * a)


def a111_energy_from_counts(
    n_jki: int,
    n_jk: int,
    n_ji: int,
    n_ki: int,
    n_j: int,
    n_k: int,
    n_i: int,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    return a111_from_counts(
        n_jki, n_jk, n_ji, n_ki, n_j, n_k, n_i, n_users, eps=eps
    )


def a111_energy_from_n_jki_matrix(
    n_jki: np.ndarray,
    n_jk: np.ndarray,
    n_ji: np.ndarray,
    n_ki: np.ndarray,
    n_j: np.ndarray,
    n_k: np.ndarray,
    n_i: np.ndarray,
    n_users: int,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized a111 / energy. Shapes broadcast to (M, C)."""

    N = float(n_users)
    if N <= 0:
        z = np.zeros_like(n_jki, dtype=np.float64)
        return z, z.copy()
    p_j = np.asarray(n_j, dtype=np.float64) / N
    p_k = np.asarray(n_k, dtype=np.float64) / N
    p_i = np.asarray(n_i, dtype=np.float64) / N
    p_jk = np.asarray(n_jk, dtype=np.float64) / N
    p_ji = np.asarray(n_ji, dtype=np.float64) / N
    p_ki = np.asarray(n_ki, dtype=np.float64) / N
    p_jki = np.asarray(n_jki, dtype=np.float64) / N
    # broadcast: n_j, n_k, n_jk → (M,1); n_i → (C,); n_ji,n_ki,n_jki → (M,C)
    denom = np.sqrt(
        np.maximum(p_j * (1.0 - p_j), 0.0)
        * np.maximum(p_k * (1.0 - p_k), 0.0)
        * np.maximum(p_i * (1.0 - p_i), 0.0)
    )
    num = p_jki - p_j * p_ki - p_k * p_ji - p_i * p_jk + 2.0 * p_j * p_k * p_i
    a = np.zeros_like(p_jki, dtype=np.float64)
    ok = denom > eps
    a[ok] = num[ok] / denom[ok]
    return a, a * a


def history_unordered_pairs(history_items: Iterable[int], max_history: int = 25) -> np.ndarray:
    """Return (M, 2) array of pairs (j,k) with j < k after H2 truncation."""

    hist = _truncate_history(history_items, max_history)
    ids = sorted({int(x) for x in hist})
    if len(ids) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pairs.append((ids[a], ids[b]))
    return np.asarray(pairs, dtype=np.int64)


def pool_h3_raw_matrix(a111: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Pool M triples → 8-D float32 (same layout as H2 a11_energy).

    ``a111`` / ``energy`` shape (M,) or (M, C). wmean ≡ mean for d=2.
    """

    a = np.asarray(a111, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        e = e.reshape(-1, 1)
    return pool_a11_energy_matrix(a, e, kind="a11_energy")
