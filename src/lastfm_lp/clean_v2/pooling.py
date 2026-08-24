"""CLEAN V2 HCR pool: exactly [mean, max, top3mean] of signed A11 — no wmean."""

from __future__ import annotations

import numpy as np

from src.lastfm_lp.clean_v2.constants import CLEAN_V2_HCR_DIM


def pool_signed_a11_3d(values: np.ndarray, *, top_k: int = 3) -> np.ndarray:
    """Pool signed A11 vector v → (3,) float32: mean, max, top3mean.

    top3mean = mean of the largest min(3, K) **signed** values (not abs).
    Empty → zeros.
    """

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(CLEAN_V2_HCR_DIM, dtype=np.float64)
    if v.size == 0:
        return out.astype(np.float32)
    out[0] = float(v.mean())
    out[1] = float(v.max())
    k = min(int(top_k), int(v.size))
    # largest signed values
    order = np.argsort(-v)
    out[2] = float(v[order[:k]].mean())
    # materialize as float32 for feature matrices; values stay exact in float64 math
    return out.astype(np.float32)


def pool_signed_a11_3d_matrix(a11_selected: np.ndarray, *, top_k: int = 3) -> np.ndarray:
    """Pool each column of (K, C) selected signed A11 → (C, 3)."""

    a = np.asarray(a11_selected, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("a11_selected must be KxC")
    K, C = a.shape
    out = np.zeros((C, CLEAN_V2_HCR_DIM), dtype=np.float32)
    if K == 0:
        return out
    out[:, 0] = a.mean(axis=0).astype(np.float32)
    out[:, 1] = a.max(axis=0).astype(np.float32)
    kk = min(int(top_k), K)
    order = np.argsort(-a, axis=0)
    top = np.take_along_axis(a, order[:kk, :], axis=0)
    out[:, 2] = top.mean(axis=0).astype(np.float32)
    return out
