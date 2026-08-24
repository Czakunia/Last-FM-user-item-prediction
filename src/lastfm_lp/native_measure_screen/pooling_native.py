"""Native Top25 encodings: A11 signed 8-D; MI/Cosine nonneg 7-D."""

from __future__ import annotations

import numpy as np

from src.lastfm_lp.native_measure_screen.constants import A11_DIM, NONNEG_DIM


def encode_a11_8d(values: np.ndarray) -> np.ndarray:
    """Signed A11 vector → 8-D float32 (population std, top3/bottom3 signed)."""

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(A11_DIM, dtype=np.float64)
    k = int(v.size)
    if k == 0:
        return out.astype(np.float32)
    mean = float(v.mean())
    out[0] = mean
    out[1] = float(np.std(v, ddof=0))
    out[2] = float(v.min())
    out[3] = float(v.max())
    kk = min(3, k)
    order_desc = np.argsort(-v)
    order_asc = np.argsort(v)
    out[4] = float(v[order_desc[:kk]].mean())
    out[5] = float(v[order_asc[:kk]].mean())
    out[6] = float(np.mean(v > 0.0))
    out[7] = float(np.mean(v < 0.0))
    if not np.isfinite(out).all():
        raise ValueError("NaN/Inf in encode_a11_8d")
    return out.astype(np.float32)


def encode_nonneg_7d(values: np.ndarray) -> np.ndarray:
    """MI or Cosine vector → 7-D float32 (quantiles method=linear)."""

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros(NONNEG_DIM, dtype=np.float64)
    k = int(v.size)
    if k == 0:
        return out.astype(np.float32)
    out[0] = float(v.mean())
    out[1] = float(np.std(v, ddof=0))
    # NumPy 1.22+ uses method=; older uses interpolation=
    try:
        q = np.quantile(v, [0.25, 0.50, 0.75], method="linear")
    except TypeError:
        q = np.quantile(v, [0.25, 0.50, 0.75], interpolation="linear")
    out[2] = float(q[1])
    out[3] = float(q[0])
    out[4] = float(q[2])
    out[5] = float(v.max())
    kk = min(3, k)
    order_desc = np.argsort(-v)
    out[6] = float(v[order_desc[:kk]].mean())
    if not np.isfinite(out).all():
        raise ValueError("NaN/Inf in encode_nonneg_7d")
    return out.astype(np.float32)


def encode_a11_8d_matrix(selected: np.ndarray) -> np.ndarray:
    """(K, C) → (C, 8)."""

    a = np.asarray(selected, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("selected must be KxC")
    k, c = a.shape
    out = np.zeros((c, A11_DIM), dtype=np.float32)
    if k == 0:
        return out
    for j in range(c):
        out[j] = encode_a11_8d(a[:, j])
    return out


def encode_nonneg_7d_matrix(selected: np.ndarray) -> np.ndarray:
    """(K, C) → (C, 7)."""

    a = np.asarray(selected, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("selected must be KxC")
    k, c = a.shape
    out = np.zeros((c, NONNEG_DIM), dtype=np.float32)
    if k == 0:
        return out
    for j in range(c):
        out[j] = encode_nonneg_7d(a[:, j])
    return out
