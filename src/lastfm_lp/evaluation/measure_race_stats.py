"""Prepared statistical utilities for the KRAM measure race (no results yet).

Predeclared BEFORE seeing full-race outcomes:
  - paired user-level deltas (HCR − baseline)
  - paired sign-flip / permutation test
  - Holm multiplicity correction over the 6 primary HCR comparisons
  - seeds = stability only (not inferential N)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

PRIMARY_COMPARATORS = (
    "cooc_top25",
    "jaccard_top25",
    "cosine_top25",
    "mi_top25",
    "npmi_top25",
    "g2_top25",
)

PERMUTATION_SEED = 2026
N_PERMUTATIONS = 10_000
MULTIPLICITY_METHOD = "holm"


@dataclass(frozen=True)
class PairedDeltaSummary:
    mean: float
    median: float
    std: float
    ci95_low: float
    ci95_high: float
    frac_pos: float
    frac_zero: float
    frac_neg: float
    n: int


def paired_deltas(hcr: np.ndarray, other: np.ndarray) -> np.ndarray:
    h = np.asarray(hcr, dtype=np.float64)
    o = np.asarray(other, dtype=np.float64)
    if h.shape != o.shape:
        raise ValueError("paired arrays must match")
    return h - o


def summarize_deltas(delta: np.ndarray) -> PairedDeltaSummary:
    d = np.asarray(delta, dtype=np.float64)
    n = int(d.size)
    if n == 0:
        nan = float("nan")
        return PairedDeltaSummary(nan, nan, nan, nan, nan, nan, nan, nan, 0)
    mean = float(d.mean())
    # normal approx CI on mean (predeclared); for skewed metrics prefer bootstrap later if frozen
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = 1.959963984540054
    return PairedDeltaSummary(
        mean=mean,
        median=float(np.median(d)),
        std=float(d.std(ddof=1)) if n > 1 else 0.0,
        ci95_low=mean - z * se,
        ci95_high=mean + z * se,
        frac_pos=float(np.mean(d > 0)),
        frac_zero=float(np.mean(d == 0)),
        frac_neg=float(np.mean(d < 0)),
        n=n,
    )


def paired_signflip_pvalue(
    delta: np.ndarray,
    *,
    n_perm: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> float:
    """Two-sided paired sign-flip test on mean(delta).

    Null: methods exchangeable within user ⇒ sign(delta_u) random.
    """

    d = np.asarray(delta, dtype=np.float64)
    if d.size == 0:
        return float("nan")
    obs = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), d.size))
    null = np.abs((signs * d[None, :]).mean(axis=1))
    # add-one smoothing for discrete p
    return float((1 + np.sum(null >= obs)) / (n_perm + 1))


def holm_correction(p_raw: Iterable[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""

    p = np.asarray(list(p_raw), dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        # Holm: (m-rank)*p_(rank)
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return [float(x) for x in adj]


def seed_stability(
    values_by_seed: dict[int, float],
) -> dict[str, float]:
    seeds = sorted(values_by_seed)
    arr = np.asarray([values_by_seed[s] for s in seeds], dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": float(arr.size),
    }
