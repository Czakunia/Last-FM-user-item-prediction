"""Canonical 2×2 contingency tables for KRAM measure race.

Notation (consistent across code + docs):

                H=1      H=0
C=1             a=n11    b=n10
C=0             c=n01    d=n00

N = a+b+c+d
history_support   = a+b = pop[h]
candidate_support = a+c = pop[c]
expected_n11      = (history_support * candidate_support) / N
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contingency2x2:
    """Scalar contingency for one (candidate, history) pair."""

    a: float  # n11
    b: float  # n10
    c: float  # n01
    d: float  # n00

    @property
    def n11(self) -> float:
        return self.a

    @property
    def n10(self) -> float:
        return self.b

    @property
    def n01(self) -> float:
        return self.c

    @property
    def n00(self) -> float:
        return self.d

    @property
    def N(self) -> float:
        return self.a + self.b + self.c + self.d

    @property
    def history_support(self) -> float:
        return self.a + self.b

    @property
    def candidate_support(self) -> float:
        return self.a + self.c

    @property
    def expected_n11(self) -> float:
        n = self.N
        if n <= 0:
            return 0.0
        return (self.history_support * self.candidate_support) / n


@dataclass(frozen=True)
class ContingencyBatch:
    """HxC batch of contingency tables from the same population."""

    a: np.ndarray  # n11
    b: np.ndarray  # n10
    c: np.ndarray  # n01
    d: np.ndarray  # n00
    N: float
    history_support: np.ndarray  # (H,1) or (H,)
    candidate_support: np.ndarray  # (1,C) or (C,)

    @property
    def n11(self) -> np.ndarray:
        return self.a

    @property
    def n10(self) -> np.ndarray:
        return self.b

    @property
    def n01(self) -> np.ndarray:
        return self.c

    @property
    def n00(self) -> np.ndarray:
        return self.d

    @property
    def expected_n11(self) -> np.ndarray:
        hs = np.asarray(self.history_support, dtype=np.float64)
        cs = np.asarray(self.candidate_support, dtype=np.float64)
        if hs.ndim == 1:
            hs = hs.reshape(-1, 1)
        if cs.ndim == 1:
            cs = cs.reshape(1, -1)
        n = float(self.N)
        if n <= 0:
            return np.zeros_like(self.a, dtype=np.float64)
        return (hs * cs) / n


def contingency_from_counts(
    n11: float | int,
    n10: float | int,
    n01: float | int,
    n00: float | int,
) -> Contingency2x2:
    return Contingency2x2(
        a=float(n11), b=float(n10), c=float(n01), d=float(n00)
    )


def contingency_from_n11_pop(
    n11: float | int,
    *,
    history_support: float | int,
    candidate_support: float | int,
    n_users: int,
) -> Contingency2x2:
    """Build 2×2 from n11 + marginal supports (train population size N)."""

    a = float(n11)
    hs = float(history_support)
    cs = float(candidate_support)
    n = float(n_users)
    b = max(hs - a, 0.0)
    c = max(cs - a, 0.0)
    d = max(n - hs - cs + a, 0.0)
    return Contingency2x2(a=a, b=b, c=c, d=d)


def contingency_batch_from_n11_matrix(
    n11: np.ndarray,
    pop_j: np.ndarray,
    pop_i: np.ndarray,
    n_users: int,
) -> ContingencyBatch:
    """Canonical HxC contingency from co-occurrence + popularity.

    pop_j = history_support (length H)
    pop_i = candidate_support (length C)
    N = n_users (fit population)
    """

    a = np.asarray(n11, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("n11 must be 2-d (H×C)")
    hs = np.asarray(pop_j, dtype=np.float64).reshape(-1, 1)
    cs = np.asarray(pop_i, dtype=np.float64).reshape(1, -1)
    if hs.shape[0] != a.shape[0] or cs.shape[1] != a.shape[1]:
        raise ValueError("popularity shapes must broadcast to n11")
    n = float(n_users)
    b = np.maximum(hs - a, 0.0)
    c = np.maximum(cs - a, 0.0)
    d = np.maximum(n - hs - cs + a, 0.0)
    return ContingencyBatch(
        a=a,
        b=b,
        c=c,
        d=d,
        N=n,
        history_support=hs,
        candidate_support=cs,
    )
