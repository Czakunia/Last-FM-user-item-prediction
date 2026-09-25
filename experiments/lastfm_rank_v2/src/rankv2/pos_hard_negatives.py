"""R3 positive-local semi-hard negatives.

Reconstructed from the lost ``lastfm_rank_v2`` tree. The call sites in
``01_build_hardneg_data.py``, ``01b_audit_negatives.py`` and
``sealed_external.py`` fix the contract:

* constructor kwargs: neighbors, neighbor_scores, item_pop, model_train,
  n_items, rng, band_lo=0.05, band_hi=0.30
* ``sample(u, i, n_neg=4)`` → ``{"items": [...], "sources": [...]}``
* sources ∈ {hard_local, pop, random}
* target mix for n_neg=4: 2 hard_local + 1 pop + 1 random
* never return the positive item or anything in the user's train history
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PositiveLocalHardNegativeSampler:
    def __init__(
        self,
        *,
        neighbors: np.ndarray,
        neighbor_scores: np.ndarray,
        item_pop: np.ndarray,
        model_train: dict[int, set[int]],
        n_items: int,
        rng: np.random.Generator,
        band_lo: float = 0.05,
        band_hi: float = 0.30,
    ) -> None:
        self.neighbors = np.asarray(neighbors)
        self.neighbor_scores = np.asarray(neighbor_scores)
        self.item_pop = np.asarray(item_pop, dtype=np.float64).reshape(-1)
        self.model_train = model_train
        self.n_items = int(n_items)
        self.rng = rng
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        if self.item_pop.shape[0] != self.n_items:
            raise ValueError("item_pop length must equal n_items")
        weights = np.maximum(self.item_pop, 0.0)
        total = float(weights.sum())
        self._pop_p = (weights / total) if total > 0 else None

    def sample(self, u: int, i: int, n_neg: int = 4) -> dict[str, Any]:
        n_hard, n_pop, n_rand = _mix_for(int(n_neg))
        forbidden = set(self.model_train.get(int(u), ()))
        forbidden.add(int(i))
        items: list[int] = []
        sources: list[str] = []

        hard_pool = self._band_items(int(i), forbidden)
        if len(hard_pool) < n_hard:
            seen = set(hard_pool)
            for j in self._all_neighbors(int(i)):
                if j in forbidden or j in seen:
                    continue
                hard_pool.append(j)
                seen.add(j)
                if len(hard_pool) >= n_hard:
                    break
        take = min(n_hard, len(hard_pool))
        if take:
            pick = self.rng.choice(np.asarray(hard_pool, dtype=np.int64), size=take, replace=False)
            for j in pick.tolist():
                items.append(int(j))
                sources.append("hard_local")
                forbidden.add(int(j))

        for _ in range(n_pop):
            j = self._sample_pop(forbidden)
            if j is None:
                j = self._sample_uniform(forbidden)
                sources.append("random")
            else:
                sources.append("pop")
            items.append(int(j))
            forbidden.add(int(j))

        for _ in range(n_rand):
            j = self._sample_uniform(forbidden)
            items.append(int(j))
            sources.append("random")
            forbidden.add(int(j))

        while len(items) < n_neg:
            j = self._sample_uniform(forbidden)
            items.append(int(j))
            sources.append("random")
            forbidden.add(int(j))

        return {"items": items[:n_neg], "sources": sources[:n_neg]}

    def _all_neighbors(self, i: int) -> list[int]:
        row = np.asarray(self.neighbors[i])
        return [int(j) for j in row.tolist() if int(j) >= 0]

    def _band_items(self, i: int, forbidden: set[int]) -> list[int]:
        row = np.asarray(self.neighbors[i])
        scores = np.asarray(self.neighbor_scores[i], dtype=np.float64)
        valid = row >= 0
        if not np.any(valid):
            return []
        values = scores[valid]
        lo = float(np.quantile(values, self.band_lo))
        hi = float(np.quantile(values, self.band_hi))
        if hi < lo:
            lo, hi = hi, lo
        in_band = valid & (scores >= lo) & (scores <= hi)
        out: list[int] = []
        for j in row[in_band].tolist():
            j = int(j)
            if j >= 0 and j not in forbidden:
                out.append(j)
        return out

    def _sample_pop(self, forbidden: set[int], max_tries: int = 256) -> int | None:
        if self._pop_p is None:
            return None
        for _ in range(max_tries):
            j = int(self.rng.choice(self.n_items, p=self._pop_p))
            if j not in forbidden:
                return j
        return None

    def _sample_uniform(self, forbidden: set[int], max_tries: int = 4096) -> int:
        for _ in range(max_tries):
            j = int(self.rng.integers(0, self.n_items))
            if j not in forbidden:
                return j
        for j in range(self.n_items):
            if j not in forbidden:
                return j
        raise RuntimeError("no unused item left for a negative")


def _mix_for(n_neg: int) -> tuple[int, int, int]:
    if n_neg <= 0:
        raise ValueError("n_neg must be positive")
    if n_neg == 4:
        return 2, 1, 1
    n_hard = min(2, n_neg)
    rest = n_neg - n_hard
    n_pop = 1 if rest >= 2 else (1 if rest == 1 else 0)
    n_rand = rest - n_pop
    return n_hard, n_pop, n_rand
