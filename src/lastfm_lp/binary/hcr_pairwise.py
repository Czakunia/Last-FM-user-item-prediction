"""Pairwise orthonormal HCR / compact10 lookup from sparse co-occurrence."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import sparse

from src.lastfm_lp.binary.binary_measures import (
    BinaryPairMeasures,
    a11_energy_from_n11_batch,
    a11_energy_from_n11_matrix,
    legacy_assoc_from_n11_batch,
    legacy_assoc_from_n11_matrix,
    measures_from_counts,
)
from src.lastfm_lp.binary.contingency_tables import (
    contingency_from_pop,
    cooc_row_arrays,
    gather_n11_vs_candidate,
    get_n11,
)

# Backward-compatible alias
BinaryMeasures = BinaryPairMeasures


class PairwiseStatsIndex:
    def __init__(
        self,
        cooc: sparse.csr_matrix,
        popularity: np.ndarray,
        n_users: int,
        *,
        smoothing: float = 0.5,
        uncertainty_by_pair: Mapping[tuple[int, int], float] | None = None,
    ) -> None:
        self.cooc = cooc
        self.popularity = popularity.astype(np.int32)
        self.n_users = int(n_users)
        self.smoothing = float(smoothing)
        self.uncertainty_by_pair = dict(uncertainty_by_pair or {})
        self._cache: dict[tuple[int, int], BinaryPairMeasures] = {}

    def pair(self, j: int, i: int) -> BinaryPairMeasures:
        """Directed association stats for history item j → candidate i."""

        dkey = (int(j), int(i))
        if dkey in self._cache:
            return self._cache[dkey]
        n11 = get_n11(self.cooc, int(j), int(i))
        n11, n10, n01, n00 = contingency_from_pop(
            n11,
            int(self.popularity[j]),
            int(self.popularity[i]),
            self.n_users,
        )
        # Uncertainty is symmetric in (j,i) under a11 symmetry.
        unc = self.uncertainty_by_pair.get(
            dkey, self.uncertainty_by_pair.get((int(i), int(j)), 0.0)
        )
        m = measures_from_counts(
            n11,
            n10,
            n01,
            n00,
            smoothing=self.smoothing,
            uncertainty=float(unc),
        )
        self._cache[dkey] = m
        return m

    def compact10(self, j: int, i: int) -> np.ndarray:
        return self.pair(j, i).compact10()

    def compact8(self, j: int, i: int) -> np.ndarray:
        return self.pair(j, i).compact8()

    def feature_block(self, j: int, i: int) -> np.ndarray:
        return self.pair(j, i).feature_block()

    def valid_mask(self, j: int, i: int) -> float:
        return float(self.pair(j, i).valid_mask)

    def a11_energy_vs_history(
        self,
        js: np.ndarray,
        i: int,
        *,
        row_cols: np.ndarray | None = None,
        row_data: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch raw a11 and energy for history items ``js`` vs candidate ``i``."""

        js = np.asarray(js, dtype=np.int64).reshape(-1)
        if js.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty
        n11 = gather_n11_vs_candidate(
            self.cooc,
            js,
            int(i),
            row_cols=row_cols,
            row_data=row_data,
        )
        pop_j = self.popularity[js]
        return a11_energy_from_n11_batch(
            n11,
            pop_j,
            int(self.popularity[int(i)]),
            self.n_users,
        )

    def cooc_row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return cooc_row_arrays(self.cooc, int(i))

    def cooccurrence_block(
        self,
        history_ids: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        """Small dense n11 block (H×C). Diagonal hist==cand forced to 0 like ``get_n11``."""

        hist = np.asarray(history_ids, dtype=np.int64).reshape(-1)
        cands = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        H, C = int(hist.size), int(cands.size)
        if H == 0 or C == 0:
            return np.zeros((H, C), dtype=np.int32)
        # scipy CSR fancy index → tiny H×C sparse, then dense
        block = self.cooc[hist][:, cands]
        if sparse.issparse(block):
            arr = np.asarray(block.toarray(), dtype=np.int32)
        else:
            arr = np.asarray(block, dtype=np.int32)
        # j == i → 0
        same = hist[:, None] == cands[None, :]
        if same.any():
            arr = arr.copy()
            arr[same] = 0
        return arr

    def a11_energy_block(
        self,
        history_ids: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw a11 / energy matrices (H×C) for one history vs many candidates."""

        hist = np.asarray(history_ids, dtype=np.int64).reshape(-1)
        cands = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        if hist.size == 0 or cands.size == 0:
            z = np.zeros((hist.size, cands.size), dtype=np.float64)
            return z, z.copy()
        n11 = self.cooccurrence_block(hist, cands)
        return a11_energy_from_n11_matrix(
            n11,
            self.popularity[hist],
            self.popularity[cands],
            self.n_users,
        )

    def legacy_assoc_block(
        self,
        history_ids: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        """H×C LEGACY_BINARY_ASSOC_V1 for one history vs many candidates."""

        hist = np.asarray(history_ids, dtype=np.int64).reshape(-1)
        cands = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        if hist.size == 0 or cands.size == 0:
            return np.zeros((hist.size, cands.size), dtype=np.float64)
        n11 = self.cooccurrence_block(hist, cands)
        return legacy_assoc_from_n11_matrix(
            n11,
            self.popularity[hist],
            self.popularity[cands],
            self.n_users,
            smoothing=self.smoothing,
        )

    def legacy_assoc_vs_history(
        self,
        js: np.ndarray,
        i: int,
        *,
        row_cols: np.ndarray | None = None,
        row_data: np.ndarray | None = None,
    ) -> np.ndarray:
        js = np.asarray(js, dtype=np.int64).reshape(-1)
        if js.size == 0:
            return np.zeros(0, dtype=np.float64)
        n11 = gather_n11_vs_candidate(
            self.cooc,
            js,
            int(i),
            row_cols=row_cols,
            row_data=row_data,
        )
        return legacy_assoc_from_n11_batch(
            n11,
            self.popularity[js],
            int(self.popularity[int(i)]),
            self.n_users,
            smoothing=self.smoothing,
        )
