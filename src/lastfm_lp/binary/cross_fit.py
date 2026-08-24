"""User-fold cross-fitting for orthonormal HCR (LASTFM_ORTHONORMAL_HCR_V2).

Train examples for user u ∈ F_k use contingency tables estimated on
U_train \\ F_k.  Validation/test use a frozen index on full model_train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.lastfm_lp.binary.contingency_tables import build_cooccurrence
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.data.load_kgat_lastfm import item_popularity


def assign_user_folds(
    user_ids: list[int],
    *,
    n_folds: int = 5,
    seed: int = 2026,
) -> dict[int, int]:
    """Deterministic user → fold map in {0, …, n_folds-1}."""

    rng = np.random.default_rng(seed)
    users = np.asarray(sorted(int(u) for u in user_ids), dtype=np.int64)
    order = rng.permutation(len(users))
    folds = np.empty(len(users), dtype=np.int32)
    for i, idx in enumerate(order):
        folds[idx] = int(i % n_folds)
    return {int(u): int(f) for u, f in zip(users, folds)}


def _subset_train(
    model_train: dict[int, set[int]],
    keep_users: set[int],
) -> dict[int, set[int]]:
    return {u: items for u, items in model_train.items() if u in keep_users}


def build_fold_index(
    model_train: dict[int, set[int]],
    n_items: int,
    *,
    max_history_for_pairs: int = 60,
    seed: int = 2026,
    smoothing: float = 0.5,
) -> PairwiseStatsIndex:
    pop = item_popularity(model_train, n_items)
    cooc = build_cooccurrence(
        model_train,
        n_items,
        max_history_for_pairs=max_history_for_pairs,
        seed=seed,
    )
    return PairwiseStatsIndex(
        cooc,
        pop,
        n_users=len(model_train),
        smoothing=smoothing,
    )


def compute_a11_uncertainty(
    fold_indices: list[PairwiseStatsIndex],
    j: int,
    i: int,
) -> float:
    """Cross-fold SD of raw a11 for pair (j, i)."""

    vals = [float(idx.pair(j, i).hcr_a11) for idx in fold_indices]
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1))


class CrossFitHCRBundle:
    """Hold full + leave-one-fold HCR indices and resolve per-user lookups."""

    def __init__(
        self,
        *,
        full_index: PairwiseStatsIndex,
        fold_indices: list[PairwiseStatsIndex],
        user_to_fold: dict[int, int],
        n_folds: int,
    ) -> None:
        self.full_index = full_index
        self.fold_indices = fold_indices
        self.user_to_fold = user_to_fold
        self.n_folds = int(n_folds)
        self._unc_cache: dict[tuple[int, int], float] = {}

    def index_for(self, user_id: int, *, split: str) -> PairwiseStatsIndex:
        """Return the estimation index that must be used for this row."""

        if split in {"val", "valid", "validation", "test"}:
            return self.full_index
        fold = self.user_to_fold.get(int(user_id))
        if fold is None:
            # unseen train user → fall back to full (should be rare)
            return self.full_index
        return self.fold_indices[int(fold)]

    def uncertainty(self, j: int, i: int) -> float:
        key = (int(j), int(i)) if int(j) <= int(i) else (int(i), int(j))
        if key in self._unc_cache:
            return self._unc_cache[key]
        unc = compute_a11_uncertainty(self.fold_indices, int(j), int(i))
        self._unc_cache[key] = unc
        return unc

    def pair(self, user_id: int, j: int, i: int, *, split: str):
        m = self.index_for(user_id, split=split).pair(j, i)
        # Attach cross-fold uncertainty without mutating frozen dataclass:
        # callers that need unc use uncertainty() + measures_from_counts rebuild,
        # or use pair_with_uncertainty below.
        return m

    def pair_with_uncertainty(self, user_id: int, j: int, i: int, *, split: str):
        from dataclasses import replace

        m = self.index_for(user_id, split=split).pair(j, i)
        unc = self.uncertainty(j, i)
        if abs(m.uncertainty - unc) < 1e-12:
            return m
        return replace(m, uncertainty=float(unc))


def build_cross_fit_bundle(
    model_train: dict[int, set[int]],
    n_items: int,
    *,
    n_folds: int = 5,
    seed: int = 2026,
    smoothing: float = 0.5,
    max_history_for_pairs: int = 60,
    full_index: PairwiseStatsIndex | None = None,
    cache_dir: Path | str | None = None,
) -> CrossFitHCRBundle:
    from pathlib import Path as _Path

    from scipy.sparse import load_npz, save_npz

    users = list(model_train.keys())
    user_to_fold = assign_user_folds(users, n_folds=n_folds, seed=seed)
    cache: _Path | None = _Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        folds_path = cache / "user_folds.json"
        if folds_path.exists():
            import json

            cached_folds = {int(k): int(v) for k, v in json.loads(folds_path.read_text()).items()}
            if cached_folds == user_to_fold and all(
                (cache / f"fold_{k}_cooc.npz").exists() for k in range(n_folds)
            ):
                print(f"[cross-fit] loading cached fold coocs from {cache}")
                fold_indices = []
                for k in range(n_folds):
                    cooc = load_npz(cache / f"fold_{k}_cooc.npz")
                    pop = np.load(cache / f"fold_{k}_pop.npy")
                    n_users_k = int(np.load(cache / f"fold_{k}_n_users.npy")[0])
                    fold_indices.append(
                        PairwiseStatsIndex(cooc, pop, n_users=n_users_k, smoothing=smoothing)
                    )
                    print(f"[cross-fit] fold {k}: n_users={n_users_k} cooc_nnz={cooc.nnz} (cache)")
                if full_index is None:
                    full_index = build_fold_index(
                        model_train,
                        n_items,
                        max_history_for_pairs=max_history_for_pairs,
                        seed=seed,
                        smoothing=smoothing,
                    )
                return CrossFitHCRBundle(
                    full_index=full_index,
                    fold_indices=fold_indices,
                    user_to_fold=user_to_fold,
                    n_folds=n_folds,
                )

    if full_index is None:
        full_index = build_fold_index(
            model_train,
            n_items,
            max_history_for_pairs=max_history_for_pairs,
            seed=seed,
            smoothing=smoothing,
        )

    fold_indices: list[PairwiseStatsIndex] = []
    for k in range(n_folds):
        keep = {u for u, f in user_to_fold.items() if f != k}
        sub = _subset_train(model_train, keep)
        # Distinct seed offset so pair-subsampling differs per fold but is frozen.
        idx = build_fold_index(
            sub,
            n_items,
            max_history_for_pairs=max_history_for_pairs,
            seed=seed + 17 * (k + 1),
            smoothing=smoothing,
        )
        fold_indices.append(idx)
        print(f"[cross-fit] fold {k}: n_users={len(sub)} cooc_nnz={idx.cooc.nnz}")
        if cache is not None:
            import json

            save_npz(cache / f"fold_{k}_cooc.npz", idx.cooc)
            np.save(cache / f"fold_{k}_pop.npy", idx.popularity)
            np.save(cache / f"fold_{k}_n_users.npy", np.array([idx.n_users], dtype=np.int64))
            (cache / "user_folds.json").write_text(
                json.dumps({str(u): f for u, f in user_to_fold.items()}), encoding="utf-8"
            )

    return CrossFitHCRBundle(
        full_index=full_index,
        fold_indices=fold_indices,
        user_to_fold=user_to_fold,
        n_folds=n_folds,
    )


def bundle_meta(cf: CrossFitHCRBundle) -> dict[str, Any]:
    return {
        "n_folds": cf.n_folds,
        "n_users_mapped": len(cf.user_to_fold),
        "full_cooc_nnz": int(cf.full_index.cooc.nnz),
        "fold_cooc_nnz": [int(idx.cooc.nnz) for idx in cf.fold_indices],
    }
