"""H3_FAST_USER_BATCH: history-pair × candidate blocks for orthonormal a111.

Unit of work = one user:
  resolve cross-fit once → history pairs once → S_jk once per pair →
  n_jki over all candidates → vectorized a111 → pool → write all rows.

Backends for n_jki[P×C] (batch, not single-triple):
  A sorted_batch — S_jk via sorted ∩; count via user→items scan over S_jk
  B sparse_local_batch — same S_jk; count via inverted postings ∩ S_jk per cand
  C bitset_batch — S_jk as bitset AND; candidate counts via bitset ∩ postings
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.lastfm_lp.binary.contingency_tables import get_n11
from src.lastfm_lp.binary.cross_fit import CrossFitHCRBundle
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.hcr_triple import (
    H3_RAW_FEATURE_ORDER,
    a111_energy_from_n_jki_matrix,
    history_unordered_pairs,
    pool_h3_raw_matrix,
)
from src.lastfm_lp.binary.user_hcr_aggregation import _truncate_history
from src.lastfm_lp.data.load_kgat_lastfm import item_popularity

BatchBackend = Literal["sorted_batch", "sparse_local_batch", "bitset_batch"]


def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.int32)
    return np.intersect1d(a, b, assume_unique=True)


def _popcount_u64(words: np.ndarray) -> int:
    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(words).sum())
    return int(sum(int(x).bit_count() for x in words.tolist()))


@dataclass
class SjkCacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return float(self.hits / t) if t else 0.0


class BoundedSjkCache:
    """LRU cache for S_jk keyed by (min(j,k), max(j,k))."""

    def __init__(self, max_size: int = 50_000) -> None:
        self.max_size = int(max_size)
        self._d: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.stats = SjkCacheStats()

    def get(self, j: int, k: int) -> np.ndarray | None:
        a, b = (j, k) if j <= k else (k, j)
        key = (a, b)
        if key in self._d:
            self.stats.hits += 1
            self._d.move_to_end(key)
            return self._d[key]
        self.stats.misses += 1
        return None

    def put(self, j: int, k: int, users: np.ndarray) -> None:
        a, b = (j, k) if j <= k else (k, j)
        key = (a, b)
        self._d[key] = users
        self._d.move_to_end(key)
        while len(self._d) > self.max_size:
            self._d.popitem(last=False)


@dataclass
class TripleUserBatchIndex:
    """Fit-population inverted index for batched n_jki."""

    postings: list[np.ndarray]
    user_items: list[np.ndarray]
    popularity: np.ndarray
    n_users: int
    n_items: int
    backend: BatchBackend = "sorted_batch"
    sjk_cache: BoundedSjkCache = field(default_factory=lambda: BoundedSjkCache(50_000))
    bitsets: np.ndarray | None = None
    words: int = 0
    # profiling counters (seconds accumulated by caller optionally)
    n_sjk_builds: int = 0
    n_block_calls: int = 0

    def s_jk(self, j: int, k: int) -> np.ndarray:
        j, k = int(j), int(k)
        if j == k:
            return np.zeros(0, dtype=np.int32)
        cached = self.sjk_cache.get(j, k)
        if cached is not None:
            return cached
        self.n_sjk_builds += 1
        if self.backend == "bitset_batch" and self.bitsets is not None:
            users = self._users_from_bitset_and(j, k)
        else:
            users = _intersect_sorted(self.postings[j], self.postings[k])
        self.sjk_cache.put(j, k, users)
        return users

    def _users_from_bitset_and(self, j: int, k: int) -> np.ndarray:
        assert self.bitsets is not None
        acc = self.bitsets[j] & self.bitsets[k]
        out: list[int] = []
        for w_idx, word in enumerate(acc.tolist()):
            w = int(word)
            while w:
                b = (w & -w).bit_length() - 1
                out.append(w_idx * 64 + b)
                w &= w - 1
        if not out:
            return np.zeros(0, dtype=np.int32)
        arr = np.asarray(out, dtype=np.int32)
        return arr[arr < self.n_users]

    def n_jki_row_for_pair(self, j: int, k: int, cands: np.ndarray) -> np.ndarray:
        """Return length-C n_jki for one history pair vs many candidates."""

        cands = np.asarray(cands, dtype=np.int64).reshape(-1)
        C = int(cands.size)
        out = np.zeros(C, dtype=np.int32)
        if C == 0 or j == k:
            return out
        s = self.s_jk(j, k)
        if s.size == 0:
            return out

        if self.backend == "sparse_local_batch":
            # For each candidate: |S_jk ∩ postings[i]|
            for c, i in enumerate(cands.tolist()):
                ii = int(i)
                if ii == j or ii == k or ii < 0 or ii >= self.n_items:
                    continue
                out[c] = int(_intersect_sorted(s, self.postings[ii]).size)
            return out

        if self.backend == "bitset_batch" and self.bitsets is not None:
            # S_jk bitset already from j&k
            sjk_bits = self.bitsets[j] & self.bitsets[k]
            for c, i in enumerate(cands.tolist()):
                ii = int(i)
                if ii == j or ii == k or ii < 0 or ii >= self.n_items:
                    continue
                out[c] = _popcount_u64(sjk_bits & self.bitsets[ii])
            return out

        # sorted_batch (default): scan users in S_jk, increment candidate hits
        cand_pos = {int(i): t for t, i in enumerate(cands.tolist())}
        for u in s.tolist():
            items = self.user_items[int(u)]
            if items.size == 0:
                continue
            for it in items.tolist():
                t = cand_pos.get(int(it))
                if t is not None:
                    out[t] += 1
        # j/k themselves should not count as candidate self (already 0 if j/k not in cands
        # or forced below)
        for c, i in enumerate(cands.tolist()):
            if int(i) == j or int(i) == k:
                out[c] = 0
        return out

    def n_jki_block(self, pairs: np.ndarray, cands: np.ndarray) -> np.ndarray:
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        cands = np.asarray(cands, dtype=np.int64).reshape(-1)
        P, C = int(pairs.shape[0]), int(cands.size)
        out = np.zeros((P, C), dtype=np.int32)
        self.n_block_calls += 1
        for p in range(P):
            out[p] = self.n_jki_row_for_pair(int(pairs[p, 0]), int(pairs[p, 1]), cands)
        return out

    def n_jki_scalar(self, j: int, k: int, i: int) -> int:
        if j == k or j == i or k == i:
            return 0
        s = self.s_jk(j, k)
        return int(_intersect_sorted(s, self.postings[int(i)]).size)


def build_triple_user_batch_index(
    train_by_user: dict[int, set[int]],
    n_items: int,
    *,
    backend: BatchBackend = "sorted_batch",
    max_history_for_pairs: int = 60,
    seed: int = 2026,
    sjk_cache_size: int = 50_000,
) -> TripleUserBatchIndex:
    rng = np.random.default_rng(seed)
    users = sorted(int(u) for u in train_by_user.keys())
    uid_to_dense = {u: i for i, u in enumerate(users)}
    n_users = len(users)
    buckets: list[list[int]] = [[] for _ in range(n_items)]
    user_items: list[np.ndarray] = [np.zeros(0, dtype=np.int32) for _ in range(n_users)]
    for u in users:
        items = train_by_user[u]
        arr = np.fromiter(items, dtype=np.int32)
        if arr.size == 0:
            continue
        if arr.size > max_history_for_pairs:
            arr = rng.choice(arr, size=max_history_for_pairs, replace=False)
        arr = np.unique(arr)
        d = uid_to_dense[u]
        user_items[d] = arr.astype(np.int32, copy=False)
        for it in arr.tolist():
            ii = int(it)
            if 0 <= ii < n_items:
                buckets[ii].append(d)
    postings = [
        np.unique(np.asarray(b, dtype=np.int32)) if b else np.zeros(0, dtype=np.int32)
        for b in buckets
    ]
    pop = item_popularity(train_by_user, n_items)
    bitsets = None
    words = 0
    if backend == "bitset_batch":
        words = (n_users + 63) // 64
        bitsets = np.zeros((n_items, words), dtype=np.uint64)
        for i, posts in enumerate(postings):
            for u in posts.tolist():
                bitsets[i, u >> 6] |= np.uint64(1) << np.uint64(u & 63)
    return TripleUserBatchIndex(
        postings=postings,
        user_items=user_items,
        popularity=pop.astype(np.int32),
        n_users=n_users,
        n_items=n_items,
        backend=backend,
        sjk_cache=BoundedSjkCache(sjk_cache_size),
        bitsets=bitsets,
        words=words,
    )


@dataclass
class CrossFitTripleUserBatch:
    full: TripleUserBatchIndex
    folds: list[TripleUserBatchIndex]
    cf: CrossFitHCRBundle

    def index_for(self, user_id: int, split: str) -> TripleUserBatchIndex:
        if split in {"val", "valid", "validation", "test"}:
            return self.full
        return self.folds[int(self.cf.user_to_fold[int(user_id)])]

    def expected_index_key(self, user_id: int, split: str) -> int:
        if split in {"val", "valid", "validation", "test"}:
            return -1
        return int(self.cf.user_to_fold[int(user_id)])


def build_cross_fit_triple_user_batch(
    bundle: dict[str, Any],
    *,
    backend: BatchBackend = "sorted_batch",
    max_history_for_pairs: int = 60,
    sjk_cache_size: int = 50_000,
) -> CrossFitTripleUserBatch:
    from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit

    cfg = bundle["cfg"]
    model_train = bundle["model_train"]
    n_items = int(len(bundle["popularity"]))
    seed = int(cfg.get("hcr", {}).get("cross_fitting", {}).get("seed", cfg["split"]["seed"]))
    cf = ensure_cross_fit(bundle)
    full = build_triple_user_batch_index(
        model_train,
        n_items,
        backend=backend,
        max_history_for_pairs=max_history_for_pairs,
        seed=seed,
        sjk_cache_size=sjk_cache_size,
    )
    folds: list[TripleUserBatchIndex] = []
    for k in range(cf.n_folds):
        keep = {u for u, f in cf.user_to_fold.items() if f != k}
        sub = {u: model_train[u] for u in keep if u in model_train}
        folds.append(
            build_triple_user_batch_index(
                sub,
                n_items,
                backend=backend,
                max_history_for_pairs=max_history_for_pairs,
                seed=seed,
                sjk_cache_size=sjk_cache_size,
            )
        )
    return CrossFitTripleUserBatch(full=full, folds=folds, cf=cf)


def _filter_pairs_exclude_item(pairs: np.ndarray, exclude: int) -> np.ndarray:
    if pairs.size == 0:
        return pairs
    mask = (pairs[:, 0] != exclude) & (pairs[:, 1] != exclude)
    return pairs[mask]


def compute_h3_user_block(
    *,
    user_id: int,
    history_items: set[int] | list[int],
    candidate_items: np.ndarray,
    labels: np.ndarray | None,
    split: str,
    triples: CrossFitTripleUserBatch,
    pairwise: PairwiseStatsIndex,
    max_history: int = 25,
    return_raw: bool = False,
) -> dict[str, Any]:
    """Compute H3_RAW (C,8) for one user × all candidates.

    Train self-exclusion: for y=1 and candidate ∈ history, drop pairs containing i.
    Val/test: shared history for all candidates (no self-exclusion).
    """

    cands = np.asarray(candidate_items, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    t_idx = triples.index_for(user_id, split)
    # assertion helper: all rows share this index
    index_key = triples.expected_index_key(user_id, split)
    base = set(int(x) for x in history_items)
    empty = {
        "features": np.zeros((C, 8), dtype=np.float32),
        "index_key": index_key,
        "n_pairs_used": np.zeros(C, dtype=np.int32),
        "min_a111": np.zeros(C, dtype=np.float64),
        "max_abs_a111": np.zeros(C, dtype=np.float64),
    }
    if C == 0:
        return empty

    # Fast path: val/test or all rows share same history (no train positives in hist)
    need_per_cand = False
    if labels is not None and split == "train":
        y = np.asarray(labels).reshape(-1)
        for c in range(C):
            if int(y[c]) == 1 and int(cands[c]) in base:
                need_per_cand = True
                break

    feats = np.zeros((C, 8), dtype=np.float32)
    n_pairs_used = np.zeros(C, dtype=np.int32)
    min_a = np.zeros(C, dtype=np.float64)
    max_abs = np.zeros(C, dtype=np.float64)
    raw_out = None

    if not need_per_cand:
        pairs = history_unordered_pairs(base, max_history=max_history)
        if pairs.shape[0] == 0:
            return empty
        a, e, audit = _a111_block_from_pairs(pairs, cands, t_idx, pairwise)
        for c in range(C):
            feats[c] = pool_h3_raw_matrix(a[:, c], e[:, c])[0]
            n_pairs_used[c] = pairs.shape[0]
            min_a[c] = float(a[:, c].min()) if a.shape[0] else 0.0
            max_abs[c] = float(np.abs(a[:, c]).max()) if a.shape[0] else 0.0
        if return_raw:
            raw_out = {"a111": a, "energy": e, "pairs": pairs, "n_jki": audit["n_jki"]}
    else:
        # Group: normal candidates share base; removal candidates unique hist pairs
        y = np.asarray(labels).reshape(-1)
        normal = [c for c in range(C) if not (int(y[c]) == 1 and int(cands[c]) in base)]
        removal = [c for c in range(C) if int(y[c]) == 1 and int(cands[c]) in base]
        if normal:
            pairs = history_unordered_pairs(base, max_history=max_history)
            if pairs.shape[0]:
                sub = cands[np.asarray(normal, dtype=np.int64)]
                a, e, _ = _a111_block_from_pairs(pairs, sub, t_idx, pairwise)
                for local, c in enumerate(normal):
                    feats[c] = pool_h3_raw_matrix(a[:, local], e[:, local])[0]
                    n_pairs_used[c] = pairs.shape[0]
                    min_a[c] = float(a[:, local].min())
                    max_abs[c] = float(np.abs(a[:, local]).max())
        for c in removal:
            i = int(cands[c])
            pairs = history_unordered_pairs(base - {i}, max_history=max_history)
            # also drop any pair that somehow still contains i
            pairs = _filter_pairs_exclude_item(pairs, i)
            if pairs.shape[0] == 0:
                continue
            a, e, _ = _a111_block_from_pairs(pairs, np.asarray([i], dtype=np.int64), t_idx, pairwise)
            feats[c] = pool_h3_raw_matrix(a[:, 0], e[:, 0])[0]
            n_pairs_used[c] = pairs.shape[0]
            min_a[c] = float(a[:, 0].min())
            max_abs[c] = float(np.abs(a[:, 0]).max())

    out = {
        "features": feats,
        "index_key": index_key,
        "n_pairs_used": n_pairs_used,
        "min_a111": min_a,
        "max_abs_a111": max_abs,
        "feature_names": H3_RAW_FEATURE_ORDER,
        "wmean_equals_mean": True,
    }
    if raw_out is not None:
        out["raw"] = raw_out
    return out


def _a111_block_from_pairs(
    pairs: np.ndarray,
    cands: np.ndarray,
    t_idx: TripleUserBatchIndex,
    pairwise: PairwiseStatsIndex,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    P, C = int(pairs.shape[0]), int(cands.size)
    n_jki = t_idx.n_jki_block(pairs, cands)
    # pairwise from cooc index (reuse H2 tables)
    hist_ids = np.unique(pairs.reshape(-1))
    pos = {int(h): t for t, h in enumerate(hist_ids.tolist())}
    n11_hc = pairwise.cooccurrence_block(hist_ids, cands)
    n11_hh = pairwise.cooccurrence_block(hist_ids, hist_ids)
    n_jk = np.array(
        [int(n11_hh[pos[int(pairs[m, 0])], pos[int(pairs[m, 1])]]) for m in range(P)],
        dtype=np.int64,
    ).reshape(P, 1)
    n_ji = np.zeros((P, C), dtype=np.int64)
    n_ki = np.zeros((P, C), dtype=np.int64)
    for m in range(P):
        n_ji[m] = n11_hc[pos[int(pairs[m, 0])]]
        n_ki[m] = n11_hc[pos[int(pairs[m, 1])]]
    # popularity from pairwise index (fit population) for consistency with H2
    n_j = pairwise.popularity[pairs[:, 0]].astype(np.int64).reshape(P, 1)
    n_k = pairwise.popularity[pairs[:, 1]].astype(np.int64).reshape(P, 1)
    n_i = pairwise.popularity[cands].astype(np.int64).reshape(1, C)
    a, e = a111_energy_from_n_jki_matrix(
        n_jki, n_jk, n_ji, n_ki, n_j, n_k, n_i, pairwise.n_users
    )
    return a, e, {"n_jki": n_jki, "n_jk": n_jk, "n_ji": n_ji, "n_ki": n_ki}


def a111_scalar_reference(
    j: int,
    k: int,
    i: int,
    t_idx: TripleUserBatchIndex,
    pairwise: PairwiseStatsIndex,
) -> float:
    """Scalar a111 using same counts convention as batch."""

    from src.lastfm_lp.binary.hcr_triple import a111_from_counts

    n_jki = t_idx.n_jki_scalar(j, k, i)
    n_jk = get_n11(pairwise.cooc, j, k)
    n_ji = get_n11(pairwise.cooc, j, i)
    n_ki = get_n11(pairwise.cooc, k, i)
    a, _ = a111_from_counts(
        n_jki,
        n_jk,
        n_ji,
        n_ki,
        int(pairwise.popularity[j]),
        int(pairwise.popularity[k]),
        int(pairwise.popularity[i]),
        pairwise.n_users,
    )
    return a
