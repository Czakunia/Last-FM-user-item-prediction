"""Sparse n_jki backends for orthonormal H3 (LASTFM_FULL_SOTA_V2).

Backends (prototype selection by wall-time + RAM on a sample):
  A sorted_intersect — sorted user-id postings + merge intersect
  B bitset_popcount — dense bitsets + AND popcount (+ optional fold mask)
  C sparse_local — intersect only over users of the rarest of {j,k,i}
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import numpy as np

from src.lastfm_lp.binary.hcr_triple import a111_from_counts
from src.lastfm_lp.data.load_kgat_lastfm import item_popularity

BackendName = Literal["sorted_intersect", "bitset_popcount", "sparse_local"]


def _popcount_u64(words: np.ndarray) -> int:
    """Population count for a 1-D uint64 word array (NumPy 1.x / 2.x)."""

    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(words).sum())
    total = 0
    for x in words.tolist():
        total += int(x).bit_count()
    return total


def build_item_user_postings(
    train_by_user: dict[int, set[int]],
    n_items: int,
    *,
    max_history_for_pairs: int = 60,
    seed: int = 2026,
) -> tuple[list[np.ndarray], np.ndarray, dict[int, int], int]:
    """Return (postings[item]=sorted user remap ids, popularity, uid→dense, n_users).

    Truncation matches ``build_cooccurrence`` so n_jk ≈ |postings[j] ∩ postings[k]|.
    """

    rng = np.random.default_rng(seed)
    users = sorted(int(u) for u in train_by_user.keys())
    uid_to_dense = {u: i for i, u in enumerate(users)}
    n_users = len(users)
    buckets: list[list[int]] = [[] for _ in range(n_items)]
    for u in users:
        items = train_by_user[u]
        arr = np.fromiter(items, dtype=np.int32)
        if arr.size == 0:
            continue
        if arr.size > max_history_for_pairs:
            arr = rng.choice(arr, size=max_history_for_pairs, replace=False)
        d = uid_to_dense[u]
        for it in arr:
            ii = int(it)
            if 0 <= ii < n_items:
                buckets[ii].append(d)
    postings: list[np.ndarray] = []
    for b in buckets:
        if b:
            postings.append(np.unique(np.asarray(b, dtype=np.int32)))
        else:
            postings.append(np.zeros(0, dtype=np.int32))
    pop = item_popularity(train_by_user, n_items)
    return postings, pop, uid_to_dense, n_users


def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.int32)
    return np.intersect1d(a, b, assume_unique=True)


def _n_intersect3_sorted(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    return int(_intersect_sorted(_intersect_sorted(a, b), c).size)


class TripleStatsIndex:
    """n_jki + a111 lookup with LRU cache keyed by (min(j,k), max(j,k), i)."""

    def __init__(
        self,
        postings: list[np.ndarray],
        popularity: np.ndarray,
        n_users: int,
        *,
        backend: BackendName = "bitset_popcount",
        cache_size: int = 250_000,
        bitsets: np.ndarray | None = None,
        words: int | None = None,
    ) -> None:
        self.postings = postings
        self.popularity = popularity.astype(np.int32)
        self.n_users = int(n_users)
        self.backend: BackendName = backend
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[tuple[int, int, int], int] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_items = len(postings)
        self._bitsets = bitsets
        self._words = words
        if backend == "bitset_popcount" and self._bitsets is None:
            self._bitsets, self._words = self._build_bitsets()

    def _build_bitsets(self) -> tuple[np.ndarray, int]:
        words = (self.n_users + 63) // 64
        bits = np.zeros((self.n_items, words), dtype=np.uint64)
        for i, posts in enumerate(self.postings):
            if posts.size == 0:
                continue
            w = posts >> 6
            b = posts & 63
            # scatter OR
            for ww, bb in zip(w, b):
                bits[i, int(ww)] |= np.uint64(1) << np.uint64(int(bb))
        return bits, words

    def _cache_get(self, j: int, k: int, i: int) -> int | None:
        a, b = (j, k) if j <= k else (k, j)
        key = (a, b, int(i))
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.cache_misses += 1
        return None

    def _cache_put(self, j: int, k: int, i: int, val: int) -> None:
        a, b = (j, k) if j <= k else (k, j)
        key = (a, b, int(i))
        self._cache[key] = int(val)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def n_jki(self, j: int, k: int, i: int) -> int:
        j, k, i = int(j), int(k), int(i)
        if j == k or j == i or k == i:
            return 0
        cached = self._cache_get(j, k, i)
        if cached is not None:
            return cached
        if self.backend == "sorted_intersect":
            val = _n_intersect3_sorted(self.postings[j], self.postings[k], self.postings[i])
        elif self.backend == "bitset_popcount":
            assert self._bitsets is not None
            acc = self._bitsets[j] & self._bitsets[k] & self._bitsets[i]
            val = _popcount_u64(acc)
        elif self.backend == "sparse_local":
            pj, pk, pi = self.postings[j], self.postings[k], self.postings[i]
            lens = [(pj.size, pj), (pk.size, pk), (pi.size, pi)]
            lens.sort(key=lambda t: t[0])
            base = lens[0][1]
            o1, o2 = lens[1][1], lens[2][1]
            count = 0
            for u in base:
                if o1.size and o2.size:
                    i1 = np.searchsorted(o1, u)
                    i2 = np.searchsorted(o2, u)
                    if i1 < o1.size and o1[i1] == u and i2 < o2.size and o2[i2] == u:
                        count += 1
            val = count
        else:
            raise ValueError(self.backend)
        self._cache_put(j, k, i, val)
        return val

    def n_jki_block(
        self,
        pairs_jk: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        """Return (M, C) n_jki for history pairs × candidates (no row×j×k Python)."""

        pairs = np.asarray(pairs_jk, dtype=np.int64).reshape(-1, 2)
        cands = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        M, C = int(pairs.shape[0]), int(cands.size)
        out = np.zeros((M, C), dtype=np.int32)
        if M == 0 or C == 0:
            return out

        if self.backend == "bitset_popcount" and self._bitsets is not None:
            for m in range(M):
                j, k = int(pairs[m, 0]), int(pairs[m, 1])
                if j == k:
                    continue
                jk = self._bitsets[j] & self._bitsets[k]
                for c, i in enumerate(cands):
                    ii = int(i)
                    if ii == j or ii == k:
                        continue
                    cached = self._cache_get(j, k, ii)
                    if cached is not None:
                        out[m, c] = cached
                        continue
                    acc = jk & self._bitsets[ii]
                    val = _popcount_u64(acc)
                    self._cache_put(j, k, ii, val)
                    out[m, c] = val
            return out

        # sorted / sparse: precompute jk posting once per pair
        for m in range(M):
            j, k = int(pairs[m, 0]), int(pairs[m, 1])
            if j == k:
                continue
            jk = _intersect_sorted(self.postings[j], self.postings[k])
            for c, i in enumerate(cands):
                ii = int(i)
                if ii == j or ii == k:
                    continue
                cached = self._cache_get(j, k, ii)
                if cached is not None:
                    out[m, c] = cached
                    continue
                if self.backend == "sparse_local" and jk.size > self.postings[ii].size:
                    val = int(_intersect_sorted(self.postings[ii], jk).size)
                else:
                    val = int(_intersect_sorted(jk, self.postings[ii]).size)
                self._cache_put(j, k, ii, val)
                out[m, c] = val
        return out

    def a111_energy_block(
        self,
        pairs_jk: np.ndarray,
        candidate_ids: np.ndarray,
        *,
        cooc_n11_fn=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(M,C) a111 and energy. Pairwise n11 from cooc_n11_fn(j,i) or |∩| postings."""

        pairs = np.asarray(pairs_jk, dtype=np.int64).reshape(-1, 2)
        cands = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        M, C = int(pairs.shape[0]), int(cands.size)
        a = np.zeros((M, C), dtype=np.float64)
        e = np.zeros((M, C), dtype=np.float64)
        if M == 0 or C == 0:
            return a, e
        n_jki = self.n_jki_block(pairs, cands)
        N = self.n_users
        for m in range(M):
            j, k = int(pairs[m, 0]), int(pairs[m, 1])
            n_j = int(self.popularity[j])
            n_k = int(self.popularity[k])
            if cooc_n11_fn is not None:
                n_jk = int(cooc_n11_fn(j, k))
            else:
                n_jk = int(_intersect_sorted(self.postings[j], self.postings[k]).size)
            for c in range(C):
                i = int(cands[c])
                n_i = int(self.popularity[i])
                if cooc_n11_fn is not None:
                    n_ji = int(cooc_n11_fn(j, i))
                    n_ki = int(cooc_n11_fn(k, i))
                else:
                    n_ji = int(_intersect_sorted(self.postings[j], self.postings[i]).size)
                    n_ki = int(_intersect_sorted(self.postings[k], self.postings[i]).size)
                aa, ee = a111_from_counts(
                    int(n_jki[m, c]), n_jk, n_ji, n_ki, n_j, n_k, n_i, N
                )
                a[m, c] = aa
                e[m, c] = ee
        return a, e

    def cache_stats(self) -> dict[str, float | int]:
        total = self.cache_hits + self.cache_misses
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": float(self.cache_hits / total) if total else 0.0,
            "size": len(self._cache),
            "backend": self.backend,
        }


def build_triple_index(
    train_by_user: dict[int, set[int]],
    n_items: int,
    *,
    backend: BackendName = "bitset_popcount",
    max_history_for_pairs: int = 60,
    seed: int = 2026,
    cache_size: int = 250_000,
) -> TripleStatsIndex:
    postings, pop, _, n_users = build_item_user_postings(
        train_by_user,
        n_items,
        max_history_for_pairs=max_history_for_pairs,
        seed=seed,
    )
    return TripleStatsIndex(
        postings,
        pop,
        n_users,
        backend=backend,
        cache_size=cache_size,
    )
