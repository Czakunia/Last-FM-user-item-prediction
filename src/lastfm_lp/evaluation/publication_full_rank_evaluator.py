"""Publication all-ranking evaluator — single source of truth for SOTA track.

Protocol LASTFM_PUBLICATION_PROTOCOL_V1:
  candidates(u) = {0..n_items-1} \\ train_history(u)
  score all candidates, rank descending
  tie-break: higher score first; if equal, lower item_id first
  metrics: Recall@K, NDCG@K (K in {5,10,20}), optional MRR
  mean over users with ≥1 test positive (KGAT-faithful: masked train∩test
  positives stay in the denominator and are unrankable)

Models only supply score(u, item_ids) → scores. Never implement their own metrics.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

ScoreFn = Callable[[int, np.ndarray], np.ndarray]


def tie_break_key(score: float, item_id: int) -> tuple[float, int]:
    """Sort key: higher score better; lower item_id wins ties."""
    return (-float(score), int(item_id))


def dense_topk(
    scores: np.ndarray,
    *,
    k: int,
    mask_items: Iterable[int] | None = None,
) -> np.ndarray:
    """Return top-k item ids under publication tie-break (exact dense)."""

    s = np.asarray(scores, dtype=np.float64).copy()
    if mask_items:
        for i in mask_items:
            ii = int(i)
            if 0 <= ii < s.size:
                s[ii] = -np.inf
    # lexsort: primary -score, secondary item_id (ascending)
    item_ids = np.arange(s.size, dtype=np.int64)
    order = np.lexsort((item_ids, -s))
    return order[:k]


def chunked_topk(
    *,
    user_id: int,
    n_items: int,
    score_fn: ScoreFn,
    k: int,
    chunk_size: int,
    mask_items: set[int] | None,
) -> np.ndarray:
    """Merge chunk scores into exact top-k with same tie-break as dense_topk."""

    # min-heap of (-key) equivalent: store ( -score_for_heap? )
    # We keep heap of size k with worst at top: entries are (sort_key_tuple, item)
    # Python heapq is min-heap; we push ( -score, item_id ) wait:
    # want pop smallest "quality". Quality key for min-heap of keepers:
    # use (score, -item_id) as heap key where SMALLER = worse
    # worse = lower score, or same score and higher item_id
    # heap entry: (score, -item_id, item_id) — min is worst among kept
    heap: list[tuple[float, int, int]] = []

    mask = mask_items or set()
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        chunk = np.arange(start, end, dtype=np.int64)
        scores = np.asarray(score_fn(user_id, chunk), dtype=np.float64).reshape(-1)
        if scores.shape[0] != chunk.shape[0]:
            raise ValueError("score_fn must return one score per item id")
        for item, sc in zip(chunk.tolist(), scores.tolist()):
            if item in mask:
                continue
            entry = (float(sc), -int(item), int(item))
            if len(heap) < k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    if not heap:
        return np.zeros(0, dtype=np.int64)
    # sort best-first: reverse of heap order
    ranked = sorted(heap, key=lambda t: (t[0], t[1]), reverse=True)
    return np.asarray([t[2] for t in ranked[:k]], dtype=np.int64)


def metrics_from_topk(
    top_items: np.ndarray,
    positive_items: set[int],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    full_rank_order_for_mrr: np.ndarray | None = None,
) -> dict[str, float]:
    """Recall/NDCG/HitRate@K from a ranked list; positives may be unrankable (still in denom)."""

    pos = set(int(x) for x in positive_items)
    if not pos:
        nan = float("nan")
        out = {f"Recall@{k}": nan for k in ks}
        out.update({f"NDCG@{k}": nan for k in ks})
        out.update({f"HitRate@{k}": nan for k in ks})
        out["MRR"] = nan
        return out

    out: dict[str, float] = {}
    top_list = [int(x) for x in np.asarray(top_items).tolist()]
    for k in ks:
        topk = top_list[:k]
        hits = sum(1 for i in topk if i in pos)
        out[f"Recall@{k}"] = float(hits / len(pos))
        out[f"HitRate@{k}"] = 1.0 if hits > 0 else 0.0
        dcg = 0.0
        for rank, item in enumerate(topk, start=1):
            if item in pos:
                dcg += 1.0 / np.log2(rank + 1)
        ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(k, len(pos)) + 1))
        out[f"NDCG@{k}"] = float(dcg / ideal) if ideal > 0 else 0.0

    # MRR: first positive in full order if provided, else in top_list (may miss if K small)
    order = (
        [int(x) for x in np.asarray(full_rank_order_for_mrr).tolist()]
        if full_rank_order_for_mrr is not None
        else top_list
    )
    mrr = 0.0
    for rank, item in enumerate(order, start=1):
        if item in pos:
            mrr = 1.0 / rank
            break
    out["MRR"] = float(mrr)
    return out


def evaluate_user_dense(
    scores: np.ndarray,
    *,
    positive_items: set[int],
    train_items: set[int],
    ks: tuple[int, ...] = (5, 10, 20),
    max_k: int | None = None,
) -> dict[str, float]:
    k_max = max(ks) if max_k is None else max(max_k, max(ks))
    top = dense_topk(scores, k=k_max, mask_items=train_items)
    # Full order for MRR among eligible candidates only (train-masked items excluded),
    # matching chunked_topk which skips train items rather than ranking them at -inf.
    s = np.asarray(scores, dtype=np.float64).copy()
    train = {int(i) for i in train_items}
    for i in train:
        if 0 <= i < s.size:
            s[i] = -np.inf
    item_ids = np.arange(s.size, dtype=np.int64)
    full_order = np.lexsort((item_ids, -s))
    eligible = np.asarray([int(i) for i in full_order if int(i) not in train], dtype=np.int64)
    return metrics_from_topk(top, positive_items, ks=ks, full_rank_order_for_mrr=eligible)


def evaluate_user_chunked(
    *,
    user_id: int,
    n_items: int,
    score_fn: ScoreFn,
    positive_items: set[int],
    train_items: set[int],
    ks: tuple[int, ...] = (5, 10, 20),
    chunk_size: int = 2048,
    keep_k: int = 100,
) -> dict[str, float]:
    k_max = max(max(ks), keep_k)
    top = chunked_topk(
        user_id=user_id,
        n_items=n_items,
        score_fn=score_fn,
        k=k_max,
        chunk_size=chunk_size,
        mask_items=train_items,
    )
    # MRR needs full order — for publication primary we use Recall/NDCG@20 from topk;
    # optional MRR approximated from keep_k (document limitation) unless keep_k == n_items
    return metrics_from_topk(top, positive_items, ks=ks, full_rank_order_for_mrr=top)


@dataclass
class PublicationEvalResult:
    metrics_mean: dict[str, float]
    n_users: int
    per_user: dict[int, dict[str, float]] | None = None


def evaluate_publication_full_rank(
    *,
    eval_users: list[int],
    test_positives: dict[int, set[int]],
    model_train: dict[int, set[int]],
    n_items: int,
    score_fn: ScoreFn,
    chunk_size: int = 2048,
    ks: tuple[int, ...] = (5, 10, 20),
    keep_k: int = 100,
    store_per_user: bool = False,
    mode: str = "chunked",  # "chunked" | "dense"
) -> PublicationEvalResult:
    """All-ranking evaluation. No sampled negatives. No positive capping."""

    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({f"HitRate@{k}": 0.0 for k in ks})
    sums["MRR"] = 0.0
    per_user: dict[int, dict[str, float]] = {}
    n = 0
    for u in eval_users:
        pos = set(int(x) for x in test_positives.get(u, ()))
        if not pos:
            continue
        train = set(int(x) for x in model_train.get(u, ()))
        if mode == "dense":
            scores = np.asarray(score_fn(u, np.arange(n_items, dtype=np.int64)), dtype=np.float64)
            m = evaluate_user_dense(scores, positive_items=pos, train_items=train, ks=ks)
        else:
            m = evaluate_user_chunked(
                user_id=u,
                n_items=n_items,
                score_fn=score_fn,
                positive_items=pos,
                train_items=train,
                ks=ks,
                chunk_size=chunk_size,
                keep_k=keep_k,
            )
        for k, v in m.items():
            if k not in sums:
                sums[k] = 0.0
            sums[k] += float(v)
        if store_per_user:
            per_user[int(u)] = m
        n += 1
    if n == 0:
        return PublicationEvalResult({k: float("nan") for k in sums}, 0, per_user or None)
    mean = {k: float(v / n) for k, v in sums.items()}
    mean["n_users"] = float(n)
    return PublicationEvalResult(mean, n, per_user if store_per_user else None)


def exactness_chunked_vs_dense(
    *,
    user_ids: list[int],
    n_items: int,
    score_fn: ScoreFn,
    model_train: dict[int, set[int]],
    test_positives: dict[int, set[int]] | None = None,
    k: int = 20,
    chunk_size: int = 2048,
    metric_atol: float = 1e-12,
) -> dict[str, object]:
    """Require identical top-k item sequences (and metrics if test_positives given)."""

    mismatches = []
    metric_mismatches = []
    ks = tuple(sorted({5, 10, 20, int(k)}))
    for u in user_ids:
        train = set(int(x) for x in model_train.get(u, ()))
        dense_scores = np.asarray(score_fn(u, np.arange(n_items, dtype=np.int64)), dtype=np.float64)
        top_d = dense_topk(dense_scores, k=k, mask_items=train)
        top_c = chunked_topk(
            user_id=u,
            n_items=n_items,
            score_fn=score_fn,
            k=k,
            chunk_size=chunk_size,
            mask_items=train,
        )
        if not np.array_equal(top_d, top_c):
            mismatches.append(
                {
                    "user": int(u),
                    "dense": top_d.tolist(),
                    "chunked": top_c.tolist(),
                }
            )
        if test_positives is not None:
            pos = set(int(x) for x in test_positives.get(u, ()))
            if not pos:
                continue
            md = metrics_from_topk(top_d, pos, ks=ks, full_rank_order_for_mrr=top_d)
            mc = metrics_from_topk(top_c, pos, ks=ks, full_rank_order_for_mrr=top_c)
            for key in md:
                if abs(float(md[key]) - float(mc[key])) > metric_atol:
                    metric_mismatches.append(
                        {
                            "user": int(u),
                            "metric": key,
                            "dense": float(md[key]),
                            "chunked": float(mc[key]),
                        }
                    )
                    break
    return {
        "n_users": len(user_ids),
        "k": k,
        "chunk_size": chunk_size,
        "n_mismatches": len(mismatches),
        "n_metric_mismatches": len(metric_mismatches),
        "passed": len(mismatches) == 0 and len(metric_mismatches) == 0,
        "mismatches_head": mismatches[:5],
        "metric_mismatches_head": metric_mismatches[:5],
    }
