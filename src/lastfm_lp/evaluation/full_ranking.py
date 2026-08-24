"""Chunked full-catalog ranking (FULL_SOTA final test)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def full_rank_user_metrics(
    *,
    scores: np.ndarray,
    positive_items: np.ndarray,
    train_items: set[int] | None,
    ks: tuple[int, ...] = (5, 10, 20),
    exclude_train: bool = True,
) -> dict[str, float]:
    """scores: (n_items,) logits/scores for all item ids 0..n_items-1."""
    s = scores.astype(np.float64).copy()
    if exclude_train and train_items:
        for i in train_items:
            if 0 <= i < len(s):
                s[i] = -np.inf
    # also mask positives temporarily for ranking among candidates? No — keep pos,
    # exclude only train. Positives must be scorable.
    order = np.argsort(-s, kind="stable")
    pos_set = set(int(x) for x in positive_items)
    # relevance vector in ranked order
    # NDCG/Recall via ranks of positives
    out: dict[str, float] = {}
    for k in ks:
        top = order[:k]
        hits = sum(1 for i in top if int(i) in pos_set)
        out[f"Recall@{k}"] = float(hits / max(len(pos_set), 1))
        # DCG
        dcg = 0.0
        for rank, item in enumerate(top, start=1):
            if int(item) in pos_set:
                dcg += 1.0 / np.log2(rank + 1)
        ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(k, len(pos_set)) + 1))
        out[f"NDCG@{k}"] = float(dcg / ideal) if ideal > 0 else 0.0
    # MRR
    mrr = 0.0
    for rank, item in enumerate(order, start=1):
        if int(item) in pos_set:
            mrr = 1.0 / rank
            break
    out["MRR"] = float(mrr)
    return out


def evaluate_full_ranking(
    *,
    eval_users: list[int],
    test_positives: dict[int, set[int]],
    model_train: dict[int, set[int]],
    n_items: int,
    score_fn: Callable[[int, np.ndarray], np.ndarray],
    chunk_size: int = 2048,
    ks: tuple[int, ...] = (5, 10, 20),
    exclude_train: bool = True,
    max_positives_per_user: int = 10,
) -> dict[str, float]:
    """score_fn(user_id, item_ids_chunk) -> scores for that chunk."""
    metrics_sum = {f"NDCG@{k}": 0.0 for k in ks}
    metrics_sum.update({f"Recall@{k}": 0.0 for k in ks})
    metrics_sum["MRR"] = 0.0
    n_users = 0
    for u in eval_users:
        pos = sorted(test_positives.get(u, ()))
        if not pos:
            continue
        if len(pos) > max_positives_per_user:
            pos = pos[:max_positives_per_user]
        scores = np.full(n_items, -np.inf, dtype=np.float64)
        for start in range(0, n_items, chunk_size):
            chunk = np.arange(start, min(start + chunk_size, n_items), dtype=np.int64)
            scores[chunk] = score_fn(u, chunk)
        m = full_rank_user_metrics(
            scores=scores,
            positive_items=np.asarray(pos, dtype=np.int64),
            train_items=model_train.get(u, set()),
            ks=ks,
            exclude_train=exclude_train,
        )
        for k, v in m.items():
            metrics_sum[k] += v
        n_users += 1
        if n_users % 500 == 0:
            print(f"[full-rank] users {n_users}/{len(eval_users)}")
    if n_users == 0:
        return {k: float("nan") for k in metrics_sum}
    return {k: float(v / n_users) for k, v in metrics_sum.items()}
