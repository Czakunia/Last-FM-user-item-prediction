"""User-wise ranking metrics: NDCG@K, Recall@K, MRR."""

from __future__ import annotations

import numpy as np


def _dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float((rels * discounts).sum())


def ranking_metrics_for_users(
    user_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    ks: list[int] | tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Compute mean metrics over users that have ≥1 positive in the table."""
    ks = list(ks)
    buckets: dict[int, list[tuple[float, int]]] = {}
    for u, y, s in zip(user_ids.tolist(), labels.tolist(), scores.tolist()):
        buckets.setdefault(int(u), []).append((float(s), int(y)))

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    mrrs: list[float] = []

    for rows in buckets.values():
        rows.sort(key=lambda t: t[0], reverse=True)
        ys = np.asarray([y for _, y in rows], dtype=np.float64)
        n_pos = int(ys.sum())
        if n_pos == 0:
            continue
        # MRR
        ranks = np.where(ys > 0)[0]
        mrrs.append(1.0 / (ranks[0] + 1))
        for k in ks:
            hits = float(ys[:k].sum())
            recalls[k].append(hits / n_pos)
            dcg = _dcg_at_k(ys, k)
            ideal = _dcg_at_k(np.ones(min(n_pos, k), dtype=np.float64), k)
            ndcgs[k].append(dcg / ideal if ideal > 0 else 0.0)

    out: dict[str, float] = {
        "n_users_ranked": float(len(mrrs)),
        "MRR": float(np.mean(mrrs)) if mrrs else float("nan"),
    }
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else float("nan")
        out[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else float("nan")
    return out
