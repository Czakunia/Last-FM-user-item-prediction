"""Unit tests for publication all-ranking evaluator."""

from __future__ import annotations

import numpy as np

from src.lastfm_lp.evaluation.publication_full_rank_evaluator import (
    chunked_topk,
    dense_topk,
    evaluate_publication_full_rank,
    evaluate_user_dense,
    exactness_chunked_vs_dense,
    metrics_from_topk,
)


def test_handcrafted_recall_ndcg_and_train_mask():
    # 6 items, user train={0,1}, test positives={2,4}
    # scores: prefer 1 (train, must be masked), then 4, 2, 3, 5, 0
    scores = np.array([0.1, 9.0, 0.8, 0.5, 0.9, 0.2], dtype=np.float64)
    train = {0, 1}
    pos = {2, 4}
    m = evaluate_user_dense(scores, positive_items=pos, train_items=train, ks=(2, 3))
    # After mask, ranking by score: 4 (0.9), 2 (0.8), 3 (0.5), 5 (0.2), 0(-inf), 1(-inf)
    # top2 = {4,2} → both positives → Recall@2 = 1.0
    assert abs(m["Recall@2"] - 1.0) < 1e-12
    # NDCG@2: hits at rank1 and rank2 → dcg = 1 + 1/log2(3), ideal same
    assert abs(m["NDCG@2"] - 1.0) < 1e-12
    assert abs(m["MRR"] - 1.0) < 1e-12  # first pos at rank 1


def test_tie_break_prefers_lower_item_id():
    scores = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float64)
    top = dense_topk(scores, k=3, mask_items=None)
    assert top.tolist() == [0, 1, 2]


def test_unrankable_positive_stays_in_denominator():
    # positive 1 is also in train → masked → unrankable but still in denom
    scores = np.array([0.2, 0.9, 0.8], dtype=np.float64)
    m = evaluate_user_dense(scores, positive_items={1, 2}, train_items={1}, ks=(1,))
    # only item 2 can be ranked; top1={2} → hits=1, denom=2 → Recall@1=0.5
    assert abs(m["Recall@1"] - 0.5) < 1e-12


def test_chunked_equals_dense_toy():
    n_items = 50
    rng = np.random.default_rng(0)
    base = rng.normal(size=n_items)

    def score_fn(u, items):
        return base[items] + 0.01 * u

    train = {u: set(rng.choice(n_items, size=5, replace=False).tolist()) for u in range(10)}
    users = list(range(10))
    report = exactness_chunked_vs_dense(
        user_ids=users,
        n_items=n_items,
        score_fn=score_fn,
        model_train=train,
        k=20,
        chunk_size=7,
    )
    assert report["passed"], report


def test_evaluate_publication_mean():
    n_items = 20

    def score_fn(u, items):
        # higher score for item == u % n_items and neighbors
        return -np.abs(items.astype(np.float64) - (u % n_items))

    train = {0: {0, 1}, 1: {5}}
    test = {0: {2, 3}, 1: {6, 7}}
    res = evaluate_publication_full_rank(
        eval_users=[0, 1],
        test_positives=test,
        model_train=train,
        n_items=n_items,
        score_fn=score_fn,
        chunk_size=5,
        ks=(5, 10, 20),
        mode="chunked",
    )
    assert res.n_users == 2
    assert res.metrics_mean["Recall@20"] > 0.0


def test_metrics_from_topk_empty_pos():
    m = metrics_from_topk(np.array([1, 2, 3]), set(), ks=(5,))
    assert np.isnan(m["Recall@5"])
    assert np.isnan(m["HitRate@5"])


def test_hitrate_at_k():
    m = metrics_from_topk(np.array([9, 8, 7, 1]), {1, 2}, ks=(2, 4))
    assert m["HitRate@2"] == 0.0
    assert m["HitRate@4"] == 1.0
    assert abs(m["Recall@4"] - 0.5) < 1e-12
