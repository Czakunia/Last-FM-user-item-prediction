#!/usr/bin/env python3
"""RP3β full-catalog baseline via frozen publication evaluator (audit only).

Does not modify TRUE FINAL. Fits RP3β on model_train only; evaluates on validation
users through evaluate_publication_full_rank (same protocol as neural models).

Output: reports/rp3beta_fullrank_20260824.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_repos" / "IntentAwareRS"))

from src.lastfm_lp.data.build_splits import load_user_sets
from src.lastfm_lp.evaluation.publication_full_rank_evaluator import (
    evaluate_publication_full_rank,
    evaluate_user_dense,
    dense_topk,
)
from topn_baselines_neurals.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

SPLITS = ROOT / "outputs" / "lastfm_star" / "splits"
N_ITEMS = 48123
REFERENCE_RP3BETA = {
    "topK": 350,
    "alpha": 0.7681732734954694,
    "beta": 0.4181395996963926,
    "normalize_similarity": True,
    "implicit": False,
    "min_rating": 0,
}
SANITY_USERS = [0, 1, 2, 3, 4]


def build_urm(model_train: dict[int, set[int]], n_users: int, n_items: int) -> sps.csr_matrix:
    rows, cols, data = [], [], []
    for u, items in model_train.items():
        for i in items:
            rows.append(int(u))
            cols.append(int(i))
            data.append(1.0)
    return sps.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)


def main() -> None:
    model_train = load_user_sets(SPLITS / "model_train.txt")
    valid = load_user_sets(SPLITS / "valid.txt")
    eval_users = sorted(json.loads((SPLITS / "eval_users.json").read_text())["eval_users"])
    n_users = max(max(model_train) if model_train else 0, max(valid) if valid else 0) + 1

    print(f"[rp3beta] building URM n_users={n_users} n_items={N_ITEMS} pairs={sum(len(v) for v in model_train.values())}")
    urm = build_urm(model_train, n_users, N_ITEMS)

    print(f"[rp3beta] fitting REFERENCE_RP3BETA={REFERENCE_RP3BETA}")
    t0 = time.time()
    rec = RP3betaRecommender(urm, verbose=True)
    rec.fit(**REFERENCE_RP3BETA)
    fit_s = time.time() - t0
    print(f"[rp3beta] fit done in {fit_s:.1f}s W_nonzero={rec.W_sparse.nnz}")

    def score_fn(u: int, items: np.ndarray) -> np.ndarray:
        sc = rec._compute_item_score(np.asarray([u], dtype=np.int32), items_to_compute=items.tolist())
        return sc[0].astype(np.float64)

    # Sanity on 5 users
    sanity = []
    for u in SANITY_USERS:
        if u not in valid or not valid[u]:
            continue
        train = model_train.get(u, set())
        pos = valid[u]
        scores = np.full(N_ITEMS, -np.inf, dtype=np.float64)
        sc = rec._compute_item_score(np.asarray([u], dtype=np.int32))[0]
        scores[:] = sc
        m = evaluate_user_dense(scores, positive_items=pos, train_items=train, ks=(5, 10, 20))
        top20 = dense_topk(scores, k=20, mask_items=train).tolist()
        sanity.append(
            {
                "user": u,
                "train_size": len(train),
                "n_val_pos": len(pos),
                "n_candidates": N_ITEMS - len(train),
                "top20": top20,
                "hits@20": sum(1 for i in top20 if i in pos),
                **{k: float(v) for k, v in m.items()},
            }
        )

    print(f"[rp3beta] full-catalog eval on {len(eval_users)} validation users …")
    t1 = time.time()
    result = evaluate_publication_full_rank(
        eval_users=eval_users,
        test_positives=valid,
        model_train=model_train,
        n_items=N_ITEMS,
        score_fn=score_fn,
        mode="dense",
        ks=(5, 10, 20),
        store_per_user=False,
    )
    eval_s = time.time() - t1
    print("[rp3beta] mean metrics:", result.metrics_mean)

    out = {
        "frozen_date": "2026-08-24",
        "reference_hyperparameters": REFERENCE_RP3BETA,
        "intentawarers_commit": "63ece4444659b9505c36058be219c2db951ea087",
        "implementation": "external_repos/IntentAwareRS/topn_baselines_neurals/Recommenders/GraphBased/RP3betaRecommender.py",
        "evaluator": "src/lastfm_lp/evaluation/publication_full_rank_evaluator.py",
        "n_items": N_ITEMS,
        "n_eval_users": result.n_users,
        "fit_seconds": fit_s,
        "eval_seconds": eval_s,
        "metrics_mean": result.metrics_mean,
        "sanity_users": sanity,
    }
    report_path = ROOT / "reports" / "rp3beta_fullrank_20260824.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[rp3beta] wrote {report_path}")


if __name__ == "__main__":
    main()
