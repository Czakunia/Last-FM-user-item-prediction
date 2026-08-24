"""CLEAN V2 unit tests (TEST 1–10). Must all PASS before training."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import sparse

from src.lastfm_lp.binary.binary_measures import a11_from_contingency
from src.lastfm_lp.binary.cross_fit import CrossFitHCRBundle, assign_user_folds
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_FUSION_DIM, CLEAN_V2_HCR_DIM
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for
from src.lastfm_lp.clean_v2.fusion import concat_clean_v2_fusion
from src.lastfm_lp.clean_v2.legacy_guards import assert_legacy_proxies_not_used_in_clean_call_stack
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d
from src.lastfm_lp.clean_v2.routing import (
    CLEAN_V2_ROUTING_POLICIES,
    routing_history_exclude_self,
    select_top25_ids,
)
from src.lastfm_lp.clean_v2.tabular_true import (
    candidate_neighborhood_N_X,
    true_tabular_cosine,
    true_tabular_jaccard,
)
from src.lastfm_lp.features import tabular_pair_features as legacy_tab


def test_1_true_tabular_jaccard_cosine():
    H_u = {10, 20, 30, 40}
    N_X = {20, 30, 50, 60}
    j = true_tabular_jaccard(H_u, N_X)
    c = true_tabular_cosine(H_u, N_X)
    assert abs(j - 2.0 / 6.0) <= 1e-12
    assert abs(c - 0.5) <= 1e-12


def test_2_empty_sets():
    assert true_tabular_jaccard({}, {1, 2, 3}) == 0.0
    assert true_tabular_cosine({}, {1, 2, 3}) == 0.0
    assert true_tabular_jaccard({}, {}) == 0.0
    assert true_tabular_cosine({}, {}) == 0.0
    assert true_tabular_jaccard({1}, {}) == 0.0
    assert true_tabular_cosine({1}, {}) == 0.0


def test_3_item_item_routing_jaccard_cosine():
    # U_h={1,2,3,4}, U_X={2,3,5} → n_h=4, n_X=3, n11=2
    from src.lastfm_lp.binary.binary_measures import (
        cosine_from_n11_matrix,
        jaccard_from_n11_matrix,
    )

    n11 = np.array([[2.0]])
    pop_j = np.array([4.0])
    pop_i = np.array([3.0])
    j = float(jaccard_from_n11_matrix(n11, pop_j, pop_i)[0, 0])
    c = float(cosine_from_n11_matrix(n11, pop_j, pop_i)[0, 0])
    assert abs(j - 0.4) <= 1e-12
    assert abs(c - (2.0 / math.sqrt(12.0))) <= 1e-12


def test_4_a11_equals_phi():
    # N=10, n_h=4, n_X=5, n11=3 → n10=1, n01=2, n00=4
    a11, _ = a11_from_contingency(3, 1, 2, 4)
    expected = 10.0 / math.sqrt(600.0)
    assert abs(a11 - expected) <= 1e-12
    # binary vectors
    # users 0..9: h in 0..3, X in 0..2 and 3,4? construct matching contingency
    h = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    # n11=3, n_X=5 → X on three of h's and two of non-h
    x = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.float64)
    assert int(((h == 1) & (x == 1)).sum()) == 3
    assert int(h.sum()) == 4 and int(x.sum()) == 5
    phi = float(np.corrcoef(h, x)[0, 1])
    assert abs(a11 - phi) <= 1e-12


def test_5_negative_a11():
    # below independence: N=100, n_h=50, n_X=40, n11=5 → expected 20
    # n10=45, n01=35, n00=15
    a11, _ = a11_from_contingency(5, 45, 35, 15)
    assert a11 < 0.0


def test_6_self_selection_all_policies():
    # tiny cooc: items 5,10,20,30
    n_items = 40
    # users: each pair among {5,10,20,30} co-occurs
    cooc = sparse.lil_matrix((n_items, n_items), dtype=np.int32)
    items = [5, 10, 20, 30]
    for a in items:
        for b in items:
            if a != b:
                cooc[a, b] = 3
    pop = np.zeros(n_items, dtype=np.int32)
    for i in items:
        pop[i] = 5
    index = PairwiseStatsIndex(cooc.tocsr(), pop, n_users=10, smoothing=0.0)
    H = {5, 10, 20, 30}
    X = 20
    rh = routing_history_exclude_self(H, X)
    assert 20 not in set(rh.tolist())
    assert set(rh.tolist()) == {5, 10, 30}
    for pol in CLEAN_V2_ROUTING_POLICIES:
        sel = select_top25_ids(H, X, index, policy=pol, assert_no_self=True)
        assert 20 not in set(int(x) for x in sel.tolist())


def test_7_crossfit_excludes_own_fold():
    # 6 users, 2 folds, 3 items
    model_train = {
        0: {0, 1},
        1: {0, 1},
        2: {0, 1},
        3: {1, 2},
        4: {1, 2},
        5: {1, 2},
    }
    user_to_fold = assign_user_folds(sorted(model_train), n_folds=2, seed=2026)
    # Build fold indices manually like cross_fit
    from src.lastfm_lp.binary.contingency_tables import build_cooccurrence
    from src.lastfm_lp.binary.cross_fit import build_fold_index

    full = build_fold_index(model_train, 3, max_history_for_pairs=80, seed=2026, smoothing=0.0)
    fold_indices = []
    for k in range(2):
        keep = {u for u, f in user_to_fold.items() if f != k}
        sub = {u: model_train[u] for u in keep}
        fold_indices.append(
            build_fold_index(sub, 3, max_history_for_pairs=80, seed=2026 + 17 * (k + 1), smoothing=0.0)
        )
    cf = CrossFitHCRBundle(
        full_index=full,
        fold_indices=fold_indices,
        user_to_fold=user_to_fold,
        n_folds=2,
    )
    u = 0
    f = user_to_fold[u]
    idx = clean_v2_index_for(cf, u)
    n11_before = int(idx.cooc[0, 1])
    # mutate a user IN fold f — CLEAN index must ignore fold f users
    # Adding a new co-occurrence only among fold-f users should not change idx
    fold_f_users = [uu for uu, ff in user_to_fold.items() if ff == f]
    assert u in fold_f_users
    # change interactions of fold-f user (not used in idx)
    mutated = {uu: set(items) for uu, items in model_train.items()}
    for uu in fold_f_users:
        mutated[uu] = {0, 1, 2}  # add item 2 everywhere in fold f
    # rebuild fold index for fold f from keep = fold != f (unchanged users outside f)
    keep = {uu for uu, ff in user_to_fold.items() if ff != f}
    sub = {uu: mutated[uu] for uu in keep}  # outside fold unchanged
    idx2 = build_fold_index(sub, 3, max_history_for_pairs=80, seed=2026 + 17 * (f + 1), smoothing=0.0)
    assert int(idx2.cooc[0, 1]) == n11_before
    # If we wrongly included fold f, n11 would grow — verify mutated fold-f alone would change full
    full_mut = build_cooccurrence(mutated, 3, max_history_for_pairs=80, seed=2026)
    assert int(full_mut[0, 2]) >= int(full.cooc[0, 2])


def test_8_clean_hcr_pool():
    v = np.array([-0.4, 0.1, 0.3, 0.8])
    out = pool_signed_a11_3d(v)
    assert out.shape == (3,)
    # float32 storage after float64 reductions
    assert abs(float(out[0]) - 0.2) <= 1e-7
    assert abs(float(out[1]) - 0.8) <= 1e-7
    assert abs(float(out[2]) - 0.4) <= 1e-7


def test_9_fusion_shape_265():
    B = 4
    gctx = torch.zeros(B, 256)
    gscore = torch.zeros(B)
    A = torch.zeros(B, 5)
    H = torch.zeros(B, 3)
    fused = concat_clean_v2_fusion(gctx, gscore, A, H)
    assert fused.shape == (B, CLEAN_V2_FUSION_DIM)
    assert CLEAN_V2_FUSION_DIM == 265
    assert CLEAN_V2_HCR_DIM == 3


def test_10_legacy_proxies_not_called():
    # CLEAN true formulas do not invoke legacy helpers
    called: set[str] = set()
    orig_cos = legacy_tab._cosine_pop
    orig_build = legacy_tab.build_tabular_pair_features

    def wrap_cos(*a, **k):
        called.add("_cosine_pop")
        return orig_cos(*a, **k)

    def wrap_build(*a, **k):
        called.add("build_tabular_pair_features")
        return orig_build(*a, **k)

    legacy_tab._cosine_pop = wrap_cos  # type: ignore
    legacy_tab.build_tabular_pair_features = wrap_build  # type: ignore
    try:
        _ = true_tabular_jaccard({1, 2}, {2, 3})
        _ = true_tabular_cosine({1, 2}, {2, 3})
        _ = pool_signed_a11_3d(np.array([0.1, 0.2]))
        assert_legacy_proxies_not_used_in_clean_call_stack(called)
        assert "_cosine_pop" not in called
        assert "build_tabular_pair_features" not in called
    finally:
        legacy_tab._cosine_pop = orig_cos  # type: ignore
        legacy_tab.build_tabular_pair_features = orig_build  # type: ignore


def test_N_X_excludes_self():
    n_items = 5
    cooc = sparse.lil_matrix((n_items, n_items), dtype=np.int32)
    cooc[1, 2] = 4
    cooc[2, 1] = 4
    cooc[1, 3] = 2
    cooc[3, 1] = 2
    pop = np.array([0, 5, 5, 3, 0], dtype=np.int32)
    index = PairwiseStatsIndex(cooc.tocsr(), pop, n_users=10, smoothing=0.0)
    N = candidate_neighborhood_N_X(1, index)
    assert 1 not in N
    assert N == {2, 3}
