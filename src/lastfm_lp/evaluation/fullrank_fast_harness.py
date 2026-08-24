"""Shared OLD vs NEW full-rank feature comparison (no model required)."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy import sparse

from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_TOP_K
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix
from src.lastfm_lp.clean_v2.tabular_true import neighborhood_sizes, vectorized_clean_v2_A_for_user
from src.lastfm_lp.evaluation.fullrank_fast_features import (
    eligible_candidates,
    eligible_candidates_reference,
    iter_feature_blocks,
    precompute_candidate_a5_statics,
    scale_arr,
)


def activity_group(hist_len: int) -> str:
    if hist_len <= 3:
        return "VERY_LIGHT"
    if hist_len <= 10:
        return "LIGHT"
    if hist_len <= 25:
        return "MEDIUM"
    if hist_len <= 80:
        return "HEAVY"
    return "VERY_HEAVY"


def reference_h3_l2(
    hist: np.ndarray,
    cands: np.ndarray,
    index: PairwiseStatsIndex,
    *,
    max_history: int = CLEAN_V2_TOP_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copy of the reference h3_and_l2 plus Top25 IDs / signed values."""

    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    h3 = np.zeros((C, 3), dtype=np.float32)
    l2 = np.zeros((C, 1), dtype=np.float32)
    hist = np.asarray(hist, dtype=np.int64).reshape(-1)
    if hist.size == 0 or C == 0:
        return h3, l2, np.zeros((0, C), dtype=np.int64), np.zeros((0, C), dtype=np.float64)
    n11 = index.cooccurrence_block(hist, cands)
    a11, _ = a11_energy_from_n11_matrix(
        n11, index.popularity[hist], index.popularity[cands], index.n_users
    )
    if hist.size <= max_history:
        a_sel = a11
        top_ids = np.broadcast_to(hist[:, None], (hist.size, C)).copy()
    else:
        idx = deterministic_topk_indices(a11, hist, k=max_history)
        a_sel = np.take_along_axis(a11, idx, axis=0)
        top_ids = hist[idx]
    h3 = pool_signed_a11_3d_matrix(a_sel)
    p2 = 0.5 * (3.0 * np.square(a_sel.astype(np.float64)) - 1.0)
    l2[:, 0] = p2.mean(axis=0).astype(np.float32)
    return h3, l2, top_ids, a_sel


def max_abs_rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 and b.size == 0:
        return 0.0, 0.0
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(a.astype(np.float64)), 1e-30)
    rel = float((diff / denom).max()) if diff.size else 0.0
    return mx, rel


def compare_user_features(
    *,
    user_id: int,
    n_items: int,
    hist: set[int],
    index: PairwiseStatsIndex,
    model_train: dict[int, set[int]],
    popularity: np.ndarray,
    item_kg_degree: np.ndarray,
    n_x_sizes: np.ndarray,
    cand_batch: int,
    a_mean: np.ndarray,
    a_scale: np.ndarray,
    h_mean: np.ndarray,
    h_scale: np.ndarray,
    l2_mean: np.ndarray,
    l2_scale: np.ndarray,
    sample_cands_for_top25: int = 64,
) -> dict[str, Any]:
    hist_arr = np.asarray(sorted(hist), dtype=np.int64)
    log1p_pop, log1p_kg = precompute_candidate_a5_statics(popularity, item_kg_degree)

    t0 = perf_counter()
    cands_old = eligible_candidates_reference(n_items, hist)
    t_cand_old = perf_counter() - t0
    t0 = perf_counter()
    cands_new = eligible_candidates(n_items, hist)
    t_cand_new = perf_counter() - t0
    cand_equal = bool(np.array_equal(cands_old, cands_new))

    A_old, H_old, L2_old = [], [], []
    top_old_ids, top_old_a11 = [], []
    t_a5 = t_a11 = t_h3 = t_scale_old = 0.0
    sample = cands_old[: min(int(sample_cands_for_top25), int(cands_old.size))]

    for start in range(0, int(cands_old.size), int(cand_batch)):
        end = min(start + int(cand_batch), int(cands_old.size))
        cands = cands_old[start:end]
        t0 = perf_counter()
        A = vectorized_clean_v2_A_for_user(
            int(user_id),
            cands,
            model_train,
            popularity,
            item_kg_degree,
            index,
            n_x_sizes=n_x_sizes,
        )
        t_a5 += perf_counter() - t0
        t0 = perf_counter()
        H, L2, top_ids, a_sel = reference_h3_l2(hist_arr, cands, index)
        t_h3 += perf_counter() - t0
        # A11+Top25 is inside h3; split is not exact in the reference (shared n11
        # is computed twice). Report h3 wall as A11+Top25+H3+L2.
        t0 = perf_counter()
        _ = scale_arr(A, a_mean, a_scale)
        _ = scale_arr(H, h_mean, h_scale)
        _ = scale_arr(L2, l2_mean, l2_scale)
        t_scale_old += perf_counter() - t0
        A_old.append(A)
        H_old.append(H)
        L2_old.append(L2)
        if sample.size:
            mask = np.isin(cands, sample)
            if mask.any():
                top_old_ids.append(top_ids[:, mask])
                top_old_a11.append(a_sel[:, mask])

    t_a11 = t_h3  # reference folds A11/Top25 into h3_and_l2

    A_new, H_new, L2_new = [], [], []
    top_new_ids, top_new_a11 = [], []
    t_feat_new = 0.0
    t0 = perf_counter()
    for blk in iter_feature_blocks(
        n_items=n_items,
        hist=hist,
        index=index,
        n_x_sizes=n_x_sizes,
        log1p_pop=log1p_pop,
        log1p_kg=log1p_kg,
        cand_batch=cand_batch,
    ):
        A_new.append(blk["A"])
        H_new.append(blk["H"])
        L2_new.append(blk["L2"])
        cands = blk["cands"]
        if sample.size:
            mask = np.isin(cands, sample)
            if mask.any():
                top_new_ids.append(blk["top25_ids"][:, mask])
                a11 = blk["a11"]
                hist_local = blk["hist_arr"]
                if hist_local.size == 0:
                    continue
                if hist_local.size <= CLEAN_V2_TOP_K:
                    a_sel = a11[:, mask]
                else:
                    idx = deterministic_topk_indices(a11, hist_local, k=CLEAN_V2_TOP_K)
                    a_sel = np.take_along_axis(a11, idx, axis=0)[:, mask]
                top_new_a11.append(a_sel)
    t_feat_new = perf_counter() - t0

    A_o = np.concatenate(A_old, axis=0) if A_old else np.zeros((0, 5), np.float32)
    H_o = np.concatenate(H_old, axis=0) if H_old else np.zeros((0, 3), np.float32)
    L_o = np.concatenate(L2_old, axis=0) if L2_old else np.zeros((0, 1), np.float32)
    A_n = np.concatenate(A_new, axis=0) if A_new else np.zeros((0, 5), np.float32)
    H_n = np.concatenate(H_new, axis=0) if H_new else np.zeros((0, 3), np.float32)
    L_n = np.concatenate(L2_new, axis=0) if L2_new else np.zeros((0, 1), np.float32)

    A_o_s = scale_arr(A_o, a_mean, a_scale)
    A_n_s = scale_arr(A_n, a_mean, a_scale)
    H_o_s = scale_arr(H_o, h_mean, h_scale)
    H_n_s = scale_arr(H_n, h_mean, h_scale)
    L_o_s = scale_arr(L_o, l2_mean, l2_scale)
    L_n_s = scale_arr(L_n, l2_mean, l2_scale)

    def pack(name: str, old: np.ndarray, new: np.ndarray, atol: float = 0.0) -> dict[str, Any]:
        mx, rel = max_abs_rel(old, new)
        return {
            "name": name,
            "max_abs_diff": mx,
            "max_rel_diff": rel,
            "allclose": bool(np.allclose(old, new, rtol=0.0, atol=max(atol, 0.0))),
            "shape_old": list(old.shape),
            "shape_new": list(new.shape),
        }

    top_id_eq = True
    top_a11_eq = True
    if top_old_ids:
        oi = np.concatenate(top_old_ids, axis=1)
        ni = np.concatenate(top_new_ids, axis=1) if top_new_ids else np.zeros_like(oi)
        top_id_eq = bool(np.array_equal(oi, ni))
        oa = np.concatenate(top_old_a11, axis=1)
        na = np.concatenate(top_new_a11, axis=1) if top_new_a11 else np.zeros_like(oa)
        top_a11_eq = bool(np.allclose(oa, na, rtol=0.0, atol=0.0))

    arrays = [
        pack("A5_raw", A_o, A_n),
        pack("A5_scaled", A_o_s, A_n_s, atol=0.0),
        pack("H3_raw", H_o, H_n),
        pack("H3_scaled", H_o_s, H_n_s),
        pack("L2_raw", L_o, L_n),
        pack("L2_scaled", L_o_s, L_n_s),
    ]
    feat_pass = cand_equal and top_id_eq and top_a11_eq and all(r["allclose"] for r in arrays)
    n_cands = int(cands_old.size)
    t_old = t_cand_old + t_a5 + t_h3 + t_scale_old
    t_new = t_cand_new + t_feat_new
    return {
        "user": int(user_id),
        "hist_len": int(len(hist)),
        "activity_group": activity_group(len(hist)),
        "n_cands": n_cands,
        "cand_equal": cand_equal,
        "top25_ids_equal": top_id_eq,
        "top25_a11_equal": top_a11_eq,
        "PASS": feat_pass,
        "arrays": arrays,
        "t_cand_old": t_cand_old,
        "t_cand_new": t_cand_new,
        "t_a5_old": t_a5,
        "t_a11_h3_l2_old": t_a11,
        "t_scale_old": t_scale_old,
        "t_feat_new": t_feat_new,
        "t_total_old": t_old,
        "t_total_new": t_new,
        "cands_per_sec_old": n_cands / t_old if t_old > 0 else float("nan"),
        "cands_per_sec_new": n_cands / t_new if t_new > 0 else float("nan"),
    }


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = perf_counter()
    out = fn()
    return out, perf_counter() - t0


def make_synth_index(
    rng: np.random.Generator,
    *,
    n_items: int,
    n_users: int,
    nnz: int,
    symmetric: bool = True,
    zipf: bool = False,
) -> tuple[PairwiseStatsIndex, np.ndarray, np.ndarray]:
    rows = rng.integers(0, n_items, size=int(nnz), dtype=np.int64)
    cols = rng.integers(0, n_items, size=int(nnz), dtype=np.int64)
    data = rng.integers(1, 6, size=int(nnz), dtype=np.int32)
    keep = rows != cols
    if zipf:
        rows = np.minimum(rng.zipf(1.7, int(nnz)), n_items).astype(np.int64) - 1
        cols = np.minimum(rng.zipf(1.7, int(nnz)), n_items).astype(np.int64) - 1
        data = rng.integers(1, 6, size=int(nnz), dtype=np.int32)
        keep = rows != cols
    cooc = sparse.csr_matrix(
        (data[keep], (rows[keep], cols[keep])),
        shape=(n_items, n_items),
        dtype=np.int32,
    )
    if symmetric:
        cooc = cooc.maximum(cooc.T)
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    pop = np.asarray(cooc.sum(axis=1)).reshape(-1).astype(np.int32)
    pop = np.maximum(pop, 1)
    kg = rng.integers(0, 40, size=n_items, dtype=np.int32)
    index = PairwiseStatsIndex(cooc, pop, n_users=int(n_users), smoothing=0.5)
    return index, pop, kg


def make_synth_users(
    rng: np.random.Generator,
    *,
    n_items: int,
    counts: dict[str, int],
) -> list[tuple[int, set[int]]]:
    """Deterministic activity-stratified users. Item IDs are unique per user."""

    specs = {
        "VERY_LIGHT": (1, 3),
        "LIGHT": (4, 10),
        "MEDIUM": (11, 25),
        "HEAVY": (26, 80),
        "VERY_HEAVY": (81, 160),
    }
    users: list[tuple[int, set[int]]] = []
    uid = 0
    for group, n_u in counts.items():
        lo, hi = specs[group]
        hi = min(hi, n_items - 1)
        lo = min(lo, hi)
        for _ in range(int(n_u)):
            hlen = int(rng.integers(lo, hi + 1))
            hist = set(int(x) for x in rng.choice(n_items, size=hlen, replace=False))
            users.append((uid, hist))
            uid += 1
    return users


def dummy_scaler_params(dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(dim, dtype=np.float32)
    scale = np.ones(dim, dtype=np.float32)
    return mean, scale
