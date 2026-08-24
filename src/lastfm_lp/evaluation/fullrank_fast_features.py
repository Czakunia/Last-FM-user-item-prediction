"""Exact full-rank feature path with cheaper execution.

Scientific formulas are identical to
``scripts/run_lastfm_noleak_fullrank_validation_v1.py``.
This module only avoids repeated CSR slices and Python candidate scans.

Cross-fit: always use ``clean_v2_index_for(cf, u)`` (fold 0–4), never a
global co-occurrence matrix.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from scipy import sparse
import torch

from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_TOP_K
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix


def eligible_candidates(n_items: int, hist: set[int]) -> np.ndarray:
    """C_u = {0..n_items-1} \\ H_u. Same order as a filtered range()."""

    n_items = int(n_items)
    eligible = np.ones(n_items, dtype=bool)
    if hist:
        vals = [int(x) for x in hist if 0 <= int(x) < n_items]
        if vals:
            eligible[np.asarray(vals, dtype=np.int64)] = False
    return np.flatnonzero(eligible).astype(np.int64, copy=False)


def eligible_candidates_reference(n_items: int, hist: set[int]) -> np.ndarray:
    """Reference scan used by the original evaluator."""

    return np.asarray([i for i in range(int(n_items)) if i not in hist], dtype=np.int64)


def precompute_candidate_a5_statics(
    popularity: np.ndarray,
    item_kg_degree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    log1p_pop = np.log1p(np.asarray(popularity, dtype=np.float64))
    log1p_kg = np.log1p(np.asarray(item_kg_degree, dtype=np.float64))
    return log1p_pop, log1p_kg


def n11_from_hist_csr(
    hist_csr: sparse.csr_matrix,
    hist_arr: np.ndarray,
    cands: np.ndarray,
) -> np.ndarray:
    """Same dense H×C n11 as PairwiseStatsIndex.cooccurrence_block."""

    hist_arr = np.asarray(hist_arr, dtype=np.int64).reshape(-1)
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    H, C = int(hist_arr.size), int(cands.size)
    if H == 0 or C == 0:
        return np.zeros((H, C), dtype=np.int32)
    block = hist_csr[:, cands]
    if sparse.issparse(block):
        arr = np.asarray(block.toarray(), dtype=np.int32)
    else:
        arr = np.asarray(block, dtype=np.int32)
    same = hist_arr[:, None] == cands[None, :]
    if same.any():
        arr = arr.copy()
        arr[same] = 0
    return arr


def a5_from_n11(
    *,
    n11: np.ndarray,
    cands: np.ndarray,
    h_len: float,
    n_x_sizes: np.ndarray,
    log1p_pop: np.ndarray,
    log1p_kg: np.ndarray,
) -> np.ndarray:
    """CLEAN V2 A5 from a shared n11 block. Matches vectorized_clean_v2_A_for_user."""

    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    X = np.zeros((C, 5), dtype=np.float32)
    X[:, 0] = np.log1p(h_len)
    if C == 0:
        return X
    X[:, 1] = log1p_pop[cands].astype(np.float32)
    X[:, 2] = log1p_kg[cands].astype(np.float32)
    sizes = n_x_sizes[cands].astype(np.float64)
    if h_len <= 0 or n11.size == 0:
        return X
    inter = (n11 > 0).sum(axis=0).astype(np.float64)
    union = float(h_len) + sizes - inter
    j = np.zeros(C, dtype=np.float64)
    ok_u = union > 0
    j[ok_u] = inter[ok_u] / union[ok_u]
    c = np.zeros(C, dtype=np.float64)
    ok_c = (h_len > 0) & (sizes > 0)
    c[ok_c] = inter[ok_c] / np.sqrt(h_len * sizes[ok_c])
    X[:, 3] = j.astype(np.float32)
    X[:, 4] = c.astype(np.float32)
    return X


def top25_ids_from_a11(
    a11: np.ndarray,
    hist_arr: np.ndarray,
    *,
    max_history: int = CLEAN_V2_TOP_K,
) -> np.ndarray:
    """History item IDs selected for each candidate. Shape (K, C)."""

    hist_arr = np.asarray(hist_arr, dtype=np.int64).reshape(-1)
    a11 = np.asarray(a11, dtype=np.float64)
    C = int(a11.shape[1]) if a11.ndim == 2 else 0
    if hist_arr.size == 0 or C == 0:
        return np.zeros((0, C), dtype=np.int64)
    if hist_arr.size <= max_history:
        return np.broadcast_to(hist_arr[:, None], (hist_arr.size, C)).copy()
    idx = deterministic_topk_indices(a11, hist_arr, k=max_history)
    return hist_arr[idx]


def h3_l2_from_a11(
    a11: np.ndarray,
    hist_arr: np.ndarray,
    *,
    max_history: int = CLEAN_V2_TOP_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H3 and LEG L2 from signed A11 matrix. Same Top25 rule as the reference."""

    hist_arr = np.asarray(hist_arr, dtype=np.int64).reshape(-1)
    a11 = np.asarray(a11, dtype=np.float64)
    C = int(a11.shape[1]) if a11.ndim == 2 else 0
    h3 = np.zeros((C, 3), dtype=np.float32)
    l2 = np.zeros((C, 1), dtype=np.float32)
    if hist_arr.size == 0 or C == 0:
        return h3, l2, np.zeros((0, C), dtype=np.int64)
    if hist_arr.size <= max_history:
        a_sel = a11
        top_ids = np.broadcast_to(hist_arr[:, None], (hist_arr.size, C)).copy()
    else:
        top_idx = deterministic_topk_indices(a11, hist_arr, k=max_history)
        a_sel = np.take_along_axis(a11, top_idx, axis=0)
        top_ids = hist_arr[top_idx]
    h3 = pool_signed_a11_3d_matrix(a_sel)
    p2 = 0.5 * (3.0 * np.square(a_sel.astype(np.float64)) - 1.0)
    l2[:, 0] = p2.mean(axis=0).astype(np.float32)
    return h3, l2, top_ids


def scale_arr(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / np.maximum(scale, 1e-12)).astype(np.float32)


def fold_index_cached(cf, u: int, ctx: dict[str, Any]) -> PairwiseStatsIndex:
    cache: dict[Any, PairwiseStatsIndex] = ctx.setdefault("_fold_index", {})
    fold = cf.user_to_fold.get(int(u))
    key: Any = "full" if fold is None else int(fold)
    if key not in cache:
        cache[key] = clean_v2_index_for(cf, int(u))
    return cache[key]


def ensure_a5_statics(ctx: dict[str, Any], bundle: dict[str, Any]) -> None:
    if "log1p_pop" in ctx and "log1p_kg" in ctx:
        return
    ctx["log1p_pop"], ctx["log1p_kg"] = precompute_candidate_a5_statics(
        bundle["popularity"], bundle["item_kg_degree"]
    )


def iter_feature_blocks(
    *,
    n_items: int,
    hist: set[int],
    index: PairwiseStatsIndex,
    n_x_sizes: np.ndarray,
    log1p_pop: np.ndarray,
    log1p_kg: np.ndarray,
    cand_batch: int = 8192,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield per-batch A5/H3/L2/A11 from one CSR row-slice of H_u.

    ``index.cooc[H_u]`` is taken once. Each batch only slices those rows.
    n11 is shared by A5 and A11/Top25/H3/L2 (the reference gathers it twice).
    """

    cands_all = eligible_candidates(n_items, hist)
    hist_arr = np.asarray(sorted(hist), dtype=np.int64)
    h_len = float(len(hist))
    hist_csr = index.cooc[hist_arr] if hist_arr.size else None
    pop_h = index.popularity[hist_arr] if hist_arr.size else index.popularity[:0]
    for start in range(0, int(cands_all.size), int(cand_batch)):
        end = min(start + int(cand_batch), int(cands_all.size))
        cands = cands_all[start:end]
        if hist_csr is None:
            n11 = np.zeros((0, int(cands.size)), dtype=np.int32)
            a11 = np.zeros((0, int(cands.size)), dtype=np.float64)
        else:
            n11 = n11_from_hist_csr(hist_csr, hist_arr, cands)
            a11, _ = a11_energy_from_n11_matrix(
                n11, pop_h, index.popularity[cands], index.n_users
            )
        A = a5_from_n11(
            n11=n11,
            cands=cands,
            h_len=h_len,
            n_x_sizes=n_x_sizes,
            log1p_pop=log1p_pop,
            log1p_kg=log1p_kg,
        )
        H, L2, top_ids = h3_l2_from_a11(a11, hist_arr)
        yield {
            "cands": cands,
            "A": A,
            "H": H,
            "L2": L2,
            "n11": n11,
            "a11": a11,
            "top25_ids": top_ids,
            "hist_arr": hist_arr,
        }


@torch.no_grad()
def score_user_catalog_fast(
    u: int,
    *,
    n_items: int,
    item_offset: int,
    hist: set[int],
    bundle,
    cf,
    ctx: dict[str, Any],
    n_x_sizes: np.ndarray,
    cand_batch: int = 8192,
) -> np.ndarray:
    """Same scores as the reference scorer; one CSR row-slice per user."""

    ensure_a5_statics(ctx, bundle)
    device = ctx["device"]
    model = ctx["model"]
    scores = np.full(n_items, -np.inf, dtype=np.float64)
    index = fold_index_cached(cf, int(u), ctx)
    z = ctx["z"]
    for blk in iter_feature_blocks(
        n_items=n_items,
        hist=hist,
        index=index,
        n_x_sizes=n_x_sizes,
        log1p_pop=ctx["log1p_pop"],
        log1p_kg=ctx["log1p_kg"],
        cand_batch=cand_batch,
    ):
        cands = blk["cands"]
        A_s = scale_arr(blk["A"], ctx["a_mean"], ctx["a_scale"])
        H_s = scale_arr(blk["H"], ctx["h_mean"], ctx["h_scale"])
        L2_s = scale_arr(blk["L2"], ctx["l2_mean"], ctx["l2_scale"])
        item_idx = torch.from_numpy(cands.astype(np.int64) + item_offset).to(device)
        # Scalar once per user; expand avoids per-batch torch.full on MPS.
        # Cache keyed by Python int — never .item() (MPS sync) in the inner loop.
        if ctx.get("_u_scalar_id") != int(u):
            ctx["_u_scalar"] = torch.tensor([int(u)], dtype=torch.int64, device=device)
            ctx["_u_scalar_id"] = int(u)
        user_idx = ctx["_u_scalar"].expand(int(cands.size))
        out = model(
            user_idx,
            item_idx,
            torch.from_numpy(A_s).to(device),
            torch.from_numpy(H_s).to(device),
            torch.from_numpy(L2_s).to(device),
            z=z,
        )
        scores[cands] = out["logits"].detach().cpu().numpy().astype(np.float64)
    return scores
