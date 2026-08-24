"""Exact full-rank scoring in candidate batches. Semantics unchanged."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.lastfm_lp.artist_a11.core import (
    pool_artist_for_selection,
    select_top25_and_item_a11,
)
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_TOP_K
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix
from src.lastfm_lp.clean_v2.routing import aggregate_clean_v2_a11_pool
from src.lastfm_lp.clean_v2.tabular_true import (
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)

N_ITEMS = 48123
CAND_BATCH = 8192
ORACLE_MAX_ABS = 1e-6


def _scale(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / np.maximum(scale, 1e-12)).astype(np.float32)


def _h_for_cands(
    hist: set[int],
    cands: np.ndarray,
    idx,
    mapping,
    fold,
    variant: str,
) -> np.ndarray:
    """Item-A11 Top25 on ``cands``, then B0/B1/B2 pool. Same sel for item+artist."""
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    if variant == "B0":
        return aggregate_clean_v2_a11_pool(
            hist, cands, idx, policy="a11_top25", max_history=CLEAN_V2_TOP_K
        )
    sel, item_a = select_top25_and_item_a11(hist, cands, idx)
    Ha, _ = pool_artist_for_selection(sel, cands, mapping, fold)
    if variant == "B1":
        return Ha
    if item_a.size == 0:
        Hi = np.zeros((C, 3), dtype=np.float32)
    else:
        Hi = pool_signed_a11_3d_matrix(item_a)
    return np.concatenate([Hi, Ha], axis=1)


@torch.no_grad()
def _decode_batch(ctx: dict[str, Any], zu, zi, graph_score, A_s, H_s) -> np.ndarray:
    device = ctx["device"]
    model = ctx["model"]
    n = int(zi.shape[0])
    gctx = torch.cat([zu.expand(n, -1), zi, zu * zi, (zu - zi).abs()], dim=-1)
    logits = model.fusion_head(
        graph_context=gctx,
        graph_score=graph_score,
        base_features=torch.from_numpy(A_s).to(device),
        hcr_features=torch.from_numpy(H_s).to(device),
    )
    return np.asarray(logits.detach().cpu().numpy(), dtype=np.float64)


@torch.no_grad()
def score_user_batched(
    u: int,
    bundle: dict[str, Any],
    ctx: dict[str, Any],
    cf,
    mapping,
    folds,
    fmap: dict[int, int],
    variant: str,
    *,
    candidates: np.ndarray | None = None,
    n_x_sizes: np.ndarray | None = None,
    cand_batch: int = CAND_BATCH,
) -> np.ndarray:
    """Exact scores for ``candidates`` (default: full catalog 0..n_items-1)."""
    model_train = bundle["model_train"]
    pop = bundle["popularity"]
    kg_deg = bundle["item_kg_degree"]
    if candidates is None:
        cands_all = np.arange(N_ITEMS, dtype=np.int64)
    else:
        cands_all = np.asarray(candidates, dtype=np.int64).reshape(-1)
    n_out = int(cands_all.size)
    scores = np.empty(n_out, dtype=np.float64)
    idx = clean_v2_index_for(cf, int(u))
    hist = model_train.get(int(u), set())
    fold = folds[int(fmap[int(u)])]
    if n_x_sizes is None:
        n_x_sizes = neighborhood_sizes(idx)
    zu = ctx["zemb"][int(u)]
    zi_all = ctx["zi_all"]
    for start in range(0, n_out, int(cand_batch)):
        end = min(start + int(cand_batch), n_out)
        cands = cands_all[start:end]
        A = vectorized_clean_v2_A_for_user(
            int(u), cands, model_train, pop, kg_deg, idx, n_x_sizes=n_x_sizes
        )
        A_s = _scale(A, ctx["a_mean"], ctx["a_scale"])
        H = _h_for_cands(hist, cands, idx, mapping, fold, variant)
        H_s = _scale(H, ctx["h_mean"], ctx["h_scale"])
        zi = zi_all[cands]
        gscore = (zi * zu).sum(dim=-1)
        scores[start:end] = _decode_batch(ctx, zu, zi, gscore, A_s, H_s)
    return scores


@torch.no_grad()
def score_user_oracle(
    u: int,
    bundle: dict[str, Any],
    ctx: dict[str, Any],
    cf,
    mapping,
    folds,
    fmap: dict[int, int],
    variant: str,
    candidates: np.ndarray,
    *,
    n_x_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Scalar/oracle: one candidate at a time. Same formulas, no batching."""
    out = np.empty(int(candidates.size), dtype=np.float64)
    for i, x in enumerate(np.asarray(candidates, dtype=np.int64).tolist()):
        out[i] = float(
            score_user_batched(
                u,
                bundle,
                ctx,
                cf,
                mapping,
                folds,
                fmap,
                variant,
                candidates=np.asarray([int(x)], dtype=np.int64),
                n_x_sizes=n_x_sizes,
                cand_batch=1,
            )[0]
        )
    return out


def _features_for_cands(
    u: int,
    bundle: dict[str, Any],
    ctx: dict[str, Any],
    cf,
    mapping,
    folds,
    fmap: dict[int, int],
    variant: str,
    cands: np.ndarray,
    n_x_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_train = bundle["model_train"]
    idx = clean_v2_index_for(cf, int(u))
    hist = model_train.get(int(u), set())
    fold = folds[int(fmap[int(u)])]
    A = vectorized_clean_v2_A_for_user(
        int(u),
        cands,
        model_train,
        bundle["popularity"],
        bundle["item_kg_degree"],
        idx,
        n_x_sizes=n_x_sizes,
    )
    H = _h_for_cands(hist, cands, idx, mapping, fold, variant)
    return A, H


def numerical_equivalence_check(
    bundle: dict[str, Any],
    ctx: dict[str, Any],
    cf,
    mapping,
    folds,
    fmap: dict[int, int],
    variant: str,
    *,
    n_users: int = 10,
    n_cands: int = 1000,
    seed: int = 20260815,
    cand_batch: int = CAND_BATCH,
) -> dict[str, Any]:
    val_pos = bundle.get("val_pos")
    if val_pos is None:
        from src.lastfm_lp.data.build_splits import load_user_sets
        from pathlib import Path

        val_pos = load_user_sets(
            Path(__file__).resolve().parents[3]
            / "outputs"
            / "lastfm_star"
            / "splits"
            / "valid.txt"
        )
    users = sorted(u for u, items in val_pos.items() if items)
    rng = np.random.default_rng(seed)
    pick_u = rng.choice(np.asarray(users, dtype=np.int64), size=min(n_users, len(users)), replace=False)
    fold_sizes: dict[int, np.ndarray] = {}
    max_feat = 0.0
    max_same_dec = 0.0
    max_gemm = 0.0
    n_checked = 0
    for u in pick_u.tolist():
        idx = clean_v2_index_for(cf, int(u))
        fold = cf.user_to_fold.get(int(u), -1)
        if fold not in fold_sizes:
            fold_sizes[fold] = neighborhood_sizes(idx)
        nx = fold_sizes[fold]
        cands = rng.choice(N_ITEMS, size=int(n_cands), replace=False).astype(np.int64)
        cands.sort()
        A_b, H_b = _features_for_cands(
            int(u), bundle, ctx, cf, mapping, folds, fmap, variant, cands, nx
        )
        A_o = np.zeros_like(A_b)
        H_o = np.zeros_like(H_b)
        for i, x in enumerate(cands.tolist()):
            a1, h1 = _features_for_cands(
                int(u),
                bundle,
                ctx,
                cf,
                mapping,
                folds,
                fmap,
                variant,
                np.asarray([int(x)], dtype=np.int64),
                nx,
            )
            A_o[i] = a1[0]
            H_o[i] = h1[0]
        max_feat = max(
            max_feat,
            float(np.max(np.abs(A_b - A_o))),
            float(np.max(np.abs(H_b - H_o))),
        )
        zu = ctx["zemb"][int(u)]
        zi = ctx["zi_all"][cands]
        gscore = (zi * zu).sum(dim=-1)
        A_s = _scale(A_b, ctx["a_mean"], ctx["a_scale"])
        H_s = _scale(H_b, ctx["h_mean"], ctx["h_scale"])
        A_so = _scale(A_o, ctx["a_mean"], ctx["a_scale"])
        H_so = _scale(H_o, ctx["h_mean"], ctx["h_scale"])
        s_b = _decode_batch(ctx, zu, zi, gscore, A_s, H_s)
        s_o = _decode_batch(ctx, zu, zi, gscore, A_so, H_so)
        max_same_dec = max(max_same_dec, float(np.max(np.abs(s_b - s_o))))
        s_1 = np.empty_like(s_b)
        for i in range(cands.size):
            s_1[i] = _decode_batch(
                ctx,
                zu,
                zi[i : i + 1],
                gscore[i : i + 1],
                A_s[i : i + 1],
                H_s[i : i + 1],
            )[0]
        max_gemm = max(max_gemm, float(np.max(np.abs(s_b - s_1))))
        n_checked += int(cands.size)
    ok = max_feat <= ORACLE_MAX_ABS and max_same_dec <= ORACLE_MAX_ABS
    return {
        "n_users": int(pick_u.size),
        "n_cands_per_user": int(n_cands),
        "n_scores": n_checked,
        "max_abs_diff": max_same_dec,
        "max_abs_features": max_feat,
        "max_abs_decoder_gemm_ulp": max_gemm,
        "threshold": ORACLE_MAX_ABS,
        "cand_batch": int(cand_batch),
        "variant": variant,
        "pass": ok,
    }
