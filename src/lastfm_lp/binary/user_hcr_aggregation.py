"""User-conditioned empirical association aggregates over history H_u.

Legacy Stage B–F features use LEGACY_BINARY_ASSOC_V1 (= a11² + MI) under the
historical column names ``hcr_history_*`` so cached matrices stay valid.

Stage H / V2 uses compact blocks + explicit pooling of orthonormal HCR a11.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from src.lastfm_lp.binary.binary_measures import COMPACT8_ORDER, COMPACT10_ORDER
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex

BlockKind = Literal["legacy_scalar", "a11", "a11_energy", "compact8", "compact10"]

# Historical B-stage feature names (values = legacy_binary_assoc aggregates).
BINARY_DEPENDENCE_FEATURES = [
    "hcr_history_mean",
    "hcr_history_max",
    "hcr_history_top3_mean",
    "hcr_history_top5_mean",
    "hcr_history_weighted_mean",
    "hcr_positive_fraction",
    "hcr_support_sum",
    "conditional_probability_max",
    "lift_max",
    "odds_ratio_max",
    "npmi_mean",
]


def _truncate_history(history_items: Iterable[int], max_history: int) -> list[int]:
    hist = list(history_items)
    if len(hist) > max_history:
        hist = sorted(hist)[:max_history]
    return hist


def aggregate_user_hcr(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    max_history: int = 50,
) -> dict[str, float]:
    """Legacy B-stage aggregation over LEGACY_BINARY_ASSOC_V1."""

    hist = _truncate_history(history_items, max_history)
    if not hist:
        return {k: 0.0 for k in BINARY_DEPENDENCE_FEATURES}

    legacy = []
    conds = []
    lifts = []
    odds = []
    npmis = []
    weights = []
    support_sum = 0.0

    for j in hist:
        m = index.pair(int(j), int(candidate_item))
        legacy.append(m.legacy_binary_assoc)
        conds.append(m.conditional_y_given_x)
        lifts.append(m.lift)
        odds.append(m.odds_ratio)
        npmis.append(m.npmi)
        w = float(m.support)
        weights.append(w)
        support_sum += m.n11

    h = np.asarray(legacy, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(-h)

    def topk_mean(k: int) -> float:
        k = min(k, len(h))
        return float(h[order[:k]].mean()) if k else 0.0

    wsum = float(w.sum())
    weighted = float((w * h).sum() / wsum) if wsum > 0 else float(h.mean())

    return {
        "hcr_history_mean": float(h.mean()),
        "hcr_history_max": float(h.max()),
        "hcr_history_top3_mean": topk_mean(3),
        "hcr_history_top5_mean": topk_mean(5),
        "hcr_history_weighted_mean": weighted,
        "hcr_positive_fraction": float((h > 0).mean()),
        "hcr_support_sum": float(support_sum),
        "conditional_probability_max": float(np.max(conds)),
        "lift_max": float(np.max(lifts)),
        "odds_ratio_max": float(np.max(odds)),
        "npmi_mean": float(np.mean(npmis)),
    }


def history_pair_blocks(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    max_history: int = 25,
    kind: BlockKind = "compact8",
    pair_fn=None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (B[k,d], mask[k], history_ids) for Stage-H style aggregation.

    ``pair_fn(j, i)`` optional override returning BinaryPairMeasures
    (used with cross-fit / shuffled ablations).
    """

    hist = _truncate_history(history_items, max_history)
    if kind == "legacy_scalar":
        dim = 1
    elif kind == "a11":
        dim = 1
    elif kind == "a11_energy":
        dim = 2
    elif kind == "compact8":
        dim = len(COMPACT8_ORDER)
    elif kind == "compact10":
        dim = len(COMPACT10_ORDER)
    else:
        raise ValueError(kind)

    B = np.zeros((max_history, dim), dtype=np.float32)
    mask = np.zeros(max_history, dtype=np.float32)
    ids = [-1] * max_history
    if not hist:
        return B, mask, ids

    lookup = pair_fn if pair_fn is not None else (lambda j, i: index.pair(j, i))
    for t, j in enumerate(hist):
        m = lookup(int(j), int(candidate_item))
        if kind == "legacy_scalar":
            vec = np.asarray([m.legacy_binary_assoc], dtype=np.float32)
        elif kind == "a11":
            vec = np.asarray([m.hcr_a11], dtype=np.float32)
        elif kind == "a11_energy":
            vec = np.asarray([m.hcr_a11, m.hcr_energy], dtype=np.float32)
        elif kind == "compact8":
            vec = m.compact8()
        else:
            vec = m.compact10()
        B[t] = vec
        # Presence mask for history slot; pair validity is a compact channel / field.
        mask[t] = 1.0
        ids[t] = int(j)
    return B, mask, ids


def pool_history_blocks(
    B: np.ndarray,
    mask: np.ndarray,
    *,
    support_weights: np.ndarray | None = None,
    top_k: int = 3,
) -> np.ndarray:
    """Channel-wise [mean, max, topk_mean, support_weighted_mean] → 4*d vector."""

    active = mask > 0
    d = B.shape[1]
    out = np.zeros(4 * d, dtype=np.float32)
    if not active.any():
        return out
    X = B[active]
    # mean / max
    out[0:d] = X.mean(axis=0)
    out[d : 2 * d] = X.max(axis=0)
    # top-k by first channel (a11 or legacy) mean of rows
    key = X[:, 0]
    order = np.argsort(-key)
    k = min(int(top_k), len(order))
    out[2 * d : 3 * d] = X[order[:k]].mean(axis=0)
    # support-weighted mean
    if support_weights is None:
        # compact8/10: support is channel index 6 / 7
        if d >= 8:
            w = np.maximum(X[:, 6 if d == 8 else 7], 0.0)
        else:
            w = np.ones(len(X), dtype=np.float64)
    else:
        w = np.asarray(support_weights[active], dtype=np.float64)
    wsum = float(w.sum())
    if wsum > 0:
        out[3 * d : 4 * d] = (X * w[:, None]).sum(axis=0) / wsum
    else:
        out[3 * d : 4 * d] = out[0:d]
    return out


def aggregate_block_raw(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    max_history: int = 25,
    kind: BlockKind = "compact8",
    pair_fn=None,
) -> np.ndarray:
    B, mask, _ = history_pair_blocks(
        history_items,
        candidate_item,
        index,
        max_history=max_history,
        kind=kind,
        pair_fn=pair_fn,
    )
    return pool_history_blocks(B, mask)


def pool_a11_energy_matrix(
    a11: np.ndarray,
    energy: np.ndarray,
    *,
    kind: BlockKind = "a11_energy",
    top_k: int = 3,
) -> np.ndarray:
    """Pool H×C a11/energy → (C, 4*d) float32 matching ``pool_history_blocks``.

    Casts to float32 *before* reductions (same as filling ``B`` then pooling).
    For ``a11_energy`` (d=2), wmean uses uniform weights → identical to mean.
    """

    a11 = np.asarray(a11)
    energy = np.asarray(energy)
    if a11.ndim != 2 or energy.shape != a11.shape:
        raise ValueError("a11/energy must be H×C with matching shapes")
    H, C = a11.shape
    d = 1 if kind == "a11" else 2
    out = np.zeros((C, 4 * d), dtype=np.float32)
    if H == 0:
        return out
    a = a11.astype(np.float32, copy=False)
    if kind == "a11":
        # single channel
        mean_a = a.mean(axis=0)
        max_a = a.max(axis=0)
        k = min(int(top_k), H)
        order = np.argsort(-a, axis=0)
        top3_a = np.take_along_axis(a, order[:k, :], axis=0).mean(axis=0)
        out[:, 0] = mean_a
        out[:, 1] = max_a
        out[:, 2] = top3_a
        out[:, 3] = mean_a  # wmean ≡ mean for d < 8
        return out

    e = energy.astype(np.float32, copy=False)
    mean_a = a.mean(axis=0)
    mean_e = e.mean(axis=0)
    max_a = a.max(axis=0)
    max_e = e.max(axis=0)
    k = min(int(top_k), H)
    order = np.argsort(-a, axis=0)  # top-k by a11 channel (same as pool_history_blocks)
    top3_a = np.take_along_axis(a, order[:k, :], axis=0).mean(axis=0)
    top3_e = np.take_along_axis(e, order[:k, :], axis=0).mean(axis=0)
    out[:, 0] = mean_a
    out[:, 1] = mean_e
    out[:, 2] = max_a
    out[:, 3] = max_e
    out[:, 4] = top3_a
    out[:, 5] = top3_e
    out[:, 6] = mean_a  # wmean with ones
    out[:, 7] = mean_e
    return out


def aggregate_a11_energy_user_batch(
    history_items: Iterable[int],
    candidate_items: np.ndarray | Iterable[int],
    index: PairwiseStatsIndex,
    *,
    max_history: int = 25,
    kind: BlockKind = "a11_energy",
    drop_a11: bool = False,
    shuffle_a11: dict[tuple[int, int], float] | None = None,
) -> np.ndarray:
    """FAST V2: one history × many candidates → (n_candidates, 4*d) float32."""

    if kind not in {"a11", "a11_energy", "legacy_scalar"}:
        raise ValueError(f"user batch only supports a11/a11_energy/legacy_scalar, got {kind}")
    hist = _truncate_history(history_items, max_history)
    cands = np.asarray(list(candidate_items) if not isinstance(candidate_items, np.ndarray) else candidate_items, dtype=np.int64).reshape(-1)
    n_out = int(cands.size)
    d = 1 if kind in {"a11", "legacy_scalar"} else 2
    if n_out == 0:
        return np.zeros((0, 4 * d), dtype=np.float32)
    if not hist:
        return np.zeros((n_out, 4 * d), dtype=np.float32)

    js = np.asarray(hist, dtype=np.int64)
    if kind == "legacy_scalar":
        legacy = index.legacy_assoc_block(js, cands)
        if shuffle_a11 is not None:
            for hi, j in enumerate(js.tolist()):
                for ci, i in enumerate(cands.tolist()):
                    key = (j, i) if j <= i else (i, j)
                    if key in shuffle_a11:
                        legacy[hi, ci] = float(shuffle_a11[key])
        if drop_a11:
            legacy = np.zeros_like(legacy)
        # Pool as single-channel (same layout as a11).
        return pool_a11_energy_matrix(legacy, np.zeros_like(legacy), kind="a11")

    a11, energy = index.a11_energy_block(js, cands)
    if shuffle_a11 is not None:
        for hi, j in enumerate(js.tolist()):
            for ci, i in enumerate(cands.tolist()):
                key = (j, i) if j <= i else (i, j)
                if key in shuffle_a11:
                    aa = float(shuffle_a11[key])
                    a11[hi, ci] = aa
                    energy[hi, ci] = aa * aa
    if drop_a11:
        a11 = np.zeros_like(a11)
        energy = np.zeros_like(energy)
    return pool_a11_energy_matrix(a11, energy, kind=kind)


def aggregate_a11_energy_vectorized(
    history_items: Iterable[int],
    candidate_item: int,
    index: PairwiseStatsIndex,
    *,
    max_history: int = 25,
    kind: BlockKind = "a11_energy",
    drop_a11: bool = False,
    shuffle_a11: dict[tuple[int, int], float] | None = None,
    row_cols: np.ndarray | None = None,
    row_data: np.ndarray | None = None,
) -> np.ndarray:
    """Fast path for ``a11`` / ``a11_energy``: batch n11 + vector a11, same pooling.

    Output matches ``aggregate_block_raw`` with scalar ``index.pair`` lookups for
    these kinds (float32 pooled features via ``pool_history_blocks``).
    """

    if kind not in {"a11", "a11_energy", "legacy_scalar"}:
        raise ValueError(f"vectorized path only supports a11/a11_energy/legacy_scalar, got {kind}")
    hist = _truncate_history(history_items, max_history)
    d = 1 if kind in {"a11", "legacy_scalar"} else 2
    B = np.zeros((max_history, d), dtype=np.float32)
    mask = np.zeros(max_history, dtype=np.float32)
    if not hist:
        return pool_history_blocks(B, mask)

    js = np.asarray(hist, dtype=np.int64)
    if kind == "legacy_scalar":
        legacy = index.legacy_assoc_vs_history(
            js,
            int(candidate_item),
            row_cols=row_cols,
            row_data=row_data,
        )
        if shuffle_a11 is not None:
            cand = int(candidate_item)
            for t, j in enumerate(js.tolist()):
                key = (j, cand) if j <= cand else (cand, j)
                if key in shuffle_a11:
                    legacy[t] = float(shuffle_a11[key])
        if drop_a11:
            legacy = np.zeros_like(legacy)
        k = int(js.shape[0])
        B[:k, 0] = legacy.astype(np.float32)
        mask[:k] = 1.0
        return pool_history_blocks(B, mask)

    a11, energy = index.a11_energy_vs_history(
        js,
        int(candidate_item),
        row_cols=row_cols,
        row_data=row_data,
    )
    if shuffle_a11 is not None:
        cand = int(candidate_item)
        for t, j in enumerate(js.tolist()):
            key = (j, cand) if j <= cand else (cand, j)
            if key in shuffle_a11:
                a = float(shuffle_a11[key])
                a11[t] = a
                energy[t] = a * a
    if drop_a11:
        a11 = np.zeros_like(a11)
        energy = np.zeros_like(energy)

    k = int(js.shape[0])
    B[:k, 0] = a11.astype(np.float32)
    if kind == "a11_energy":
        B[:k, 1] = energy.astype(np.float32)
    mask[:k] = 1.0
    return pool_history_blocks(B, mask)


def raw_pool_feature_names(kind: BlockKind) -> list[str]:
    if kind == "legacy_scalar":
        base = ["legacy_assoc"]
    elif kind == "a11":
        base = ["hcr_a11"]
    elif kind == "a11_energy":
        base = ["hcr_a11", "hcr_energy"]
    elif kind == "compact8":
        base = list(COMPACT8_ORDER)
    else:
        base = list(COMPACT10_ORDER)
    prefixes = ["mean", "max", "top3mean", "wmean"]
    return [f"{p}:{c}" for p in prefixes for c in base]
