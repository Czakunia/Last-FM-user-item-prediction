"""Overtime V2 experimental A11 pooling — does NOT change canonical a11 math.

Operates on raw HxC signed a11 from PairwiseStatsIndex.a11_energy_block.
Writes caches under outputs/lastfm_overtime_v2/features/ only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from src.lastfm_lp.binary.user_hcr_aggregation import _truncate_history
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit

NEAR_ZERO = 0.05

# Canonical experimental blocks (exact feature order — frozen for this session)
P1_NAMES = [
    "POS_MAX",
    "POS_TOP3_MEAN",
    "POS_MEAN_NONZERO",
    "POS_COUNT",
    "POS_FRACTION",
]
P2_NAMES = ["POS_MAX"]
P3_NAMES = ["POS_MAX", "POS_TOP3_MEAN", "POS_FRACTION"]
P4_NAMES = [
    "pos_max",
    "pos_top3mean",
    "pos_fraction",
    "neg_strength_max",
    "neg_top3mean",
    "neg_fraction",
]

SCHEME_NAMES: dict[str, list[str]] = {
    "P1_POSITIVE_ONLY": P1_NAMES,
    "P2_POSITIVE_MAX": P2_NAMES,
    "P3_POSITIVE_TOPK": P3_NAMES,
    "P4_POS_NEG_SPLIT": P4_NAMES,
}


def _topk_mean_sorted_desc(vals: np.ndarray, k: int) -> float:
    if vals.size == 0:
        return 0.0
    k = min(k, int(vals.size))
    # vals already positive strengths
    part = np.partition(vals, -k)[-k:]
    return float(part.mean())


def pool_column_p1(a11_col: np.ndarray) -> np.ndarray:
    """a11_col: (H,) signed → P1 features."""
    h = a11_col.astype(np.float64).reshape(-1)
    n = float(h.size) if h.size else 1.0
    pos = h[h > 0]
    out = np.zeros(5, dtype=np.float32)
    if pos.size:
        out[0] = float(pos.max())
        out[1] = _topk_mean_sorted_desc(pos, 3)
        out[2] = float(pos.mean())
        out[3] = float(pos.size)
        out[4] = float(pos.size) / n
    return out


def pool_column_p2(a11_col: np.ndarray) -> np.ndarray:
    h = a11_col.astype(np.float64).reshape(-1)
    pos = h[h > 0]
    return np.asarray([float(pos.max()) if pos.size else 0.0], dtype=np.float32)


def pool_column_p3(a11_col: np.ndarray) -> np.ndarray:
    h = a11_col.astype(np.float64).reshape(-1)
    n = float(h.size) if h.size else 1.0
    pos = h[h > 0]
    out = np.zeros(3, dtype=np.float32)
    if pos.size:
        out[0] = float(pos.max())
        out[1] = _topk_mean_sorted_desc(pos, 3)
        out[2] = float(pos.size) / n
    return out


def pool_column_p4(a11_col: np.ndarray) -> np.ndarray:
    h = a11_col.astype(np.float64).reshape(-1)
    n = float(h.size) if h.size else 1.0
    pos = h[h > 0]
    neg = h[h < 0]
    out = np.zeros(6, dtype=np.float32)
    if pos.size:
        out[0] = float(pos.max())
        out[1] = _topk_mean_sorted_desc(pos, 3)
        out[2] = float(pos.size) / n
    if neg.size:
        strength = -neg  # positive magnitudes
        out[3] = float(strength.max())
        out[4] = _topk_mean_sorted_desc(strength, 3)
        out[5] = float(neg.size) / n
    return out


POOLERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "P1_POSITIVE_ONLY": pool_column_p1,
    "P2_POSITIVE_MAX": pool_column_p2,
    "P3_POSITIVE_TOPK": pool_column_p3,
    "P4_POS_NEG_SPLIT": pool_column_p4,
}


def audit_stats_from_vector(a11: np.ndarray) -> dict[str, float]:
    h = np.asarray(a11, dtype=np.float64).reshape(-1)
    if h.size == 0:
        return {
            "history_length": 0.0,
            "mean_signed": 0.0,
            "max_positive": 0.0,
            "min_negative": 0.0,
            "max_abs": 0.0,
            "mean_positive": 0.0,
            "mean_negative": 0.0,
            "top1_positive": 0.0,
            "top3_positive_mean": 0.0,
            "top1_negative_abs": 0.0,
            "top3_negative_abs_mean": 0.0,
            "fraction_positive": 0.0,
            "fraction_negative": 0.0,
            "fraction_near_zero": 0.0,
        }
    pos = h[h > 0]
    neg = h[h < 0]
    abs_h = np.abs(h)
    return {
        "history_length": float(h.size),
        "mean_signed": float(h.mean()),
        "max_positive": float(pos.max()) if pos.size else 0.0,
        "min_negative": float(neg.min()) if neg.size else 0.0,
        "max_abs": float(abs_h.max()),
        "mean_positive": float(pos.mean()) if pos.size else 0.0,
        "mean_negative": float(neg.mean()) if neg.size else 0.0,
        "top1_positive": float(pos.max()) if pos.size else 0.0,
        "top3_positive_mean": _topk_mean_sorted_desc(pos, 3) if pos.size else 0.0,
        "top1_negative_abs": float((-neg).max()) if neg.size else 0.0,
        "top3_negative_abs_mean": _topk_mean_sorted_desc(-neg, 3) if neg.size else 0.0,
        "fraction_positive": float((h > 0).mean()),
        "fraction_negative": float((h < 0).mean()),
        "fraction_near_zero": float((abs_h < NEAR_ZERO).mean()),
    }


def materialize_scheme(
    bundle: dict[str, Any],
    scheme: str,
    out_dir: Path,
    *,
    max_history: int = 25,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Path:
    """Materialize pooled experimental features for train/val/test pair tables."""
    if scheme not in POOLERS:
        raise KeyError(scheme)
    pooler = POOLERS[scheme]
    names = SCHEME_NAMES[scheme]
    dim = len(names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")

    cf = ensure_cross_fit(bundle)
    model_train = bundle["model_train"]

    for split in splits:
        out_x = out_dir / f"X_{split}.npy"
        if out_x.exists():
            print(f"[{scheme}] reuse {out_x}", flush=True)
            continue
        pairs = bundle[f"{split}_pairs"]
        users = pairs["user_id"].astype(np.int64)
        items = pairs["item_id"].astype(np.int64)
        n = len(users)
        X = np.zeros((n, dim), dtype=np.float32)
        by_u: dict[int, list[int]] = defaultdict(list)
        for idx, u in enumerate(users.tolist()):
            by_u[int(u)].append(idx)
        for u, idxs in tqdm(by_u.items(), desc=f"{scheme}:{split}", leave=False):
            pw = cf.index_for(int(u), split=split)
            hist = _truncate_history(model_train.get(int(u), ()), max_history)
            if not hist:
                continue
            cands = items[idxs]
            a11, _ = pw.a11_energy_block(np.asarray(hist, dtype=np.int64), cands)
            for local, row_i in enumerate(idxs):
                X[row_i] = pooler(a11[:, local])
        np.save(out_x, X)
        print(f"[{scheme}] wrote {out_x} shape={X.shape}", flush=True)
    needed = [f"X_{s}.npy" for s in splits]
    if all((out_dir / f).exists() for f in needed):
        (out_dir / "COMPLETE").write_text("1\n", encoding="utf-8")
    return out_dir
