"""Evaluate a scorer and package metrics for a split."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.lastfm_lp.evaluation.binary_metrics import binary_metrics
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users


def evaluate_split(
    pairs: dict[str, np.ndarray],
    scores: np.ndarray,
    ks: list[int],
) -> dict[str, float]:
    y = pairs["label"]
    u = pairs["user_id"]
    out = {}
    out.update(binary_metrics(y, scores))
    out.update(ranking_metrics_for_users(u, y, scores, ks=ks))
    return out


def delta(a: dict[str, float], b: dict[str, float], keys: list[str]) -> dict[str, float]:
    return {k: float(a.get(k, np.nan) - b.get(k, np.nan)) for k in keys}
