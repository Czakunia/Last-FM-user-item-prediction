"""Pointwise classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int)
    s = scores.astype(float)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    if len(np.unique(y)) < 2:
        out.update({"AUPRC": float("nan"), "ROC_AUC": float("nan")})
        return out
    out["AUPRC"] = float(average_precision_score(y, s))
    out["ROC_AUC"] = float(roc_auc_score(y, s))
    return out
