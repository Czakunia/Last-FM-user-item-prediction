"""Probability calibration helpers (Platt on validation)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=200)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        X = scores.reshape(-1, 1)
        self.model.fit(X, y.astype(int))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        X = scores.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1].astype(np.float64)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y_true.astype(float)) ** 2))
