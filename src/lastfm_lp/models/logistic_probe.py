"""A0 / B0 logistic regression probe."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticProbe:
    def __init__(self, C: float = 1.0, max_iter: int = 200, seed: int = 2026) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y.astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1].astype(np.float64)

    def n_parameters(self) -> int:
        return int(self.model.coef_.size + self.model.intercept_.size)
