"""Item-level features from training statistics only."""

from __future__ import annotations

import numpy as np


def log1p_popularity(popularity: np.ndarray, item_id: int) -> float:
    return float(np.log1p(popularity[item_id]))


def log1p_kg_degree(item_kg_degree: np.ndarray, item_id: int) -> float:
    return float(np.log1p(item_kg_degree[item_id]))
