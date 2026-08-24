"""Frozen constants for native-measure × architecture screen."""

from __future__ import annotations

from typing import Literal

MeasureName = Literal["a11", "mi", "cosine"]
ArchitectureName = Literal["flat", "branch"]

MEASURES: tuple[MeasureName, ...] = ("a11", "mi", "cosine")
ARCHITECTURES: tuple[ArchitectureName, ...] = ("flat", "branch")

SCENARIOS: tuple[str, ...] = (
    "A11_FLAT",
    "A11_BRANCH",
    "MI_FLAT",
    "MI_BRANCH",
    "COS_FLAT",
    "COS_BRANCH",
)

SCENARIO_SPEC: dict[str, dict[str, object]] = {
    "A11_FLAT": {"measure": "a11", "architecture": "flat", "encoding_dim": 8},
    "A11_BRANCH": {"measure": "a11", "architecture": "branch", "encoding_dim": 8},
    "MI_FLAT": {"measure": "mi", "architecture": "flat", "encoding_dim": 7},
    "MI_BRANCH": {"measure": "mi", "architecture": "branch", "encoding_dim": 7},
    "COS_FLAT": {"measure": "cosine", "architecture": "flat", "encoding_dim": 7},
    "COS_BRANCH": {"measure": "cosine", "architecture": "branch", "encoding_dim": 7},
}

# Map measure → clean_v2 routing policy name (Top25 selection only)
MEASURE_TO_ROUTING = {
    "a11": "a11_top25",
    "mi": "mi_top25",
    "cosine": "cosine_top25",
}

A11_8D_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "top3_mean",
    "bottom3_mean",
    "positive_fraction",
    "negative_fraction",
]
NONNEG_7D_NAMES = [
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "max",
    "top3_mean",
]

A11_DIM = 8
NONNEG_DIM = 7
TABULAR_A_DIM = 5
GRAPH_CTX_DIM = 256
GRAPH_SCORE_DIM = 1
FLAT_A11_IN = GRAPH_CTX_DIM + GRAPH_SCORE_DIM + TABULAR_A_DIM + A11_DIM  # 270
FLAT_NONNEG_IN = GRAPH_CTX_DIM + GRAPH_SCORE_DIM + TABULAR_A_DIM + NONNEG_DIM  # 269
BRANCH_FUSION_IN = 64 + 16 + 16 + 1  # 97
TOP_K = 25
SEEDS = (101, 202, 303)
FOLD_SEED = 2026
N_FOLDS = 5
EPS = 1e-8
SCALE_EPS = 1e-8
