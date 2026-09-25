"""Paths for TRUE FINAL hardneg joint retrain."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[2]
# Optional overrides for hparam cells (must not clobber primary hardneg ART)
ART = Path(os.environ["TFHN_ART"]) if os.environ.get("TFHN_ART") else (EXP / "artifacts")
REP = Path(os.environ["TFHN_REP"]) if os.environ.get("TFHN_REP") else (EXP / "reports")
LOG = EXP / "logs"
SPLITS_DIR = ROOT / "outputs" / "lastfm_star" / "splits"
# Local copy (do not depend on lastfm_rank_v2 tree)
NEIGHBORS_SRC = EXP / "artifacts" / "item_a11_neighbors.npz"
# Official JOINT-compatible tree (does NOT overwrite original JOINT_TRAINING_V1)
_JOINT_DEFAULT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HARDNEG_R3_FROM_SCRATCH_V1"
JOINT_OUT = Path(os.environ["TFHN_JOINT_OUT"]) if os.environ.get("TFHN_JOINT_OUT") else _JOINT_DEFAULT
FROZEN_JOINT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_TRAINING_V1"
B0_SEED303 = {"NDCG@20": 0.271886, "Recall@20": 0.376197, "MRR": 0.343391}
B0_MEAN = {"NDCG@20": 0.2764, "Recall@20": 0.3768}
R3_REF = {"NDCG@20": 0.2868}
RP3BETA_DEV = {"NDCG@20": 0.3227}
HP_DONE_FLAG = ROOT / "experiments" / "lastfm_m1_final_v1" / "artifacts" / "AWAITING_GO_EXTERNAL.flag"


def resolve_hgt_dims() -> tuple[int, int, int]:
    """HGT width/depth/heads from env (defaults = official hardneg claim)."""
    d = int(os.environ.get("TFHN_D", "64"))
    layers = int(os.environ.get("TFHN_LAYERS", "2"))
    heads = int(os.environ.get("TFHN_HEADS", "2"))
    if d % heads != 0:
        raise ValueError(f"TFHN_D={d} must be divisible by TFHN_HEADS={heads}")
    return d, layers, heads


def resolve_arch_name(d: int | None = None, layers: int | None = None, heads: int | None = None) -> str:
    if d is None or layers is None or heads is None:
        d, layers, heads = resolve_hgt_dims()
    return f"HGT_{d}D_{layers}L_{heads}H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL"


ARCH = resolve_arch_name()
