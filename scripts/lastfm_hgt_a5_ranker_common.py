"""Ablation rankers: HGT-only / HGT+A5 / HGT+A5+H3. No LEG residual."""

from __future__ import annotations

from torch import Tensor, nn

from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    GRAPH_CTX,
    build_race_model,
)
from src.lastfm_lp.models.fusion import LateFusionHead  # noqa: E402

ARCH_NAME = "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5_NO_H3_NO_LEG"
DECODER_INPUT_DIM = 262  # 256 + 1 + 5
ARCH_NAME_HGT_ONLY = "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1_NO_A5_NO_H3_NO_LEG"
DECODER_INPUT_DIM_HGT_ONLY = 257  # 256 + 1
ARCH_NAME_H3 = "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3_NO_LEG"
DECODER_INPUT_DIM_H3 = 265  # 256 + 1 + 5 + 3


class ZeroLeg(nn.Module):
    """API-compatible stub: δ = 0, no parameters."""

    def forward(self, r: Tensor) -> Tensor:
        if r.ndim == 1:
            return r * 0
        return r.squeeze(-1) * 0


def build_hgt_a5_ranker(bundle, graph, device, *, d: int = 64, layers: int = 2, heads: int = 2):
    """Fresh HGT + 262-D ranker. H3/LEG tensors may still be passed; they are ignored."""
    model = build_race_model(bundle, graph, device, d=d, layers=layers, heads=heads)
    cfg = bundle["cfg"]
    fcfg = cfg.get("fusion", {})
    model.fusion_head = LateFusionHead(
        graph_dim=GRAPH_CTX,
        base_feature_dim=5,
        hcr_feature_dim=3,
        hidden_dim=int(fcfg.get("hidden_dim", 128)),
        dropout=float(fcfg.get("dropout", 0.2)),
        use_hcr=False,
        activation=str(fcfg.get("activation", "gelu")),
    ).to(device)
    model.leg = ZeroLeg().to(device)
    return model


def build_hgt_only_ranker(bundle, graph, device, *, d: int = 64, layers: int = 2, heads: int = 2):
    """Fresh HGT + 257-D ranker. A5/H3/LEG ignored (empty A, zero H/L2)."""
    model = build_race_model(bundle, graph, device, d=d, layers=layers, heads=heads)
    cfg = bundle["cfg"]
    fcfg = cfg.get("fusion", {})
    model.fusion_head = LateFusionHead(
        graph_dim=GRAPH_CTX,
        base_feature_dim=0,
        hcr_feature_dim=3,
        hidden_dim=int(fcfg.get("hidden_dim", 128)),
        dropout=float(fcfg.get("dropout", 0.2)),
        use_hcr=False,
        activation=str(fcfg.get("activation", "gelu")),
    ).to(device)
    model.leg = ZeroLeg().to(device)
    return model


def build_hgt_a5_h3_noleg(bundle, graph, device, *, d: int = 64, layers: int = 2, heads: int = 2):
    """Fresh HGT + 265-D ranker with H3. LEG stub is identically zero."""
    model = build_race_model(bundle, graph, device, d=d, layers=layers, heads=heads)
    model.leg = ZeroLeg().to(device)
    return model
