"""Shared helpers for HGT residual-aggregation experiments."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from scripts.run_artist_a11_residual_branch_v1 import scale_split
from scripts.run_hgt_artist_edge_ablation_v1 import abort
from scripts.run_hgt_node_semantic_role_features_v1 import (
    logits_on_idx,
    score_all,
    train_variant,
)
from scripts.run_publication_our_hgt_fullrank import build_fusion_head
from src.lastfm_lp.models.encoders import HGTGraphEncoder
from src.lastfm_lp.models.fusion import RecommendationModel
from src.lastfm_lp.models.hgt_conv_residual import HGTConvAddResidual

HASH_FULL = "3b59bac939fea39fc1ed294b52840b057285efa80a37c008c0d6986a0d421970"
SEEDS = (101, 202, 303)
AUX_MARKERS = ("beta.", "aux_aggr", "aux_proj")


def build_model(
    bundle,
    graph,
    device,
    aux_kind: str | None,
    deg_hist=None,
    message_norm: bool = False,
    hetero_ln: bool = False,
):
    cfg = bundle["cfg"]
    fcfg = cfg.get("fusion", {})
    edge_index_dict = {k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0}
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    dh = deg_hist.to(device) if deg_hist is not None else None
    encoder = HGTGraphEncoder(
        graph["meta"]["n_users"],
        graph["meta"]["n_entities"],
        metadata,
        edge_index_dict,
        embed_dim=64,
        n_layers=2,
        heads=2,
        dropout=0.1,
        entity_role=None,
        aux_kind=aux_kind,
        deg_hist=dh,
        message_norm=message_norm,
        hetero_ln=hetero_ln,
    ).to(device)
    fusion = build_fusion_head(encoder=encoder, hcr_dim=3, use_a11=True, fcfg=fcfg).to(device)
    return RecommendationModel(encoder, fusion).to(device)


def is_shared_key(k: str) -> bool:
    return (not any(m in k for m in AUX_MARKERS)) and ("role_proj" not in k)


def copy_shared(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    s, d = src.state_dict(), dst.state_dict()
    merged = {k: s[k] for k in s if k in d and is_shared_key(k)}
    for k, v in d.items():
        if k not in merged:
            merged[k] = v
    dst.load_state_dict(merged)


def bind_train_out(out_dir: Path) -> None:
    import scripts.run_hgt_node_semantic_role_features_v1 as role_mod

    role_mod.OUT = out_dir


def n_beta(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if ".beta." in n)


def branch_usage(model) -> list[dict[str, Any]]:
    rows = []
    for li, conv in enumerate(model.encoder.core.convs):
        if not isinstance(conv, HGTConvAddResidual):
            continue
        betas, gammas = conv.beta_dict(), conv.gamma_dict()
        extra: dict[str, Any] = {"aux_kind": conv.aux_kind}
        if conv.aux_kind == "softmax" and hasattr(conv.aux_aggr, "t"):
            t = conv.aux_aggr.t
            extra["softmax_t"] = float(t.detach().mean()) if torch.is_tensor(t) else float(t)
        if conv.aux_kind == "powermean" and hasattr(conv.aux_aggr, "p"):
            p = conv.aux_aggr.p
            extra["powermean_p"] = float(p.detach().mean()) if torch.is_tensor(p) else float(p)
        if conv.aux_proj is not None:
            extra["proj_frob"] = float(conv.aux_proj.weight.detach().norm())
        for nt in betas:
            rows.append(
                {
                    "layer": li,
                    "node_type": nt,
                    "beta": betas[nt],
                    "gamma": gammas[nt],
                    **extra,
                }
            )
    return rows


def max_abs_gamma(model) -> float:
    return max((abs(r["gamma"]) for r in branch_usage(model)), default=0.0)


@torch.no_grad()
def stock_equivalence(
    a0,
    a1,
    bundle,
    A_tr_s,
    H_tr_s,
    item_offset,
    device,
    seed: int,
) -> dict[str, float]:
    a0.eval()
    a1.eval()
    z0 = a0.encoder.encode_all()
    z1 = a1.encoder.encode_all(aux_enabled=True)
    emb = float((z0 - z1).abs().max())
    rng = np.random.default_rng(20260816 + seed)
    n_tr = len(bundle["train_pairs"]["label"])
    pick = np.sort(rng.choice(n_tr, size=min(1000, n_tr), replace=False))
    l0 = logits_on_idx(
        a0, bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"],
        A_tr_s, H_tr_s, item_offset, device, pick,
    )
    l1 = logits_on_idx(
        a1, bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"],
        A_tr_s, H_tr_s, item_offset, device, pick, aux_enabled=True,
    )
    logit = float(np.max(np.abs(l0 - l1)))
    return {"max_abs_embedding_diff": emb, "max_abs_logit_diff": logit, "n_pairs": int(pick.size)}


def train_a1_from_a0(
    *,
    bundle,
    graph,
    device,
    aux_kind: str,
    deg_hist,
    seed: int,
    A_tr,
    A_va,
    Hi_tr,
    Hi_va,
    item_offset: int,
    out_dir: Path,
    stage: str,
) -> tuple[Any, np.ndarray, dict[str, float], list[str]]:
    bind_train_out(out_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)
    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(Hi_tr)
    A_tr_s, A_va_s = scale_split(a_scaler, A_tr), scale_split(a_scaler, A_va)
    H_tr_s, H_va_s = scale_split(h_scaler, Hi_tr), scale_split(h_scaler, Hi_va)
    a0 = build_model(bundle, graph, device, None)
    try:
        a1 = build_model(bundle, graph, device, aux_kind, deg_hist)
    except Exception as exc:
        del a0
        raise RuntimeError(f"build failed for {aux_kind}: {exc}") from exc
    copy_shared(a0, a1)
    if n_beta(a1) != 4:
        abort(f"{stage} expected 4 beta scalars, got {n_beta(a1)}")
    for conv in a1.encoder.core.convs:
        if isinstance(conv, HGTConvAddResidual):
            for p in conv.beta.values():
                if float(p.detach().abs().max()) > 0:
                    abort(f"{stage} beta not zero at init")
    eq = stock_equivalence(a0, a1, bundle, A_tr_s, H_tr_s, item_offset, device, seed)
    notes = [f"seed {seed} {stage}: emb={eq['max_abs_embedding_diff']:.3e} logit={eq['max_abs_logit_diff']:.3e}"]
    if eq["max_abs_embedding_diff"] > 2e-7 or eq["max_abs_logit_diff"] > 2e-7:
        del a0, a1
        raise RuntimeError(f"STOCK_EQUIVALENCE_FAIL {notes[-1]}")
    del a0
    gc.collect()
    train_variant(
        bundle, a1, stage=stage, seed=seed,
        A_tr_s=A_tr_s, A_va_s=A_va_s, H_tr_s=H_tr_s, H_va_s=H_va_s,
        a_scaler=a_scaler, h_scaler=h_scaler, item_offset=item_offset, device=device,
    )
    va_u, va_i = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"]
    logits = score_all(a1, va_u, va_i, A_va_s, H_va_s, item_offset, device, aux_enabled=True)
    return a1, logits, eq, notes


def val_scalers(A_tr, A_va, Hi_tr, Hi_va):
    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(Hi_tr)
    return scale_split(a_scaler, A_va), scale_split(h_scaler, Hi_va)


def empty_cache() -> None:
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
