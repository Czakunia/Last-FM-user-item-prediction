"""Stage E — R-GCN / HGT with best decoder (MLP) ± user-conditioned HCR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes
from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.models.hgt_encoder import HGTEncoder
from src.lastfm_lp.models.mlp_decoder import MLPDecoder
from src.lastfm_lp.models.rgcn_encoder import RGCNEncoder
from src.lastfm_lp.pipeline.run_graph_stage import _load_side_features

STAGE_SPECS: dict[str, dict[str, Any]] = {
    "E0": {"encoder": "rgcn", "hcr": False, "tabular_side": True},
    "E1": {"encoder": "rgcn", "hcr": True, "tabular_side": True},
    "E2": {"encoder": "hgt", "hcr": False, "tabular_side": True},
    "E3": {"encoder": "hgt", "hcr": True, "tabular_side": True},
}


def _score_with_encode(
    encode_fn: Callable[[], torch.Tensor],
    decoder: nn.Module,
    users: np.ndarray,
    items: np.ndarray,
    item_offset: int,
    side: np.ndarray | None,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    decoder.eval()
    with torch.no_grad():
        z = encode_fn()
        u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
        scores = []
        for start in range(0, len(users), batch_size):
            sl = slice(start, start + batch_size)
            zu = z[u_idx[sl].to(device)]
            zi = z[i_idx[sl].to(device)]
            s_side = torch.from_numpy(side[sl]).to(device) if side is not None else None
            scores.append(decoder(zu, zi, s_side).detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float64)


def run_stage_e(bundle: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage not in STAGE_SPECS:
        raise ValueError(stage)
    spec = STAGE_SPECS[stage]
    cfg = bundle["cfg"]
    ecfg = cfg.get("models", {}).get("stage_e", {})
    gcfg = cfg.get("models", {}).get("graphsage", {})
    seed = int(cfg["models"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    if spec["encoder"] == "hgt":
        max_kg = ecfg.get("max_kg_edges", 250_000)
        embed_dim = int(ecfg.get("hgt_embed_dim", 32))
    else:
        max_kg = ecfg.get("rgcn_max_kg_edges", None)
        embed_dim = int(ecfg.get("embed_dim", gcfg.get("embed_dim", 64)))
    graph = load_data_and_typed_graph(cfg, bundle["model_train"], max_kg_edges=max_kg)
    meta = graph["meta"]
    item_offset = meta["item_offset"]

    n_layers = int(ecfg.get("n_layers", 2))
    dropout = float(ecfg.get("dropout", 0.1))
    lr = float(ecfg.get("lr", gcfg.get("lr", 1e-3)))
    weight_decay = float(ecfg.get("weight_decay", 1e-4))
    max_epochs = int(ecfg.get("max_epochs", 20))
    patience = int(ecfg.get("patience", 5))

    if spec["encoder"] == "rgcn":
        edge_index = graph["edge_index"].to(device)
        edge_type = graph["edge_type"].to(device)
        encoder: nn.Module = RGCNEncoder(
            meta["n_nodes"],
            meta["n_relations"],
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            n_layers=n_layers,
            dropout=dropout,
            num_bases=int(ecfg.get("num_bases", 8)),
        ).to(device)

        def encode_fn() -> torch.Tensor:
            return encoder(edge_index, edge_type)

    else:
        edge_index_dict = {
            k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0
        }
        # HGT metadata must match present edge types
        metadata = (
            ["user", "entity"],
            list(edge_index_dict.keys()),
        )
        encoder = HGTEncoder(
            meta["n_users"],
            meta["n_entities"],
            metadata=metadata,
            embed_dim=embed_dim,
            n_layers=n_layers,
            heads=int(ecfg.get("heads", 2)),
            dropout=dropout,
        ).to(device)

        def encode_fn() -> torch.Tensor:
            return encoder(edge_index_dict)

    side_pack = _load_side_features(
        cfg, tabular=bool(spec["tabular_side"]), hcr=bool(spec["hcr"])
    )
    side_dim = len(side_pack["names"]) if side_pack else 0
    if side_pack:
        scaler = StandardScaler()
        side_train = scaler.fit_transform(side_pack["train"]).astype(np.float32)
        side_val = scaler.transform(side_pack["val"]).astype(np.float32)
        side_test = scaler.transform(side_pack["test"]).astype(np.float32)
    else:
        side_train = side_val = side_test = None

    mcfg = cfg["models"]["mlp"]
    decoder = MLPDecoder(
        embed_dim, side_dim=side_dim, hidden_dims=mcfg["hidden_dims"], dropout=mcfg["dropout"]
    ).to(device)

    train_y = bundle["train_pairs"]["label"].astype(np.float32)
    u_t, i_t = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    u_t, i_t = u_t.to(device), i_t.to(device)
    y_t = torch.from_numpy(train_y).to(device)
    side_t = torch.from_numpy(side_train).to(device) if side_train is not None else None

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=weight_decay
    )
    n_pos = float((train_y > 0.5).sum())
    n_neg = float((train_y <= 0.5).sum())
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_ndcg = -1.0
    patience_left = patience
    history = []
    k_primary = 20

    for epoch in range(max_epochs):
        encoder.train()
        decoder.train()
        opt.zero_grad()
        z = encode_fn()
        loss = loss_fn(decoder(z[u_t], z[i_t], side_t), y_t)
        loss.backward()
        opt.step()

        val_scores = _score_with_encode(
            encode_fn,
            decoder,
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["item_id"],
            item_offset,
            side_val,
            device,
        )
        val_rank = ranking_metrics_for_users(
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["label"],
            val_scores,
            ks=[k_primary],
        )
        ndcg = float(val_rank.get(f"NDCG@{k_primary}", 0.0))
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_ndcg": ndcg})
        print(f"[{stage}] epoch {epoch} train_loss={loss.item():.4f} val_NDCG@{k_primary}={ndcg:.4f}")
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        encoder.to(device)
        decoder.to(device)

    ks = cfg["evaluation"]["ranking_k"]
    val_scores = _score_with_encode(
        encode_fn,
        decoder,
        bundle["val_pairs"]["user_id"],
        bundle["val_pairs"]["item_id"],
        item_offset,
        side_val,
        device,
    )
    test_scores = _score_with_encode(
        encode_fn,
        decoder,
        bundle["test_pairs"]["user_id"],
        bundle["test_pairs"]["item_id"],
        item_offset,
        side_test,
        device,
    )
    cal = PlattCalibrator().fit(val_scores, bundle["val_pairs"]["label"])
    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, ks)
    val_metrics["Brier_calibrated"] = brier_score(
        bundle["val_pairs"]["label"], cal.transform(val_scores)
    )
    test_metrics["Brier_calibrated"] = brier_score(
        bundle["test_pairs"]["label"], cal.transform(test_scores)
    )

    result = {
        "stage": stage,
        "spec": spec,
        "graph_meta": {k: v for k, v in meta.items() if k != "metadata"},
        "side_features": side_pack["names"] if side_pack else [],
        "train_info": {
            "history": history,
            "best_val_ndcg": best_ndcg,
            "epochs_ran": len(history),
            "n_parameters": int(
                sum(p.numel() for p in encoder.parameters())
                + sum(p.numel() for p in decoder.parameters())
            ),
            "max_kg_edges": max_kg,
            "embed_dim": embed_dim,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
    }
    out = Path(cfg["paths"]["stage_e"]) / stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "test_scores.npy", test_scores)
    print(
        f"[{stage}] test {result['primary_metric']}={result['primary_test']:.4f} "
        f"AUPRC={test_metrics['AUPRC']:.4f} MRR={test_metrics['MRR']:.4f}"
    )
    return result
