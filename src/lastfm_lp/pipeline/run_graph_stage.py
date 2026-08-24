"""Train/eval Stage C/D graph models on frozen pair tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.lastfm_lp.data.build_ckg_graph import load_data_and_graph, user_item_to_nodes
from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.models.graph_sage_encoder import GraphSAGEEncoder, IDEmbeddingEncoder
from src.lastfm_lp.models.kan_decoder import KANDecoder
from src.lastfm_lp.models.mlp_decoder import MLPDecoder

DecoderKind = Literal["dot", "mlp", "kan"]
EncoderKind = Literal["id", "graphsage"]


STAGE_SPECS: dict[str, dict[str, Any]] = {
    "C0": {"encoder": "id", "decoder": "dot", "hcr": False, "tabular_side": False},
    "C1": {"encoder": "graphsage", "decoder": "dot", "hcr": False, "tabular_side": False},
    "C2": {"encoder": "graphsage", "decoder": "mlp", "hcr": False, "tabular_side": True},
    "C3": {"encoder": "graphsage", "decoder": "kan", "hcr": False, "tabular_side": True},
    "D0": {"encoder": "graphsage", "decoder": "mlp", "hcr": False, "tabular_side": True},
    "D2": {"encoder": "graphsage", "decoder": "mlp", "hcr": True, "tabular_side": True},
    "D3": {"encoder": "graphsage", "decoder": "kan", "hcr": False, "tabular_side": True},
    "D5": {"encoder": "graphsage", "decoder": "kan", "hcr": True, "tabular_side": True},
}


class DotDecoder(nn.Module):
    def forward(
        self,
        zu: torch.Tensor,
        zi: torch.Tensor,
        side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (zu * zi).sum(dim=-1)


def _load_side_features(cfg: dict[str, Any], *, tabular: bool, hcr: bool) -> dict[str, Any] | None:
    """Build side feature pack from cached Stage A / B_hcr matrices."""
    feat_root = Path(cfg["paths"]["features"])
    cols: list[tuple[str, Path, list[str] | None]] = []
    # (tag, dir, optional name filter)
    if tabular:
        cols.append(("A", feat_root / "A", None))
    if hcr:
        from src.lastfm_lp.binary.user_hcr_aggregation import BINARY_DEPENDENCE_FEATURES

        cols.append(("B_hcr", feat_root / "B_hcr", list(BINARY_DEPENDENCE_FEATURES)))

    if not cols:
        return None

    pieces_train, pieces_val, pieces_test = [], [], []
    names_out: list[str] = []
    for tag, d, filt in cols:
        names = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))
        Xtr = np.load(d / "X_train.npy")
        Xva = np.load(d / "X_val.npy")
        Xte = np.load(d / "X_test.npy")
        if filt is None:
            idx = list(range(len(names)))
            use_names = names
        else:
            idx = [names.index(n) for n in filt if n in names]
            use_names = [names[i] for i in idx]
        pieces_train.append(Xtr[:, idx])
        pieces_val.append(Xva[:, idx])
        pieces_test.append(Xte[:, idx])
        names_out.extend([f"{tag}:{n}" for n in use_names])

    return {
        "names": names_out,
        "train": np.concatenate(pieces_train, axis=1).astype(np.float32),
        "val": np.concatenate(pieces_val, axis=1).astype(np.float32),
        "test": np.concatenate(pieces_test, axis=1).astype(np.float32),
    }


def _score_pairs(
    encoder: nn.Module,
    decoder: nn.Module,
    edge_index: torch.Tensor,
    users: np.ndarray,
    items: np.ndarray,
    item_offset: int,
    side: np.ndarray | None,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(edge_index.to(device))
        u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
        scores = []
        for start in range(0, len(users), batch_size):
            sl = slice(start, start + batch_size)
            zu = z[u_idx[sl].to(device)]
            zi = z[i_idx[sl].to(device)]
            s_side = None
            if side is not None:
                s_side = torch.from_numpy(side[sl]).to(device)
            scores.append(decoder(zu, zi, s_side).detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float64)


def run_graph_stage(bundle: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage not in STAGE_SPECS:
        raise ValueError(f"Unknown graph stage {stage}")
    spec = STAGE_SPECS[stage]
    cfg = bundle["cfg"]
    gcfg = cfg.get("models", {}).get("graphsage", {})
    seed = int(cfg["models"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    graph = load_data_and_graph(cfg, bundle["model_train"])
    edge_index = graph["edge_index"].to(device)
    meta = graph["meta"]
    item_offset = meta["item_offset"]
    n_nodes = meta["n_nodes"]

    embed_dim = int(gcfg.get("embed_dim", 64))
    hidden_dim = int(gcfg.get("hidden_dim", 64))
    n_layers = int(gcfg.get("n_layers", 2))
    dropout = float(gcfg.get("dropout", 0.1))
    lr = float(gcfg.get("lr", 1e-3))
    weight_decay = float(gcfg.get("weight_decay", 1e-4))
    max_epochs = int(gcfg.get("max_epochs", 25))
    patience = int(gcfg.get("patience", 6))

    if spec["encoder"] == "id":
        encoder: nn.Module = IDEmbeddingEncoder(n_nodes, embed_dim).to(device)
    else:
        encoder = GraphSAGEEncoder(
            n_nodes, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout
        ).to(device)

    side_pack = _load_side_features(
        cfg, tabular=bool(spec.get("tabular_side")), hcr=bool(spec.get("hcr"))
    )
    side_dim = len(side_pack["names"]) if side_pack else 0
    if side_pack:
        side_scaler = StandardScaler()
        side_train = side_scaler.fit_transform(side_pack["train"]).astype(np.float32)
        side_val = side_scaler.transform(side_pack["val"]).astype(np.float32)
        side_test = side_scaler.transform(side_pack["test"]).astype(np.float32)
    else:
        side_train = side_val = side_test = None

    if spec["decoder"] == "dot":
        decoder: nn.Module = DotDecoder().to(device)
        if side_dim > 0:
            raise ValueError("dot decoder does not support side features")
    elif spec["decoder"] == "mlp":
        mcfg = cfg["models"]["mlp"]
        decoder = MLPDecoder(
            embed_dim, side_dim=side_dim, hidden_dims=mcfg["hidden_dims"], dropout=mcfg["dropout"]
        ).to(device)
    else:
        kcfg = cfg["models"]["kan"]
        decoder = KANDecoder(
            embed_dim,
            side_dim=side_dim,
            hidden_dim=kcfg["hidden_dim"],
            dropout=kcfg["dropout"],
            grid_size=kcfg["grid_size"],
            spline_order=kcfg["spline_order"],
        ).to(device)

    train_u = bundle["train_pairs"]["user_id"]
    train_i = bundle["train_pairs"]["item_id"]
    train_y = bundle["train_pairs"]["label"].astype(np.float32)
    val_u = bundle["val_pairs"]["user_id"]
    val_i = bundle["val_pairs"]["item_id"]
    val_y = bundle["val_pairs"]["label"].astype(np.float32)

    u_t, i_t = user_item_to_nodes(train_u, train_i, item_offset)
    u_t, i_t = u_t.to(device), i_t.to(device)
    y_t = torch.from_numpy(train_y).to(device)
    side_t = torch.from_numpy(side_train).to(device) if side_train is not None else None

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    n_pos = float((train_y > 0.5).sum())
    n_neg = float((train_y <= 0.5).sum())
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_ndcg = -1.0
    patience_left = patience
    history = []
    primary = cfg["evaluation"]["primary_metric"]
    k_primary = int(str(primary).split("@")[-1]) if "@" in str(primary) else 20

    for epoch in range(max_epochs):
        encoder.train()
        decoder.train()
        opt.zero_grad()
        z = encoder(edge_index)
        logits = decoder(z[u_t], z[i_t], side_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()

        val_scores = _score_pairs(
            encoder, decoder, edge_index, val_u, val_i, item_offset, side_val, device
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
    val_scores = _score_pairs(
        encoder, decoder, edge_index, val_u, val_i, item_offset, side_val, device
    )
    test_scores = _score_pairs(
        encoder,
        decoder,
        edge_index,
        bundle["test_pairs"]["user_id"],
        bundle["test_pairs"]["item_id"],
        item_offset,
        side_test,
        device,
    )
    cal = PlattCalibrator().fit(val_scores, bundle["val_pairs"]["label"])
    val_probs = cal.transform(val_scores)
    test_probs = cal.transform(test_scores)

    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, ks)
    val_metrics["Brier_calibrated"] = brier_score(bundle["val_pairs"]["label"], val_probs)
    test_metrics["Brier_calibrated"] = brier_score(bundle["test_pairs"]["label"], test_probs)

    n_params = int(
        sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    )
    result = {
        "stage": stage,
        "spec": spec,
        "graph_meta": meta,
        "side_features": side_pack["names"] if side_pack else [],
        "train_info": {
            "history": history,
            "best_val_ndcg": best_ndcg,
            "epochs_ran": len(history),
            "n_parameters": n_params,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
    }

    out_root = Path(cfg["paths"]["stage_c"] if stage.startswith("C") else cfg["paths"]["stage_d"])
    out = out_root / stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "test_scores.npy", test_scores)
    print(
        f"[{stage}] test {cfg['evaluation']['primary_metric']}="
        f"{result['primary_test']:.4f}  AUPRC={test_metrics['AUPRC']:.4f}  "
        f"MRR={test_metrics['MRR']:.4f}"
    )
    return result
