"""Architecture benchmark with frozen H2 HCR (LASTFM_ARCHITECTURE_HCR_V1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.lastfm_lp.data.build_ckg_graph import (
    build_ckg_edge_index,
    load_data_and_typed_graph,
    user_item_to_nodes,
)
from src.lastfm_lp.data.load_kgat_lastfm import load_lastfm
from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.features.hcr_h2_contract import (
    load_base_and_hcr,
    load_h2_contract,
    shuffle_hcr_block,
)
from src.lastfm_lp.models.encoders import (
    HGTGraphEncoder,
    KGATEncoder,
    LightGCNEncoder,
    RGCNGraphEncoder,
)
from src.lastfm_lp.models.fusion import KANFusionHead, LateFusionHead, RecommendationModel

# Base encoder triplets + Stage-G fusion variants (HCR still frozen H2).
STAGE_SPECS: dict[str, dict[str, Any]] = {
    "HGT_A_ONLY": {"encoder": "hgt", "hcr": False, "shuffle": "none"},
    "HGT_H2": {"encoder": "hgt", "hcr": True, "shuffle": "none"},
    "HGT_H2_SHUFFLED": {"encoder": "hgt", "hcr": True, "shuffle": "popularity_stratified"},
    "KGAT_A_ONLY": {"encoder": "kgat", "hcr": False, "shuffle": "none"},
    "KGAT_H2": {"encoder": "kgat", "hcr": True, "shuffle": "none"},
    "KGAT_H2_SHUFFLED": {"encoder": "kgat", "hcr": True, "shuffle": "popularity_stratified"},
    "RGCN_A_ONLY": {"encoder": "rgcn", "hcr": False, "shuffle": "none"},
    "RGCN_H2": {"encoder": "rgcn", "hcr": True, "shuffle": "none"},
    "RGCN_H2_SHUFFLED": {"encoder": "rgcn", "hcr": True, "shuffle": "popularity_stratified"},
    "LIGHTGCN_A_ONLY": {"encoder": "lightgcn", "hcr": False, "shuffle": "none"},
    "LIGHTGCN_H2": {"encoder": "lightgcn", "hcr": True, "shuffle": "none"},
    "LIGHTGCN_H2_SHUFFLED": {
        "encoder": "lightgcn",
        "hcr": True,
        "shuffle": "popularity_stratified",
    },
    # Stage G: activation / KAN (encoder=RGCN or HGT, HCR=H2)
    "RGCN_H2_ACT_GELU": {
        "encoder": "rgcn",
        "hcr": True,
        "shuffle": "none",
        "activation": "gelu",
    },
    "RGCN_H2_ACT_RELU": {
        "encoder": "rgcn",
        "hcr": True,
        "shuffle": "none",
        "activation": "relu",
    },
    "RGCN_H2_ACT_SILU": {
        "encoder": "rgcn",
        "hcr": True,
        "shuffle": "none",
        "activation": "silu",
    },
    "RGCN_H2_ACT_TANH": {
        "encoder": "rgcn",
        "hcr": True,
        "shuffle": "none",
        "activation": "tanh",
    },
    "RGCN_H2_ACT_LEAKY_RELU": {
        "encoder": "rgcn",
        "hcr": True,
        "shuffle": "none",
        "activation": "leaky_relu",
    },
    "RGCN_H2_KAN": {
        "encoder": "rgcn",
        "hcr": True,
        "shuffle": "none",
        "fusion": "kan",
    },
    "HGT_H2_KAN": {
        "encoder": "hgt",
        "hcr": True,
        "shuffle": "none",
        "fusion": "kan",
    },
    "HGT_H2_ACT_SILU": {
        "encoder": "hgt",
        "hcr": True,
        "shuffle": "none",
        "activation": "silu",
    },
    "HGT_H2_ACT_RELU": {
        "encoder": "hgt",
        "hcr": True,
        "shuffle": "none",
        "activation": "relu",
    },
}


def _build_encoder(name: str, cfg: dict[str, Any], bundle: dict[str, Any], device: torch.device):
    acfg = cfg.get("models", {}).get("architecture", {})
    embed_dim = int(acfg.get("embedding_dim", 64))
    n_layers = int(acfg.get("num_layers", 2))
    dropout = float(acfg.get("dropout", 0.1))
    max_kg = acfg.get("max_kg_edges", 250_000)

    if name == "hgt":
        graph = load_data_and_typed_graph(cfg, bundle["model_train"], max_kg_edges=max_kg)
        meta = graph["meta"]
        edge_index_dict = {
            k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0
        }
        metadata = (["user", "entity"], list(edge_index_dict.keys()))
        enc = HGTGraphEncoder(
            meta["n_users"],
            meta["n_entities"],
            metadata,
            edge_index_dict,
            embed_dim=embed_dim,
            n_layers=n_layers,
            heads=int(acfg.get("heads", 2)),
            dropout=dropout,
        ).to(device)
        return enc, meta

    if name == "rgcn":
        graph = load_data_and_typed_graph(cfg, bundle["model_train"], max_kg_edges=max_kg)
        meta = graph["meta"]
        enc = RGCNGraphEncoder(
            meta["n_nodes"],
            meta["n_relations"],
            graph["edge_index"].to(device),
            graph["edge_type"].to(device),
            embed_dim=embed_dim,
            n_layers=n_layers,
            dropout=dropout,
            num_bases=int(acfg.get("num_bases", 8)),
        ).to(device)
        return enc, meta

    if name == "kgat":
        graph = load_data_and_typed_graph(cfg, bundle["model_train"], max_kg_edges=max_kg)
        meta = graph["meta"]
        enc = KGATEncoder(
            meta["n_nodes"],
            meta["n_relations"],
            graph["edge_index"].to(device),
            graph["edge_type"].to(device),
            embed_dim=embed_dim,
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)
        return enc, meta

    if name == "lightgcn":
        data = load_lastfm(cfg["data"]["path"])
        edge_index, meta = build_ckg_edge_index(
            data, bundle["model_train"], max_kg_edges=max_kg, seed=cfg["split"]["seed"]
        )
        enc = LightGCNEncoder(
            meta["n_nodes"],
            edge_index.to(device),
            embed_dim=embed_dim,
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)
        return enc, meta

    raise ValueError(name)


def _build_fusion(
    *,
    graph_dim: int,
    use_hcr: bool,
    fcfg: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    fusion_kind = str(spec.get("fusion", fcfg.get("type", "late_concat"))).lower()
    if fusion_kind in {"kan", "kan_fusion"}:
        return KANFusionHead(
            graph_dim=graph_dim,
            base_feature_dim=5,
            hcr_feature_dim=8,
            hidden_dim=int(fcfg.get("kan_hidden_dim", fcfg.get("hidden_dim", 64))),
            dropout=float(fcfg.get("kan_dropout", fcfg.get("dropout", 0.1))),
            use_hcr=use_hcr,
            grid_size=int(fcfg.get("kan_grid_size", 5)),
            spline_order=int(fcfg.get("kan_spline_order", 3)),
        ).to(device)

    activation = str(spec.get("activation", fcfg.get("activation", "gelu")))
    return LateFusionHead(
        graph_dim=graph_dim,
        base_feature_dim=5,
        hcr_feature_dim=8,
        hidden_dim=int(fcfg.get("hidden_dim", 128)),
        dropout=float(fcfg.get("dropout", 0.2)),
        use_hcr=use_hcr,
        activation=activation,
    ).to(device)


def run_architecture_stage(
    bundle: dict[str, Any],
    stage: str,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    if stage not in STAGE_SPECS:
        raise ValueError(stage)
    spec = STAGE_SPECS[stage]
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    seed = int(seed if seed is not None else cfg["models"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    contract = load_h2_contract(cfg.get("feature_contract"))
    feats = load_base_and_hcr(cfg, contract)
    use_hcr = bool(spec["hcr"])

    H_train = feats["H_train"]
    H_val = feats["H_val"]
    H_test = feats["H_test"]
    if use_hcr and spec["shuffle"] != "none":
        H_train = shuffle_hcr_block(
            H_train,
            mode=spec["shuffle"],
            popularity=bundle["popularity"],
            item_ids=bundle["train_pairs"]["item_id"],
            seed=seed + 17,
        )
        H_val = shuffle_hcr_block(
            H_val,
            mode=spec["shuffle"],
            popularity=bundle["popularity"],
            item_ids=bundle["val_pairs"]["item_id"],
            seed=seed + 31,
        )
        H_test = shuffle_hcr_block(
            H_test,
            mode=spec["shuffle"],
            popularity=bundle["popularity"],
            item_ids=bundle["test_pairs"]["item_id"],
            seed=seed + 47,
        )

    # train-only scaling
    a_scaler = StandardScaler().fit(feats["A_train"])
    A_tr = a_scaler.transform(feats["A_train"]).astype(np.float32)
    A_va = a_scaler.transform(feats["A_val"]).astype(np.float32)
    A_te = a_scaler.transform(feats["A_test"]).astype(np.float32)
    if use_hcr:
        h_scaler = StandardScaler().fit(H_train)
        H_tr = h_scaler.transform(H_train).astype(np.float32)
        H_va = h_scaler.transform(H_val).astype(np.float32)
        H_te = h_scaler.transform(H_test).astype(np.float32)
    else:
        H_tr = H_va = H_te = None

    encoder, meta = _build_encoder(spec["encoder"], cfg, bundle, device)
    item_offset = meta["item_offset"]
    fusion = _build_fusion(
        graph_dim=encoder.context_dim,
        use_hcr=use_hcr,
        fcfg=fcfg,
        spec=spec,
        device=device,
    )
    model = RecommendationModel(encoder, fusion).to(device)

    u_t, i_t = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    u_t, i_t = u_t.to(device), i_t.to(device)
    y_t = torch.from_numpy(bundle["train_pairs"]["label"].astype(np.float32)).to(device)
    A_t = torch.from_numpy(A_tr).to(device)
    H_t = torch.from_numpy(H_tr).to(device) if H_tr is not None else None

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(acfg.get("lr", 1e-3)),
        weight_decay=float(acfg.get("weight_decay", 1e-4)),
    )
    n_pos = float((bundle["train_pairs"]["label"] > 0.5).sum())
    n_neg = float((bundle["train_pairs"]["label"] <= 0.5).sum())
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_ndcg = -1.0
    left = int(acfg.get("patience", 5))
    history = []

    def score_split(users, items, A, H):
        model.eval()
        with torch.no_grad():
            z = model.encoder.encode_all()
            u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
            scores = []
            for start in range(0, len(users), 8192):
                sl = slice(start, start + 8192)
                out = model(
                    u_idx[sl].to(device),
                    i_idx[sl].to(device),
                    torch.from_numpy(A[sl]).to(device),
                    torch.from_numpy(H[sl]).to(device) if H is not None else None,
                    z=z,
                )
                scores.append(out["logits"].detach().cpu().numpy())
        return np.concatenate(scores)

    for epoch in range(int(acfg.get("max_epochs", 20))):
        model.train()
        opt.zero_grad()
        out = model(u_t, i_t, A_t, H_t)
        loss = loss_fn(out["logits"], y_t)
        loss.backward()
        opt.step()
        val_scores = score_split(
            bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], A_va, H_va
        )
        val_rank = ranking_metrics_for_users(
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["label"],
            val_scores,
            ks=(20,),
        )
        ndcg = float(val_rank["NDCG@20"])
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_NDCG@20": ndcg})
        print(f"[{stage}] epoch {epoch} train_loss={float(loss.item()):.4f} val_NDCG@20={ndcg:.4f}")
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = int(acfg.get("patience", 5))
        else:
            left -= 1
            if left <= 0:
                break

    if best_state:
        model.load_state_dict(best_state)

    ks = cfg["evaluation"]["ranking_k"]
    val_scores = score_split(
        bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], A_va, H_va
    )
    test_scores = score_split(
        bundle["test_pairs"]["user_id"], bundle["test_pairs"]["item_id"], A_te, H_te
    )
    yva = bundle["val_pairs"]["label"].astype(np.int32)
    yte = bundle["test_pairs"]["label"].astype(np.int32)
    cal = PlattCalibrator().fit(val_scores, yva)
    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, ks)
    val_metrics["Brier_calibrated"] = brier_score(yva, cal.transform(val_scores))
    test_metrics["Brier_calibrated"] = brier_score(yte, cal.transform(test_scores))

    fusion_kind = str(spec.get("fusion", "mlp"))
    activation = getattr(fusion, "activation", spec.get("activation", fcfg.get("activation", "gelu")))
    result = {
        "stage": stage,
        "protocol": cfg["protocol"],
        "encoder": spec["encoder"],
        "fusion": fusion_kind if fusion_kind != "late_concat" else "mlp",
        "activation": activation,
        "hcr": use_hcr,
        "hcr_shuffle": spec["shuffle"],
        "feature_contract": "H2_FROZEN",
        "seed": seed,
        "train_info": {"history": history, "best_val_NDCG@20": best_ndcg},
        "validation": val_metrics,
        "test": test_metrics,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
        "hcr_names": feats["hcr_names"] if use_hcr else [],
        "base_names": feats["base_names"],
    }
    out_dir = Path(cfg["paths"].get("stage_arch", Path(cfg["paths"]["outputs_root"]) / "stage_arch"))
    out = out_dir / stage / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Legacy flat path for seed 101 (keeps prior summary scripts happy)
    if seed == 101:
        legacy = out_dir / stage
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        np.save(legacy / "test_scores.npy", test_scores)
    np.save(out / "test_scores.npy", test_scores)
    print(
        f"[{stage} seed={seed}] test {result['primary_metric']}="
        f"{result['primary_test']:.4f}  AUPRC={test_metrics['AUPRC']:.4f}  "
        f"MRR={test_metrics['MRR']:.4f}  fusion={result['fusion']} act={result['activation']}"
    )
    return result


def write_arch_summary(cfg: dict[str, Any], results: dict[str, dict]) -> Path:
    lines = [
        "# LASTFM_ARCHITECTURE_HCR_V1 / Stage G",
        "",
        "Frozen HCR = **H2** (a11 + energy). Vary encoder / fusion activation / KAN.",
        "",
        f"Reference Stage-H H2 NDCG@20 = **0.6512** | B3 = 0.6426 | RGCN_H2@101 = 0.6485",
        "",
        "| stage | seed | encoder | fusion | act | NDCG@20 | AUPRC | MRR |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for key, r in results.items():
        t = r["test"]
        lines.append(
            f"| {r.get('stage', key)} | {r.get('seed', '')} | {r['encoder']} | "
            f"{r.get('fusion', 'mlp')} | {r.get('activation', 'gelu')} | "
            f"{t.get('NDCG@20', float('nan')):.4f} | {t.get('AUPRC', float('nan')):.4f} | "
            f"{t.get('MRR', float('nan')):.4f} |"
        )
    out = Path(cfg["paths"]["outputs_root"]) / "stage_arch_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_stage_g_summary(cfg: dict[str, Any], results: dict[str, dict]) -> Path:
    """Dedicated Stage-G summary (activations / KAN / multi-seed)."""
    lines = [
        "# Stage G — fusion activations + KAN (H2 frozen)",
        "",
        "| run | seed | encoder | fusion | act | NDCG@20 | AUPRC | MRR |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for key, r in results.items():
        t = r["test"]
        lines.append(
            f"| {key} | {r.get('seed', '')} | {r['encoder']} | {r.get('fusion', 'mlp')} | "
            f"{r.get('activation', '')} | {t.get('NDCG@20', float('nan')):.4f} | "
            f"{t.get('AUPRC', float('nan')):.4f} | {t.get('MRR', float('nan')):.4f} |"
        )
    out = Path(cfg["paths"]["outputs_root"]) / "stage_g_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
