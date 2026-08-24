"""Stage P: path descriptors × a11/energy on frozen HGT+H2+GELU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes
from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.features.hcr_h2_contract import (
    load_base_and_hcr,
    load_h2_contract,
    shuffle_hcr_block,
)
from src.lastfm_lp.models.encoders import HGTGraphEncoder
from src.lastfm_lp.models.fusion import (
    BlockLateFusionHead,
    LateFusionHead,
    RecommendationModel,
)
from src.lastfm_lp.pipeline.features_path_h2 import materialize_path_h2

# P0–P4 + controls
STAGE_SPECS: dict[str, dict[str, Any]] = {
    "P0": {"hcr": "h2", "path": False, "inter": False, "shuffle": "none"},
    "P1": {"hcr": "h2", "path": True, "inter": False, "shuffle": "none"},
    "P2": {"hcr": "h2", "path": False, "inter": True, "shuffle": "none"},
    "P3": {"hcr": "h2", "path": True, "inter": True, "shuffle": "none"},
    "P4": {"hcr": "h2_split", "path": False, "inter": False, "shuffle": "none"},
    # controls
    "P_PATH_ONLY": {"hcr": "none", "path": True, "inter": False, "shuffle": "none"},
    "P3_PATH_SHUFFLE": {
        "hcr": "h2",
        "path": True,
        "inter": True,
        "shuffle": "path_popularity",
    },
    "P3_INTER_DECOUPLE": {
        "hcr": "h2",
        "path": True,
        "inter": True,
        "shuffle": "inter_decouple",
    },
}


def _build_hgt(cfg: dict[str, Any], bundle: dict[str, Any], device: torch.device):
    acfg = cfg.get("models", {}).get("architecture", {})
    graph = load_data_and_typed_graph(
        cfg, bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
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
        embed_dim=int(acfg.get("embedding_dim", 64)),
        n_layers=int(acfg.get("num_layers", 2)),
        heads=int(acfg.get("heads", 2)),
        dropout=float(acfg.get("dropout", 0.1)),
    ).to(device)
    return enc, meta


def _scale(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    sc = StandardScaler().fit(train)
    return (
        sc.transform(train).astype(np.float32),
        sc.transform(val).astype(np.float32),
        sc.transform(test).astype(np.float32),
    )


def _maybe_shuffle(
    X: np.ndarray,
    *,
    mode: str,
    popularity: np.ndarray,
    item_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    if mode in {"none", "inter_decouple"}:
        return X
    if mode == "path_popularity":
        return shuffle_hcr_block(
            X,
            mode="popularity_stratified",
            popularity=popularity,
            item_ids=item_ids,
            seed=seed,
        )
    raise ValueError(mode)


def run_stage_p(
    bundle: dict[str, Any],
    stage: str,
    *,
    seed: int,
    path_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in STAGE_SPECS:
        raise ValueError(f"Unknown stage {stage}; known={sorted(STAGE_SPECS)}")
    spec = STAGE_SPECS[stage]
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    if path_cache is None:
        path_cache = materialize_path_h2(
            bundle, max_history=int(cfg.get("features", {}).get("max_history", 25))
        )

    contract = load_h2_contract(cfg.get("feature_contract"))
    feats = load_base_and_hcr(cfg, contract)
    A_tr, A_va, A_te = _scale(feats["A_train"], feats["A_val"], feats["A_test"])

    hcr_mode = spec["hcr"]
    if hcr_mode == "h2":
        H_tr, H_va, H_te = _scale(feats["H_train"], feats["H_val"], feats["H_test"])
        hcr_dim = H_tr.shape[1]
    elif hcr_mode == "h2_split":
        H_tr, H_va, H_te = _scale(
            path_cache["h2_split"]["train"],
            path_cache["h2_split"]["val"],
            path_cache["h2_split"]["test"],
        )
        hcr_dim = H_tr.shape[1]
    else:
        H_tr = H_va = H_te = None
        hcr_dim = 0

    P_tr = P_va = P_te = None
    path_dim = 0
    if spec["path"]:
        P_tr0 = path_cache["struct"]["train"]
        P_va0 = path_cache["struct"]["val"]
        P_te0 = path_cache["struct"]["test"]
        if spec["shuffle"] == "path_popularity":
            P_tr0 = _maybe_shuffle(
                P_tr0,
                mode="path_popularity",
                popularity=bundle["popularity"],
                item_ids=bundle["train_pairs"]["item_id"],
                seed=seed + 17,
            )
            P_va0 = _maybe_shuffle(
                P_va0,
                mode="path_popularity",
                popularity=bundle["popularity"],
                item_ids=bundle["val_pairs"]["item_id"],
                seed=seed + 31,
            )
            P_te0 = _maybe_shuffle(
                P_te0,
                mode="path_popularity",
                popularity=bundle["popularity"],
                item_ids=bundle["test_pairs"]["item_id"],
                seed=seed + 47,
            )
        P_tr, P_va, P_te = _scale(P_tr0, P_va0, P_te0)
        path_dim = P_tr.shape[1]

    C_tr = C_va = C_te = None
    inter_dim = 0
    if spec["inter"]:
        C_tr0 = path_cache["inter"]["train"].copy()
        C_va0 = path_cache["inter"]["val"].copy()
        C_te0 = path_cache["inter"]["test"].copy()
        if spec["shuffle"] == "path_popularity":
            C_tr0 = _maybe_shuffle(
                C_tr0,
                mode="path_popularity",
                popularity=bundle["popularity"],
                item_ids=bundle["train_pairs"]["item_id"],
                seed=seed + 117,
            )
            C_va0 = _maybe_shuffle(
                C_va0,
                mode="path_popularity",
                popularity=bundle["popularity"],
                item_ids=bundle["val_pairs"]["item_id"],
                seed=seed + 131,
            )
            C_te0 = _maybe_shuffle(
                C_te0,
                mode="path_popularity",
                popularity=bundle["popularity"],
                item_ids=bundle["test_pairs"]["item_id"],
                seed=seed + 147,
            )
        elif spec["shuffle"] == "inter_decouple":
            # Approximate decoupling control: shuffle inter rows vs keep H2/path aligned
            rng = np.random.default_rng(seed + 999)
            C_tr0 = C_tr0[rng.permutation(len(C_tr0))]
            C_va0 = C_va0[rng.permutation(len(C_va0))]
            C_te0 = C_te0[rng.permutation(len(C_te0))]
        C_tr, C_va, C_te = _scale(C_tr0, C_va0, C_te0)
        inter_dim = C_tr.shape[1]

    encoder, meta = _build_hgt(cfg, bundle, device)
    item_offset = meta["item_offset"]
    # P0 (no path/inter, standard H2): exact LateFusionHead to match HGT_H2.
    if path_dim == 0 and inter_dim == 0 and hcr_mode == "h2":
        fusion = LateFusionHead(
            graph_dim=encoder.context_dim,
            base_feature_dim=5,
            hcr_feature_dim=hcr_dim,
            hidden_dim=int(fcfg.get("hidden_dim", 128)),
            dropout=float(fcfg.get("dropout", 0.2)),
            use_hcr=True,
            activation=str(fcfg.get("activation", "gelu")),
        ).to(device)
    else:
        fusion = BlockLateFusionHead(
            graph_dim=encoder.context_dim,
            base_feature_dim=5,
            hcr_feature_dim=hcr_dim,
            path_feature_dim=path_dim,
            inter_feature_dim=inter_dim,
            hidden_dim=int(fcfg.get("hidden_dim", 128)),
            dropout=float(fcfg.get("dropout", 0.2)),
            use_hcr=hcr_mode != "none",
            use_base=True,
            activation=str(fcfg.get("activation", "gelu")),
        ).to(device)
    model = RecommendationModel(encoder, fusion).to(device)

    u_t, i_t = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    u_t, i_t = u_t.to(device), i_t.to(device)
    y_t = torch.from_numpy(bundle["train_pairs"]["label"].astype(np.float32)).to(device)
    A_t = torch.from_numpy(A_tr).to(device)
    H_t = torch.from_numpy(H_tr).to(device) if H_tr is not None else None
    P_t = torch.from_numpy(P_tr).to(device) if P_tr is not None else None
    C_t = torch.from_numpy(C_tr).to(device) if C_tr is not None else None

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

    def score_split(users, items, A, H, P, C):
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
                    torch.from_numpy(P[sl]).to(device) if P is not None else None,
                    torch.from_numpy(C[sl]).to(device) if C is not None else None,
                    z=z,
                )
                scores.append(out["logits"].detach().cpu().numpy())
        return np.concatenate(scores)

    for epoch in range(int(acfg.get("max_epochs", 20))):
        model.train()
        opt.zero_grad()
        out = model(u_t, i_t, A_t, H_t, P_t, C_t)
        loss = loss_fn(out["logits"], y_t)
        loss.backward()
        opt.step()
        val_scores = score_split(
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["item_id"],
            A_va,
            H_va,
            P_va,
            C_va,
        )
        val_rank = ranking_metrics_for_users(
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["label"],
            val_scores,
            ks=(20,),
        )
        ndcg = float(val_rank["NDCG@20"])
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_NDCG@20": ndcg})
        print(f"[{stage} s={seed}] epoch {epoch} loss={float(loss.item()):.4f} val_NDCG@20={ndcg:.4f}")
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
        bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], A_va, H_va, P_va, C_va
    )
    test_scores = score_split(
        bundle["test_pairs"]["user_id"], bundle["test_pairs"]["item_id"], A_te, H_te, P_te, C_te
    )
    yva = bundle["val_pairs"]["label"].astype(np.int32)
    yte = bundle["test_pairs"]["label"].astype(np.int32)
    cal = PlattCalibrator().fit(val_scores, yva)
    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, ks)
    val_metrics["Brier_calibrated"] = brier_score(yva, cal.transform(val_scores))
    test_metrics["Brier_calibrated"] = brier_score(yte, cal.transform(test_scores))

    result = {
        "stage": stage,
        "protocol": cfg["protocol"],
        "seed": seed,
        "spec": spec,
        "dims": {"hcr": hcr_dim, "path": path_dim, "inter": inter_dim},
        "train_info": {"history": history, "best_val_NDCG@20": best_ndcg},
        "validation": val_metrics,
        "test": test_metrics,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_val": val_metrics.get(cfg["evaluation"]["primary_metric"]),
        "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
    }
    out_dir = Path(cfg["paths"].get("stage_p", "outputs/lastfm/stage_p")) / stage / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out_dir / "test_scores.npy", test_scores)
    print(
        f"[{stage} s={seed}] val NDCG@20={result['primary_val']:.4f}  "
        f"test NDCG@20={result['primary_test']:.4f}  "
        f"AUPRC={test_metrics['AUPRC']:.4f}  MRR={test_metrics['MRR']:.4f}"
    )
    return result


def write_stage_p_summary(cfg: dict[str, Any], results: dict[str, dict]) -> Path:
    lines = [
        "# Stage P — path × H2 on frozen HGT+GELU",
        "",
        "Selection uses **val NDCG@20**; test reported for frozen runs.",
        "",
        "| run | seed | val NDCG@20 | test NDCG@20 | AUPRC | MRR | dims |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, r in results.items():
        d = r.get("dims", {})
        lines.append(
            f"| {key} | {r.get('seed')} | {r.get('primary_val', float('nan')):.4f} | "
            f"{r.get('primary_test', float('nan')):.4f} | "
            f"{r['test'].get('AUPRC', float('nan')):.4f} | "
            f"{r['test'].get('MRR', float('nan')):.4f} | "
            f"h{d.get('hcr',0)}/p{d.get('path',0)}/c{d.get('inter',0)} |"
        )
    out = Path(cfg["paths"]["outputs_root"]) / "stage_p_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
