"""FULL_SOTA_V2 ladder: HGT + A + optional H2 / H3_RAW (separate train-only scalers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.features.hcr_h2_contract import shuffle_hcr_block
from src.lastfm_lp.models.encoders import HGTGraphEncoder
from src.lastfm_lp.models.fusion import LateFusionHead, RecommendationModel
from src.lastfm_lp.torch_device import resolve_torch_device


def _load_block(feat_dir: Path) -> dict[str, Any]:
    names = json.loads((feat_dir / "feature_names.json").read_text(encoding="utf-8"))
    return {
        "names": names,
        "train": np.load(feat_dir / "X_train.npy").astype(np.float32),
        "val": np.load(feat_dir / "X_val.npy").astype(np.float32),
        "test": np.load(feat_dir / "X_test.npy").astype(np.float32),
    }


def _scale_block(block: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sc = StandardScaler().fit(block["train"])
    return (
        sc.transform(block["train"]).astype(np.float32),
        sc.transform(block["val"]).astype(np.float32),
        sc.transform(block["test"]).astype(np.float32),
        list(block["names"]),
    )


def _load_block_cols(feat_dir: Path, cols: list[int] | None = None, names_want: list[str] | None = None) -> dict[str, Any]:
    """Load HCR block; optional column subset or dedicated signed / signed_energy files."""

    feat_dir = Path(feat_dir)
    if names_want is not None:
        want_energy = any("energy" in n for n in names_want)
        signed_names_p = feat_dir / (
            "feature_names_signed_energy.json" if want_energy else "feature_names_signed.json"
        )
        signed_x = {
            "train": feat_dir / ("X_train_signed_energy.npy" if want_energy else "X_train_signed.npy"),
            "val": feat_dir / ("X_val_signed_energy.npy" if want_energy else "X_val_signed.npy"),
            "test": feat_dir / ("X_test_signed_energy.npy" if want_energy else "X_test_signed.npy"),
        }
        if signed_names_p.exists() and all(p.exists() for p in signed_x.values()):
            names = json.loads(signed_names_p.read_text(encoding="utf-8"))
            if list(names_want) == list(names):
                return {
                    "names": list(names),
                    "train": np.load(signed_x["train"]).astype(np.float32),
                    "val": np.load(signed_x["val"]).astype(np.float32),
                    "test": np.load(signed_x["test"]).astype(np.float32),
                }
        names = json.loads((feat_dir / "feature_names.json").read_text(encoding="utf-8"))
        idx = [names.index(n) for n in names_want]
        return {
            "names": list(names_want),
            "train": np.load(feat_dir / "X_train.npy").astype(np.float32)[:, idx],
            "val": np.load(feat_dir / "X_val.npy").astype(np.float32)[:, idx],
            "test": np.load(feat_dir / "X_test.npy").astype(np.float32)[:, idx],
        }
    names = json.loads((feat_dir / "feature_names.json").read_text(encoding="utf-8"))
    Xtr = np.load(feat_dir / "X_train.npy").astype(np.float32)
    Xva = np.load(feat_dir / "X_val.npy").astype(np.float32)
    Xte = np.load(feat_dir / "X_test.npy").astype(np.float32)
    if cols is not None:
        Xtr, Xva, Xte = Xtr[:, cols], Xva[:, cols], Xte[:, cols]
        names = [names[i] for i in cols]
    return {"names": names, "train": Xtr, "val": Xva, "test": Xte}


def _shuffle_block(
    block: dict[str, Any],
    bundle: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "names": block["names"],
        "train": shuffle_hcr_block(
            block["train"],
            mode="popularity_stratified",
            popularity=bundle["popularity"],
            item_ids=bundle["train_pairs"]["item_id"],
            seed=seed + 17,
        ),
        "val": shuffle_hcr_block(
            block["val"],
            mode="popularity_stratified",
            popularity=bundle["popularity"],
            item_ids=bundle["val_pairs"]["item_id"],
            seed=seed + 31,
        ),
        "test": shuffle_hcr_block(
            block["test"],
            mode="popularity_stratified",
            popularity=bundle["popularity"],
            item_ids=bundle["test_pairs"]["item_id"],
            seed=seed + 47,
        ),
    }


def run_v2_hgt_stage(
    bundle: dict[str, Any],
    *,
    stage: str,
    use_h2: bool = False,
    use_h3: bool = False,
    shuffle_h3: bool = False,
    use_a11: bool = False,
    shuffle_a11: bool = False,
    h3_dir: Path | str | None = None,
    h3_names: list[str] | None = None,
    a11_dir: Path | str | None = None,
    hidden_dim: int | None = None,
    seed: int = 101,
) -> dict[str, Any]:
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_name = (
        acfg.get("device")
        or cfg.get("models", {}).get("device")
        or "auto"
    )
    device = resolve_torch_device(str(device_name))
    print(f"[{stage} s={seed}] device={device} (requested={device_name})")

    feat_root = Path(cfg["paths"]["features"])
    A = _load_block(feat_root / "A")
    A_tr, A_va, A_te, _ = _scale_block(A)

    h_parts_tr: list[np.ndarray] = []
    h_parts_va: list[np.ndarray] = []
    h_parts_te: list[np.ndarray] = []
    h_names: list[str] = []

    if use_a11:
        a11_path = Path(a11_dir or feat_root / "stage_h" / "H1_a11")
        A11 = _load_block(a11_path)
        if shuffle_a11:
            A11 = _shuffle_block(A11, bundle, seed=seed + 100)
        t, v, e, names = _scale_block(A11)
        h_parts_tr.append(t)
        h_parts_va.append(v)
        h_parts_te.append(e)
        h_names.extend([f"A11:{n}" for n in names])

    if use_h2:
        H2 = _load_block(Path(cfg["paths"]["hcr_h2_cache"]))
        t, v, e, names = _scale_block(H2)
        h_parts_tr.append(t)
        h_parts_va.append(v)
        h_parts_te.append(e)
        h_names.extend([f"H2:{n}" for n in names])

    if use_h3:
        h3_path = Path(h3_dir or cfg["paths"]["hcr_h3_cache"])
        if h3_names is not None:
            H3 = _load_block_cols(h3_path, names_want=h3_names)
        else:
            H3 = _load_block(h3_path)
        if shuffle_h3:
            H3 = _shuffle_block(H3, bundle, seed=seed)
        t, v, e, names = _scale_block(H3)
        h_parts_tr.append(t)
        h_parts_va.append(v)
        h_parts_te.append(e)
        h_names.extend([f"H3:{n}" for n in names])

    if h_parts_tr:
        H_tr = np.concatenate(h_parts_tr, axis=1)
        H_va = np.concatenate(h_parts_va, axis=1)
        H_te = np.concatenate(h_parts_te, axis=1)
        hcr_dim = int(H_tr.shape[1])
        use_hcr = True
    else:
        H_tr = H_va = H_te = None
        hcr_dim = 0
        use_hcr = False

    hid = int(hidden_dim if hidden_dim is not None else fcfg.get("hidden_dim", 128))

    graph = load_data_and_typed_graph(
        cfg, bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
    meta = graph["meta"]
    edge_index_dict = {
        k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0
    }
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    encoder = HGTGraphEncoder(
        meta["n_users"],
        meta["n_entities"],
        metadata,
        edge_index_dict,
        embed_dim=int(acfg.get("embedding_dim", 64)),
        n_layers=int(acfg.get("num_layers", 2)),
        heads=int(acfg.get("heads", 2)),
        dropout=float(acfg.get("dropout", 0.1)),
    ).to(device)
    fusion = LateFusionHead(
        graph_dim=encoder.context_dim,
        base_feature_dim=5,
        hcr_feature_dim=hcr_dim,
        hidden_dim=hid,
        dropout=float(fcfg.get("dropout", 0.2)),
        use_hcr=use_hcr,
        activation=str(fcfg.get("activation", "gelu")),
    ).to(device)
    model = RecommendationModel(encoder, fusion).to(device)
    item_offset = meta["item_offset"]

    u_all, i_all = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    y_all = bundle["train_pairs"]["label"].astype(np.float32)
    n_train = len(y_all)
    batch_size = int(acfg.get("batch_size", 4096))

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(acfg.get("lr", 1e-3)),
        weight_decay=float(acfg.get("weight_decay", 1e-4)),
    )
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_ndcg = -1.0
    left = int(acfg.get("patience", 4))
    history = []

    def score_split(users, items, A_x, H_x):
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
                    torch.from_numpy(A_x[sl]).to(device),
                    torch.from_numpy(H_x[sl]).to(device) if H_x is not None else None,
                    z=z,
                )
                scores.append(out["logits"].detach().cpu().numpy())
        return np.concatenate(scores)

    for epoch in range(int(acfg.get("max_epochs", 15))):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        z_live = model.encoder.encode_all()
        batch_starts = list(range(0, n_train, batch_size))
        n_b = max(len(batch_starts), 1)
        for bi, start in enumerate(batch_starts):
            idx = perm[start : start + batch_size]
            z = z_live if bi == 0 else z_live.detach()
            out = model(
                u_all[idx].to(device),
                i_all[idx].to(device),
                torch.from_numpy(A_tr[idx]).to(device),
                torch.from_numpy(H_tr[idx]).to(device) if H_tr is not None else None,
                z=z,
            )
            loss = loss_fn(out["logits"], torch.from_numpy(y_all[idx]).to(device))
            (loss / n_b).backward()
            epoch_loss += float(loss.item())
            n_batches += 1
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
        history.append(
            {"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_NDCG@20": ndcg}
        )
        print(
            f"[{stage} s={seed}] epoch {epoch} loss={epoch_loss/max(n_batches,1):.4f} "
            f"val_NDCG@20={ndcg:.4f}"
        )
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = int(acfg.get("patience", 4))
        else:
            left -= 1
            if left <= 0:
                break

    if best_state:
        model.load_state_dict(best_state)

    ks = tuple(cfg["evaluation"]["ranking_k"])
    val_scores = score_split(
        bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], A_va, H_va
    )
    test_scores = score_split(
        bundle["test_pairs"]["user_id"], bundle["test_pairs"]["item_id"], A_te, H_te
    )
    val_m = ranking_metrics_for_users(
        bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"], val_scores, ks=ks
    )
    test_m = ranking_metrics_for_users(
        bundle["test_pairs"]["user_id"], bundle["test_pairs"]["label"], test_scores, ks=ks
    )
    from src.lastfm_lp.evaluation.paired_runs import evaluate_split

    test_extra = evaluate_split(bundle["test_pairs"], test_scores, list(ks))

    out_dir = Path(cfg["paths"]["stage_full"]) / stage / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_scores_sampled.npy", test_scores.astype(np.float32))
    n_params = int(sum(p.numel() for p in model.parameters()))
    n_fusion = int(sum(p.numel() for p in fusion.parameters()))
    metrics = {
        "stage": stage,
        "seed": seed,
        "hidden_dim": hid,
        "hcr_names": h_names,
        "hcr_dim": hcr_dim,
        "n_parameters": n_params,
        "n_fusion_parameters": n_fusion,
        "flags": {
            "use_a11": use_a11,
            "shuffle_a11": shuffle_a11,
            "use_h2": use_h2,
            "use_h3": use_h3,
            "shuffle_h3": shuffle_h3,
        },
        "primary_val": float(val_m.get("NDCG@20", best_ndcg)),
        "primary_test_sampled": float(test_m.get("NDCG@20", 0.0)),
        "validation_sampled": {k: float(v) for k, v in val_m.items()},
        "test_sampled": {k: float(v) for k, v in test_m.items()},
        "test_extra": {k: float(v) for k, v in test_extra.items()},
        "train_info": {"history": history, "best_val_NDCG@20": best_ndcg},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[{stage} s={seed}] sampled test NDCG@20={metrics['primary_test_sampled']:.4f} "
        f"AUPRC={test_extra.get('AUPRC', float('nan')):.4f} "
        f"MRR={test_extra.get('MRR', float('nan')):.4f}"
    )
    return metrics
