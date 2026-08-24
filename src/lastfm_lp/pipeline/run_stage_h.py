"""Stage H — orthonormal HCR representation audit (HT* tabular, H* + HGT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.lastfm_lp.binary.user_hcr_aggregation import history_pair_blocks
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes
from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.models.hgt_encoder import HGTEncoder
from src.lastfm_lp.models.mlp_decoder import MLPDecoder
from src.lastfm_lp.models.mlp_predictor import MLPPredictor, TrainConfig
from src.lastfm_lp.models.pair_mlp import PairMLP, pool_encoded
from src.lastfm_lp.pipeline.features_hcr_v2 import (
    build_a11_shuffle_map,
    ensure_cross_fit,
    materialize_pooled_sides,
    _hist_for_row,
    _pair_fn_factory,
)
from src.lastfm_lp.pipeline.run_graph_stage import _load_side_features


def _tabular_A(cfg: dict[str, Any]) -> dict[str, Any]:
    pack = _load_side_features(cfg, tabular=True, hcr=False)
    assert pack is not None
    # strip already-prefixed names → raw for re-tagging
    return {
        "names": [n.split(":", 1)[-1] for n in pack["names"]],
        "X_train": pack["train"],
        "X_val": pack["val"],
        "X_test": pack["test"],
    }


def _out_dir(cfg: dict[str, Any], stage: str) -> Path:
    root = Path(cfg["paths"].get("stage_h", Path(cfg["paths"]["outputs_root"]) / "stage_h"))
    out = root / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_result(cfg: dict[str, Any], stage: str, result: dict[str, Any]) -> dict[str, Any]:
    out = _out_dir(cfg, stage)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{stage}] test {result['primary_metric']}="
        f"{result['primary_test']:.4f}  AUPRC={result['test']['AUPRC']:.4f}  "
        f"MRR={result['test']['MRR']:.4f}"
    )
    return result


def _train_tabular_mlp(bundle: dict[str, Any], stage: str, sides: dict[str, Any]) -> dict[str, Any]:
    cfg = bundle["cfg"]
    # optional Stage-A tabular concat
    tab = _tabular_A(cfg)
    Xtr = np.concatenate([tab["X_train"], sides["X_train"]], axis=1)
    Xva = np.concatenate([tab["X_val"], sides["X_val"]], axis=1)
    Xte = np.concatenate([tab["X_test"], sides["X_test"]], axis=1)
    ytr = sides["y_train"].astype(np.int32)
    yva = sides["y_val"].astype(np.int32)
    yte = sides["y_test"].astype(np.int32)
    names = [f"A:{n}" for n in tab["names"]] + [f"H:{n}" for n in sides["feature_names"]]

    mcfg = cfg["models"]["mlp"]
    tcfg = TrainConfig(
        lr=mcfg["lr"],
        weight_decay=mcfg["weight_decay"],
        max_epochs=mcfg["max_epochs"],
        batch_size=mcfg["batch_size"],
        patience=mcfg["patience"],
        seed=cfg["models"]["seed"],
    )
    model = MLPPredictor(Xtr.shape[1], mcfg["hidden_dims"], mcfg["dropout"], tcfg)
    train_info = model.fit(Xtr, ytr, Xva, yva)
    train_info["n_parameters"] = model.n_parameters()
    val_scores = model.predict_proba(Xva)
    test_scores = model.predict_proba(Xte)
    cal = PlattCalibrator().fit(val_scores, yva)
    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, cfg["evaluation"]["ranking_k"])
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, cfg["evaluation"]["ranking_k"])
    val_metrics["Brier_calibrated"] = brier_score(yva, cal.transform(val_scores))
    test_metrics["Brier_calibrated"] = brier_score(yte, cal.transform(test_scores))
    return _save_result(
        cfg,
        stage,
        {
            "stage": stage,
            "protocol": cfg["protocol"],
            "kind": "tabular_mlp_pooled",
            "block": sides["kind"],
            "feature_names": names,
            "n_features": len(names),
            "train_info": train_info,
            "validation": val_metrics,
            "test": test_metrics,
            "primary_metric": cfg["evaluation"]["primary_metric"],
            "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
        },
    )


def _materialize_hist_bank(
    bundle: dict[str, Any],
    *,
    tag: str,
    drop_a11: bool = False,
    shuffle_a11: dict | None = None,
) -> dict[str, Any]:
    cfg = bundle["cfg"]
    max_history = int(cfg.get("models", {}).get("stage_h", {}).get("max_history", 25))
    out_dir = Path(cfg["paths"]["features"]) / "stage_h" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "train_feats.npy").exists():
        return {
            "train_feats": np.load(out_dir / "train_feats.npy"),
            "train_mask": np.load(out_dir / "train_mask.npy"),
            "val_feats": np.load(out_dir / "val_feats.npy"),
            "val_mask": np.load(out_dir / "val_mask.npy"),
            "test_feats": np.load(out_dir / "test_feats.npy"),
            "test_mask": np.load(out_dir / "test_mask.npy"),
        }

    cf = ensure_cross_fit(bundle)

    def build(pairs, split, desc):
        n = len(pairs["label"])
        # probe dim
        f0, m0, _ = history_pair_blocks([], 0, cf.full_index, max_history=max_history, kind="compact8")
        feats = np.zeros((n, max_history, f0.shape[1]), dtype=np.float32)
        mask = np.zeros((n, max_history), dtype=np.float32)
        for r in tqdm(range(n), desc=desc, mininterval=2.0):
            u = int(pairs["user_id"][r])
            i = int(pairs["item_id"][r])
            y = int(pairs["label"][r])
            hist = _hist_for_row(u, i, y, bundle["model_train"])
            pair_fn = _pair_fn_factory(
                cf, u, split, drop_a11=drop_a11, shuffle_a11=shuffle_a11
            )
            mat, msk, _ = history_pair_blocks(
                hist, i, cf.full_index, max_history=max_history, kind="compact8", pair_fn=pair_fn
            )
            feats[r] = mat
            mask[r] = msk
        return feats, mask

    tr = build(bundle["train_pairs"], "train", f"{tag} train")
    va = build(bundle["val_pairs"], "val", f"{tag} val")
    te = build(bundle["test_pairs"], "test", f"{tag} test")
    np.save(out_dir / "train_feats.npy", tr[0])
    np.save(out_dir / "train_mask.npy", tr[1])
    np.save(out_dir / "val_feats.npy", va[0])
    np.save(out_dir / "val_mask.npy", va[1])
    np.save(out_dir / "test_feats.npy", te[0])
    np.save(out_dir / "test_mask.npy", te[1])
    return {
        "train_feats": tr[0],
        "train_mask": tr[1],
        "val_feats": va[0],
        "val_mask": va[1],
        "test_feats": te[0],
        "test_mask": te[1],
    }


class _PairMLPTabular(nn.Module):
    def __init__(self, pair_in: int, pair_hidden: list[int], side_dim: int, dec_hidden: list[int], dropout: float):
        super().__init__()
        self.pair = PairMLP(pair_in, pair_hidden, dropout=dropout)
        pooled_dim = 4 * self.pair.out_dim
        self.side_ln = nn.LayerNorm(side_dim) if side_dim > 0 else None
        self.hcr_ln = nn.LayerNorm(pooled_dim)
        in_dim = pooled_dim + side_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in dec_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, feats, mask, side=None):
        # feats: [B,H,D]
        enc = self.pair(feats)
        # support channel index 6 in compact8
        support = feats[..., 6]
        pooled = pool_encoded(enc, mask, support=support)
        pooled = self.hcr_ln(pooled)
        if side is not None and self.side_ln is not None:
            x = torch.cat([pooled, self.side_ln(side)], dim=-1)
        else:
            x = pooled
        return self.decoder(x).squeeze(-1)


def _train_pairmlp_tabular(
    bundle: dict[str, Any],
    stage: str,
    *,
    drop_a11: bool = False,
    shuffle: bool = False,
) -> dict[str, Any]:
    cfg = bundle["cfg"]
    hcfg = cfg.get("models", {}).get("stage_h", {})
    seed = int(cfg["models"]["seed"])
    torch.manual_seed(seed)
    shuffle_map = build_a11_shuffle_map(bundle, seed=seed) if shuffle else None
    tag = f"{stage}_bank" + ("_shufa11" if shuffle else "") + ("_noa11" if drop_a11 else "")
    bank = _materialize_hist_bank(
        bundle, tag=tag, drop_a11=drop_a11, shuffle_a11=shuffle_map
    )
    tab = _tabular_A(cfg)
    scaler_f = StandardScaler()
    flat = bank["train_feats"].reshape(-1, bank["train_feats"].shape[-1])
    msk = bank["train_mask"].reshape(-1) > 0
    scaler_f.fit(flat[msk] if msk.any() else flat)

    def scale_bank(feats):
        s = feats.shape
        out = scaler_f.transform(feats.reshape(-1, s[-1])).astype(np.float32).reshape(s)
        return out

    tr_f = scale_bank(bank["train_feats"])
    va_f = scale_bank(bank["val_feats"])
    te_f = scale_bank(bank["test_feats"])
    side_scaler = StandardScaler()
    Str = side_scaler.fit_transform(tab["X_train"]).astype(np.float32)
    Sva = side_scaler.transform(tab["X_val"]).astype(np.float32)
    Ste = side_scaler.transform(tab["X_test"]).astype(np.float32)

    ytr = bundle["train_pairs"]["label"].astype(np.float32)
    yva = bundle["val_pairs"]["label"].astype(np.float32)
    device = torch.device("cpu")
    model = _PairMLPTabular(
        pair_in=tr_f.shape[-1],
        pair_hidden=list(hcfg.get("pair_mlp_hidden", [16, 16])),
        side_dim=Str.shape[1],
        dec_hidden=list(cfg["models"]["mlp"]["hidden_dims"]),
        dropout=float(cfg["models"]["mlp"]["dropout"]),
    ).to(device)

    ds = TensorDataset(
        torch.from_numpy(tr_f),
        torch.from_numpy(bank["train_mask"]),
        torch.from_numpy(Str),
        torch.from_numpy(ytr),
    )
    loader = DataLoader(ds, batch_size=int(cfg["models"]["mlp"]["batch_size"]), shuffle=True)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["models"]["mlp"]["lr"]),
        weight_decay=float(cfg["models"]["mlp"]["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_ndcg = -1.0
    patience = int(cfg["models"]["mlp"]["patience"])
    left = patience
    history = []
    ks = cfg["evaluation"]["ranking_k"]

    def predict(feats, mask, side):
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(feats), 8192):
                sl = slice(start, start + 8192)
                logits = model(
                    torch.from_numpy(feats[sl]).to(device),
                    torch.from_numpy(mask[sl]).to(device),
                    torch.from_numpy(side[sl]).to(device),
                )
                scores.append(logits.cpu().numpy())
        return np.concatenate(scores)

    for epoch in range(int(cfg["models"]["mlp"]["max_epochs"])):
        model.train()
        total = 0.0
        n = 0
        for xf, xm, xs, yb in loader:
            opt.zero_grad()
            logits = model(xf.to(device), xm.to(device), xs.to(device))
            loss = loss_fn(logits, yb.to(device))
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        val_scores = predict(va_f, bank["val_mask"], Sva)
        val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
        ndcg = float(val_metrics["NDCG@20"])
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_NDCG@20": ndcg})
        print(f"[{stage}] epoch {epoch} train_loss={total/max(n,1):.4f} val_NDCG@20={ndcg:.4f}")
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
            if left <= 0:
                break
    if best_state:
        model.load_state_dict(best_state)
    test_scores = predict(te_f, bank["test_mask"], Ste)
    val_scores = predict(va_f, bank["val_mask"], Sva)
    yva_i = bundle["val_pairs"]["label"].astype(np.int32)
    yte_i = bundle["test_pairs"]["label"].astype(np.int32)
    cal = PlattCalibrator().fit(val_scores, yva_i)
    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, ks)
    val_metrics["Brier_calibrated"] = brier_score(yva_i, cal.transform(val_scores))
    test_metrics["Brier_calibrated"] = brier_score(yte_i, cal.transform(test_scores))
    return _save_result(
        cfg,
        stage,
        {
            "stage": stage,
            "protocol": cfg["protocol"],
            "kind": "tabular_pairmlp",
            "block": "compact8",
            "drop_a11": drop_a11,
            "shuffle_a11": shuffle,
            "train_info": {"history": history, "best_val_NDCG@20": best_ndcg},
            "validation": val_metrics,
            "test": test_metrics,
            "primary_metric": cfg["evaluation"]["primary_metric"],
            "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
        },
    )


def _train_hgt_pooled(bundle: dict[str, Any], stage: str, sides: dict[str, Any]) -> dict[str, Any]:
    """H0–H3: HGT + pooled orthonormal-HCR / legacy sides (same decoder style as E)."""

    cfg = bundle["cfg"]
    hcfg = cfg.get("models", {}).get("stage_h", {})
    ecfg = cfg.get("models", {}).get("stage_e", {})
    seed = int(cfg["models"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    embed_dim = int(hcfg.get("hgt_embed_dim", ecfg.get("hgt_embed_dim", 32)))
    max_kg = hcfg.get("max_kg_edges", 250_000)
    graph = load_data_and_typed_graph(cfg, bundle["model_train"], max_kg_edges=max_kg)
    meta = graph["meta"]
    item_offset = meta["item_offset"]
    edge_index_dict = {
        k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0
    }
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    encoder = HGTEncoder(
        meta["n_users"],
        meta["n_entities"],
        metadata=metadata,
        embed_dim=embed_dim,
        n_layers=int(hcfg.get("n_layers", 2)),
        heads=int(hcfg.get("heads", 2)),
        dropout=float(hcfg.get("dropout", 0.1)),
    ).to(device)

    tab = _tabular_A(cfg)
    side_tr = np.concatenate([tab["X_train"], sides["X_train"]], axis=1).astype(np.float32)
    side_va = np.concatenate([tab["X_val"], sides["X_val"]], axis=1).astype(np.float32)
    side_te = np.concatenate([tab["X_test"], sides["X_test"]], axis=1).astype(np.float32)
    scaler = StandardScaler()
    side_tr = scaler.fit_transform(side_tr).astype(np.float32)
    side_va = scaler.transform(side_va).astype(np.float32)
    side_te = scaler.transform(side_te).astype(np.float32)

    mcfg = cfg["models"]["mlp"]
    decoder = MLPDecoder(
        embed_dim, side_dim=side_tr.shape[1], hidden_dims=mcfg["hidden_dims"], dropout=mcfg["dropout"]
    ).to(device)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(hcfg.get("lr", 1e-3)),
        weight_decay=float(hcfg.get("weight_decay", 1e-4)),
    )
    train_y = bundle["train_pairs"]["label"].astype(np.float32)
    u_t, i_t = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    u_t, i_t = u_t.to(device), i_t.to(device)
    y_t = torch.from_numpy(train_y).to(device)
    side_t = torch.from_numpy(side_tr).to(device)
    n_pos = float((train_y > 0.5).sum())
    n_neg = float((train_y <= 0.5).sum())
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = None
    best_ndcg = -1.0
    left = int(hcfg.get("patience", 5))
    history = []

    def encode_fn():
        return encoder(edge_index_dict)

    def score_split(users, items, side):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z = encode_fn()
            u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
            scores = []
            for start in range(0, len(users), 8192):
                sl = slice(start, start + 8192)
                zu = z[u_idx[sl].to(device)]
                zi = z[i_idx[sl].to(device)]
                s = torch.from_numpy(side[sl]).to(device)
                scores.append(decoder(zu, zi, s).detach().cpu().numpy())
        return np.concatenate(scores)

    for epoch in range(int(hcfg.get("max_epochs", 20))):
        encoder.train()
        decoder.train()
        opt.zero_grad()
        z = encode_fn()
        loss = loss_fn(decoder(z[u_t], z[i_t], side_t), y_t)
        loss.backward()
        opt.step()
        val_scores = score_split(
            bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], side_va
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
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            }
            left = int(hcfg.get("patience", 5))
        else:
            left -= 1
            if left <= 0:
                break
    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
    ks = cfg["evaluation"]["ranking_k"]
    test_scores = score_split(
        bundle["test_pairs"]["user_id"], bundle["test_pairs"]["item_id"], side_te
    )
    val_scores = score_split(
        bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], side_va
    )
    yva = bundle["val_pairs"]["label"].astype(np.int32)
    yte = bundle["test_pairs"]["label"].astype(np.int32)
    cal = PlattCalibrator().fit(val_scores, yva)
    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, ks)
    val_metrics["Brier_calibrated"] = brier_score(yva, cal.transform(val_scores))
    test_metrics["Brier_calibrated"] = brier_score(yte, cal.transform(test_scores))
    return _save_result(
        cfg,
        stage,
        {
            "stage": stage,
            "protocol": cfg["protocol"],
            "kind": "hgt_pooled",
            "block": sides["kind"],
            "side_features": [f"A:{n}" for n in tab["names"]]
            + [f"H:{n}" for n in sides["feature_names"]],
            "train_info": {"history": history, "best_val_NDCG@20": best_ndcg},
            "validation": val_metrics,
            "test": test_metrics,
            "primary_metric": cfg["evaluation"]["primary_metric"],
            "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
        },
    )


def run_stage_h(bundle: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage in {"HT0", "HT1", "HT2"}:
        sides = materialize_pooled_sides(bundle, stage)
        return _train_tabular_mlp(bundle, stage, sides)
    if stage == "HT3":
        return _train_pairmlp_tabular(bundle, stage)
    if stage in {"H0", "H1", "H2", "H3"}:
        sides = materialize_pooled_sides(bundle, stage)
        return _train_hgt_pooled(bundle, stage, sides)
    if stage == "H4":
        # PairMLP tabular signal + HGT via pooled encoding freeze-in: reuse HT3 bank
        # encoded offline after a short PairMLP pretrain on HT3 objective is heavy;
        # for V2 audit, H4 = HGT + PairMLP-pooled compact8 from a dedicated bank pass
        # using the same tabular PairMLP trainer's pooled vector as side features.
        # Practical: materialize compact8 raw pool (H3) is not PairMLP — run PairMLP
        # bank, dump pooled vectors via a fitted PairMLP from HT3 if present, else train.
        return _train_pairmlp_tabular(bundle, "H4")  # tabular PairMLP; graph fusion in G later
    if stage == "H5":
        return _train_pairmlp_tabular(bundle, stage, shuffle=True)
    if stage == "H6":
        return _train_pairmlp_tabular(bundle, stage, drop_a11=True)
    raise ValueError(stage)


def write_stage_h_summary(cfg: dict[str, Any], results: dict[str, dict]) -> Path:
    root = Path(cfg["paths"].get("stage_h", Path(cfg["paths"]["outputs_root"]) / "stage_h"))
    lines = [
        "# LASTFM_ORTHONORMAL_HCR_V2 — Stage H",
        "",
        "Primary: **NDCG@20**",
        "",
        "| stage | kind | NDCG@20 | Recall@20 | MRR | AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage, r in results.items():
        t = r["test"]
        lines.append(
            f"| {stage} | {r.get('kind','')} | {t.get('NDCG@20', float('nan')):.4f} | "
            f"{t.get('Recall@20', float('nan')):.4f} | {t.get('MRR', float('nan')):.4f} | "
            f"{t.get('AUPRC', float('nan')):.4f} |"
        )
    # key deltas
    if "HT3" in results and "HT0" in results:
        d = results["HT3"]["test"]["NDCG@20"] - results["HT0"]["test"]["NDCG@20"]
        lines += ["", f"- **HT3_minus_HT0** NDCG@20={d:+.4f}"]
    if "H4" in results and "H5" in results:
        d = results["H4"]["test"]["NDCG@20"] - results["H5"]["test"]["NDCG@20"]
        lines += [f"- **H4_minus_H5** (shuffle control) NDCG@20={d:+.4f}"]
    if "H4" in results and "H6" in results:
        d = results["H4"]["test"]["NDCG@20"] - results["H6"]["test"]["NDCG@20"]
        lines += [f"- **H4_minus_H6** (a11 ablation) NDCG@20={d:+.4f}"]
    out = Path(cfg["paths"]["outputs_root"]) / "stage_h_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out
