"""Sampled-only train for flat / branch native-measure screen (NO full-rank)."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.models.encoders import HGTGraphEncoder
from src.lastfm_lp.models.fusion import LateFusionHead, RecommendationModel
from src.lastfm_lp.native_measure_screen.branch_fusion import BranchFusionHead
from src.lastfm_lp.native_measure_screen.constants import (
    A11_DIM,
    BRANCH_FUSION_IN,
    FLAT_A11_IN,
    FLAT_NONNEG_IN,
    NONNEG_DIM,
    SCALE_EPS,
)
from src.lastfm_lp.torch_device import resolve_torch_device


def user_item_to_nodes(users, items, item_offset: int):
    u = torch.as_tensor(users, dtype=torch.long)
    i = torch.as_tensor(items, dtype=torch.long) + int(item_offset)
    return u, i


def _hitrate_at_k(user_ids, labels, scores, *, k: int = 20) -> float:
    buckets: dict[int, list[tuple[float, int]]] = {}
    for u, y, s in zip(user_ids.tolist(), labels.tolist(), scores.tolist()):
        buckets.setdefault(int(u), []).append((float(s), int(y)))
    hits = []
    for rows in buckets.values():
        rows.sort(key=lambda t: t[0], reverse=True)
        ys = np.asarray([y for _, y in rows], dtype=np.float64)
        if ys.sum() <= 0:
            continue
        hits.append(1.0 if ys[:k].sum() > 0 else 0.0)
    return float(np.mean(hits)) if hits else float("nan")


def build_fusion(architecture: str, graph_dim: int, measure_dim: int, fcfg: dict) -> nn.Module:
    if architecture == "flat":
        expected = FLAT_A11_IN if measure_dim == A11_DIM else FLAT_NONNEG_IN
        got = graph_dim + 1 + 5 + measure_dim
        if got != expected:
            raise AssertionError(f"flat input dim {got} != {expected}")
        return LateFusionHead(
            graph_dim=graph_dim,
            base_feature_dim=5,
            hcr_feature_dim=measure_dim,
            hidden_dim=int(fcfg.get("hidden_dim", 128)),
            dropout=float(fcfg.get("dropout", 0.2)),
            use_hcr=True,
            activation=str(fcfg.get("activation", "gelu")),
        )
    if architecture == "branch":
        if BRANCH_FUSION_IN != 97:
            raise AssertionError("branch fusion dim constant corrupted")
        return BranchFusionHead(
            graph_dim=graph_dim,
            base_feature_dim=5,
            measure_dim=measure_dim,
            dropout=float(fcfg.get("dropout", 0.2)),
            measure_dropout=0.1,
        )
    raise ValueError(architecture)


def train_sampled_scenario(
    bundle: dict[str, Any],
    *,
    stage: str,
    architecture: str,
    seed: int,
    a_feature_dir: Path,
    measure_feature_dir: Path,
    ckpt_root: Path,
    max_epochs: int | None = None,
    max_train_rows: int | None = None,
    max_val_users: int | None = None,
) -> dict[str, Any]:
    """Train one scenario; early-stop on sampled val NDCG@20. Never runs full-rank."""

    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    t0 = time.time()

    A_names = json.loads((a_feature_dir / "feature_names.json").read_text())
    A_tr = np.load(a_feature_dir / "X_train.npy").astype(np.float32)
    A_va = np.load(a_feature_dir / "X_val.npy").astype(np.float32)
    H_tr = np.load(measure_feature_dir / "X_train.npy").astype(np.float32)
    H_va = np.load(measure_feature_dir / "X_val.npy").astype(np.float32)
    measure_dim = int(H_tr.shape[1])
    if measure_dim not in (A11_DIM, NONNEG_DIM):
        raise AssertionError(f"unexpected measure dim {measure_dim}")

    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(H_tr)
    # population-style: sklearn StandardScaler uses ddof=0 for scale with with_std
    A_tr_s = a_scaler.transform(A_tr).astype(np.float32)
    A_va_s = a_scaler.transform(A_va).astype(np.float32)
    H_tr_s = h_scaler.transform(H_tr).astype(np.float32)
    H_va_s = h_scaler.transform(H_va).astype(np.float32)
    # match screen eps note (sklearn already uses ~1e-8 via scale_)
    _ = SCALE_EPS

    if max_train_rows is not None:
        A_tr_s = A_tr_s[:max_train_rows]
        H_tr_s = H_tr_s[:max_train_rows]
        train_users = bundle["train_pairs"]["user_id"][:max_train_rows]
        train_items = bundle["train_pairs"]["item_id"][:max_train_rows]
        y_all = bundle["train_pairs"]["label"][:max_train_rows].astype(np.float32)
    else:
        train_users = bundle["train_pairs"]["user_id"]
        train_items = bundle["train_pairs"]["item_id"]
        y_all = bundle["train_pairs"]["label"].astype(np.float32)

    val_users = bundle["val_pairs"]["user_id"]
    val_items = bundle["val_pairs"]["item_id"]
    val_labels = bundle["val_pairs"]["label"]
    if max_val_users is not None:
        keep_u = set(np.unique(val_users)[: int(max_val_users)].tolist())
        mask = np.array([int(u) in keep_u for u in val_users.tolist()], dtype=bool)
        val_users, val_items, val_labels = val_users[mask], val_items[mask], val_labels[mask]
        A_va_s, H_va_s = A_va_s[mask], H_va_s[mask]

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
    fusion = build_fusion(architecture, encoder.context_dim, measure_dim, fcfg).to(device)
    model = RecommendationModel(encoder, fusion).to(device)
    item_offset = meta["item_offset"]

    u_all, i_all = user_item_to_nodes(train_users, train_items, item_offset)
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

    def score_split(users, items, A, H, zero_measure: bool = False):
        model.eval()
        with torch.no_grad():
            z = model.encoder.encode_all()
            u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
            scores = []
            for start in range(0, len(users), 8192):
                sl = slice(start, start + 8192)
                Hh = np.zeros_like(H[sl]) if zero_measure else H[sl]
                out = model(
                    u_idx[sl].to(device),
                    i_idx[sl].to(device),
                    torch.from_numpy(A[sl]).to(device),
                    torch.from_numpy(Hh).to(device),
                    z=z,
                )
                scores.append(out["logits"].detach().cpu().numpy())
        return np.concatenate(scores)

    best_state = None
    best_ndcg = -1.0
    best_metrics: dict[str, float] = {}
    left = int(acfg.get("patience", 4))
    history = []
    grad_log: list[dict[str, float]] = []
    epochs = int(max_epochs if max_epochs is not None else acfg.get("max_epochs", 15))

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        z_live = model.encoder.encode_all()
        batch_starts = list(range(0, n_train, batch_size))
        n_b = max(len(batch_starts), 1)
        last_grads = {"grad_graph": 0.0, "grad_tabular": 0.0, "grad_measure": 0.0}
        for bi, start in enumerate(batch_starts):
            idx = perm[start : start + batch_size]
            z = z_live if bi == 0 else z_live.detach()
            out = model(
                u_all[idx].to(device),
                i_all[idx].to(device),
                torch.from_numpy(A_tr_s[idx]).to(device),
                torch.from_numpy(H_tr_s[idx]).to(device),
                z=z,
            )
            loss = loss_fn(out["logits"], torch.from_numpy(y_all[idx]).to(device))
            (loss / n_b).backward()
            epoch_loss += float(loss.item())
            n_batches += 1
            if architecture == "branch" and isinstance(model.fusion_head, BranchFusionHead):
                last_grads = model.fusion_head.grad_norms()
        opt.step()
        if architecture == "branch":
            grad_log.append({"epoch": epoch, **last_grads})
            if last_grads.get("grad_measure", 0.0) == 0.0 and epoch == 0:
                print(f"[{stage}] WARN measure grad=0 at epoch 0", flush=True)

        val_scores = score_split(val_users, val_items, A_va_s, H_va_s)
        val_rank = ranking_metrics_for_users(
            val_users, val_labels, val_scores, ks=(20,)
        )
        ndcg = float(val_rank["NDCG@20"])
        recall = float(val_rank.get("Recall@20", float("nan")))
        mrr = float(val_rank.get("MRR", float("nan")))
        hit = _hitrate_at_k(val_users, val_labels, val_scores, k=20)
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(n_batches, 1),
                "val_NDCG@20": ndcg,
                "val_Recall@20": recall,
                "val_MRR": mrr,
                "val_HitRate@20": hit,
            }
        )
        print(
            f"[{stage}] epoch {epoch} loss={epoch_loss/max(n_batches,1):.4f} "
            f"val_NDCG@20={ndcg:.4f}",
            flush=True,
        )
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_metrics = {
                "sampled_ndcg20": ndcg,
                "sampled_recall20": recall,
                "sampled_mrr20": mrr,
                "sampled_hitrate20": hit,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = int(acfg.get("patience", 4))
            best_epoch = epoch
        else:
            left -= 1
            if left <= 0:
                break
    if "best_epoch" not in locals():
        best_epoch = history[-1]["epoch"] if history else 0

    if best_state:
        model.load_state_dict(best_state)

    # Branch utilization: measure zeroed
    ablation = None
    if architecture == "branch":
        scores_n = score_split(val_users, val_items, A_va_s, H_va_s, zero_measure=False)
        scores_z = score_split(val_users, val_items, A_va_s, H_va_s, zero_measure=True)
        m_n = ranking_metrics_for_users(val_users, val_labels, scores_n, ks=(20,))
        m_z = ranking_metrics_for_users(val_users, val_labels, scores_z, ks=(20,))
        ablation = {
            "normal_ndcg": float(m_n["NDCG@20"]),
            "zero_measure_ndcg": float(m_z["NDCG@20"]),
            "delta_measure": float(m_n["NDCG@20"]) - float(m_z["NDCG@20"]),
        }

    # Per-user metrics at best model
    val_scores = score_split(val_users, val_items, A_va_s, H_va_s)
    from collections import defaultdict

    by_u: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for u, y, s in zip(val_users.tolist(), val_labels.tolist(), val_scores.tolist()):
        by_u[int(u)].append((int(y), float(s)))
    user_rows = []
    for u, rows in by_u.items():
        labels = np.array([r[0] for r in rows], dtype=np.int64)
        scores = np.array([r[1] for r in rows], dtype=np.float64)
        # reuse batch metric on single user
        um = ranking_metrics_for_users(
            np.array([u] * len(labels)), labels, scores, ks=(20,)
        )
        user_rows.append(
            {
                "user_id": u,
                "ndcg20": float(um["NDCG@20"]),
                "recall20": float(um.get("Recall@20", float("nan"))),
                "mrr20": float(um.get("MRR", float("nan"))),
                "hitrate20": _hitrate_at_k(
                    np.array([u] * len(labels)), labels, scores, k=20
                ),
            }
        )

    ckpt_dir = ckpt_root / stage / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt_dir / "model.pt")
    with (ckpt_dir / "a_scaler.pkl").open("wb") as f:
        pickle.dump(a_scaler, f)
    with (ckpt_dir / "h_scaler.pkl").open("wb") as f:
        pickle.dump(h_scaler, f)
    runtime = time.time() - t0
    meta_out = {
        "stage": stage,
        "architecture": architecture,
        "seed": seed,
        "best_val_NDCG@20_sampled": best_ndcg,
        "best_epoch": int(best_epoch),
        "history": history,
        "item_offset": item_offset,
        "n_users": meta["n_users"],
        "n_entities": meta["n_entities"],
        "measure_dim": measure_dim,
        "A_feature_names": A_names,
        "early_stop": "sampled_val_NDCG@20",
        "runtime_sec": runtime,
        "n_val_users_ranked": len(by_u),
        "grad_log": grad_log,
        "branch_ablation": ablation,
        "metrics": best_metrics,
        "note": "SAMPLED SCREEN ONLY — do not treat as full-rank primary",
    }
    (ckpt_dir / "train_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    (ckpt_dir / "user_metrics.json").write_text(json.dumps(user_rows), encoding="utf-8")
    print(f"[{stage}] saved → {ckpt_dir} NDCG@20={best_ndcg:.4f}", flush=True)
    return {
        "ckpt_dir": str(ckpt_dir),
        "metrics": best_metrics,
        "best_epoch": meta_out["best_epoch"],
        "runtime_sec": runtime,
        "n_users": len(by_u),
        "branch_ablation": ablation,
        "grad_log": grad_log,
        "user_rows": user_rows,
    }
