#!/usr/bin/env python3
"""Publication full-ranking for OUR_F1 (HGT+A) and primary (HGT+A+A11).

Retrains (V1 had no saved weights), saves checkpoints+scalers, then evaluates
with REAL on-demand A / A11 under publication_full_rank_evaluator.

Never uses the zeroed-A/H graphprobe path.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG, materialized_root  # noqa: E402

from src.lastfm_lp.binary.user_hcr_aggregation import aggregate_a11_energy_user_batch
from src.lastfm_lp.config import load_protocol_config
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes
from src.lastfm_lp.data.build_splits import load_user_sets
from src.lastfm_lp.evaluation.publication_full_rank_evaluator import (
    evaluate_user_dense,
)
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.features.tabular_pair_features import TABULAR_BASE_FEATURES
from src.lastfm_lp.models.encoders import HGTGraphEncoder
from src.lastfm_lp.models.fusion import KANFusionHead, LateFusionHead, RecommendationModel
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit
from src.lastfm_lp.pipeline.prepare import load_prepared
from src.lastfm_lp.torch_device import resolve_torch_device

OUT = ROOT / "outputs" / "publication_sota" / "our_models"
CFG_PATH = "configs/lastfm_full_sota_v1.yaml"


def _vectorized_A_for_user(
    user_id: int,
    candidates: np.ndarray,
    model_train: dict[int, set[int]],
    popularity: np.ndarray,
    item_kg_degree: np.ndarray,
    pairwise,
) -> np.ndarray:
    """(C,5) tabular A features — same semantics as build_tabular_pair_features."""
    hist = model_train.get(user_id, set())
    cands = np.asarray(candidates, dtype=np.int64).reshape(-1)
    C = len(cands)
    X = np.zeros((C, 5), dtype=np.float32)
    X[:, 0] = np.log1p(len(hist))
    pop_c = popularity[cands].astype(np.float64)
    X[:, 1] = np.log1p(pop_c)
    X[:, 2] = np.log1p(item_kg_degree[cands].astype(np.float64))
    if not hist:
        return X
    sample = np.asarray(sorted(hist)[:40], dtype=np.int64)
    if pairwise is not None and sample.size:
        n11 = pairwise.cooccurrence_block(sample, cands)  # (H, C)
        X[:, 3] = (n11 > 0).mean(axis=0).astype(np.float32)
    hist_pops = popularity[list(hist)].astype(np.float64)
    hist_norm = float(np.linalg.norm(hist_pops))
    mean_hist_pop = float(hist_pops.mean())
    den = hist_norm * np.maximum(pop_c, 1.0)
    X[:, 4] = np.where(den > 0, (pop_c * mean_hist_pop) / den, 0.0).astype(np.float32)
    return X


def build_fusion_head(
    *,
    encoder: HGTGraphEncoder,
    hcr_dim: int,
    use_a11: bool,
    fcfg: dict[str, Any],
) -> nn.Module:
    """MLP LateFusionHead or in-repo KANFusionHead from fusion cfg."""
    kind = str(fcfg.get("type", "late_concat")).lower()
    if kind in {"kan", "kan_fusion"}:
        return KANFusionHead(
            graph_dim=encoder.context_dim,
            base_feature_dim=5,
            hcr_feature_dim=hcr_dim,
            hidden_dim=int(fcfg.get("kan_hidden_dim", 64)),
            dropout=float(fcfg.get("kan_dropout", 0.1)),
            use_hcr=use_a11,
            grid_size=int(fcfg.get("kan_grid_size", 5)),
            spline_order=int(fcfg.get("kan_spline_order", 3)),
        )
    return LateFusionHead(
        graph_dim=encoder.context_dim,
        base_feature_dim=5,
        hcr_feature_dim=hcr_dim,
        hidden_dim=int(fcfg.get("hidden_dim", 128)),
        dropout=float(fcfg.get("dropout", 0.2)),
        use_hcr=use_a11,
        activation=str(fcfg.get("activation", "gelu")),
    )


def train_and_save(
    bundle: dict[str, Any],
    *,
    stage: str,
    use_a11: bool,
    seed: int = 101,
    hcr_feature_dir: Path | None = None,
    a_feature_dir: Path | None = None,
    ckpt_root: Path | None = None,
    typed_graph: dict[str, Any] | None = None,
    zero_h_after_scale: bool = False,
) -> Path:
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    print(f"[{stage}] train device={device}", flush=True)

    feat_root = Path(cfg["paths"]["features"])
    a_dir = Path(a_feature_dir) if a_feature_dir is not None else (feat_root / "A")
    A_names = json.loads((a_dir / "feature_names.json").read_text())
    A_tr = np.load(a_dir / "X_train.npy").astype(np.float32)
    A_va = np.load(a_dir / "X_val.npy").astype(np.float32)
    a_scaler = StandardScaler().fit(A_tr)
    A_tr_s = a_scaler.transform(A_tr).astype(np.float32)
    A_va_s = a_scaler.transform(A_va).astype(np.float32)

    h_scaler = None
    H_tr_s = H_va_s = None
    hcr_dim = 0
    if use_a11:
        hdir = Path(hcr_feature_dir) if hcr_feature_dir is not None else (feat_root / "stage_h" / "H1_a11")
        H_tr = np.load(hdir / "X_train.npy").astype(np.float32)
        H_va = np.load(hdir / "X_val.npy").astype(np.float32)
        h_scaler = StandardScaler().fit(H_tr)
        H_tr_s = h_scaler.transform(H_tr).astype(np.float32)
        H_va_s = h_scaler.transform(H_va).astype(np.float32)
        hcr_dim = H_tr_s.shape[1]
        if zero_h_after_scale:
            H_tr_s = np.zeros_like(H_tr_s)
            H_va_s = np.zeros_like(H_va_s)

    if typed_graph is not None:
        graph = typed_graph
    else:
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
    fusion = build_fusion_head(
        encoder=encoder, hcr_dim=hcr_dim, use_a11=use_a11, fcfg=fcfg
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

    best_state = None
    best_ndcg = -1.0
    left = int(acfg.get("patience", 4))
    history = []
    # Early stop on sampled val (documented); full-rank test is the publication metric
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
                torch.from_numpy(A_tr_s[idx]).to(device),
                torch.from_numpy(H_tr_s[idx]).to(device) if H_tr_s is not None else None,
                z=z,
            )
            loss = loss_fn(out["logits"], torch.from_numpy(y_all[idx]).to(device))
            (loss / n_b).backward()
            epoch_loss += float(loss.item())
            n_batches += 1
        opt.step()
        val_scores = score_split(
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["item_id"],
            A_va_s,
            H_va_s,
        )
        val_rank = ranking_metrics_for_users(
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["label"],
            val_scores,
            ks=(20,),
        )
        ndcg = float(val_rank["NDCG@20"])
        history.append({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1), "val_NDCG@20": ndcg})
        print(f"[{stage}] epoch {epoch} loss={epoch_loss/max(n_batches,1):.4f} val_NDCG@20={ndcg:.4f}", flush=True)
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

    root = Path(ckpt_root) if ckpt_root is not None else OUT
    ckpt_dir = root / stage / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt_dir / "model.pt")
    with (ckpt_dir / "a_scaler.pkl").open("wb") as f:
        pickle.dump(a_scaler, f)
    if h_scaler is not None:
        with (ckpt_dir / "h_scaler.pkl").open("wb") as f:
            pickle.dump(h_scaler, f)
    meta_out = {
        "stage": stage,
        "use_a11": use_a11,
        "seed": seed,
        "best_val_NDCG@20_sampled": best_ndcg,
        "history": history,
        "item_offset": item_offset,
        "n_users": meta["n_users"],
        "n_entities": meta["n_entities"],
        "hcr_dim": hcr_dim,
        "A_feature_names": A_names,
        "early_stop": "sampled_val_NDCG@20",
        "hcr_feature_dir": str(hcr_feature_dir) if hcr_feature_dir is not None else None,
        "zero_h_after_scale": bool(zero_h_after_scale),
        "decoder_input_note": "265D kept; H- zeros the scaled 3D item-A11 block",
        "fusion_type": str(fcfg.get("type", "late_concat")),
        "fusion": {k: fcfg.get(k) for k in (
            "type", "activation", "hidden_dim", "dropout",
            "kan_hidden_dim", "kan_dropout", "kan_grid_size", "kan_spline_order",
        ) if k in fcfg or k == "type"},
        "note": "Publication test uses all-ranking with real on-demand A/A11",
    }
    (ckpt_dir / "train_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"[{stage}] saved checkpoint → {ckpt_dir}", flush=True)
    return ckpt_dir


def fullrank_evaluate(
    bundle: dict[str, Any],
    ckpt_dir: Path,
    *,
    stage: str,
    use_a11: bool,
    seed: int = 101,
    max_users: int | None = None,
    shard_users: int | None = None,
) -> dict[str, Any]:
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    # MPS silently dies mid-eval (~800 users) under memory pressure on this host.
    # CPU eval is stable; encode+fusion for 48k items is still ~10+ users/s.
    device = resolve_torch_device("cpu")
    print(f"[{stage}] full-rank eval device={device}", flush=True)

    meta_t = json.loads((ckpt_dir / "train_meta.json").read_text())
    with (ckpt_dir / "a_scaler.pkl").open("rb") as f:
        a_scaler: StandardScaler = pickle.load(f)
    h_scaler = None
    if use_a11:
        with (ckpt_dir / "h_scaler.pkl").open("rb") as f:
            h_scaler = pickle.load(f)

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
    hcr_dim = int(meta_t["hcr_dim"])
    eval_fcfg = dict(fcfg)
    if meta_t.get("fusion_type"):
        eval_fcfg["type"] = meta_t["fusion_type"]
    if isinstance(meta_t.get("fusion"), dict):
        eval_fcfg.update(meta_t["fusion"])
    fusion = build_fusion_head(
        encoder=encoder, hcr_dim=hcr_dim, use_a11=use_a11, fcfg=eval_fcfg
    ).to(device)
    model = RecommendationModel(encoder, fusion).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.eval()
    item_offset = meta["item_offset"]
    n_items = int(bundle["popularity"].shape[0])

    cf = ensure_cross_fit(bundle)
    pop = bundle["popularity"]
    kg_deg = bundle["item_kg_degree"]
    model_train = bundle["model_train"]

    splits_dir = Path(cfg["paths"]["splits"])
    test_by_user = load_user_sets(splits_dir / "test.txt")
    # Publication mask = official train history (includes val carve), KGAT-faithful
    official_train = load_user_sets(ROOT / "data" / "KGAT_LastFM" / "train.txt")

    eval_users = sorted(u for u in test_by_user if test_by_user[u])
    if max_users is not None:
        rng = np.random.default_rng(2026)
        eval_users = [
            int(x)
            for x in rng.choice(eval_users, size=min(max_users, len(eval_users)), replace=False)
        ]

    with torch.no_grad():
        z = model.encoder.encode_all()
        # Precompute all item embeddings once (entities[item_offset : item_offset+n_items])
        zi_all = z[item_offset : item_offset + n_items].detach()
    fusion_head = model.fusion_head
    fusion_head.eval()

    a_mean = a_scaler.mean_.astype(np.float32)
    a_scale = a_scaler.scale_.astype(np.float32)
    h_mean = h_scaler.mean_.astype(np.float32) if h_scaler is not None else None
    h_scale = h_scaler.scale_.astype(np.float32) if h_scaler is not None else None
    all_items = np.arange(n_items, dtype=np.int64)

    t0 = time.time()
    ks = (5, 10, 20)
    sums = {f"Recall@{k}": 0.0 for k in ks}
    sums.update({f"NDCG@{k}": 0.0 for k in ks})
    sums.update({f"HitRate@{k}": 0.0 for k in ks})
    sums["MRR"] = 0.0
    n_u = 0
    fuse_chunk = 16384
    start_ui = 0
    partial_path = OUT / f"{stage}_FULLRANK.partial.json"
    if partial_path.exists() and max_users is None:
        try:
            prev = json.loads(partial_path.read_text())
            if int(prev.get("next_ui", 0)) > 0 and prev.get("stage") == stage:
                # Require HitRate sums so mean is not diluted by pre-HitRate resume state
                if all(f"sum_HitRate@{k}" in prev for k in ks) or int(prev.get("n_users_done", 0)) == 0:
                    start_ui = int(prev["next_ui"])
                    for k in sums:
                        sums[k] = float(prev.get("sum_" + k, 0.0))
                    n_u = int(prev.get("n_users_done", 0))
                    print(f"[{stage}] RESUME from user index {start_ui} (n_done={n_u})", flush=True)
                else:
                    print(
                        f"[{stage}] partial lacks HitRate — restarting full-rank from user 0",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[{stage}] resume skipped: {exc}", flush=True)

    def _save_partial(ui: int) -> dict[str, Any]:
        partial = {k: float(v / max(n_u, 1)) for k, v in sums.items()}
        partial.update(
            {
                "n_users_done": float(n_u),
                "next_ui": float(ui + 1),
                "partial": True,
                "stage": stage,
                **{f"sum_{k}": float(v) for k, v in sums.items()},
            }
        )
        partial_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        return partial

    new_in_shard = 0
    for ui, u in enumerate(eval_users):
        if ui < start_ui:
            continue
        pw = cf.index_for(int(u), split="test")
        hist = model_train.get(int(u), set())
        A = _vectorized_A_for_user(int(u), all_items, model_train, pop, kg_deg, pw)
        A_s = ((A - a_mean) / a_scale).astype(np.float32)
        H_s = None
        if use_a11:
            H = aggregate_a11_energy_user_batch(hist, all_items, pw, max_history=25, kind="a11")
            H_s = ((H - h_mean) / h_scale).astype(np.float32)

        with torch.no_grad():
            zu = z[int(u)]
            graph_score = (zi_all * zu).sum(dim=-1)
            scores = np.empty(n_items, dtype=np.float64)
            for start in range(0, n_items, fuse_chunk):
                end = min(start + fuse_chunk, n_items)
                zi = zi_all[start:end]
                ctx = torch.cat(
                    [zu.expand(end - start, -1), zi, zu * zi, (zu - zi).abs()], dim=-1
                )
                base = torch.from_numpy(A_s[start:end]).to(device)
                hcr = (
                    torch.from_numpy(H_s[start:end]).to(device)
                    if H_s is not None
                    else None
                )
                logits = fusion_head(
                    graph_context=ctx,
                    graph_score=graph_score[start:end],
                    base_features=base,
                    hcr_features=hcr,
                )
                scores[start:end] = logits.detach().cpu().numpy()

        m = evaluate_user_dense(
            scores,
            positive_items=set(test_by_user[u]),
            train_items=set(official_train.get(u, ())),
            ks=ks,
        )
        for k, v in m.items():
            if k not in sums:
                sums[k] = 0.0
            sums[k] += float(v)
        n_u += 1
        new_in_shard += 1
        if (ui + 1) % 50 == 0 or ui == 0:
            rate = new_in_shard / max(time.time() - t0, 1e-9)
            eta = (len(eval_users) - ui - 1) / max(rate, 1e-9)
            print(
                f"[{stage}] users {ui+1}/{len(eval_users)} "
                f"NDCG@20={sums['NDCG@20']/n_u:.4f} "
                f"users/s={rate:.3f} eta_h={eta/3600:.2f}",
                flush=True,
            )
            _save_partial(ui)
        if shard_users is not None and new_in_shard >= int(shard_users):
            partial = _save_partial(ui)
            print(
                f"[{stage}] SHARD DONE at {ui+1}/{len(eval_users)} "
                f"(processed {new_in_shard} this process)",
                flush=True,
            )
            return partial

    metrics = {k: float(v / n_u) for k, v in sums.items()}
    metrics.update(
        {
            "n_users": float(n_u),
            "stage": stage,
            "use_a11": use_a11,
            "seed": seed,
            "wall_hours": (time.time() - t0) / 3600.0,
            "protocol": "LASTFM_PUBLICATION_PROTOCOL_V1",
            "evaluator": "publication_full_rank_evaluator.evaluate_user_dense",
            "features": "real_on_demand_A" + ("_A11" if use_a11 else ""),
            "train_checkpoint": str(ckpt_dir),
            "sampled_val_NDCG@20": meta_t.get("best_val_NDCG@20_sampled"),
            "eval_mode": "dense_fast_fusion",
        }
    )
    out_path = OUT / f"{stage}_FULLRANK.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "outputs/publication_sota" / f"{stage}_FULLRANK.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if partial_path.exists():
        partial_path.unlink()
    print(f"[{stage}] FULLRANK DONE", json.dumps({k: metrics[k] for k in metrics if k.startswith(("Recall", "NDCG", "MRR", "n_"))}, indent=2), flush=True)
    return metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_protocol_config(CFG_PATH)
    # write publication stage outputs separately
    cfg = dict(cfg)
    cfg.setdefault("paths", {})
    # keep features/splits from V1
    bundle = load_prepared(cfg, verify=False)
    bundle["cfg"] = cfg

    stages = [
        ("OUR_F1_HGT_A", False),
        ("OUR_F2_HGT_A_A11", True),
    ]
    # optional: --smoke N | --stage NAME | --shard N (exit after N new users; watchdog loops)
    smoke = None
    if "--smoke" in sys.argv:
        i = sys.argv.index("--smoke")
        smoke = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 100
    shard = 400
    if "--shard" in sys.argv:
        shard = int(sys.argv[sys.argv.index("--shard") + 1])
    if smoke is not None:
        shard = None  # smoke runs in one shot
    only_stage = None
    if "--stage" in sys.argv:
        only_stage = sys.argv[sys.argv.index("--stage") + 1]
    if only_stage:
        stages = [(s, u) for s, u in stages if s == only_stage]

    results = {}
    for stage, use_a11 in stages:
        final = ROOT / "outputs" / "publication_sota" / f"{stage}_FULLRANK.json"
        if smoke is None and final.exists():
            try:
                prev = json.loads(final.read_text())
                if float(prev.get("n_users", 0)) >= 20000 and not prev.get("partial"):
                    print(f"[{stage}] SKIP — final full-rank already present", flush=True)
                    results[stage] = prev
                    continue
            except Exception:
                pass
        ckpt = OUT / stage / "seed_101"
        if not (ckpt / "model.pt").exists():
            train_and_save(bundle, stage=stage, use_a11=use_a11, seed=101)
        else:
            print(f"[{stage}] reuse checkpoint {ckpt}", flush=True)
        results[stage] = fullrank_evaluate(
            bundle,
            ckpt,
            stage=stage,
            use_a11=use_a11,
            seed=101,
            max_users=smoke,
            shard_users=shard,
        )

    summary = {
        "popularity_ref": json.loads((ROOT / "outputs/publication_sota/POPULARITY_FULLRANK.json").read_text())
        if (ROOT / "outputs/publication_sota/POPULARITY_FULLRANK.json").exists()
        else None,
        "bprmf_ref": json.loads((ROOT / "outputs/publication_sota/BPRMF_FULLRANK.json").read_text())
        if (ROOT / "outputs/publication_sota/BPRMF_FULLRANK.json").exists()
        else None,
        "ours": results,
    }
    (ROOT / "outputs/publication_sota/OUR_FULLRANK_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
