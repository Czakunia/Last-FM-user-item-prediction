"""FULL_SOTA: minibatch HGT + A + optional HCR block (sampled eval + optional full-rank)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes
from src.lastfm_lp.data.build_splits import load_user_sets
from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.full_ranking import evaluate_full_ranking
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.features.hcr_h2_contract import shuffle_hcr_block
from src.lastfm_lp.models.encoders import HGTGraphEncoder
from src.lastfm_lp.models.fusion import LateFusionHead, RecommendationModel
from src.lastfm_lp.torch_device import resolve_torch_device


def _load_side_block(feat_dir: Path, names_want: list[str] | None = None) -> dict[str, Any]:
    names = json.loads((feat_dir / "feature_names.json").read_text(encoding="utf-8"))
    Xtr = np.load(feat_dir / "X_train.npy").astype(np.float32)
    Xva = np.load(feat_dir / "X_val.npy").astype(np.float32)
    Xte = np.load(feat_dir / "X_test.npy").astype(np.float32)
    if names_want is not None:
        idx = [names.index(n) for n in names_want]
        Xtr, Xva, Xte = Xtr[:, idx], Xva[:, idx], Xte[:, idx]
        names = list(names_want)
    return {"names": names, "train": Xtr, "val": Xva, "test": Xte}


def run_full_hgt_stage(
    bundle: dict[str, Any],
    *,
    stage: str,
    hcr_dir: Path | None,
    shuffle_hcr: bool = False,
    seed: int = 101,
    do_full_rank: bool = False,
) -> dict[str, Any]:
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_name = (
        acfg.get("device")
        or cfg.get("models", {}).get("device")
        or cfg.get("device")
        or "auto"
    )
    device = resolve_torch_device(str(device_name))
    print(f"[{stage} s={seed}] device={device} (requested={device_name})")

    feat_root = Path(cfg["paths"]["features"])
    A = _load_side_block(feat_root / "A")
    use_hcr = hcr_dir is not None
    if use_hcr:
        H = _load_side_block(Path(hcr_dir))
        if shuffle_hcr:
            H = {
                "names": H["names"],
                "train": shuffle_hcr_block(
                    H["train"],
                    mode="popularity_stratified",
                    popularity=bundle["popularity"],
                    item_ids=bundle["train_pairs"]["item_id"],
                    seed=seed + 17,
                ),
                "val": shuffle_hcr_block(
                    H["val"],
                    mode="popularity_stratified",
                    popularity=bundle["popularity"],
                    item_ids=bundle["val_pairs"]["item_id"],
                    seed=seed + 31,
                ),
                "test": shuffle_hcr_block(
                    H["test"],
                    mode="popularity_stratified",
                    popularity=bundle["popularity"],
                    item_ids=bundle["test_pairs"]["item_id"],
                    seed=seed + 47,
                ),
            }
        h_scaler = StandardScaler().fit(H["train"])
        H_tr = h_scaler.transform(H["train"]).astype(np.float32)
        H_va = h_scaler.transform(H["val"]).astype(np.float32)
        H_te = h_scaler.transform(H["test"]).astype(np.float32)
        hcr_dim = H_tr.shape[1]
    else:
        H_tr = H_va = H_te = None
        hcr_dim = 0

    a_scaler = StandardScaler().fit(A["train"])
    A_tr = a_scaler.transform(A["train"]).astype(np.float32)
    A_va = a_scaler.transform(A["val"]).astype(np.float32)
    A_te = a_scaler.transform(A["test"]).astype(np.float32)

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
        hidden_dim=int(fcfg.get("hidden_dim", 128)),
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

    for epoch in range(int(acfg.get("max_epochs", 15))):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        # Encode once per epoch. First minibatch backprops through the live
        # encoder tape; later batches use z.detach() so we do not need
        # retain_graph across hundreds of FULL batches (too slow / heavy).
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
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "val_NDCG@20": ndcg,
            }
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

    result: dict[str, Any] = {
        "stage": stage,
        "protocol": cfg["protocol"],
        "seed": seed,
        "use_hcr": use_hcr,
        "hcr_shuffle": shuffle_hcr,
        "hcr_dim": hcr_dim,
        "train_info": {"history": history, "best_val_NDCG@20": best_ndcg},
        "validation_sampled": val_metrics,
        "test_sampled": test_metrics,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_val": val_metrics.get("NDCG@20"),
        "primary_test_sampled": test_metrics.get("NDCG@20"),
    }

    if do_full_rank:
        print(f"[{stage}] starting full-ranking test …")
        splits_dir = Path(cfg["paths"]["splits"])
        test_by_user = load_user_sets(splits_dir / "test.txt")
        n_items = int(bundle["popularity"].shape[0])
        # Precompute side features on the fly is heavy; for full-rank use graph-only
        # score + mean HCR proxy is wrong. Instead: score graph+A+H2 with
        # on-the-fly H2 would need index. Here we use encoder pair score only
        # fused with popularity baselines via a lightweight path:
        # re-score using model with A/H from nearest train-row stats is invalid.
        # Practical FULL overnight: full-rank with graph context + zeros for missing
        # side features (conservative lower bound) OR skip if no on-the-fly.
        # Better: use sampled metrics as primary overnight; full-rank graph-only.
        model.eval()
        with torch.no_grad():
            z = model.encoder.encode_all()

        def score_fn(u: int, items: np.ndarray) -> np.ndarray:
            # graph-only fusion: A/H zeros (document as graph-rank probe)
            uu = np.full(len(items), u, dtype=np.int64)
            u_idx, i_idx = user_item_to_nodes(uu, items, item_offset)
            A0 = np.zeros((len(items), 5), dtype=np.float32)
            H0 = np.zeros((len(items), hcr_dim), dtype=np.float32) if use_hcr else None
            out = model(
                u_idx.to(device),
                i_idx.to(device),
                torch.from_numpy(A0).to(device),
                torch.from_numpy(H0).to(device) if H0 is not None else None,
                z=z,
            )
            return out["logits"].detach().cpu().numpy()

        fr = evaluate_full_ranking(
            eval_users=bundle["eval_users"],
            test_positives=test_by_user,
            model_train=bundle["model_train"],
            n_items=n_items,
            score_fn=score_fn,
            chunk_size=int(cfg["evaluation"].get("candidate_chunk_size", 2048)),
            ks=tuple(ks),
            exclude_train=bool(cfg["evaluation"].get("exclude_train_items", True)),
            max_positives_per_user=int(
                cfg["evaluation"].get("max_test_positives_per_user", 10)
            ),
        )
        result["test_full_ranking_graphprobe"] = fr
        result["full_ranking_note"] = (
            "Graph probe with zeroed A/H side features; sampled test is primary overnight."
        )
        print(f"[{stage}] full-rank probe NDCG@20={fr.get('NDCG@20', float('nan')):.4f}")

    out_dir = Path(cfg["paths"]["stage_full"]) / stage / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out_dir / "test_scores_sampled.npy", test_scores)
    print(
        f"[{stage} s={seed}] sampled test NDCG@20={result['primary_test_sampled']:.4f} "
        f"AUPRC={test_metrics['AUPRC']:.4f} MRR={test_metrics['MRR']:.4f}"
    )
    return result
