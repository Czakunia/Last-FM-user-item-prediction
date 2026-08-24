"""Stage F — KG-conditioned HCR (KG-HCR) on top of HGT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.lastfm_lp.binary.kg_hcr_aggregation import history_pair_matrix
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes
from src.lastfm_lp.data.load_kgat_lastfm import load_lastfm
from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users
from src.lastfm_lp.kg.path_index import KGPathIndex
from src.lastfm_lp.models.hgt_encoder import HGTEncoder
from src.lastfm_lp.models.kg_hcr_fusion import GatedKGHCRModel, HistoryAttentionAggregator
from src.lastfm_lp.models.mlp_decoder import MLPDecoder
from src.lastfm_lp.pipeline.features_kg_hcr import materialize_kg_hcr_sides
from src.lastfm_lp.pipeline.run_graph_stage import _load_side_features

STAGE_SPECS = {
    "F0": {"kind": "reference_e3"},
    "F1": {"kind": "flat", "mask_no_path": False, "tag": "F1_kg_hcr"},
    "F2": {"kind": "flat", "mask_no_path": True, "tag": "F2_kg_masked_hcr"},
    "F3": {"kind": "attn", "mask_no_path": False},
    "F4": {"kind": "gated", "mask_no_path": False},
}


def _load_tabular_A(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    pack = _load_side_features(cfg, tabular=True, hcr=False)
    assert pack is not None
    return pack


def _hgt_encode_fns(cfg: dict[str, Any], bundle: dict[str, Any], device: torch.device):
    ecfg = cfg.get("models", {}).get("stage_e", {})
    fcfg = cfg.get("models", {}).get("stage_f", {})
    max_kg = fcfg.get("max_kg_edges", ecfg.get("max_kg_edges", 250_000))
    embed_dim = int(fcfg.get("hgt_embed_dim", ecfg.get("hgt_embed_dim", 32)))
    graph = load_data_and_typed_graph(cfg, bundle["model_train"], max_kg_edges=max_kg)
    meta = graph["meta"]
    edge_index_dict = {k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel()}
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    encoder = HGTEncoder(
        meta["n_users"],
        meta["n_entities"],
        metadata=metadata,
        embed_dim=embed_dim,
        n_layers=int(fcfg.get("n_layers", ecfg.get("n_layers", 2))),
        heads=int(fcfg.get("heads", ecfg.get("heads", 2))),
        dropout=float(fcfg.get("dropout", 0.1)),
    ).to(device)

    def encode_fn() -> torch.Tensor:
        return encoder(edge_index_dict)

    return encoder, encode_fn, meta, embed_dim


def _train_flat_hgt(
    bundle: dict[str, Any],
    stage: str,
    side_extra: dict[str, np.ndarray],
) -> dict[str, Any]:
    """F1/F2: HGT + MLP on [pair_repr, tabular_A, kg_hcr_agg]."""
    cfg = bundle["cfg"]
    fcfg = cfg.get("models", {}).get("stage_f", {})
    seed = int(cfg["models"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    encoder, encode_fn, meta, embed_dim = _hgt_encode_fns(cfg, bundle, device)
    item_offset = meta["item_offset"]
    tab = _load_tabular_A(cfg)

    def stack(split: str) -> np.ndarray:
        return np.concatenate([tab[split], side_extra[split]], axis=1).astype(np.float32)

    side_train = stack("train")
    side_val = stack("val")
    side_test = stack("test")
    scaler = StandardScaler()
    side_train = scaler.fit_transform(side_train).astype(np.float32)
    side_val = scaler.transform(side_val).astype(np.float32)
    side_test = scaler.transform(side_test).astype(np.float32)
    side_dim = side_train.shape[1]
    side_names = [f"A:{n}" for n in tab["names"]] + [f"KGHCR:{n}" for n in side_extra["names"]]

    mcfg = cfg["models"]["mlp"]
    decoder = MLPDecoder(
        embed_dim, side_dim=side_dim, hidden_dims=mcfg["hidden_dims"], dropout=mcfg["dropout"]
    ).to(device)

    return _fit_eval_hgt_decoder(
        bundle,
        stage,
        encoder,
        encode_fn,
        decoder,
        meta,
        side_train,
        side_val,
        side_test,
        side_names,
        device,
        fcfg,
    )


def _fit_eval_hgt_decoder(
    bundle,
    stage,
    encoder,
    encode_fn,
    decoder,
    meta,
    side_train,
    side_val,
    side_test,
    side_names,
    device,
    fcfg,
) -> dict[str, Any]:
    item_offset = meta["item_offset"]
    train_y = bundle["train_pairs"]["label"].astype(np.float32)
    u_t, i_t = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    u_t, i_t = u_t.to(device), i_t.to(device)
    y_t = torch.from_numpy(train_y).to(device)
    side_t = torch.from_numpy(side_train).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(fcfg.get("lr", 1e-3)),
        weight_decay=float(fcfg.get("weight_decay", 1e-4)),
    )
    n_pos = float((train_y > 0.5).sum())
    n_neg = float(len(train_y) - n_pos)
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state, best_ndcg, patience_left = None, -1.0, int(fcfg.get("patience", 5))
    history = []
    max_epochs = int(fcfg.get("max_epochs", 20))

    for epoch in range(max_epochs):
        encoder.train()
        decoder.train()
        opt.zero_grad()
        z = encode_fn()
        loss = loss_fn(decoder(z[u_t], z[i_t], side_t), y_t)
        loss.backward()
        opt.step()

        val_scores = _score_flat(encode_fn, decoder, bundle, "val", side_val, item_offset, device)
        ndcg = ranking_metrics_for_users(
            bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"], val_scores, ks=[20]
        )["NDCG@20"]
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_ndcg": float(ndcg)})
        print(f"[{stage}] epoch {epoch} train_loss={loss.item():.4f} val_NDCG@20={ndcg:.4f}")
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = float(ndcg)
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            }
            patience_left = int(fcfg.get("patience", 5))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        encoder.to(device)
        decoder.to(device)

    return _pack_result(
        bundle,
        stage,
        encode_fn,
        decoder,
        meta,
        side_val,
        side_test,
        side_names,
        history,
        best_ndcg,
        item_offset,
        device,
        model_kind="flat_hgt",
    )


def _score_flat(encode_fn, decoder, bundle, split, side, item_offset, device):
    pairs = bundle[f"{split}_pairs"]
    decoder.eval()
    with torch.no_grad():
        z = encode_fn()
        u_idx, i_idx = user_item_to_nodes(pairs["user_id"], pairs["item_id"], item_offset)
        scores = []
        bs = 8192
        for start in range(0, len(pairs["label"]), bs):
            sl = slice(start, start + bs)
            s_side = torch.from_numpy(side[sl]).to(device)
            scores.append(
                decoder(z[u_idx[sl].to(device)], z[i_idx[sl].to(device)], s_side).cpu().numpy()
            )
    return np.concatenate(scores).astype(np.float64)


def _pack_result(
    bundle,
    stage,
    encode_fn,
    decoder,
    meta,
    side_val,
    side_test,
    side_names,
    history,
    best_ndcg,
    item_offset,
    device,
    model_kind: str,
    score_fn=None,
):
    cfg = bundle["cfg"]
    if score_fn is None:
        val_scores = _score_flat(encode_fn, decoder, bundle, "val", side_val, item_offset, device)
        test_scores = _score_flat(encode_fn, decoder, bundle, "test", side_test, item_offset, device)
    else:
        val_scores = score_fn("val")
        test_scores = score_fn("test")

    ks = cfg["evaluation"]["ranking_k"]
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
        "model_kind": model_kind,
        "side_features": side_names,
        "graph_meta": {k: v for k, v in meta.items() if k != "metadata"},
        "train_info": {
            "history": history,
            "best_val_ndcg": best_ndcg,
            "epochs_ran": len(history),
        },
        "validation": val_metrics,
        "test": test_metrics,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
    }
    out = Path(cfg["paths"].get("stage_f", str(Path(cfg["paths"]["outputs_root"]) / "stage_f")))
    out = Path(out) / stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "test_scores.npy", test_scores)
    print(
        f"[{stage}] test {result['primary_metric']}={result['primary_test']:.4f} "
        f"AUPRC={test_metrics['AUPRC']:.4f} MRR={test_metrics['MRR']:.4f}"
    )
    return result


def _materialize_hist_bank(
    bundle: dict[str, Any],
    *,
    mask_no_path: bool,
    max_history: int = 25,
    tag: str = "F_hist_bank",
) -> dict[str, Any]:
    cfg = bundle["cfg"]
    out_dir = Path(cfg["paths"]["features"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "train_feats.npy").exists():
        return {
            "train_feats": np.load(out_dir / "train_feats.npy"),
            "train_mask": np.load(out_dir / "train_mask.npy"),
            "train_ids": np.load(out_dir / "train_ids.npy"),
            "val_feats": np.load(out_dir / "val_feats.npy"),
            "val_mask": np.load(out_dir / "val_mask.npy"),
            "val_ids": np.load(out_dir / "val_ids.npy"),
            "test_feats": np.load(out_dir / "test_feats.npy"),
            "test_mask": np.load(out_dir / "test_mask.npy"),
            "test_ids": np.load(out_dir / "test_ids.npy"),
        }

    data = load_lastfm(cfg["data"]["path"])
    kg_index = KGPathIndex(data.kg, data.n_entities, data.n_relations)
    hcr_index = bundle["index"]
    model_train = bundle["model_train"]

    def build(pairs, desc):
        n = len(pairs["label"])
        # probe dim
        f0, m0, i0 = history_pair_matrix([], 0, hcr_index, kg_index, max_history=max_history)
        feats = np.zeros((n, max_history, f0.shape[1]), dtype=np.float32)
        mask = np.zeros((n, max_history), dtype=np.float32)
        ids = -np.ones((n, max_history), dtype=np.int32)
        for r in tqdm(range(n), desc=desc, mininterval=2.0):
            u = int(pairs["user_id"][r])
            i = int(pairs["item_id"][r])
            y = int(pairs["label"][r])
            hist = set(model_train.get(u, ()))
            if y == 1 and i in hist:
                hist = hist - {i}
            mat, msk, hid = history_pair_matrix(
                hist, i, hcr_index, kg_index, max_history=max_history, mask_no_path=mask_no_path
            )
            feats[r] = mat
            mask[r] = msk
            ids[r] = np.array(hid, dtype=np.int32)
        return feats, mask, ids

    tr = build(bundle["train_pairs"], f"{tag} train")
    va = build(bundle["val_pairs"], f"{tag} val")
    te = build(bundle["test_pairs"], f"{tag} test")
    np.save(out_dir / "train_feats.npy", tr[0])
    np.save(out_dir / "train_mask.npy", tr[1])
    np.save(out_dir / "train_ids.npy", tr[2])
    np.save(out_dir / "val_feats.npy", va[0])
    np.save(out_dir / "val_mask.npy", va[1])
    np.save(out_dir / "val_ids.npy", va[2])
    np.save(out_dir / "test_feats.npy", te[0])
    np.save(out_dir / "test_mask.npy", te[1])
    np.save(out_dir / "test_ids.npy", te[2])
    return {
        "train_feats": tr[0],
        "train_mask": tr[1],
        "train_ids": tr[2],
        "val_feats": va[0],
        "val_mask": va[1],
        "val_ids": va[2],
        "test_feats": te[0],
        "test_mask": te[1],
        "test_ids": te[2],
    }


def _train_attn_or_gated(bundle: dict[str, Any], stage: str, kind: str) -> dict[str, Any]:
    cfg = bundle["cfg"]
    fcfg = cfg.get("models", {}).get("stage_f", {})
    seed = int(cfg["models"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    max_history = int(fcfg.get("max_history", 25))

    bank = _materialize_hist_bank(
        bundle, mask_no_path=False, max_history=max_history, tag="F_hist_bank"
    )
    encoder, encode_fn, meta, embed_dim = _hgt_encode_fns(cfg, bundle, device)
    item_offset = meta["item_offset"]
    pair_feat_dim = bank["train_feats"].shape[-1]

    # scale pair feats using train
    flat = bank["train_feats"].reshape(-1, pair_feat_dim)
    msk = bank["train_mask"].reshape(-1) > 0
    scaler = StandardScaler()
    scaler.fit(flat[msk] if msk.any() else flat)
    for split in ("train", "val", "test"):
        x = bank[f"{split}_feats"]
        sh = x.shape
        bank[f"{split}_feats"] = scaler.transform(x.reshape(-1, sh[-1])).reshape(sh).astype(
            np.float32
        )

    attn = HistoryAttentionAggregator(embed_dim, pair_feat_dim, hidden=64).to(device)
    mcfg = cfg["models"]["mlp"]

    if kind == "attn":
        # F3: concat attended KG-HCR with tabular A into MLP decoder
        tab = _load_tabular_A(cfg)
        tab_scaler = StandardScaler()
        tab_train = tab_scaler.fit_transform(tab["train"]).astype(np.float32)
        tab_val = tab_scaler.transform(tab["val"]).astype(np.float32)
        tab_test = tab_scaler.transform(tab["test"]).astype(np.float32)
        side_dim = pair_feat_dim + tab_train.shape[1]
        decoder = MLPDecoder(
            embed_dim, side_dim=side_dim, hidden_dims=mcfg["hidden_dims"], dropout=mcfg["dropout"]
        ).to(device)
        params = list(encoder.parameters()) + list(attn.parameters()) + list(decoder.parameters())
        gate_model = None
    else:
        # F4 gated residual
        gate_dim = 4  # path_coverage, support, uncertainty, |s_kg| proxy via hcr_mean
        gate_model = GatedKGHCRModel(
            embed_dim,
            kg_hcr_dim=pair_feat_dim,
            gate_dim=gate_dim,
            hidden_dims=mcfg["hidden_dims"],
            dropout=mcfg["dropout"],
        ).to(device)
        decoder = gate_model  # type: ignore
        params = list(encoder.parameters()) + list(attn.parameters()) + list(gate_model.parameters())
        tab_train = tab_val = tab_test = None

    opt = torch.optim.Adam(
        params, lr=float(fcfg.get("lr", 1e-3)), weight_decay=float(fcfg.get("weight_decay", 1e-4))
    )
    train_y = bundle["train_pairs"]["label"].astype(np.float32)
    u_t, i_t = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    u_t, i_t = u_t.to(device), i_t.to(device)
    y_t = torch.from_numpy(train_y).to(device)
    feats_t = torch.from_numpy(bank["train_feats"]).to(device)
    mask_t = torch.from_numpy(bank["train_mask"]).to(device)
    ids_t = torch.from_numpy(bank["train_ids"]).to(device)

    n_pos = float((train_y > 0.5).sum())
    n_neg = float(len(train_y) - n_pos)
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state, best_ndcg, patience_left = None, -1.0, int(fcfg.get("patience", 5))
    history = []
    max_epochs = int(fcfg.get("max_epochs", 20))

    def forward_scores(z, u_idx, i_idx, feats, mask, ids, tab_side=None):
        zu = z[u_idx]
        zi = z[i_idx]
        # gather z_hist; invalid ids -> zeros
        B, H = ids.shape
        flat_ids = ids.clamp(min=0) + item_offset
        z_hist = z[flat_ids]  # [B,H,D]
        z_hist = z_hist * mask.unsqueeze(-1)
        agg, _ = attn(zu, zi, z_hist, feats, mask)
        if kind == "attn":
            assert tab_side is not None
            side = torch.cat([agg, tab_side], dim=-1)
            return decoder(zu, zi, side)
        # gated
        path_cov = mask.sum(dim=-1, keepdim=True)  # placeholder replaced below
        # gate feats from agg stats
        # agg columns: hcr,cond,lift,npmi,support,uncertainty,kg_connected,...
        gfeat = torch.stack(
            [
                agg[:, 6],  # kg_connected (mean-attended)
                torch.log1p(agg[:, 4].clamp_min(0)),  # support
                agg[:, 5],  # uncertainty
                agg[:, 0].abs(),  # |hcr|
            ],
            dim=-1,
        )
        s, _, _ = gate_model(zu, zi, agg, gfeat)
        return s

    for epoch in range(max_epochs):
        encoder.train()
        attn.train()
        decoder.train()
        opt.zero_grad()
        z = encode_fn()
        if kind == "attn":
            tab_t = torch.from_numpy(tab_train).to(device)
            logits = forward_scores(z, u_t, i_t, feats_t, mask_t, ids_t, tab_t)
        else:
            logits = forward_scores(z, u_t, i_t, feats_t, mask_t, ids_t, None)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()

        def score_split(split: str) -> np.ndarray:
            encoder.eval()
            attn.eval()
            decoder.eval()
            with torch.no_grad():
                z = encode_fn()
                pairs = bundle[f"{split}_pairs"]
                u_idx, i_idx = user_item_to_nodes(
                    pairs["user_id"], pairs["item_id"], item_offset
                )
                feats = torch.from_numpy(bank[f"{split}_feats"]).to(device)
                mask = torch.from_numpy(bank[f"{split}_mask"]).to(device)
                ids = torch.from_numpy(bank[f"{split}_ids"]).to(device)
                tab_side = None
                if kind == "attn":
                    arr = {"train": tab_train, "val": tab_val, "test": tab_test}[split]
                    tab_side = torch.from_numpy(arr).to(device)
                scores = []
                bs = 2048
                for start in range(0, len(pairs["label"]), bs):
                    sl = slice(start, start + bs)
                    ts = tab_side[sl] if tab_side is not None else None
                    s = forward_scores(
                        z,
                        u_idx[sl].to(device),
                        i_idx[sl].to(device),
                        feats[sl],
                        mask[sl],
                        ids[sl],
                        ts,
                    )
                    scores.append(s.detach().cpu().numpy())
            return np.concatenate(scores).astype(np.float64)

        val_scores = score_split("val")
        ndcg = ranking_metrics_for_users(
            bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"], val_scores, ks=[20]
        )["NDCG@20"]
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_ndcg": float(ndcg)})
        print(f"[{stage}] epoch {epoch} train_loss={loss.item():.4f} val_NDCG@20={ndcg:.4f}")
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = float(ndcg)
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "attn": {k: v.detach().cpu().clone() for k, v in attn.state_dict().items()},
                "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            }
            patience_left = int(fcfg.get("patience", 5))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        attn.load_state_dict(best_state["attn"])
        decoder.load_state_dict(best_state["decoder"])
        encoder.to(device)
        attn.to(device)
        decoder.to(device)

    def score_fn(split: str) -> np.ndarray:
        encoder.eval()
        attn.eval()
        decoder.eval()
        with torch.no_grad():
            z = encode_fn()
            pairs = bundle[f"{split}_pairs"]
            u_idx, i_idx = user_item_to_nodes(pairs["user_id"], pairs["item_id"], item_offset)
            feats = torch.from_numpy(bank[f"{split}_feats"]).to(device)
            mask = torch.from_numpy(bank[f"{split}_mask"]).to(device)
            ids = torch.from_numpy(bank[f"{split}_ids"]).to(device)
            tab_side = None
            if kind == "attn":
                arr = {"val": tab_val, "test": tab_test}[split]
                tab_side = torch.from_numpy(arr).to(device)
            scores = []
            bs = 2048
            for start in range(0, len(pairs["label"]), bs):
                sl = slice(start, start + bs)
                ts = tab_side[sl] if tab_side is not None else None
                s = forward_scores(
                    z,
                    u_idx[sl].to(device),
                    i_idx[sl].to(device),
                    feats[sl],
                    mask[sl],
                    ids[sl],
                    ts,
                )
                scores.append(s.detach().cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    return _pack_result(
        bundle,
        stage,
        encode_fn,
        decoder,
        meta,
        None,
        None,
        [f"hist_pair:{i}" for i in range(pair_feat_dim)],
        history,
        best_ndcg,
        item_offset,
        device,
        model_kind=kind,
        score_fn=score_fn,
    )


def run_stage_f(bundle: dict[str, Any], stage: str) -> dict[str, Any]:
    spec = STAGE_SPECS[stage]
    cfg = bundle["cfg"]
    # ensure stage_f path
    cfg["paths"].setdefault("stage_f", str(Path(cfg["paths"]["outputs_root"]) / "stage_f"))

    if spec["kind"] == "reference_e3":
        e3_path = Path(cfg["paths"]["stage_e"]) / "E3" / "metrics.json"
        e3 = json.loads(e3_path.read_text(encoding="utf-8"))
        result = {
            "stage": "F0",
            "model_kind": "reference_e3",
            "note": "F0 := frozen E3 (HGT + flat user-HCR)",
            "test": e3["test"],
            "validation": e3.get("validation", {}),
            "primary_metric": e3["primary_metric"],
            "primary_test": e3["primary_test"],
            "train_info": {"reference": "E3"},
            "side_features": e3.get("side_features", []),
        }
        out = Path(cfg["paths"]["stage_f"]) / "F0"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[F0] reference E3 primary_test={result['primary_test']:.4f}")
        return result

    if spec["kind"] == "flat":
        sides = materialize_kg_hcr_sides(
            bundle,
            mask_no_path=bool(spec["mask_no_path"]),
            tag=spec["tag"],
            max_history=int(cfg.get("models", {}).get("stage_f", {}).get("max_history", 25)),
        )
        return _train_flat_hgt(bundle, stage, sides)

    if spec["kind"] == "attn":
        return _train_attn_or_gated(bundle, stage, "attn")
    if spec["kind"] == "gated":
        return _train_attn_or_gated(bundle, stage, "gated")
    raise ValueError(stage)
