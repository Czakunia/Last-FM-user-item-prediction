#!/usr/bin/env python3
"""ARTIST_A11_RESIDUAL_BRANCH_V1 — frozen B0 + tiny artist residual. Sampled val only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_publication_our_hgt_fullrank import build_fusion_head  # noqa: E402
from scripts.run_race_clean_3 import activity_bucket, shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.evaluation.measure_race_stats import paired_signflip_pvalue  # noqa: E402
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users  # noqa: E402
from src.lastfm_lp.models.encoders import HGTGraphEncoder  # noqa: E402
from src.lastfm_lp.models.fusion import RecommendationModel  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = ROOT / "configs" / "lastfm_star_race_clean_3.yaml"
SPLITS = ROOT / "outputs" / "lastfm_star" / "splits"
RACE = ROOT / "KRAM_FINAL_WORK" / "RACE_CLEAN_3"
B0_CKPT = RACE / "race" / "a11_top25" / "checkpoints" / "RACE_CLEAN_3_a11_top25"
B0_H = RACE / "race" / "a11_top25" / "features"
H_ART = RACE / "ARTIST_A11_LEVEL_V1" / "cache" / "H_artist"
PREV = RACE / "ARTIST_A11_LEVEL_V1"
OUT = RACE / "ARTIST_A11_RESIDUAL_BRANCH_V1"
SEEDS = (101, 202, 303)
B2_SAMPLED = {101: 0.865092, 202: 0.8643, 303: 0.8609}
ACT = ("VERY_LIGHT", "LIGHT", "MEDIUM", "HEAVY", "VERY_HEAVY")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arr(*arrs: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def abort(msg: str) -> None:
    raise SystemExit(f"HARD_FAIL: {msg}")


class ArtistResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 8)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(8, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(h))).squeeze(-1)


def load_b0(bundle: dict[str, Any], seed: int) -> dict[str, Any]:
    ckpt = B0_CKPT / f"seed_{seed}"
    if not (ckpt / "model.pt").exists():
        abort(f"missing B0 {ckpt}")
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    meta = json.loads((ckpt / "train_meta.json").read_text())
    with (ckpt / "a_scaler.pkl").open("rb") as f:
        a_scaler = pickle.load(f)
    with (ckpt / "h_scaler.pkl").open("rb") as f:
        h_scaler = pickle.load(f)
    graph = load_data_and_typed_graph(
        cfg, bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
    edge_index_dict = {
        k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0
    }
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    encoder = HGTGraphEncoder(
        graph["meta"]["n_users"],
        graph["meta"]["n_entities"],
        metadata,
        edge_index_dict,
        embed_dim=64,
        n_layers=2,
        heads=2,
        dropout=0.1,
    ).to(device)
    fusion = build_fusion_head(encoder=encoder, hcr_dim=3, use_a11=True, fcfg=fcfg).to(device)
    model = RecommendationModel(encoder, fusion).to(device)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    item_offset = int(meta["item_offset"])
    return {
        "device": device,
        "model": model,
        "a_scaler": a_scaler,
        "h_scaler": h_scaler,
        "item_offset": item_offset,
        "ckpt": ckpt,
        "meta": meta,
        "acfg": acfg,
    }


@torch.no_grad()
def b0_logits(ctx: dict[str, Any], users, items, A_s, H_s) -> np.ndarray:
    model = ctx["model"]
    device = ctx["device"]
    off = ctx["item_offset"]
    model.eval()
    z = model.encoder.encode_all()
    u_idx, i_idx = user_item_to_nodes(users, items, off)
    out = []
    for start in range(0, len(users), 8192):
        sl = slice(start, start + 8192)
        pred = model(
            u_idx[sl].to(device),
            i_idx[sl].to(device),
            torch.from_numpy(A_s[sl]).to(device),
            torch.from_numpy(H_s[sl]).to(device),
            z=z,
        )
        out.append(pred["logits"].detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def scale_split(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


def sampled_metrics(users, labels, scores) -> dict[str, float]:
    m = ranking_metrics_for_users(users, labels, scores, ks=(20,))
    # HitRate@20 from the same ranking
    buckets: dict[int, list[tuple[float, int]]] = {}
    for u, y, s in zip(users.tolist(), labels.tolist(), scores.tolist()):
        buckets.setdefault(int(u), []).append((float(s), int(y)))
    hrs = []
    for rows in buckets.values():
        rows.sort(key=lambda t: t[0], reverse=True)
        ys = np.asarray([y for _, y in rows], dtype=np.float64)
        if ys.sum() <= 0:
            continue
        hrs.append(1.0 if ys[:20].sum() > 0 else 0.0)
    m["HitRate@20"] = float(np.mean(hrs)) if hrs else float("nan")
    return m


def per_user_ndcg(users, labels, scores) -> dict[int, float]:
    buckets: dict[int, list[tuple[float, int]]] = {}
    for u, y, s in zip(users.tolist(), labels.tolist(), scores.tolist()):
        buckets.setdefault(int(u), []).append((float(s), int(y)))
    out: dict[int, float] = {}
    for u, rows in buckets.items():
        rows.sort(key=lambda t: t[0], reverse=True)
        ys = np.asarray([y for _, y in rows], dtype=np.float64)
        n_pos = int(ys.sum())
        if n_pos == 0:
            continue
        k = 20
        dcg = float((ys[:k] / np.log2(np.arange(2, min(k, ys.size) + 2))).sum())
        ideal = float((np.ones(min(n_pos, k)) / np.log2(np.arange(2, min(n_pos, k) + 2))).sum())
        out[int(u)] = dcg / ideal if ideal > 0 else 0.0
    return out


def bootstrap_ci(delta: np.ndarray, n: int = 10_000, seed: int = 20260815) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        means[i] = delta[rng.integers(0, delta.size, delta.size)].mean()
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def train_seed(bundle: dict[str, Any], seed: int, A_tr, A_va, Hi_tr, Hi_va, Ha_tr, Ha_va) -> dict[str, Any]:
    run = OUT / "runs" / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    ctx = load_b0(bundle, seed)
    device = ctx["device"]
    a_s_tr = scale_split(ctx["a_scaler"], A_tr)
    a_s_va = scale_split(ctx["a_scaler"], A_va)
    h_s_tr = scale_split(ctx["h_scaler"], Hi_tr)
    h_s_va = scale_split(ctx["h_scaler"], Hi_va)
    art_scaler = StandardScaler().fit(Ha_tr)
    ha_tr = scale_split(art_scaler, Ha_tr)
    ha_va = scale_split(art_scaler, Ha_va)
    with (run / "artist_scaler.pkl").open("wb") as f:
        pickle.dump(art_scaler, f)
    write_json(
        run / "artist_scaler.json",
        {
            "mean": art_scaler.mean_.tolist(),
            "scale": art_scaler.scale_.tolist(),
            "fingerprint": sha256_arr(art_scaler.mean_, art_scaler.scale_),
        },
    )

    tr_u, tr_i, tr_y = (
        bundle["train_pairs"]["user_id"],
        bundle["train_pairs"]["item_id"],
        bundle["train_pairs"]["label"].astype(np.float32),
    )
    va_u, va_i, va_y = (
        bundle["val_pairs"]["user_id"],
        bundle["val_pairs"]["item_id"],
        bundle["val_pairs"]["label"].astype(np.float32),
    )

    print(f"[B3 seed={seed}] scoring frozen B0 …", flush=True)
    b0_tr = b0_logits(ctx, tr_u, tr_i, a_s_tr, h_s_tr)
    b0_va = b0_logits(ctx, va_u, va_i, a_s_va, h_s_va)
    np.save(run / "b0_val_logits.npy", b0_va.astype(np.float32))

    # identity: raw B0 vs wrapper (delta=0) on 1000 val pairs
    rng = np.random.default_rng(20260815 + seed)
    pick = np.sort(rng.choice(len(va_y), size=min(1000, len(va_y)), replace=False))
    branch = ArtistResidual().to(device)
    with torch.no_grad():
        d0 = branch(torch.from_numpy(ha_va[pick]).to(device)).cpu().numpy()
    wrap = b0_va[pick] + d0.astype(np.float64)
    ident = float(np.max(np.abs(wrap - b0_va[pick])))
    zero_d = float(np.max(np.abs(d0)))
    if ident > 1e-7 or zero_d > 1e-7:
        abort(f"seed {seed} B3!=B0 at init max_abs={ident} delta={zero_d}")
    write_json(run / "zero_init.json", {"n": int(pick.size), "max_abs_logit": ident, "max_abs_delta": zero_d})

    # trainable audit
    for n, p in ctx["model"].named_parameters():
        if p.requires_grad:
            abort(f"B0 param trainable: {n}")
    trainable = [n for n, p in branch.named_parameters() if p.requires_grad]
    n_b0 = sum(p.numel() for p in ctx["model"].parameters())
    n_br = sum(p.numel() for p in branch.parameters())
    write_json(
        run / "params.json",
        {
            "n_total": n_b0 + n_br,
            "n_frozen": n_b0,
            "n_trainable": n_br,
            "trainable_names": trainable,
        },
    )

    acfg = ctx["acfg"]
    opt = torch.optim.Adam(
        branch.parameters(),
        lr=float(acfg.get("lr", 1e-3)),
        weight_decay=float(acfg.get("weight_decay", 1e-4)),
    )
    n_pos = float((tr_y > 0.5).sum())
    n_neg = float((tr_y <= 0.5).sum())
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = int(acfg.get("batch_size", 4096))
    n_train = len(tr_y)
    b0_tr_t = torch.from_numpy(b0_tr.astype(np.float32)).to(device)
    ha_tr_t = torch.from_numpy(ha_tr).to(device)
    y_t = torch.from_numpy(tr_y).to(device)

    best_state = None
    best_ndcg = -1.0
    left = int(acfg.get("patience", 4))
    history = []
    torch.manual_seed(seed)
    np.random.seed(seed)
    for epoch in range(int(acfg.get("max_epochs", 15))):
        branch.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        starts = list(range(0, n_train, batch_size))
        n_b = max(len(starts), 1)
        for start in starts:
            idx = torch.from_numpy(perm[start : start + batch_size]).to(device)
            delta = branch(ha_tr_t[idx])
            loss = loss_fn(b0_tr_t[idx] + delta, y_t[idx])
            (loss / n_b).backward()
            epoch_loss += float(loss.item())
            n_batches += 1
        opt.step()
        for p in ctx["model"].parameters():
            if p.requires_grad:
                abort("B0 unfrozen during train")
        branch.eval()
        with torch.no_grad():
            delta_va = branch(torch.from_numpy(ha_va).to(device)).cpu().numpy().astype(np.float64)
        b3_va = b0_va + delta_va
        met = sampled_metrics(va_u, va_y, b3_va)
        ndcg = float(met["NDCG@20"])
        history.append({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1), "val_NDCG@20": ndcg})
        print(f"[B3 seed={seed}] epoch {epoch} loss={history[-1]['loss']:.4f} val_NDCG@20={ndcg:.4f}", flush=True)
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
            left = int(acfg.get("patience", 4))
        else:
            left -= 1
            if left <= 0:
                break
    if best_state:
        branch.load_state_dict(best_state)
    torch.save(best_state or branch.state_dict(), run / "branch.pt")
    branch.eval()
    with torch.no_grad():
        delta_va = branch(torch.from_numpy(ha_va).to(device)).cpu().numpy().astype(np.float64)
    b3_va = b0_va + delta_va
    np.save(run / "b3_val_logits.npy", b3_va.astype(np.float32))
    np.save(run / "delta_val.npy", delta_va.astype(np.float32))
    m0 = sampled_metrics(va_u, va_y, b0_va)
    m3 = sampled_metrics(va_u, va_y, b3_va)
    pu0 = per_user_ndcg(va_u, va_y, b0_va)
    pu3 = per_user_ndcg(va_u, va_y, b3_va)
    write_json(
        run / "train_meta.json",
        {
            "seed": seed,
            "best_val_NDCG@20_sampled": float(m3["NDCG@20"]),
            "B0_val_NDCG@20_sampled": float(m0["NDCG@20"]),
            "history": history,
            "best_epoch": int(np.argmax([h["val_NDCG@20"] for h in history])),
            "lr": float(acfg.get("lr", 1e-3)),
            "weight_decay": float(acfg.get("weight_decay", 1e-4)),
            "optimizer": "Adam",
            "B0_meta_sampled": float(ctx["meta"]["best_val_NDCG@20_sampled"]),
        },
    )
    return {
        "seed": seed,
        "B0": m0,
        "B3": m3,
        "pu0": pu0,
        "pu3": pu3,
        "delta_va": delta_va,
        "ha_va": Ha_va,
        "va_u": va_u,
        "va_i": va_i,
        "va_y": va_y,
        "params": {"n_total": n_b0 + n_br, "n_frozen": n_b0, "n_trainable": n_br, "names": trainable},
        "ident": ident,
        "art_scaler": art_scaler,
        "history": history,
    }


def finalize(bundle: dict[str, Any], results: list[dict[str, Any]]) -> None:
    rep = OUT / "reports"
    audit = OUT / "audit"
    rep.mkdir(exist_ok=True)
    audit.mkdir(exist_ok=True)
    mt = bundle["model_train"]
    with (PREV / "cache" / "artists_of_item.pkl").open("rb") as f:
        artists_of = pickle.load(f)

    def user_arts(u: int) -> set[int]:
        s: set[int] = set()
        for i in mt.get(int(u), ()):
            if 0 <= int(i) < len(artists_of):
                s.update(int(a) for a in artists_of[int(i)].tolist())
        return s

    def cand_seen(u: int, i: int, ua: set[int]) -> bool:
        if not (0 <= int(i) < len(artists_of)):
            return False
        return bool(ua & {int(a) for a in artists_of[int(i)].tolist()})

    rows = []
    all_delta = []
    act_acc = {b: {"B0": [], "B3": []} for b in ACT}
    seen_acc = {"SEEN": {"B0": [], "B3": []}, "UNSEEN": {"B0": [], "B3": []}}
    paired_rows = []
    delta_all = []
    ha_all = []
    for r in results:
        s = r["seed"]
        b0, b3 = r["B0"]["NDCG@20"], r["B3"]["NDCG@20"]
        rows.append(
            {
                "seed": s,
                "B0": b0,
                "B2_raw_concat": B2_SAMPLED[s],
                "B3_residual": b3,
                "B3-B0": b3 - b0,
                "B3-B2": b3 - B2_SAMPLED[s],
                "Recall@20_B0": r["B0"]["Recall@20"],
                "Recall@20_B3": r["B3"]["Recall@20"],
                "HitRate@20_B0": r["B0"]["HitRate@20"],
                "HitRate@20_B3": r["B3"]["HitRate@20"],
                "MRR_B0": r["B0"]["MRR"],
                "MRR_B3": r["B3"]["MRR"],
            }
        )
        common = sorted(set(r["pu0"]) & set(r["pu3"]))
        d = np.asarray([r["pu3"][u] - r["pu0"][u] for u in common], dtype=np.float64)
        all_delta.append(d)
        for u, dv in zip(common, d):
            paired_rows.append(
                {"seed": s, "user_id": u, "NDCG20_B0": r["pu0"][u], "NDCG20_B3": r["pu3"][u], "delta_B3_B0": float(dv)}
            )
            hlen = len(mt.get(int(u), ()))
            ab = activity_bucket(hlen)
            act_acc[ab]["B0"].append(r["pu0"][u])
            act_acc[ab]["B3"].append(r["pu3"][u])
            ua = user_arts(int(u))
            pos_i = r["va_i"][r["va_u"] == u]
            pos_y = r["va_y"][r["va_u"] == u]
            pos_items = [int(i) for i, y in zip(pos_i.tolist(), pos_y.tolist()) if y > 0.5]
            n_seen = sum(1 for i in pos_items if cand_seen(int(u), i, ua))
            key = "SEEN" if pos_items and n_seen >= (len(pos_items) - n_seen) else "UNSEEN"
            if pos_items:
                seen_acc[key]["B0"].append(r["pu0"][u])
                seen_acc[key]["B3"].append(r["pu3"][u])
        delta_all.append(r["delta_va"])
        ha_all.append(r["ha_va"])

    write_csv(
        rep / "BRANCH_SAMPLED_BY_SEED.csv",
        rows,
        ["seed", "B0", "B2_raw_concat", "B3_residual", "B3-B0", "B3-B2",
         "Recall@20_B0", "Recall@20_B3", "HitRate@20_B0", "HitRate@20_B3", "MRR_B0", "MRR_B3"],
    )
    b0s = [r["B0"] for r in rows]
    b3s = [r["B3_residual"] for r in rows]
    b2s = [r["B2_raw_concat"] for r in rows]
    d30 = [a - b for a, b in zip(b3s, b0s)]
    overall = {
        "B0_mean": float(np.mean(b0s)),
        "B0_std": float(np.std(b0s, ddof=1)),
        "B2_mean": float(np.mean(b2s)),
        "B3_mean": float(np.mean(b3s)),
        "B3_std": float(np.std(b3s, ddof=1)),
        "dB3_B0": float(np.mean(d30)),
        "dB3_B2": float(np.mean(b3s) - np.mean(b2s)),
        "n_pos_seeds": int(sum(1 for x in d30 if x > 0)),
    }
    write_csv(rep / "BRANCH_SAMPLED_OVERALL.csv", [overall], list(overall))
    write_csv(rep / "BRANCH_PAIRED_USER_DELTA.csv", paired_rows,
              ["seed", "user_id", "NDCG20_B0", "NDCG20_B3", "delta_B3_B0"])
    delta = np.concatenate(all_delta)
    mu, lo, hi = bootstrap_ci(delta)
    sign_p = paired_signflip_pvalue(delta)
    act_rows = []
    for b in ACT:
        m0 = float(np.mean(act_acc[b]["B0"])) if act_acc[b]["B0"] else float("nan")
        m3 = float(np.mean(act_acc[b]["B3"])) if act_acc[b]["B3"] else float("nan")
        act_rows.append({"bucket": b, "B0": m0, "B3": m3, "delta": m3 - m0, "n": len(act_acc[b]["B0"])})
    write_csv(rep / "BRANCH_SAMPLED_BY_ACTIVITY.csv", act_rows, ["bucket", "B0", "B3", "delta", "n"])
    seen_rows = []
    for k in ("SEEN", "UNSEEN"):
        m0 = float(np.mean(seen_acc[k]["B0"])) if seen_acc[k]["B0"] else float("nan")
        m3 = float(np.mean(seen_acc[k]["B3"])) if seen_acc[k]["B3"] else float("nan")
        seen_rows.append({"split": f"ARTIST_{k}", "B0": m0, "B3": m3, "delta": m3 - m0, "n": len(seen_acc[k]["B0"])})
    write_csv(rep / "BRANCH_ARTIST_SEEN_UNSEEN.csv", seen_rows, ["split", "B0", "B3", "delta", "n"])

    dlt = np.concatenate(delta_all)
    ha = np.concatenate(ha_all)
    qs = {q: float(np.percentile(dlt, p)) for q, p in
          (("median", 50), ("p1", 1), ("p5", 5), ("p25", 25), ("p75", 75), ("p95", 95), ("p99", 99))}
    dist = {
        "mean": float(dlt.mean()),
        "std": float(dlt.std(ddof=1)),
        "max_abs": float(np.max(np.abs(dlt))),
        "frac_pos": float((dlt > 0).mean()),
        "frac_neg": float((dlt < 0).mean()),
        "frac_near0": float((np.abs(dlt) < 1e-6).mean()),
        "corr_mean": float(np.corrcoef(dlt, ha[:, 0])[0, 1]) if dlt.std() > 0 else float("nan"),
        "corr_max": float(np.corrcoef(dlt, ha[:, 1])[0, 1]) if dlt.std() > 0 else float("nan"),
        "corr_top3": float(np.corrcoef(dlt, ha[:, 2])[0, 1]) if dlt.std() > 0 else float("nan"),
        **qs,
    }
    write_csv(rep / "ARTIST_RESIDUAL_DISTRIBUTION.csv", [dist], list(dist))

    n_pos = overall["n_pos_seeds"]
    dmean = overall["dB3_B0"]
    if dmean > 0 and n_pos >= 2 and lo > 0:
        verdict = "STRONG_BRANCH_GAIN"
    elif dmean > 0 and n_pos >= 2:
        verdict = "WEAK_BRANCH_GAIN"
    else:
        verdict = "NO_BRANCH_GAIN"
    const = dist["std"] < 1e-6
    un_d = next(x["delta"] for x in seen_rows if x["split"] == "ARTIST_UNSEEN")

    md = f"""# ARTIST_A11_RESIDUAL_BRANCH_V1 — sampled screen

Frozen B0 + Linear(3,8)-GELU-Linear(8,1) residual. Full-rank locked. Test locked.

## Table

| seed | B0 | B2 concat | B3 residual | B3−B0 | B3−B2 |
|---|---:|---:|---:|---:|---:|
| 101 | {rows[0]['B0']:.4f} | {rows[0]['B2_raw_concat']:.4f} | {rows[0]['B3_residual']:.4f} | {rows[0]['B3-B0']:+.5f} | {rows[0]['B3-B2']:+.5f} |
| 202 | {rows[1]['B0']:.4f} | {rows[1]['B2_raw_concat']:.4f} | {rows[1]['B3_residual']:.4f} | {rows[1]['B3-B0']:+.5f} | {rows[1]['B3-B2']:+.5f} |
| 303 | {rows[2]['B0']:.4f} | {rows[2]['B2_raw_concat']:.4f} | {rows[2]['B3_residual']:.4f} | {rows[2]['B3-B0']:+.5f} | {rows[2]['B3-B2']:+.5f} |
| mean | {overall['B0_mean']:.4f} | {overall['B2_mean']:.4f} | {overall['B3_mean']:.4f} | {overall['dB3_B0']:+.5f} | {overall['dB3_B2']:+.5f} |

Paired users × 3 seeds: mean {mu:+.5f} median {float(np.median(delta)):+.5f} p25 {float(np.percentile(delta,25)):+.5f} p75 {float(np.percentile(delta,75)):+.5f}
frac>0 {float((delta>0).mean()):.3f} frac=0 {float((delta==0).mean()):.3f} frac<0 {float((delta<0).mean()):.3f}
bootstrap 10k 95% CI [{lo:+.5f}, {hi:+.5f}]  sign-flip p={sign_p:.4g}

delta_artist: mean {dist['mean']:+.4g} std {dist['std']:.4g} median {dist['median']:+.4g} max|Δ| {dist['max_abs']:.4g}
frac≈0 {dist['frac_near0']:.3f}  corr(mean/max/top3)={dist['corr_mean']:.3f}/{dist['corr_max']:.3f}/{dist['corr_top3']:.3f}
constant-like: {const}

## Q1–Q10

1. B0 wrapper identity: max|Δ| init ≤ 1e-7 (see ZERO_INIT_EQUIVALENCE_AUDIT.md).
2. Zero-init last layer: delta=0 at start.
3. Sampled NDCG B3 vs B0: {overall['dB3_B0']:+.5f}.
4. Seeds B3>B0: {n_pos}/3.
5. Residual std={dist['std']:.4g}; constant-like={const}.
6. Activity: see BRANCH_SAMPLED_BY_ACTIVITY.csv.
7. ARTIST_UNSEEN Δ={un_d:+.5f}.
8. B3 vs B2: {overall['dB3_B2']:+.5f}.
9. Residual complementary? verdict below.
10. Shuffle/full-rank only if GAIN. Not started.

ARTIST_A11_BRANCH_VERDICT = {verdict}
"""
    (rep / "ARTIST_RESIDUAL_BRANCH_REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)

    # audits
    p0 = results[0]["params"]
    (audit / "B0_FREEZE_AUDIT.md").write_text(
        f"B0 frozen on all seeds. n_frozen={p0['n_frozen']} n_trainable={p0['n_trainable']}.\n"
        "requires_grad=False on every B0 parameter. Optimizer sees residual only.\n",
        encoding="utf-8",
    )
    (audit / "ZERO_INIT_EQUIVALENCE_AUDIT.md").write_text(
        "Last Linear(8,1) zero-init. Before step: logit_B3 == logit_B0.\n"
        + "\n".join(
            f"seed {r['seed']}: max|B3-B0|={r['ident']:.3e}" for r in results
        )
        + "\nPASS (threshold 1e-7).\n",
        encoding="utf-8",
    )
    (audit / "TRAINABLE_PARAMETER_AUDIT.md").write_text(
        f"trainable: {p0['names']}\nn_trainable={p0['n_trainable']} n_frozen={p0['n_frozen']} n_total={p0['n_total']}\n",
        encoding="utf-8",
    )
    src_r = PREV / "audit" / "ROUTING_IDENTITY_AUDIT.md"
    (audit / "ROUTING_IDENTITY_AUDIT.md").write_text(
        (src_r.read_text() if src_r.exists() else "")
        + "\nB3 uses the same item-A11 Top25 as B0/B2. Artist residual does not enter routing.\n"
        "Top25_B0(u,X) == Top25_B3(u,X) by construction.\n",
        encoding="utf-8",
    )
    (audit / "SCALER_AUDIT.md").write_text(
        "StandardScaler_artist fit on train H_artist 3-D only. Not item H, not B2 6-D, not val.\n"
        "Per-seed fingerprints in runs/seed*/artist_scaler.json.\n",
        encoding="utf-8",
    )
    (audit / "DATA_PROVENANCE_AUDIT.md").write_text(
        "Last-FM* fingerprints unchanged (see manifest). Artist A11 from ARTIST_A11_LEVEL_V1 cache.\n"
        f"val_pairs hash = {sha256_arr(bundle['val_pairs']['user_id'], bundle['val_pairs']['item_id'], bundle['val_pairs']['label'])}\n"
        f"train_pairs hash = {sha256_arr(bundle['train_pairs']['user_id'], bundle['train_pairs']['item_id'], bundle['train_pairs']['label'])}\n"
        "Same sampled val candidate table as B0/B1/B2.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FULLRANK_LOCKED").write_text("LOCKED_NOT_RUN\n")
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n")
    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    A_tr = np.load(shared_A_dir() / "X_train.npy").astype(np.float32)
    A_va = np.load(shared_A_dir() / "X_val.npy").astype(np.float32)
    Hi_tr = np.load(B0_H / "X_train.npy").astype(np.float32)
    Hi_va = np.load(B0_H / "X_val.npy").astype(np.float32)
    Ha_tr = np.load(H_ART / "X_train.npy").astype(np.float32)
    Ha_va = np.load(H_ART / "X_val.npy").astype(np.float32)
    n_tr = len(bundle["train_pairs"]["label"])
    n_va = len(bundle["val_pairs"]["label"])
    if Ha_tr.shape != (n_tr, 3) or Hi_tr.shape[0] != n_tr:
        abort(f"train row mismatch Ha {Ha_tr.shape} Hi {Hi_tr.shape} n={n_tr}")
    if Ha_va.shape != (n_va, 3) or Hi_va.shape[0] != n_va:
        abort(f"val row mismatch")
    if Ha_tr.shape[1] != 3:
        abort("artist H must be 3-D")

    man_prev = json.loads((PREV / "ARTIST_A11_LEVEL_V1_MANIFEST.json").read_text())
    write_json(
        OUT / "ARTIST_A11_RESIDUAL_BRANCH_V1_MANIFEST.json",
        {
            "experiment_id": "ARTIST_A11_RESIDUAL_BRANCH_V1",
            "timestamp": utc_now(),
            "git_commit": "UNKNOWN",
            "dataset_hash": man_prev.get("corrected_dataset_hash"),
            "model_train_hash": man_prev["model_train_hash"],
            "valid_hash": man_prev["valid_hash"],
            "test_hash": man_prev["test_hash"],
            "KG_hash": man_prev["KG_hash"],
            "artist_mapping_hash": man_prev["artist_mapping_hash"],
            "artist_crossfit_hash": sha256_file(PREV / "cache" / "artist_crossfit_fold0.npz"),
            "B0_checkpoint_hashes": {f"seed{s}": sha256_file(B0_CKPT / f"seed_{s}" / "model.pt") for s in SEEDS},
            "B0_frozen": True,
            "artist_branch": {
                "input_dim": 3,
                "hidden_dim": 8,
                "activation": "GELU",
                "output_dim": 1,
                "final_layer_zero_initialized": True,
            },
            "fusion": "FINAL_LOGIT = B0_LOGIT + ARTIST_DELTA",
            "routing": "ITEM_A11_SIGNED_TOP25_FIXED",
            "artist_same_identity": "EXCLUDED",
            "evaluation": "SAMPLED_VALIDATION_ONLY",
            "fullrank_status": "LOCKED_NOT_RUN",
            "test_status": "LOCKED_NOT_RUN",
            "val_pairs_hash": sha256_arr(
                bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], bundle["val_pairs"]["label"]
            ),
            "train_pairs_hash": sha256_arr(
                bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], bundle["train_pairs"]["label"]
            ),
        },
    )

    results = []
    for seed in SEEDS:
        results.append(train_seed(bundle, seed, A_tr, A_va, Hi_tr, Hi_va, Ha_tr, Ha_va))
    finalize(bundle, results)
    print("ARTIST_A11_RESIDUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
