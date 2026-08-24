#!/usr/bin/env python3
"""HGT_ARTIST_EDGE_ABLATION_V1 — sampled validation only.

Remove recording<->artist edges from the audited HGT graph, retrain B0 from
scratch, then attach the frozen-B0 artist residual. Full-rank and test locked.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import pickle
import subprocess
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

from scripts.run_artist_a11_residual_branch_v1 import (  # noqa: E402
    ArtistResidual,
    b0_logits,
    bootstrap_ci,
    per_user_ndcg,
    sampled_metrics,
    scale_split,
)
from scripts.run_publication_our_hgt_fullrank import (  # noqa: E402
    build_fusion_head,
    train_and_save,
)
from scripts.run_race_clean_3 import activity_bucket, shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.models.encoders import HGTGraphEncoder  # noqa: E402
from src.lastfm_lp.models.fusion import RecommendationModel  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = ROOT / "configs" / "lastfm_star_race_clean_3.yaml"
DATA = ROOT / "data" / "LastFM_star_IntentAwareRS"
SPLITS = ROOT / "outputs" / "lastfm_star" / "splits"
RACE = ROOT / "KRAM_FINAL_WORK" / "RACE_CLEAN_3"
B0_CKPT = RACE / "race" / "a11_top25" / "checkpoints" / "RACE_CLEAN_3_a11_top25"
B0_H = RACE / "race" / "a11_top25" / "features"
H_ART = RACE / "ARTIST_A11_LEVEL_V1" / "cache" / "H_artist"
PREV = RACE / "ARTIST_A11_LEVEL_V1"
B3_FULL_DIR = RACE / "ARTIST_A11_RESIDUAL_BRANCH_V1"
OUT = RACE / "HGT_ARTIST_EDGE_ABLATION_V1"
SEEDS = (101, 202, 303)
ACT = ("VERY_LIGHT", "LIGHT", "MEDIUM", "HEAVY", "VERY_HEAVY")
ARTIST_URI = "http://rdf.freebase.com/ns/music.recording.artist"
ARTIST_LOCAL_NAME = "music.recording.artist"

EXPECTED = {
    "model_train": "479d40d9e81cb18f6e47b4f2934b4832ac66764665d482c1e76baa8275917278",
    "valid": "4178acf9108ce0be0f12501ae0bce6f857bf0901c28ce0111a18095c0a02d77f",
    "test": "f684145429010deba000da57b4c6192d3787382d64ee5bdf5c7554e64d8bc03f",
    "kg": "8e3c22a89543f8827fb056044774474f9a9dddd9ff648a264b8ea316807c1f54",
    "item_list": "0e4991b5b8a292e82dc5fe3ca34bd04dc90bdeee63e0b99d71b4b8d93c129e80",
    "artist_mapping": "9992a0cf88a4ebd282ab9dcb7ea1110b64b15cf6bece4a42d69e943c063a660d",
    "val_pairs": "c3a2781408117db31af4fa11da474f81fcc8ea56012c611c7d5ecada4aedf979",
    "train_pairs": "e4a7e7b43c6256ad537e724d2c2df8cfc76940acb5a6f2dac1c44fc917de4046",
    "B0_ckpt": {
        101: "70677f47eefc56fb2b8aaedf759e6ce3bb776d2b849bb3b1b8997e5494835ed1",
        202: "4ca2ca58adc1e7a6f6df34c173c14c18627f4ce2ac8d4aeb4e0b4d47dc347489",
        303: "18a55f7b5928c246fec4d5a85c37cef44ad71ccf36766762bdb1d91bab49d128",
    },
}


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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def abort(msg: str) -> None:
    raise SystemExit(f"HARD_FAIL: {msg}")


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def parse_relation_list(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open() as f:
        header = f.readline()
        if "org_id" not in header and "remap_id" not in header:
            f.seek(0)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[-1])
            except ValueError:
                continue
            rows.append((parts[0], rid))
    return rows


def resolve_artist_relation() -> dict[str, Any]:
    rows = parse_relation_list(DATA / "relation_list.txt")
    exact = [(uri, rid) for uri, rid in rows if uri == ARTIST_URI]
    suffix = [(uri, rid) for uri, rid in rows if uri.endswith(ARTIST_LOCAL_NAME)]
    if len(exact) != 1:
        abort(f"artist URI mapping ambiguous or missing: exact={exact} suffix={suffix}")
    if suffix != exact:
        abort(f"artist suffix matches disagree with exact URI: {suffix} vs {exact}")
    uri, rid = exact[0]
    featured = [(u, i) for u, i in rows if "featured_artists" in u]
    if any(i == rid for _, i in featured):
        abort("artist relation id collides with featured_artists")
    return {
        "relation_uri": uri,
        "local_relation_id": int(rid),
        "forward_edge_type": f"rel_{rid}",
        "reverse_edge_type": f"rel_{rid}_rev",
        "forward_hetero_key": ["entity", f"rel_{rid}", "entity"],
        "reverse_hetero_key": ["entity", f"rel_{rid}_rev", "entity"],
        "other_relations": [{"uri": u, "id": i} for u, i in rows if i != rid],
        "featured_artists": featured,
    }


def hash_edge_dict(edge_index_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(edge_index_dict.keys(), key=lambda t: (t[0], t[1], t[2])):
        h.update(repr(k).encode())
        arr = edge_index_dict[k].detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def n_edges(edge_index_dict: dict) -> int:
    return int(sum(int(v.size(1)) for v in edge_index_dict.values()))


def entity_touched(edge_index_dict: dict) -> set[int]:
    touched: set[int] = set()
    for (src_t, _rel, dst_t), ei in edge_index_dict.items():
        if src_t == "entity":
            touched.update(ei[0].cpu().numpy().tolist())
        if dst_t == "entity":
            touched.update(ei[1].cpu().numpy().tolist())
    return touched


def entity_degree(edge_index_dict: dict, n_entities: int) -> np.ndarray:
    deg = np.zeros(n_entities, dtype=np.int64)
    for (src_t, _rel, dst_t), ei in edge_index_dict.items():
        if src_t == "entity":
            np.add.at(deg, ei[0].cpu().numpy().astype(np.int64), 1)
        if dst_t == "entity":
            np.add.at(deg, ei[1].cpu().numpy().astype(np.int64), 1)
    return deg


def deg_summary(deg: np.ndarray) -> dict[str, float]:
    if deg.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"),
                "max": float("nan"), "n_zero": 0}
    return {
        "n": int(deg.size),
        "mean": float(deg.mean()),
        "median": float(np.median(deg)),
        "p90": float(np.percentile(deg, 90)),
        "max": float(deg.max()),
        "n_zero": int((deg == 0).sum()),
    }


def drop_artist_edges(graph: dict[str, Any], rel_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    fwd = ("entity", f"rel_{rel_id}", "entity")
    rev = ("entity", f"rel_{rel_id}_rev", "entity")
    src = graph["edge_index_dict"]
    if fwd not in src or rev not in src:
        abort(f"missing artist edge types in G_FULL: have={list(src)} need={fwd},{rev}")
    n_fwd = int(src[fwd].size(1))
    n_rev = int(src[rev].size(1))
    if n_fwd <= 0 or n_rev <= 0:
        abort(f"artist edges empty in G_FULL fwd={n_fwd} rev={n_rev}")
    if n_fwd != n_rev:
        abort(f"artist fwd/rev count mismatch {n_fwd} vs {n_rev}")
    new_e = {k: v.clone() for k, v in src.items() if k not in (fwd, rev)}
    meta = dict(graph["meta"])
    metadata = (["user", "entity"], list(new_e.keys()))
    meta["metadata"] = metadata
    dropped = {
        "forward_key": list(fwd),
        "reverse_key": list(rev),
        "removed_forward": n_fwd,
        "removed_reverse": n_rev,
    }
    g_no = {
        "data": graph.get("data"),
        "edge_index_dict": new_e,
        "meta": meta,
        "metadata": metadata,
    }
    return g_no, dropped


def check_overlaps() -> dict[str, int]:
    mt = load_user_sets(SPLITS / "model_train.txt")
    va = load_user_sets(SPLITS / "valid.txt")
    te = load_user_sets(SPLITS / "test.txt")
    tv = tt = vv = 0
    for u in set(mt) | set(va) | set(te):
        a, b, c = mt.get(u, set()), va.get(u, set()), te.get(u, set())
        tv += len(a & b)
        tt += len(a & c)
        vv += len(b & c)
    if tv or tt or vv:
        abort(f"forbidden overlap train∩val={tv} train∩test={tt} val∩test={vv}")
    return {"model_train_cap_valid": tv, "model_train_cap_test": tt, "valid_cap_test": vv}


def verify_fingerprints(bundle: dict[str, Any]) -> dict[str, str]:
    hashes = {
        "model_train": sha256_file(SPLITS / "model_train.txt"),
        "valid": sha256_file(SPLITS / "valid.txt"),
        "test": sha256_file(SPLITS / "test.txt"),
        "kg": sha256_file(DATA / "kg_final.txt"),
        "item_list": sha256_file(DATA / "item_list.txt"),
        "artist_mapping": sha256_file(PREV / "cache" / "artist_mapping.npz"),
        "val_pairs": sha256_arr(
            bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], bundle["val_pairs"]["label"]
        ),
        "train_pairs": sha256_arr(
            bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], bundle["train_pairs"]["label"]
        ),
    }
    for k, exp in EXPECTED.items():
        if k == "B0_ckpt":
            continue
        if hashes.get(k) != exp:
            abort(f"fingerprint mismatch {k}: got {hashes.get(k)} expected {exp}")
    for seed, exp in EXPECTED["B0_ckpt"].items():
        got = sha256_file(B0_CKPT / f"seed_{seed}" / "model.pt")
        if got != exp:
            abort(f"B0_FULL ckpt hash mismatch seed {seed}")
    return hashes


def load_model(bundle: dict[str, Any], ckpt: Path, graph: dict[str, Any]) -> dict[str, Any]:
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    meta_t = json.loads((ckpt / "train_meta.json").read_text())
    with (ckpt / "a_scaler.pkl").open("rb") as f:
        a_scaler = pickle.load(f)
    with (ckpt / "h_scaler.pkl").open("rb") as f:
        h_scaler = pickle.load(f)
    edge_index_dict = {k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0}
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
    return {
        "device": device,
        "model": model,
        "a_scaler": a_scaler,
        "h_scaler": h_scaler,
        "item_offset": int(meta_t["item_offset"]),
        "ckpt": ckpt,
        "meta": meta_t,
        "acfg": acfg,
    }


def train_b0_no_artist(bundle: dict[str, Any], seed: int, g_no: dict[str, Any]) -> Path:
    ckpt_root = OUT / "checkpoints"
    stage = "B0_NO_ARTIST"
    ckpt = ckpt_root / stage / f"seed_{seed}"
    if (ckpt / "model.pt").exists() and (ckpt / "train_meta.json").exists():
        print(f"[B0_NO_ARTIST] reuse {ckpt}", flush=True)
        return ckpt
    print(f"[B0_NO_ARTIST] train from scratch seed={seed}", flush=True)
    return train_and_save(
        bundle,
        stage=stage,
        use_a11=True,
        seed=seed,
        hcr_feature_dir=B0_H,
        a_feature_dir=shared_A_dir(),
        ckpt_root=ckpt_root,
        typed_graph=g_no,
    )


def train_b3_no_artist(
    bundle: dict[str, Any],
    seed: int,
    ctx: dict[str, Any],
    A_tr: np.ndarray,
    A_va: np.ndarray,
    Hi_tr: np.ndarray,
    Hi_va: np.ndarray,
    Ha_tr: np.ndarray,
    Ha_va: np.ndarray,
) -> dict[str, Any]:
    run = OUT / "runs" / f"B3_NO_ARTIST_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    done = run / "train_meta.json"
    if done.exists() and (run / "b3_val_logits.npy").exists():
        print(f"[B3_NO_ARTIST] reuse {run}", flush=True)
        meta = json.loads(done.read_text())
        va_u = bundle["val_pairs"]["user_id"]
        va_y = bundle["val_pairs"]["label"].astype(np.float32)
        b0_va = np.load(run / "b0_val_logits.npy").astype(np.float64)
        b3_va = np.load(run / "b3_val_logits.npy").astype(np.float64)
        return {
            "seed": seed,
            "B0": sampled_metrics(va_u, va_y, b0_va),
            "B3": sampled_metrics(va_u, va_y, b3_va),
            "pu0": per_user_ndcg(va_u, va_y, b0_va),
            "pu3": per_user_ndcg(va_u, va_y, b3_va),
            "ident": float(meta.get("zero_init_max_abs", 0.0)),
            "history": meta.get("history", []),
        }

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
    print(f"[B3_NO_ARTIST seed={seed}] scoring frozen B0_NO_ARTIST …", flush=True)
    b0_tr = b0_logits(ctx, tr_u, tr_i, a_s_tr, h_s_tr)
    b0_va = b0_logits(ctx, va_u, va_i, a_s_va, h_s_va)
    np.save(run / "b0_val_logits.npy", b0_va.astype(np.float32))

    rng = np.random.default_rng(20260815 + seed)
    pick = np.sort(rng.choice(len(va_y), size=min(1000, len(va_y)), replace=False))
    branch = ArtistResidual().to(device)
    with torch.no_grad():
        d0 = branch(torch.from_numpy(ha_va[pick]).to(device)).cpu().numpy()
    wrap = b0_va[pick] + d0.astype(np.float64)
    ident = float(np.max(np.abs(wrap - b0_va[pick])))
    zero_d = float(np.max(np.abs(d0)))
    if ident > 1e-7 or zero_d > 1e-7:
        abort(f"seed {seed} B3_NO_ARTIST!=B0_NO_ARTIST at init max_abs={ident} delta={zero_d}")
    write_json(run / "zero_init.json", {"n": int(pick.size), "max_abs_logit": ident, "max_abs_delta": zero_d})

    for n, p in ctx["model"].named_parameters():
        if p.requires_grad:
            abort(f"B0_NO_ARTIST param trainable: {n}")
    trainable = [n for n, p in branch.named_parameters() if p.requires_grad]
    n_b0 = sum(p.numel() for p in ctx["model"].parameters())
    n_br = sum(p.numel() for p in branch.parameters())
    write_json(
        run / "params.json",
        {"n_total": n_b0 + n_br, "n_frozen": n_b0, "n_trainable": n_br, "trainable_names": trainable},
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
                abort("B0_NO_ARTIST unfrozen during residual train")
        branch.eval()
        with torch.no_grad():
            delta_va = branch(torch.from_numpy(ha_va).to(device)).cpu().numpy().astype(np.float64)
        b3_va = b0_va + delta_va
        met = sampled_metrics(va_u, va_y, b3_va)
        ndcg = float(met["NDCG@20"])
        history.append({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1), "val_NDCG@20": ndcg})
        print(f"[B3_NO_ARTIST seed={seed}] epoch {epoch} loss={history[-1]['loss']:.4f} val_NDCG@20={ndcg:.4f}", flush=True)
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
    write_json(
        run / "train_meta.json",
        {
            "seed": seed,
            "best_val_NDCG@20_sampled": float(m3["NDCG@20"]),
            "B0_NO_ARTIST_val_NDCG@20_sampled": float(m0["NDCG@20"]),
            "history": history,
            "zero_init_max_abs": ident,
            "lr": float(acfg.get("lr", 1e-3)),
            "weight_decay": float(acfg.get("weight_decay", 1e-4)),
            "optimizer": "Adam",
        },
    )
    return {
        "seed": seed,
        "B0": m0,
        "B3": m3,
        "pu0": per_user_ndcg(va_u, va_y, b0_va),
        "pu3": per_user_ndcg(va_u, va_y, b3_va),
        "ident": ident,
        "history": history,
    }


def load_full_condition(bundle: dict[str, Any], seed: int) -> dict[str, Any]:
    run = B3_FULL_DIR / "runs" / f"seed{seed}"
    if not (run / "b0_val_logits.npy").exists() or not (run / "b3_val_logits.npy").exists():
        abort(f"missing B0_FULL/B3_FULL logits {run}")
    va_u = bundle["val_pairs"]["user_id"]
    va_y = bundle["val_pairs"]["label"].astype(np.float32)
    b0 = np.load(run / "b0_val_logits.npy").astype(np.float64)
    b3 = np.load(run / "b3_val_logits.npy").astype(np.float64)
    if len(b0) != len(va_y) or len(b3) != len(va_y):
        abort(f"FULL logit length mismatch seed {seed}")
    return {
        "seed": seed,
        "B0": sampled_metrics(va_u, va_y, b0),
        "B3": sampled_metrics(va_u, va_y, b3),
        "pu0": per_user_ndcg(va_u, va_y, b0),
        "pu3": per_user_ndcg(va_u, va_y, b3),
    }


@torch.no_grad()
def embedding_diagnostic(
    bundle: dict[str, Any],
    g_full: dict[str, Any],
    g_no: dict[str, Any],
    artist_ids: np.ndarray,
) -> dict[str, Any]:
    n_users = int(g_full["meta"]["n_users"])
    n_items = int(g_full["meta"]["n_items"])
    off = int(g_full["meta"]["item_offset"])
    rng = np.random.default_rng(20260816)
    users = np.sort(rng.choice(n_users, size=min(500, n_users), replace=False))
    recs = np.sort(rng.choice(n_items, size=min(500, n_items), replace=False))
    arts = np.sort(rng.choice(artist_ids, size=min(500, int(artist_ids.size)), replace=False)) if artist_ids.size else np.array([], dtype=np.int64)
    rows = []
    for seed in SEEDS:
        ctx_f = load_model(bundle, B0_CKPT / f"seed_{seed}", g_full)
        ctx_n = load_model(bundle, OUT / "checkpoints" / "B0_NO_ARTIST" / f"seed_{seed}", g_no)
        zf = ctx_f["model"].encoder.encode_all().detach().cpu().numpy()
        zn = ctx_n["model"].encoder.encode_all().detach().cpu().numpy()
        del ctx_f, ctx_n
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        def stats(idx_flat: np.ndarray, role: str) -> dict[str, float]:
            a = zf[idx_flat]
            b = zn[idx_flat]
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            den = np.maximum(na * nb, 1e-12)
            cos = np.sum(a * b, axis=1) / den
            l2 = np.linalg.norm(a - b, axis=1)
            return {
                "seed": seed,
                "role": role,
                "n": int(idx_flat.size),
                "cosine_mean": float(cos.mean()),
                "cosine_median": float(np.median(cos)),
                "cosine_p10": float(np.percentile(cos, 10)),
                "cosine_p90": float(np.percentile(cos, 90)),
                "l2_mean": float(l2.mean()),
                "l2_median": float(np.median(l2)),
                "l2_p10": float(np.percentile(l2, 10)),
                "l2_p90": float(np.percentile(l2, 90)),
            }

        rows.append(stats(users.astype(np.int64), "user"))
        rows.append(stats((recs + off).astype(np.int64), "recording"))
        if arts.size:
            rows.append(stats((arts + off).astype(np.int64), "artist"))
        del zf, zn
        gc.collect()
    write_csv(
        OUT / "reports" / "ARTIST_EDGE_EMBEDDING_DIAGNOSTIC.csv",
        rows,
        ["seed", "role", "n", "cosine_mean", "cosine_median", "cosine_p10", "cosine_p90",
         "l2_mean", "l2_median", "l2_p10", "l2_p90"],
    )
    return {"rows": rows, "note": "Descriptive only. Independently trained; coordinates not identifiable."}


def write_audits(
    *,
    rel: dict[str, Any],
    hashes: dict[str, str],
    overlaps: dict[str, int],
    g_full: dict[str, Any],
    g_no: dict[str, Any],
    dropped: dict[str, Any],
    rec_deg_f: dict[str, float],
    rec_deg_n: dict[str, float],
    art_deg_f: dict[str, float],
    art_deg_n: dict[str, float],
    n_iso_f: int,
    n_iso_n: int,
    A_tr: np.ndarray,
    A_va: np.ndarray,
    Hi_tr: np.ndarray,
    Hi_va: np.ndarray,
    Ha_tr: np.ndarray,
    Ha_va: np.ndarray,
    bundle: dict[str, Any],
) -> None:
    audit = OUT / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    e_f = g_full["edge_index_dict"]
    e_n = g_no["edge_index_dict"]
    fwd = tuple(dropped["forward_key"])
    rev = tuple(dropped["reverse_key"])

    (audit / "ARTIST_RELATION_ID_AUDIT.md").write_text(
        "# ARTIST_RELATION_ID_AUDIT\n\n"
        f"Resolved from `{DATA / 'relation_list.txt'}` by exact Freebase URI.\n\n"
        f"- relation URI: `{rel['relation_uri']}`\n"
        f"- local relation ID: `{rel['local_relation_id']}`\n"
        f"- forward edge type: `{rel['forward_edge_type']}` → hetero `{rel['forward_hetero_key']}`\n"
        f"- reverse edge type: `{rel['reverse_edge_type']}` → hetero `{rel['reverse_hetero_key']}`\n"
        f"- forward edge count in G_FULL: {dropped['removed_forward']}\n"
        f"- reverse edge count in G_FULL: {dropped['removed_reverse']}\n\n"
        "Mapping is unique. `music.recording.featured_artists` is a different relation and was kept.\n"
        "Artist edges were NOT inferred from node names.\n",
        encoding="utf-8",
    )

    keys_f = set(e_f)
    keys_n = set(e_n)
    if keys_f - {fwd, rev} != keys_n:
        abort(f"non-artist key mismatch {keys_f - {fwd, rev}} vs {keys_n}")
    for k in keys_n:
        if not torch.equal(e_f[k], e_n[k]):
            abort(f"non-artist edges changed for {k}")
    if g_full["meta"]["n_users"] != g_no["meta"]["n_users"]:
        abort("n_users changed")
    if g_full["meta"]["n_entities"] != g_no["meta"]["n_entities"]:
        abort("n_entities changed")
    if g_full["meta"]["n_items"] != g_no["meta"]["n_items"]:
        abort("n_items changed")

    (audit / "GRAPH_EDGE_DIFF_AUDIT.md").write_text(
        "# GRAPH_EDGE_DIFF_AUDIT\n\n"
        f"- total directed hetero edges G_FULL: {n_edges(e_f)}\n"
        f"- total directed hetero edges G_NO_ARTIST: {n_edges(e_n)}\n"
        f"- removed forward artist edges: {dropped['removed_forward']}\n"
        f"- removed reverse artist edges: {dropped['removed_reverse']}\n"
        f"- G_FULL fingerprint: `{hash_edge_dict(e_f)}`\n"
        f"- G_NO_ARTIST fingerprint: `{hash_edge_dict(e_n)}`\n\n"
        "Allowed difference only: `rel_2` and `rel_2_rev` removed after the same "
        "`build_typed_ckg` + `max_kg_edges=250000` + seed 2026 subsample.\n"
        "Interact / interact_rev / every non-artist KG relation: identical tensors.\n"
        "HGT metadata loses those two edge types (no messages, no relation-specific HGTConv weights).\n"
        "Artist nodes remain in the entity universe. Connectivity was not repaired.\n\n"
        f"- isolated entity nodes G_FULL: {n_iso_f}\n"
        f"- isolated entity nodes G_NO_ARTIST: {n_iso_n}\n\n"
        f"Recording degree G_FULL: {rec_deg_f}\n"
        f"Recording degree G_NO_ARTIST: {rec_deg_n}\n"
        f"Artist degree G_FULL: {art_deg_f}\n"
        f"Artist degree G_NO_ARTIST: {art_deg_n}\n",
        encoding="utf-8",
    )

    (audit / "NODE_UNIVERSE_IDENTITY_AUDIT.md").write_text(
        "# NODE_UNIVERSE_IDENTITY_AUDIT\n\n"
        f"- n_users: {g_full['meta']['n_users']} (identical)\n"
        f"- n_entities: {g_full['meta']['n_entities']} (identical)\n"
        f"- n_items: {g_full['meta']['n_items']} (identical)\n"
        f"- item_offset: {g_full['meta']['item_offset']} (identical)\n\n"
        "Artist entities were not deleted. Ablation removes the recording↔artist path only.\n",
        encoding="utf-8",
    )

    hi_hash = sha256_arr(Hi_tr, Hi_va)
    ha_hash = sha256_arr(Ha_tr, Ha_va)
    a_hash = sha256_arr(A_tr, A_va)
    (audit / "ROUTING_IDENTITY_AUDIT.md").write_text(
        "# ROUTING_IDENTITY_AUDIT\n\n"
        "Routing is signed item-A11 Top25, tie-break lower item ID, self-exclusion. "
        "It does not depend on HGT topology.\n\n"
        f"- item-A11 feature hash (train+val): `{hi_hash}`\n"
        "- FULL and NO_ARTIST use the same `race/a11_top25/features` arrays.\n"
        "- Therefore Top25_FULL(u,X) == Top25_NO_ARTIST(u,X) by construction.\n"
        f"- candidate/val_pairs hash: `{hashes['val_pairs']}`\n"
        f"- candidate_hash_FULL == candidate_hash_NO_ARTIST: True\n",
        encoding="utf-8",
    )
    (audit / "FEATURE_IDENTITY_AUDIT.md").write_text(
        "# FEATURE_IDENTITY_AUDIT\n\n"
        "A block and item-A11 are unchanged files. Artist-A11 remains an external channel.\n\n"
        f"- A 5-D hash (train+val): `{a_hash}`\n"
        f"- item-A11 3-D hash: `{hi_hash}`\n"
        f"- artist-A11 3-D hash: `{ha_hash}`\n\n"
        "HGT ablation does **not** delete `artists_of_item` or H_artist. "
        "B3_NO_ARTIST still receives the true candidate-aligned artist-A11 vector.\n"
        "Two channels are separate: graph topology vs explicit artist-A11.\n",
        encoding="utf-8",
    )
    (audit / "DATA_PROVENANCE_AUDIT.md").write_text(
        "# DATA_PROVENANCE_AUDIT\n\n"
        "Corrected Last-FM* only. Splits not rebuilt. Test inaccessible.\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in hashes.items())
        + "\n\n"
        f"Overlaps (must be 0): {overlaps}\n"
        f"test_status: LOCKED_NOT_RUN\n"
        f"fullrank_status: LOCKED_NOT_RUN\n",
        encoding="utf-8",
    )
    (audit / "TRAINING_FAIRNESS_AUDIT.md").write_text(
        "# TRAINING_FAIRNESS_AUDIT\n\n"
        "B0_FULL reused from `race/a11_top25` + residual logits. Fingerprints matched "
        "(dataset, B0 checkpoints, val_pairs, architecture 64/2/2, recipe Adam 1e-3 / wd 1e-4 / "
        "batch 4096 / ≤15 ep / patience 4 / BCE+pos_weight).\n\n"
        "B3_FULL reused from `ARTIST_A11_RESIDUAL_BRANCH_V1` under the same val_pairs hash.\n\n"
        "B0_NO_ARTIST trained from scratch on G_NO_ARTIST. No FULL-graph weight load.\n"
        "B3_NO_ARTIST: freeze B0_NO_ARTIST, same Linear(3,8)-GELU-Linear(8,1), zero-init last layer.\n\n"
        "Only intended difference vs FULL: artist forward+reverse edges absent from HGT.\n"
        "Node features were not added.\n",
        encoding="utf-8",
    )
    write_json(
        OUT / "cache" / "graph_meta.json",
        {
            "relation": rel,
            "dropped": dropped,
            "n_edges_full": n_edges(e_f),
            "n_edges_no": n_edges(e_n),
            "hash_full": hash_edge_dict(e_f),
            "hash_no": hash_edge_dict(e_n),
            "n_users": g_full["meta"]["n_users"],
            "n_entities": g_full["meta"]["n_entities"],
            "isolated_full": n_iso_f,
            "isolated_no": n_iso_n,
            "recording_degree_full": rec_deg_f,
            "recording_degree_no": rec_deg_n,
            "artist_degree_full": art_deg_f,
            "artist_degree_no": art_deg_n,
            "routing_hash": hi_hash,
            "A_hash": a_hash,
            "H_artist_hash": ha_hash,
        },
    )


def finalize(
    bundle: dict[str, Any],
    full_rows: list[dict[str, Any]],
    no_rows: list[dict[str, Any]],
    hashes: dict[str, str],
    embed: dict[str, Any],
) -> None:
    rep = OUT / "reports"
    rep.mkdir(parents=True, exist_ok=True)
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

    va_u = bundle["val_pairs"]["user_id"]
    va_i = bundle["val_pairs"]["item_id"]
    va_y = bundle["val_pairs"]["label"]

    seed_rows = []
    paired = []
    ge_all, df_all, dn_all, ix_all = [], [], [], []
    act_acc = {b: {"B0F": [], "B0N": [], "B3F": [], "B3N": []} for b in ACT}
    seen_acc = {k: {"B0F": [], "B0N": [], "B3F": [], "B3N": []} for k in ("SEEN", "UNSEEN")}

    for fr, nr in zip(full_rows, no_rows):
        if fr["seed"] != nr["seed"]:
            abort("seed order mismatch")
        s = fr["seed"]
        b0f, b0n = float(fr["B0"]["NDCG@20"]), float(nr["B0"]["NDCG@20"])
        b3f, b3n = float(fr["B3"]["NDCG@20"]), float(nr["B3"]["NDCG@20"])
        ge = b0n - b0f
        df = b3f - b0f
        dn = b3n - b0n
        ix = dn - df
        seed_rows.append(
            {
                "seed": s,
                "B0_FULL": b0f,
                "B0_NO_ARTIST": b0n,
                "B3_FULL": b3f,
                "B3_NO_ARTIST": b3n,
                "GRAPH_EFFECT": ge,
                "DELTA_FULL": df,
                "DELTA_NO_ARTIST": dn,
                "INTERACTION": ix,
            }
        )
        common = sorted(set(fr["pu0"]) & set(fr["pu3"]) & set(nr["pu0"]) & set(nr["pu3"]))
        for u in common:
            g_u = nr["pu0"][u] - fr["pu0"][u]
            df_u = fr["pu3"][u] - fr["pu0"][u]
            dn_u = nr["pu3"][u] - nr["pu0"][u]
            ix_u = dn_u - df_u
            ge_all.append(g_u)
            df_all.append(df_u)
            dn_all.append(dn_u)
            ix_all.append(ix_u)
            paired.append(
                {
                    "seed": s,
                    "user_id": u,
                    "NDCG_B0_FULL": fr["pu0"][u],
                    "NDCG_B0_NO_ARTIST": nr["pu0"][u],
                    "NDCG_B3_FULL": fr["pu3"][u],
                    "NDCG_B3_NO_ARTIST": nr["pu3"][u],
                    "graph_effect_user": g_u,
                    "delta_full_user": df_u,
                    "delta_no_artist_user": dn_u,
                    "interaction_user": ix_u,
                }
            )
            ab = activity_bucket(len(mt.get(int(u), ())))
            act_acc[ab]["B0F"].append(fr["pu0"][u])
            act_acc[ab]["B0N"].append(nr["pu0"][u])
            act_acc[ab]["B3F"].append(fr["pu3"][u])
            act_acc[ab]["B3N"].append(nr["pu3"][u])
            ua = user_arts(int(u))
            mask = va_u == u
            pos_items = [int(i) for i, y in zip(va_i[mask].tolist(), va_y[mask].tolist()) if y > 0.5]
            n_seen = sum(1 for i in pos_items if cand_seen(int(u), i, ua))
            key = "SEEN" if pos_items and n_seen >= (len(pos_items) - n_seen) else "UNSEEN"
            if pos_items:
                seen_acc[key]["B0F"].append(fr["pu0"][u])
                seen_acc[key]["B0N"].append(nr["pu0"][u])
                seen_acc[key]["B3F"].append(fr["pu3"][u])
                seen_acc[key]["B3N"].append(nr["pu3"][u])

    mean_row = {
        "seed": "mean",
        "B0_FULL": float(np.mean([r["B0_FULL"] for r in seed_rows])),
        "B0_NO_ARTIST": float(np.mean([r["B0_NO_ARTIST"] for r in seed_rows])),
        "B3_FULL": float(np.mean([r["B3_FULL"] for r in seed_rows])),
        "B3_NO_ARTIST": float(np.mean([r["B3_NO_ARTIST"] for r in seed_rows])),
    }
    mean_row["GRAPH_EFFECT"] = mean_row["B0_NO_ARTIST"] - mean_row["B0_FULL"]
    mean_row["DELTA_FULL"] = mean_row["B3_FULL"] - mean_row["B0_FULL"]
    mean_row["DELTA_NO_ARTIST"] = mean_row["B3_NO_ARTIST"] - mean_row["B0_NO_ARTIST"]
    mean_row["INTERACTION"] = mean_row["DELTA_NO_ARTIST"] - mean_row["DELTA_FULL"]
    std_row = {
        "seed": "std",
        "B0_FULL": float(np.std([r["B0_FULL"] for r in seed_rows], ddof=1)),
        "B0_NO_ARTIST": float(np.std([r["B0_NO_ARTIST"] for r in seed_rows], ddof=1)),
        "B3_FULL": float(np.std([r["B3_FULL"] for r in seed_rows], ddof=1)),
        "B3_NO_ARTIST": float(np.std([r["B3_NO_ARTIST"] for r in seed_rows], ddof=1)),
        "GRAPH_EFFECT": float(np.std([r["GRAPH_EFFECT"] for r in seed_rows], ddof=1)),
        "DELTA_FULL": float(np.std([r["DELTA_FULL"] for r in seed_rows], ddof=1)),
        "DELTA_NO_ARTIST": float(np.std([r["DELTA_NO_ARTIST"] for r in seed_rows], ddof=1)),
        "INTERACTION": float(np.std([r["INTERACTION"] for r in seed_rows], ddof=1)),
    }
    fields = [
        "seed", "B0_FULL", "B0_NO_ARTIST", "B3_FULL", "B3_NO_ARTIST",
        "GRAPH_EFFECT", "DELTA_FULL", "DELTA_NO_ARTIST", "INTERACTION",
    ]
    write_csv(rep / "ARTIST_EDGE_ABLATION_BY_SEED.csv", seed_rows + [mean_row, std_row], fields)
    write_csv(rep / "ARTIST_EDGE_ABLATION_OVERALL.csv", [mean_row, std_row], fields)
    write_csv(
        rep / "ARTIST_EDGE_ABLATION_PAIRED_USERS.csv",
        paired,
        [
            "seed", "user_id", "NDCG_B0_FULL", "NDCG_B0_NO_ARTIST", "NDCG_B3_FULL", "NDCG_B3_NO_ARTIST",
            "graph_effect_user", "delta_full_user", "delta_no_artist_user", "interaction_user",
        ],
    )

    boot_rows = []
    for name, arr in (
        ("B0_NO_ARTIST - B0_FULL", np.asarray(ge_all, dtype=np.float64)),
        ("B3_FULL - B0_FULL", np.asarray(df_all, dtype=np.float64)),
        ("B3_NO_ARTIST - B0_NO_ARTIST", np.asarray(dn_all, dtype=np.float64)),
        ("INTERACTION", np.asarray(ix_all, dtype=np.float64)),
    ):
        mu, lo, hi = bootstrap_ci(arr)
        boot_rows.append(
            {
                "contrast": name,
                "n": int(arr.size),
                "mean": mu,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "ci_entirely_gt0": bool(lo > 0),
                "ci_entirely_lt0": bool(hi < 0),
            }
        )
    write_csv(
        rep / "ARTIST_EDGE_INTERACTION_BOOTSTRAP.csv",
        boot_rows,
        ["contrast", "n", "mean", "ci95_lo", "ci95_hi", "ci_entirely_gt0", "ci_entirely_lt0"],
    )

    act_rows = []
    for b in ACT:
        m0f = float(np.mean(act_acc[b]["B0F"])) if act_acc[b]["B0F"] else float("nan")
        m0n = float(np.mean(act_acc[b]["B0N"])) if act_acc[b]["B0N"] else float("nan")
        m3f = float(np.mean(act_acc[b]["B3F"])) if act_acc[b]["B3F"] else float("nan")
        m3n = float(np.mean(act_acc[b]["B3N"])) if act_acc[b]["B3N"] else float("nan")
        act_rows.append(
            {
                "bucket": b,
                "B0_FULL": m0f,
                "B0_NO_ARTIST": m0n,
                "B3_FULL": m3f,
                "B3_NO_ARTIST": m3n,
                "GRAPH_EFFECT": m0n - m0f,
                "INTERACTION": (m3n - m0n) - (m3f - m0f),
                "n": len(act_acc[b]["B0F"]),
            }
        )
    write_csv(
        rep / "ARTIST_EDGE_ABLATION_BY_ACTIVITY.csv",
        act_rows,
        ["bucket", "B0_FULL", "B0_NO_ARTIST", "B3_FULL", "B3_NO_ARTIST", "GRAPH_EFFECT", "INTERACTION", "n"],
    )

    seen_rows = []
    for k in ("SEEN", "UNSEEN"):
        m0f = float(np.mean(seen_acc[k]["B0F"])) if seen_acc[k]["B0F"] else float("nan")
        m0n = float(np.mean(seen_acc[k]["B0N"])) if seen_acc[k]["B0N"] else float("nan")
        m3f = float(np.mean(seen_acc[k]["B3F"])) if seen_acc[k]["B3F"] else float("nan")
        m3n = float(np.mean(seen_acc[k]["B3N"])) if seen_acc[k]["B3N"] else float("nan")
        ge = m0n - m0f
        df = m3f - m0f
        dn = m3n - m0n
        seen_rows.append(
            {
                "split": f"ARTIST_{k}",
                "B0_FULL": m0f,
                "B0_NO_ARTIST": m0n,
                "B3_FULL": m3f,
                "B3_NO_ARTIST": m3n,
                "GRAPH_EFFECT": ge,
                "DELTA_FULL": df,
                "DELTA_NO_ARTIST": dn,
                "INTERACTION": dn - df,
                "n": len(seen_acc[k]["B0F"]),
            }
        )
    write_csv(
        rep / "ARTIST_EDGE_ABLATION_ARTIST_SEEN_UNSEEN.csv",
        seen_rows,
        ["split", "B0_FULL", "B0_NO_ARTIST", "B3_FULL", "B3_NO_ARTIST",
         "GRAPH_EFFECT", "DELTA_FULL", "DELTA_NO_ARTIST", "INTERACTION", "n"],
    )

    ge_mean = mean_row["GRAPH_EFFECT"]
    n_full_gt = int(sum(1 for r in seed_rows if r["B0_FULL"] > r["B0_NO_ARTIST"]))
    ge_boot = next(x for x in boot_rows if x["contrast"].startswith("B0_NO"))
    if ge_mean < 0 and n_full_gt >= 2 and ge_boot["ci_entirely_lt0"]:
        edge_v = "ARTIST_EDGE_IMPORTANT"
    elif ge_mean < 0 and n_full_gt >= 2:
        edge_v = "ARTIST_EDGE_WEAK"
    elif ge_mean < 0:
        edge_v = "ARTIST_EDGE_WEAK"
    else:
        edge_v = "ARTIST_EDGE_NO_EFFECT"

    ix_mean = mean_row["INTERACTION"]
    n_ix_pos = int(sum(1 for r in seed_rows if r["DELTA_NO_ARTIST"] > r["DELTA_FULL"]))
    ix_boot = next(x for x in boot_rows if x["contrast"] == "INTERACTION")
    if ix_mean > 0 and n_ix_pos >= 2 and ix_boot["ci_entirely_gt0"]:
        red_v = "REDUNDANCY_SUPPORTED"
    elif ix_mean > 0:
        red_v = "REDUNDANCY_WEAK"
    else:
        red_v = "REDUNDANCY_NOT_SUPPORTED"

    if edge_v == "ARTIST_EDGE_IMPORTANT" and ix_mean > 0 and abs(mean_row["DELTA_NO_ARTIST"]) > 2 * max(abs(mean_row["DELTA_FULL"]), 1e-8):
        case = "CASE 1 — artist topology useful and artist-A11 substitutes when it is removed"
    elif edge_v in {"ARTIST_EDGE_IMPORTANT", "ARTIST_EDGE_WEAK"} and abs(ix_mean) <= 5e-5:
        case = "CASE 2 — artist topology matters but artist-A11 is not a substitute"
    elif edge_v == "ARTIST_EDGE_NO_EFFECT" and abs(ix_mean) <= 5e-5:
        case = "CASE 3 — current HGT is not materially using the artist relation"
    elif edge_v == "ARTIST_EDGE_NO_EFFECT" and ix_mean > 0:
        case = "CASE 4 — little graph effect but larger residual after ablation; interpret cautiously"
    else:
        case = "MIXED — see numbers; do not over-claim"

    vl = next(x for x in act_rows if x["bucket"] == "VERY_LIGHT")
    unseen = next(x for x in seen_rows if x["split"] == "ARTIST_UNSEEN")
    seen = next(x for x in seen_rows if x["split"] == "ARTIST_SEEN")
    act_sorted = sorted(act_rows, key=lambda r: r["GRAPH_EFFECT"])
    most_neg = act_sorted[0]["bucket"]

    q8 = (
        "Partially: artist topology carries ranking-useful information."
        if edge_v == "ARTIST_EDGE_IMPORTANT"
        else (
            "Weakly: point estimate says artist edges help B0, but uncertainty remains."
            if edge_v == "ARTIST_EDGE_WEAK"
            else "No: removing artist edges did not reduce B0. Artist-A11 redundancy is unlikely to be HGT artist topology."
        )
    )
    q9 = (
        "Yes, substitution is supported: artist-A11 recovers more after the path is cut."
        if red_v == "REDUNDANCY_SUPPORTED"
        else (
            "Uncertain: interaction is positive on average but not CI-separated from 0."
            if red_v == "REDUNDANCY_WEAK"
            else "More likely item-A11 / other relations: artist-A11 does not become more useful after ablation."
        )
    )
    q10 = "YES — recommend a separate HGT_NODE_ROLE_FEATURES_V1 later." if edge_v == "ARTIST_EDGE_IMPORTANT" else "NO — do not start node-role features from this result."

    def fmt(x: float) -> str:
        return f"{x:.6f}"

    md = f"""# HGT_ARTIST_EDGE_ABLATION_V1

Sampled validation only. Test locked. Full-rank locked. No node-feature enrichment.

G_NO_ARTIST = G_FULL after removing `{ARTIST_URI}` and its graph reverse. Artist nodes kept. Artist-A11 kept as an external channel.

## Primary table

| seed | B0_FULL | B0_NO_ARTIST | B3_FULL | B3_NO_ARTIST |
|---|---:|---:|---:|---:|
| 101 | {fmt(seed_rows[0]['B0_FULL'])} | {fmt(seed_rows[0]['B0_NO_ARTIST'])} | {fmt(seed_rows[0]['B3_FULL'])} | {fmt(seed_rows[0]['B3_NO_ARTIST'])} |
| 202 | {fmt(seed_rows[1]['B0_FULL'])} | {fmt(seed_rows[1]['B0_NO_ARTIST'])} | {fmt(seed_rows[1]['B3_FULL'])} | {fmt(seed_rows[1]['B3_NO_ARTIST'])} |
| 303 | {fmt(seed_rows[2]['B0_FULL'])} | {fmt(seed_rows[2]['B0_NO_ARTIST'])} | {fmt(seed_rows[2]['B3_FULL'])} | {fmt(seed_rows[2]['B3_NO_ARTIST'])} |
| mean | {fmt(mean_row['B0_FULL'])} | {fmt(mean_row['B0_NO_ARTIST'])} | {fmt(mean_row['B3_FULL'])} | {fmt(mean_row['B3_NO_ARTIST'])} |
| std | {fmt(std_row['B0_FULL'])} | {fmt(std_row['B0_NO_ARTIST'])} | {fmt(std_row['B3_FULL'])} | {fmt(std_row['B3_NO_ARTIST'])} |

## Derived effects

| seed | GRAPH_EFFECT | DELTA_FULL | DELTA_NO_ARTIST | INTERACTION |
|---|---:|---:|---:|---:|
| 101 | {seed_rows[0]['GRAPH_EFFECT']:+.6f} | {seed_rows[0]['DELTA_FULL']:+.6f} | {seed_rows[0]['DELTA_NO_ARTIST']:+.6f} | {seed_rows[0]['INTERACTION']:+.6f} |
| 202 | {seed_rows[1]['GRAPH_EFFECT']:+.6f} | {seed_rows[1]['DELTA_FULL']:+.6f} | {seed_rows[1]['DELTA_NO_ARTIST']:+.6f} | {seed_rows[1]['INTERACTION']:+.6f} |
| 303 | {seed_rows[2]['GRAPH_EFFECT']:+.6f} | {seed_rows[2]['DELTA_FULL']:+.6f} | {seed_rows[2]['DELTA_NO_ARTIST']:+.6f} | {seed_rows[2]['INTERACTION']:+.6f} |
| mean | {mean_row['GRAPH_EFFECT']:+.6f} | {mean_row['DELTA_FULL']:+.6f} | {mean_row['DELTA_NO_ARTIST']:+.6f} | {mean_row['INTERACTION']:+.6f} |

GRAPH_EFFECT = B0_NO_ARTIST − B0_FULL. INTERACTION = Δ_NO_ARTIST − Δ_FULL.

## Paired-user bootstrap (10,000)

| contrast | mean | 95% CI |
|---|---:|---|
| B0_NO_ARTIST − B0_FULL | {ge_boot['mean']:+.6f} | [{ge_boot['ci95_lo']:+.6f}, {ge_boot['ci95_hi']:+.6f}] |
| B3_FULL − B0_FULL | {boot_rows[1]['mean']:+.6f} | [{boot_rows[1]['ci95_lo']:+.6f}, {boot_rows[1]['ci95_hi']:+.6f}] |
| B3_NO_ARTIST − B0_NO_ARTIST | {boot_rows[2]['mean']:+.6f} | [{boot_rows[2]['ci95_lo']:+.6f}, {boot_rows[2]['ci95_hi']:+.6f}] |
| INTERACTION | {ix_boot['mean']:+.6f} | [{ix_boot['ci95_lo']:+.6f}, {ix_boot['ci95_hi']:+.6f}] |

## Activity (diagnostic)

| bucket | B0_FULL | B0_NO | B3_FULL | B3_NO | GRAPH_EFFECT | INTERACTION | n |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for r in act_rows:
        md += (
            f"| {r['bucket']} | {r['B0_FULL']:.4f} | {r['B0_NO_ARTIST']:.4f} | {r['B3_FULL']:.4f} | "
            f"{r['B3_NO_ARTIST']:.4f} | {r['GRAPH_EFFECT']:+.5f} | {r['INTERACTION']:+.5f} | {r['n']} |\n"
        )
    md += f"""
VERY_LIGHT GRAPH_EFFECT={vl['GRAPH_EFFECT']:+.5f} INTERACTION={vl['INTERACTION']:+.5f}.
Largest negative GRAPH_EFFECT bucket: {most_neg}.

## ARTIST_SEEN / ARTIST_UNSEEN

| split | B0_FULL | B0_NO | GRAPH_EFFECT | Δ_FULL | Δ_NO | INTERACTION | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARTIST_SEEN | {seen['B0_FULL']:.4f} | {seen['B0_NO_ARTIST']:.4f} | {seen['GRAPH_EFFECT']:+.5f} | {seen['DELTA_FULL']:+.5f} | {seen['DELTA_NO_ARTIST']:+.5f} | {seen['INTERACTION']:+.5f} | {seen['n']} |
| ARTIST_UNSEEN | {unseen['B0_FULL']:.4f} | {unseen['B0_NO_ARTIST']:.4f} | {unseen['GRAPH_EFFECT']:+.5f} | {unseen['DELTA_FULL']:+.5f} | {unseen['DELTA_NO_ARTIST']:+.5f} | {unseen['INTERACTION']:+.5f} | {unseen['n']} |

## Representation diagnostic (descriptive)

Independently trained models; do not treat coordinate differences as semantics.

"""
    for r in embed["rows"]:
        md += (
            f"- seed {r['seed']} {r['role']}: cosine mean={r['cosine_mean']:.3f} "
            f"L2 mean={r['l2_mean']:.3f} (n={r['n']})\n"
        )
    md += f"""
## Q1–Q10

1. Does removing recording↔artist edges reduce B0 sampled NDCG@20? **{'YES' if ge_mean < 0 else 'NO'}** (mean GRAPH_EFFECT={ge_mean:+.6f}).
2. Consistent across seeds? **{n_full_gt}/3** seeds have B0_FULL > B0_NO_ARTIST.
3. Does artist-A11 residual gain become larger without artist edges? **{'YES' if mean_row['DELTA_NO_ARTIST'] > mean_row['DELTA_FULL'] else 'NO'}**.
4. Is INTERACTION positive? **{'YES' if ix_mean > 0 else 'NO'}** ({ix_mean:+.6f}).
5. Is INTERACTION bootstrap CI entirely > 0? **{'YES' if ix_boot['ci_entirely_gt0'] else 'NO'}**.
6. Where is artist topology most important (most negative GRAPH_EFFECT)? **{most_neg}**.
7. ARTIST_SEEN GRAPH_EFFECT={seen['GRAPH_EFFECT']:+.5f}; ARTIST_UNSEEN GRAPH_EFFECT={unseen['GRAPH_EFFECT']:+.5f}. UNSEEN INTERACTION={unseen['INTERACTION']:+.5f}.
8. Support for “artist information is already encoded by HGT topology”? {q8}
9. Or is artist-A11 redundancy more likely from item-A11 / another source? {q9}
10. Justify HGT_NODE_ROLE_FEATURES_V1? {q10}

Case reading: {case}

B0_FULL / B3_FULL reused (fingerprints matched). B0_NO_ARTIST trained from scratch. Test not run. Full-rank not run.

HGT_ARTIST_EDGE_VERDICT = {edge_v}

HGT_ARTIST_A11_REDUNDANCY_VERDICT = {red_v}
"""
    (rep / "ARTIST_EDGE_ABLATION_REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    write_json(
        OUT / "HGT_ARTIST_EDGE_ABLATION_V1_MANIFEST.json",
        {
            "experiment_id": "HGT_ARTIST_EDGE_ABLATION_V1",
            "timestamp": utc_now(),
            "git_commit": git_commit(),
            "dataset": "LastFM_star_IntentAwareRS",
            **{f"{k}_hash": v for k, v in hashes.items()},
            "B0_FULL_reused": True,
            "B3_FULL_reused": True,
            "B0_NO_ARTIST_trained_from_scratch": True,
            "artist_relation_uri": ARTIST_URI,
            "evaluation": "SAMPLED_VALIDATION_ONLY",
            "fullrank_status": "LOCKED_NOT_RUN",
            "test_status": "LOCKED_NOT_RUN",
            "node_features_added": False,
            "HGT_ARTIST_EDGE_VERDICT": edge_v,
            "HGT_ARTIST_A11_REDUNDANCY_VERDICT": red_v,
            "GRAPH_EFFECT_mean": ge_mean,
            "INTERACTION_mean": ix_mean,
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("audit", "reports", "runs", "checkpoints", "cache"):
        (OUT / sub).mkdir(exist_ok=True)
    (OUT / "FULLRANK_LOCKED").write_text("LOCKED_NOT_RUN\n")
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n")

    print("[ablation] load protocol + prepared bundle", flush=True)
    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    overlaps = check_overlaps()
    rel = resolve_artist_relation()
    write_json(OUT / "cache" / "artist_relation.json", rel)
    print(f"[ablation] artist rel id={rel['local_relation_id']} uri={rel['relation_uri']}", flush=True)

    A_tr = np.load(shared_A_dir() / "X_train.npy").astype(np.float32)
    A_va = np.load(shared_A_dir() / "X_val.npy").astype(np.float32)
    Hi_tr = np.load(B0_H / "X_train.npy").astype(np.float32)
    Hi_va = np.load(B0_H / "X_val.npy").astype(np.float32)
    Ha_tr = np.load(H_ART / "X_train.npy").astype(np.float32)
    Ha_va = np.load(H_ART / "X_val.npy").astype(np.float32)
    n_tr = len(bundle["train_pairs"]["label"])
    n_va = len(bundle["val_pairs"]["label"])
    if A_tr.shape != (n_tr, 5) or Hi_tr.shape != (n_tr, 3) or Ha_tr.shape != (n_tr, 3):
        abort("train feature shape mismatch")
    if A_va.shape != (n_va, 5) or Hi_va.shape != (n_va, 3) or Ha_va.shape != (n_va, 3):
        abort("val feature shape mismatch")
    if not (PREV / "cache" / "artists_of_item.pkl").exists():
        abort("missing external artist mapping — would destroy the A11 channel")

    acfg = cfg.get("models", {}).get("architecture", {})
    print("[ablation] build G_FULL (same subsample as B0)", flush=True)
    g_full = load_data_and_typed_graph(
        cfg, bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
    g_no, dropped = drop_artist_edges(g_full, int(rel["local_relation_id"]))
    n_ent = int(g_full["meta"]["n_entities"])
    n_items = int(g_full["meta"]["n_items"])
    deg_f = entity_degree(g_full["edge_index_dict"], n_ent)
    deg_n = entity_degree(g_no["edge_index_dict"], n_ent)
    iso_f = n_ent - len(entity_touched(g_full["edge_index_dict"]))
    iso_n = n_ent - len(entity_touched(g_no["edge_index_dict"]))
    map_z = np.load(PREV / "cache" / "artist_mapping.npz")
    artist_ids = np.asarray(map_z["artist_entity_ids"], dtype=np.int64)
    artist_ids = artist_ids[(artist_ids >= 0) & (artist_ids < n_ent)]
    rec_ids = np.arange(n_items, dtype=np.int64)
    write_audits(
        rel=rel,
        hashes=hashes,
        overlaps=overlaps,
        g_full=g_full,
        g_no=g_no,
        dropped=dropped,
        rec_deg_f=deg_summary(deg_f[rec_ids]),
        rec_deg_n=deg_summary(deg_n[rec_ids]),
        art_deg_f=deg_summary(deg_f[artist_ids]),
        art_deg_n=deg_summary(deg_n[artist_ids]),
        n_iso_f=iso_f,
        n_iso_n=iso_n,
        A_tr=A_tr,
        A_va=A_va,
        Hi_tr=Hi_tr,
        Hi_va=Hi_va,
        Ha_tr=Ha_tr,
        Ha_va=Ha_va,
        bundle=bundle,
    )
    print(
        f"[ablation] G_FULL edges={n_edges(g_full['edge_index_dict'])} "
        f"G_NO={n_edges(g_no['edge_index_dict'])} "
        f"removed_fwd={dropped['removed_forward']} removed_rev={dropped['removed_reverse']}",
        flush=True,
    )
    if "--audit-only" in sys.argv:
        print("[ablation] --audit-only: graph/data audits written; stopping before train.", flush=True)
        return

    full_rows = [load_full_condition(bundle, s) for s in SEEDS]
    no_rows = []
    for seed in SEEDS:
        ckpt = train_b0_no_artist(bundle, seed, g_no)
        # scaler identity vs B0_FULL
        with (ckpt / "a_scaler.pkl").open("rb") as f:
            a_no = pickle.load(f)
        with (B0_CKPT / f"seed_{seed}" / "a_scaler.pkl").open("rb") as f:
            a_full = pickle.load(f)
        if not np.allclose(a_no.mean_, a_full.mean_) or not np.allclose(a_no.scale_, a_full.scale_):
            abort(f"A scaler mismatch seed {seed}")
        with (ckpt / "h_scaler.pkl").open("rb") as f:
            h_no = pickle.load(f)
        with (B0_CKPT / f"seed_{seed}" / "h_scaler.pkl").open("rb") as f:
            h_full = pickle.load(f)
        if not np.allclose(h_no.mean_, h_full.mean_) or not np.allclose(h_no.scale_, h_full.scale_):
            abort(f"H scaler mismatch seed {seed}")
        ctx = load_model(bundle, ckpt, g_no)
        no_rows.append(
            train_b3_no_artist(bundle, seed, ctx, A_tr, A_va, Hi_tr, Hi_va, Ha_tr, Ha_va)
        )
        del ctx
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("[ablation] embedding diagnostic", flush=True)
    embed = embedding_diagnostic(bundle, g_full, g_no, artist_ids)
    finalize(bundle, full_rows, no_rows, hashes, embed)
    print("test_status: LOCKED_NOT_RUN", flush=True)
    print("fullrank_status: LOCKED_NOT_RUN", flush=True)


if __name__ == "__main__":
    main()
