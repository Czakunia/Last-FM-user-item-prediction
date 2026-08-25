#!/usr/bin/env python3
"""HGT_NODE_SEMANTIC_ROLE_FEATURES_V1 — sampled validation only.

N0 ID-only vs N1 coarse residual vs N2 multi-hot residual.
Roles from KG_NODE_ROLE_AUDIT_V1. Matched init. Zero-init role projection.
No test. No full-rank. No HeteroLinear / PE / KGE / aggregation change.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG, materialized_root  # noqa: E402

from scripts.run_artist_a11_residual_branch_v1 import (  # noqa: E402
    bootstrap_ci,
    per_user_ndcg,
    sampled_metrics,
    scale_split,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    ACT,
    EXPECTED,
    abort,
    check_overlaps,
    git_commit,
    hash_edge_dict,
    sha256_arr,
    sha256_file,
    verify_fingerprints,
    write_csv,
    write_json,
)
from scripts.run_publication_our_hgt_fullrank import build_fusion_head  # noqa: E402
from scripts.run_race_clean_3 import activity_bucket, shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes  # noqa: E402
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users  # noqa: E402
from src.lastfm_lp.models.encoders import HGTGraphEncoder  # noqa: E402
from src.lastfm_lp.models.fusion import RecommendationModel  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = PROTOCOL_CONFIG
DATA = ROOT / "data" / "LastFM_star_IntentAwareRS"
RACE = materialized_root()
AUDIT_SRC = RACE / "audit" / "KG_NODE_ROLE_AUDIT_V1"
B0_H = RACE / "race" / "a11_top25" / "features"
PREV = RACE / "ARTIST_A11_LEVEL_V1"
OUT = RACE / "HGT_NODE_SEMANTIC_ROLE_FEATURES_V1"
SEEDS = (101, 202, 303)
N_ITEMS = 48123
N_ENTITIES = 106389
HASH_FULL = "3b59bac939fea39fc1ed294b52840b057285efa80a37c008c0d6986a0d421970"
COARSE_ORDER = (
    "catalog_recording",
    "noncatalog_recording",
    "release",
    "credit_person",
    "type",
    "song",
)
MULTI_ORDER = (
    "catalog_recording",
    "noncatalog_recording",
    "release",
    "type",
    "song",
    "artist",
    "producer",
    "engineer",
    "featured_artist",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_role_matrices() -> dict[str, Any]:
    kg = np.loadtxt(DATA / "kg_final.txt", dtype=np.int64)
    heads, rels, tails = kg[:, 0], kg[:, 1], kg[:, 2]
    items = set(range(N_ITEMS))

    def tgt(rid: int) -> set[int]:
        return set(tails[rels == rid].tolist())

    def src(rid: int) -> set[int]:
        return set(heads[rels == rid].tolist())

    rec_src = set()
    for rid in (1, 2, 3, 4, 5, 6, 8):
        rec_src |= src(rid)
    artist, engineer, producer, featured = tgt(2), tgt(3), tgt(4), tgt(8)
    release, types, song = tgt(1), tgt(0), tgt(6)
    credit = artist | engineer | producer | featured
    version_like = tgt(5) | tgt(7) | src(7)

    coarse = np.zeros((N_ENTITIES, 6), dtype=np.float32)
    multi = np.zeros((N_ENTITIES, 9), dtype=np.float32)
    leftover = []
    for v in range(N_ENTITIES):
        if v in items:
            coarse[v, 0] = 1.0
            multi[v, 0] = 1.0
        elif v in rec_src:
            coarse[v, 1] = 1.0
            multi[v, 1] = 1.0
        elif v in release:
            coarse[v, 2] = 1.0
            multi[v, 2] = 1.0
        elif v in credit:
            coarse[v, 3] = 1.0
            if v in artist:
                multi[v, 5] = 1.0
            if v in producer:
                multi[v, 6] = 1.0
            if v in engineer:
                multi[v, 7] = 1.0
            if v in featured:
                multi[v, 8] = 1.0
        elif v in types:
            coarse[v, 4] = 1.0
            multi[v, 3] = 1.0
        elif v in song:
            coarse[v, 5] = 1.0
            multi[v, 4] = 1.0
        elif v in version_like:
            coarse[v, 1] = 1.0
            multi[v, 1] = 1.0
            leftover.append(v)
        else:
            abort(f"entity {v} has no audited coarse role")

    if not np.allclose(coarse.sum(axis=1), 1.0):
        abort("coarse role is not exclusive one-hot")
    k = multi.sum(axis=1, keepdims=True)
    if np.any(k < 1):
        abort("multi-hot row with zero bits")
    multi_norm = multi / np.maximum(k, 1.0)
    sets = {
        "catalog_recording": items,
        "noncatalog_recording": rec_src - items,
        "release": release,
        "credit_person": credit,
        "type": types,
        "song": song,
        "artist": artist,
        "producer": producer,
        "engineer": engineer,
        "featured_artist": featured,
        "version_like_leftover_to_noncatalog": set(leftover),
    }
    return {
        "coarse": coarse,
        "multi": multi,
        "multi_norm": multi_norm.astype(np.float32),
        "k": k.reshape(-1).astype(np.float32),
        "sets": sets,
        "n_leftover_version": len(leftover),
    }


def build_model(bundle, graph, device, entity_role: torch.Tensor | None):
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    edge_index_dict = {k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0}
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    role_t = entity_role.to(device) if entity_role is not None else None
    encoder = HGTGraphEncoder(
        graph["meta"]["n_users"],
        graph["meta"]["n_entities"],
        metadata,
        edge_index_dict,
        embed_dim=64,
        n_layers=2,
        heads=2,
        dropout=0.1,
        entity_role=role_t,
    ).to(device)
    fusion = build_fusion_head(encoder=encoder, hcr_dim=3, use_a11=True, fcfg=fcfg).to(device)
    return RecommendationModel(encoder, fusion).to(device)


def shared_keys(sd: dict) -> list[str]:
    return [k for k in sd if "role_proj" not in k]


def copy_shared(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    s, d = src.state_dict(), dst.state_dict()
    merged = {k: s[k] for k in shared_keys(s)}
    for k, v in d.items():
        if k not in merged:
            merged[k] = v
    dst.load_state_dict(merged)


@torch.no_grad()
def logits_on_idx(model, users, items, A_s, H_s, item_offset, device, idx, role_enabled=True, aux_enabled=True):
    model.eval()
    z = model.encoder.encode_all(role_enabled=role_enabled, aux_enabled=aux_enabled)
    u_idx, i_idx = user_item_to_nodes(users[idx], items[idx], item_offset)
    out = model(
        u_idx.to(device),
        i_idx.to(device),
        torch.from_numpy(A_s[idx]).to(device),
        torch.from_numpy(H_s[idx]).to(device),
        z=z,
    )
    return out["logits"].detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def score_all(model, users, items, A_s, H_s, item_offset, device, role_enabled=True, aux_enabled=True) -> np.ndarray:
    model.eval()
    z = model.encoder.encode_all(role_enabled=role_enabled, aux_enabled=aux_enabled)
    u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
    chunks = []
    for start in range(0, len(users), 8192):
        sl = slice(start, start + 8192)
        out = model(
            u_idx[sl].to(device),
            i_idx[sl].to(device),
            torch.from_numpy(A_s[sl]).to(device),
            torch.from_numpy(H_s[sl]).to(device),
            z=z,
        )
        chunks.append(out["logits"].detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_variant(
    bundle,
    model,
    *,
    stage: str,
    seed: int,
    A_tr_s,
    A_va_s,
    H_tr_s,
    H_va_s,
    a_scaler,
    h_scaler,
    item_offset: int,
    device,
) -> Path:
    ckpt = OUT / "checkpoints" / stage / f"seed_{seed}"
    if (ckpt / "model.pt").exists() and (ckpt / "train_meta.json").exists():
        print(f"[{stage}] reuse {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
        return ckpt
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = None
    best_ndcg = -1.0
    left = int(acfg.get("patience", 4))
    history = []
    for epoch in range(int(acfg.get("max_epochs", 15))):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        z_live = model.encoder.encode_all()
        starts = list(range(0, n_train, batch_size))
        n_b = max(len(starts), 1)
        for bi, start in enumerate(starts):
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
        opt.step()
        del z_live
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        val = score_all(
            model,
            bundle["val_pairs"]["user_id"],
            bundle["val_pairs"]["item_id"],
            A_va_s,
            H_va_s,
            item_offset,
            device,
        )
        ndcg = float(ranking_metrics_for_users(
            bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"], val, ks=(20,)
        )["NDCG@20"])
        history.append({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1), "val_NDCG@20": ndcg})
        print(f"[{stage}] epoch {epoch} loss={history[-1]['loss']:.4f} val_NDCG@20={ndcg:.4f}", flush=True)
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
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt / "model.pt")
    with (ckpt / "a_scaler.pkl").open("wb") as f:
        pickle.dump(a_scaler, f)
    with (ckpt / "h_scaler.pkl").open("wb") as f:
        pickle.dump(h_scaler, f)
    write_json(
        ckpt / "train_meta.json",
        {
            "stage": stage,
            "seed": seed,
            "best_val_NDCG@20_sampled": best_ndcg,
            "history": history,
            "item_offset": item_offset,
            "hcr_dim": 3,
            "zero_h_after_scale": False,
        },
    )
    print(f"[{stage}] saved {ckpt}", flush=True)
    return ckpt


def write_role_audits(roles: dict[str, Any], hashes: dict[str, str], overlaps: dict[str, int]) -> None:
    audit = OUT / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    sets = roles["sets"]
    leftover = len(sets["version_like_leftover_to_noncatalog"])
    (audit / "ROLE_FEATURE_MATRIX_AUDIT.md").write_text(
        "# ROLE_FEATURE_MATRIX_AUDIT\n\n"
        "Source: KG_NODE_ROLE_AUDIT_V1 + kg_final.txt + relation_list.txt + item_list.\n"
        "No interactions / val / test / popularity / A11.\n\n"
        f"N1 order: {list(COARSE_ORDER)}\n"
        f"N2 order: {list(MULTI_ORDER)}\n"
        "N2 credit_person is NOT a separate bit.\n"
        "N2 rows are mean-normalized: r / max(1,k).\n"
        f"coarse hash: `{sha256_arr(roles['coarse'])}`\n"
        f"multi raw hash: `{sha256_arr(roles['multi'])}`\n"
        f"multi_norm hash: `{sha256_arr(roles['multi_norm'])}`\n"
        f"version-like leftovers assigned noncatalog_recording: {leftover}\n",
        encoding="utf-8",
    )
    cov_lines = ["# ROLE_COVERAGE_AUDIT\n", "| role | n |", "|---|---:|"]
    for name in COARSE_ORDER:
        cov_lines.append(f"| {name} | {int(roles['coarse'][:, COARSE_ORDER.index(name)].sum())} |")
    cov_lines.append(f"\nN1 row-sum==1: {bool(np.allclose(roles['coarse'].sum(1), 1))}")
    cov_lines.append(f"\nN2 min k: {float(roles['k'].min())} max k: {float(roles['k'].max())}")
    cov_lines.append("\nUNKNOWN=0 OTHER=0 after leftover→noncatalog_recording.\n")
    (audit / "ROLE_COVERAGE_AUDIT.md").write_text("\n".join(cov_lines), encoding="utf-8")
    art, prod, eng, feat = (sets["artist"], sets["producer"], sets["engineer"], sets["featured_artist"])
    credit = sets["credit_person"]
    n_multi_c = sum(1 for v in credit if (v in art) + (v in prod) + (v in eng) + (v in feat) >= 2)
    (audit / "MULTI_ROLE_OVERLAP_AUDIT.md").write_text(
        "# MULTI_ROLE_OVERLAP_AUDIT\n\n"
        f"credit_person={len(credit)} multi_fine={n_multi_c} "
        f"frac={n_multi_c / len(credit):.4f}\n"
        f"artist∩producer={len(art & prod)} featured⊂artist={len(feat & art)}/{len(feat)}\n"
        "Fine roles are multi-hot. Coarse credit_person is exclusive vs release/type/recording.\n",
        encoding="utf-8",
    )
    (audit / "TRAINING_PAIR_IDENTITY_AUDIT.md").write_text(
        f"# TRAINING_PAIR_IDENTITY_AUDIT\n\ntrain_pairs hash=`{hashes['train_pairs']}`\n"
        "N0/N1/N2 share the same cached train_pairs. No new negatives.\n",
        encoding="utf-8",
    )
    (audit / "VALIDATION_CANDIDATE_IDENTITY_AUDIT.md").write_text(
        f"# VALIDATION_CANDIDATE_IDENTITY_AUDIT\n\nval_pairs hash=`{hashes['val_pairs']}`\n"
        "Same frozen sampled candidates for N0/N1/N2.\n",
        encoding="utf-8",
    )
    (audit / "DATA_PROVENANCE_AUDIT.md").write_text(
        "# DATA_PROVENANCE_AUDIT\n\nLast-FM*. Role audit reused, not redone.\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in hashes.items())
        + f"\n- role_audit: `{sha256_file(AUDIT_SRC / 'KG_NODE_ROLE_AUDIT.md')}`\n"
        + f"\n- relation_list: `{sha256_file(DATA / 'relation_list.txt')}`\n"
        + f"\noverlaps={overlaps}\ntest_status: LOCKED_NOT_RUN\nfullrank_status: LOCKED_NOT_RUN\n",
        encoding="utf-8",
    )


def finalize(bundle, cells, roles, hashes) -> None:
    rep = OUT / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    mt = bundle["model_train"]
    with (PREV / "cache" / "artists_of_item.pkl").open("rb") as f:
        artists_of = pickle.load(f)
    va_u = bundle["val_pairs"]["user_id"]
    va_i = bundle["val_pairs"]["item_id"]
    va_y = bundle["val_pairs"]["label"]

    def user_arts(u: int) -> set[int]:
        s: set[int] = set()
        for i in mt.get(int(u), ()):
            if 0 <= int(i) < len(artists_of):
                s.update(int(a) for a in artists_of[int(i)].tolist())
        return s

    def cand_seen(i: int, ua: set[int]) -> bool:
        if not (0 <= int(i) < len(artists_of)):
            return False
        return bool(ua & {int(a) for a in artists_of[int(i)].tolist()})

    seed_rows = []
    paired = []
    d10, d20, d21 = [], [], []
    act = {b: {"N0": [], "N1": [], "N2": []} for b in ACT}
    seen_acc = {k: {"N0": [], "N1": [], "N2": []} for k in ("SEEN", "UNSEEN")}
    for i, seed in enumerate(SEEDS):
        n0 = float(cells["N0"][i]["metrics"]["NDCG@20"])
        n1 = float(cells["N1"][i]["metrics"]["NDCG@20"])
        n2 = float(cells["N2"][i]["metrics"]["NDCG@20"])
        seed_rows.append(
            {
                "seed": seed,
                "N0_ID": n0,
                "N1_COARSE": n1,
                "N2_MULTI_HOT": n2,
                "N1-N0": n1 - n0,
                "N2-N0": n2 - n0,
                "N2-N1": n2 - n1,
                "Recall@20_N0": cells["N0"][i]["metrics"]["Recall@20"],
                "Recall@20_N1": cells["N1"][i]["metrics"]["Recall@20"],
                "Recall@20_N2": cells["N2"][i]["metrics"]["Recall@20"],
                "HitRate@20_N0": cells["N0"][i]["metrics"]["HitRate@20"],
                "HitRate@20_N1": cells["N1"][i]["metrics"]["HitRate@20"],
                "HitRate@20_N2": cells["N2"][i]["metrics"]["HitRate@20"],
                "MRR_N0": cells["N0"][i]["metrics"]["MRR"],
                "MRR_N1": cells["N1"][i]["metrics"]["MRR"],
                "MRR_N2": cells["N2"][i]["metrics"]["MRR"],
            }
        )
        common = sorted(set(cells["N0"][i]["pu"]) & set(cells["N1"][i]["pu"]) & set(cells["N2"][i]["pu"]))
        for u in common:
            a0, a1, a2 = cells["N0"][i]["pu"][u], cells["N1"][i]["pu"][u], cells["N2"][i]["pu"][u]
            d10.append(a1 - a0)
            d20.append(a2 - a0)
            d21.append(a2 - a1)
            paired.append(
                {
                    "seed": seed,
                    "user_id": u,
                    "ndcg_N0": a0,
                    "ndcg_N1": a1,
                    "ndcg_N2": a2,
                    "delta_N1_N0": a1 - a0,
                    "delta_N2_N0": a2 - a0,
                    "delta_N2_N1": a2 - a1,
                }
            )
            ab = activity_bucket(len(mt.get(int(u), ())))
            act[ab]["N0"].append(a0)
            act[ab]["N1"].append(a1)
            act[ab]["N2"].append(a2)
            ua = user_arts(int(u))
            mask = va_u == u
            pos = [int(ii) for ii, y in zip(va_i[mask].tolist(), va_y[mask].tolist()) if y > 0.5]
            n_seen = sum(1 for ii in pos if cand_seen(ii, ua))
            key = "SEEN" if pos and n_seen >= (len(pos) - n_seen) else "UNSEEN"
            if pos:
                seen_acc[key]["N0"].append(a0)
                seen_acc[key]["N1"].append(a1)
                seen_acc[key]["N2"].append(a2)

    def ms(key: str) -> tuple[float, float]:
        xs = [r[key] for r in seed_rows]
        return float(np.mean(xs)), float(np.std(xs, ddof=1))

    mean_row, std_row = {"seed": "mean"}, {"seed": "std"}
    for k in seed_rows[0]:
        if k == "seed":
            continue
        mu, sd = ms(k)
        mean_row[k] = mu
        std_row[k] = sd
    fields = list(seed_rows[0])
    write_csv(rep / "NODE_ROLE_BY_SEED.csv", seed_rows + [mean_row, std_row], fields)
    write_csv(rep / "NODE_ROLE_OVERALL.csv", [mean_row, std_row], fields)
    write_csv(
        rep / "NODE_ROLE_PAIRED_USERS.csv",
        paired,
        ["seed", "user_id", "ndcg_N0", "ndcg_N1", "ndcg_N2", "delta_N1_N0", "delta_N2_N0", "delta_N2_N1"],
    )
    boot_rows = []
    for name, arr in (("N1-N0", np.asarray(d10)), ("N2-N0", np.asarray(d20)), ("N2-N1", np.asarray(d21))):
        mu, lo, hi = bootstrap_ci(arr)
        boot_rows.append(
            {
                "contrast": name,
                "n": int(arr.size),
                "mean": mu,
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "frac_gt0": float((arr > 0).mean()),
                "frac_eq0": float((arr == 0).mean()),
                "frac_lt0": float((arr < 0).mean()),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "ci_entirely_gt0": bool(lo > 0),
            }
        )
    write_csv(
        rep / "NODE_ROLE_BOOTSTRAP.csv",
        boot_rows,
        list(boot_rows[0]),
    )
    act_rows = []
    for b in ACT:
        m0 = float(np.mean(act[b]["N0"])) if act[b]["N0"] else float("nan")
        m1 = float(np.mean(act[b]["N1"])) if act[b]["N1"] else float("nan")
        m2 = float(np.mean(act[b]["N2"])) if act[b]["N2"] else float("nan")
        act_rows.append({"bucket": b, "N0": m0, "N1": m1, "N2": m2, "N2-N0": m2 - m0, "n": len(act[b]["N0"])})
    write_csv(rep / "NODE_ROLE_BY_ACTIVITY.csv", act_rows, ["bucket", "N0", "N1", "N2", "N2-N0", "n"])
    seen_rows = []
    for k in ("SEEN", "UNSEEN"):
        m0 = float(np.mean(seen_acc[k]["N0"])) if seen_acc[k]["N0"] else float("nan")
        m1 = float(np.mean(seen_acc[k]["N1"])) if seen_acc[k]["N1"] else float("nan")
        m2 = float(np.mean(seen_acc[k]["N2"])) if seen_acc[k]["N2"] else float("nan")
        seen_rows.append({"group": f"ARTIST_{k}", "N0": m0, "N1": m1, "N2": m2, "N2-N0": m2 - m0, "n": len(seen_acc[k]["N0"])})
    write_csv(rep / "NODE_ROLE_ARTIST_SEEN_UNSEEN.csv", seen_rows, ["group", "N0", "N1", "N2", "N2-N0", "n"])

    d20m = float(mean_row["N2-N0"])
    n_pos = int(sum(1 for r in seed_rows if r["N2-N0"] > 0))
    b20 = next(x for x in boot_rows if x["contrast"] == "N2-N0")
    if d20m > 0 and n_pos >= 2 and b20["ci_entirely_gt0"]:
        verdict = "ROLE_GAIN"
    elif d20m > 0 and n_pos >= 2:
        verdict = "ROLE_WEAK_GAIN"
    else:
        verdict = "ROLE_NO_GAIN"

    n1n0 = float(mean_row["N1-N0"])
    n2n1 = float(mean_row["N2-N1"])
    if n1n0 > 5e-5 and abs(n2n1) <= 5e-5:
        interp = "coarse useful; fine credit roles add little"
    elif n2n1 > 5e-5 and n1n0 > 5e-5:
        interp = "both coarse and fine overlapping roles contribute"
    elif d20m > 5e-5 and abs(n1n0) <= 5e-5:
        interp = "gain likely depends on fine multi-role semantics"
    elif n1n0 > 5e-5 and n2n1 < -5e-5:
        interp = "fine decomposition may add noise"
    elif abs(n1n0) <= 5e-5 and abs(d20m) <= 5e-5:
        interp = "current HGT topology already appears sufficient for these roles"
    else:
        interp = "mixed; do not over-claim"

    nxt = "Do not start HeteroLinear / PE / KGE / aggregation automatically."
    if verdict == "ROLE_GAIN":
        nxt += " Full-rank only after a separate decision. A later structural-feature or HeteroLinear study is optional."
    elif verdict == "ROLE_NO_GAIN":
        nxt += " Do not expand role features. A later structural/KGE experiment is a different question."

    def f6(x: float) -> str:
        return f"{x:.6f}"

    md = f"""# HGT_NODE_SEMANTIC_ROLE_FEATURES_V1

Sampled validation only. Test locked. Full-rank locked.

All catalog candidates share `catalog_recording`. Any N1/N2 gain is **not**
"the model learned to distinguish candidate item types." The plausible mechanism
is role information on non-catalog KG neighbors changing message passing.

## Table

| seed | N0 ID | N1 COARSE | N2 MULTI_HOT | N1−N0 | N2−N0 | N2−N1 |
|---|---:|---:|---:|---:|---:|---:|
| 101 | {f6(seed_rows[0]['N0_ID'])} | {f6(seed_rows[0]['N1_COARSE'])} | {f6(seed_rows[0]['N2_MULTI_HOT'])} | {seed_rows[0]['N1-N0']:+.6f} | {seed_rows[0]['N2-N0']:+.6f} | {seed_rows[0]['N2-N1']:+.6f} |
| 202 | {f6(seed_rows[1]['N0_ID'])} | {f6(seed_rows[1]['N1_COARSE'])} | {f6(seed_rows[1]['N2_MULTI_HOT'])} | {seed_rows[1]['N1-N0']:+.6f} | {seed_rows[1]['N2-N0']:+.6f} | {seed_rows[1]['N2-N1']:+.6f} |
| 303 | {f6(seed_rows[2]['N0_ID'])} | {f6(seed_rows[2]['N1_COARSE'])} | {f6(seed_rows[2]['N2_MULTI_HOT'])} | {seed_rows[2]['N1-N0']:+.6f} | {seed_rows[2]['N2-N0']:+.6f} | {seed_rows[2]['N2-N1']:+.6f} |
| mean | {f6(mean_row['N0_ID'])} | {f6(mean_row['N1_COARSE'])} | {f6(mean_row['N2_MULTI_HOT'])} | {n1n0:+.6f} | {d20m:+.6f} | {n2n1:+.6f} |
| std | {f6(std_row['N0_ID'])} | {f6(std_row['N1_COARSE'])} | {f6(std_row['N2_MULTI_HOT'])} | {std_row['N1-N0']:+.6f} | {std_row['N2-N0']:+.6f} | {std_row['N2-N1']:+.6f} |

Primary comparison: **N2 vs N0**. N1 is diagnostic.

## Paired-user bootstrap (10,000)

| contrast | mean | median | p25 | p75 | frac>0 | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| N2−N0 | {b20['mean']:+.6f} | {b20['median']:+.6f} | {b20['p25']:+.6f} | {b20['p75']:+.6f} | {b20['frac_gt0']:.3f} | [{b20['ci95_lo']:+.6f}, {b20['ci95_hi']:+.6f}] |
| N1−N0 | {boot_rows[0]['mean']:+.6f} | {boot_rows[0]['median']:+.6f} | {boot_rows[0]['p25']:+.6f} | {boot_rows[0]['p75']:+.6f} | {boot_rows[0]['frac_gt0']:.3f} | [{boot_rows[0]['ci95_lo']:+.6f}, {boot_rows[0]['ci95_hi']:+.6f}] |
| N2−N1 | {boot_rows[2]['mean']:+.6f} | {boot_rows[2]['median']:+.6f} | {boot_rows[2]['p25']:+.6f} | {boot_rows[2]['p75']:+.6f} | {boot_rows[2]['frac_gt0']:.3f} | [{boot_rows[2]['ci95_lo']:+.6f}, {boot_rows[2]['ci95_hi']:+.6f}] |

frac=0 N2−N0: {b20['frac_eq0']:.3f}  frac<0: {b20['frac_lt0']:.3f}

## Activity

| bucket | N0 | N1 | N2 | N2−N0 | n |
|---|---:|---:|---:|---:|---:|
"""
    for r in act_rows:
        md += f"| {r['bucket']} | {r['N0']:.4f} | {r['N1']:.4f} | {r['N2']:.4f} | {r['N2-N0']:+.5f} | {r['n']} |\n"
    unseen = next(x for x in seen_rows if x["group"] == "ARTIST_UNSEEN")
    seen = next(x for x in seen_rows if x["group"] == "ARTIST_SEEN")
    md += f"""
## ARTIST_SEEN / UNSEEN

| group | N0 | N1 | N2 | N2−N0 | n |
|---|---:|---:|---:|---:|---:|
| ARTIST_SEEN | {seen['N0']:.4f} | {seen['N1']:.4f} | {seen['N2']:.4f} | {seen['N2-N0']:+.5f} | {seen['n']} |
| ARTIST_UNSEEN | {unseen['N0']:.4f} | {unseen['N1']:.4f} | {unseen['N2']:.4f} | {unseen['N2-N0']:+.5f} | {unseen['n']} |

Interpretation: {interp}

## Q1–Q14

1. Explicit semantic roles help? **{'YES' if d20m > 0 or n1n0 > 0 else 'NO'}** (primary N2−N0={d20m:+.6f}).
2. Coarse help? **{'YES' if n1n0 > 0 else 'NO'}** ({n1n0:+.6f}).
3. Multi-hot help? **{'YES' if d20m > 0 else 'NO'}** ({d20m:+.6f}).
4. Multi-hot beyond coarse? **{'YES' if n2n1 > 0 else 'NO'}** ({n2n1:+.6f}).
5. N2>N0 seeds: **{n_pos}/3**.
6. N2−N0 CI entirely > 0? **{'YES' if b20['ci_entirely_gt0'] else 'NO'}**.
7–10. See NODE_ROLE_PROJECTION_NORMS.csv and NODE_ROLE_ROLE_ZEROED_DIAGNOSTIC.csv.
11. Activity: see table. 12. ARTIST_UNSEEN N2−N0={unseen['N2-N0']:+.5f}.
13. Later structural / HeteroLinear / GPSE / KGE / aggregation? {nxt}
14. Full-rank? **NO — locked.** Separate decision only if ROLE_GAIN.

HGT_NODE_ROLE_VERDICT = {verdict}
"""
    (rep / "HGT_NODE_SEMANTIC_ROLE_FEATURES_REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    write_json(
        OUT / "HGT_NODE_SEMANTIC_ROLE_FEATURES_V1_MANIFEST.json",
        {
            "experiment_id": "HGT_NODE_SEMANTIC_ROLE_FEATURES_V1",
            "timestamp": utc_now(),
            "git_commit": git_commit(),
            **{f"{k}_hash": v for k, v in hashes.items()},
            "role_audit_hash": sha256_file(AUDIT_SRC / "KG_NODE_ROLE_AUDIT.md"),
            "role_matrix_hash": sha256_arr(roles["coarse"], roles["multi_norm"]),
            "relation_list_hash": sha256_file(DATA / "relation_list.txt"),
            "semantic_feature_order_N1": list(COARSE_ORDER),
            "semantic_feature_order_N2": list(MULTI_ORDER),
            "role_normalization": "active_role_mean",
            "role_projection": {"bias": False, "output_dim": 64, "zero_initialized": True},
            "HGT": {"unchanged": True},
            "decoder": {"unchanged": True, "input_dim": 265},
            "item_A11": {"unchanged": True},
            "A_block": {"unchanged": True},
            "evaluation": "sampled_validation_only",
            "fullrank_status": "LOCKED_NOT_RUN",
            "test_status": "LOCKED_NOT_RUN",
            "HGT_NODE_ROLE_VERDICT": verdict,
        },
    )


def role_diagnostics(models_n1, models_n2, roles, bundle, A_va_s, H_va_s, item_offset, device) -> None:
    sets = roles["sets"]
    rows = []
    for seed, m1, m2 in zip(SEEDS, models_n1, models_n2):
        for tag, model, names, raw in (
            ("N1", m1, COARSE_ORDER, roles["coarse"]),
            ("N2", m2, MULTI_ORDER, roles["multi"]),
        ):
            W = model.encoder.core.role_proj.weight.detach().cpu().numpy()
            comp = model.encoder.core.role_component().detach().cpu().numpy()
            idw = model.encoder.core.entity_embed.weight.detach().cpu().numpy()
            idn = np.linalg.norm(idw, axis=1)
            rn = np.linalg.norm(comp, axis=1)
            ratio = rn / (idn + 1e-12)
            for j, name in enumerate(names):
                mask = raw[:, j] > 0
                rows.append(
                    {
                        "seed": seed,
                        "variant": tag,
                        "role": name,
                        "n_nodes": int(mask.sum()),
                        "proj_col_l2": float(np.linalg.norm(W[:, j])),
                        "role_norm_median": float(np.median(rn[mask])) if mask.any() else float("nan"),
                        "role_norm_p25": float(np.percentile(rn[mask], 25)) if mask.any() else float("nan"),
                        "role_norm_p75": float(np.percentile(rn[mask], 75)) if mask.any() else float("nan"),
                        "role_norm_p95": float(np.percentile(rn[mask], 95)) if mask.any() else float("nan"),
                        "ratio_median": float(np.median(ratio[mask])) if mask.any() else float("nan"),
                    }
                )
            if tag == "N2":
                cols = {n: W[:, MULTI_ORDER.index(n)] for n in ("artist", "producer", "engineer", "featured_artist")}
                for a in cols:
                    for b in cols:
                        if a < b:
                            na, nb = np.linalg.norm(cols[a]), np.linalg.norm(cols[b])
                            cos = float(np.dot(cols[a], cols[b]) / max(na * nb, 1e-12))
                            rows.append(
                                {
                                    "seed": seed,
                                    "variant": "N2_cosine",
                                    "role": f"{a}__{b}",
                                    "n_nodes": -1,
                                    "proj_col_l2": cos,
                                    "role_norm_median": float("nan"),
                                    "role_norm_p25": float("nan"),
                                    "role_norm_p75": float("nan"),
                                    "role_norm_p95": float("nan"),
                                    "ratio_median": float("nan"),
                                }
                            )
                credit = np.array(sorted(sets["credit_person"]), dtype=np.int64)
                k_c = roles["k"][credit]
                single, multi = credit[k_c == 1], credit[k_c >= 2]
                rows.append(
                    {
                        "seed": seed,
                        "variant": "N2_credit",
                        "role": "SINGLE_FINE_ROLE",
                        "n_nodes": int(single.size),
                        "proj_col_l2": float("nan"),
                        "role_norm_median": float(np.median(rn[single])) if single.size else float("nan"),
                        "role_norm_p25": float(np.percentile(rn[single], 25)) if single.size else float("nan"),
                        "role_norm_p75": float(np.percentile(rn[single], 75)) if single.size else float("nan"),
                        "role_norm_p95": float(np.percentile(rn[single], 95)) if single.size else float("nan"),
                        "ratio_median": float(np.median(ratio[single])) if single.size else float("nan"),
                    }
                )
                rows.append(
                    {
                        "seed": seed,
                        "variant": "N2_credit",
                        "role": "MULTI_FINE_ROLE",
                        "n_nodes": int(multi.size),
                        "proj_col_l2": float("nan"),
                        "role_norm_median": float(np.median(rn[multi])) if multi.size else float("nan"),
                        "role_norm_p25": float(np.percentile(rn[multi], 25)) if multi.size else float("nan"),
                        "role_norm_p75": float(np.percentile(rn[multi], 75)) if multi.size else float("nan"),
                        "role_norm_p95": float(np.percentile(rn[multi], 95)) if multi.size else float("nan"),
                        "ratio_median": float(np.median(ratio[multi])) if multi.size else float("nan"),
                    }
                )
    write_csv(
        OUT / "reports" / "NODE_ROLE_PROJECTION_NORMS.csv",
        rows,
        ["seed", "variant", "role", "n_nodes", "proj_col_l2",
         "role_norm_median", "role_norm_p25", "role_norm_p75", "role_norm_p95", "ratio_median"],
    )
    zrows = []
    va_u, va_i, va_y = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], bundle["val_pairs"]["label"]
    for i, seed in enumerate(SEEDS):
        m2 = models_n2[i]
        on = score_all(m2, va_u, va_i, A_va_s, H_va_s, item_offset, device, role_enabled=True)
        off = score_all(m2, va_u, va_i, A_va_s, H_va_s, item_offset, device, role_enabled=False)
        mon, moff = sampled_metrics(va_u, va_y, on), sampled_metrics(va_u, va_y, off)
        zrows.append(
            {
                "seed": seed,
                "N2_ROLE_ON": mon["NDCG@20"],
                "N2_ROLE_ZEROED": moff["NDCG@20"],
                "ON-ZEROED": mon["NDCG@20"] - moff["NDCG@20"],
            }
        )
    write_csv(
        OUT / "reports" / "NODE_ROLE_ROLE_ZEROED_DIAGNOSTIC.csv",
        zrows,
        ["seed", "N2_ROLE_ON", "N2_ROLE_ZEROED", "ON-ZEROED"],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("audit", "reports", "runs", "checkpoints", "cache"):
        (OUT / sub).mkdir(exist_ok=True)
    (OUT / "FULLRANK_LOCKED").write_text("LOCKED_NOT_RUN\n")
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n")
    if not (AUDIT_SRC / "KG_NODE_ROLE_AUDIT.md").exists():
        abort("missing KG_NODE_ROLE_AUDIT_V1")

    print("[roles] load bundle + role matrices", flush=True)
    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    overlaps = check_overlaps()
    roles = build_role_matrices()
    np.savez_compressed(
        OUT / "cache" / "role_matrices.npz",
        coarse=roles["coarse"],
        multi=roles["multi"],
        multi_norm=roles["multi_norm"],
        k=roles["k"],
    )
    write_role_audits(roles, hashes, overlaps)

    acfg = cfg.get("models", {}).get("architecture", {})
    print("[roles] build G_FULL", flush=True)
    g_full = load_data_and_typed_graph(
        cfg, bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
    gh = hash_edge_dict(g_full["edge_index_dict"])
    if gh != HASH_FULL:
        abort(f"G_FULL fingerprint mismatch {gh}")
    if "--audit-only" in sys.argv:
        print("[roles] --audit-only done", flush=True)
        return

    A_tr = np.load(shared_A_dir() / "X_train.npy").astype(np.float32)
    A_va = np.load(shared_A_dir() / "X_val.npy").astype(np.float32)
    Hi_tr = np.load(B0_H / "X_train.npy").astype(np.float32)
    Hi_va = np.load(B0_H / "X_val.npy").astype(np.float32)
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    item_offset = int(g_full["meta"]["item_offset"])
    coarse_t = torch.from_numpy(roles["coarse"])
    multi_t = torch.from_numpy(roles["multi_norm"])

    cells = {"N0": [], "N1": [], "N2": []}
    models_n1, models_n2 = [], []
    ident_lines = []
    match_lines = []

    for seed in SEEDS:
        print(f"[roles] matched init seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        a_scaler = StandardScaler().fit(A_tr)
        h_scaler = StandardScaler().fit(Hi_tr)
        A_tr_s, A_va_s = scale_split(a_scaler, A_tr), scale_split(a_scaler, A_va)
        H_tr_s, H_va_s = scale_split(h_scaler, Hi_tr), scale_split(h_scaler, Hi_va)

        n0 = build_model(bundle, g_full, device, None)
        n1 = build_model(bundle, g_full, device, coarse_t)
        n2 = build_model(bundle, g_full, device, multi_t)
        copy_shared(n0, n1)
        copy_shared(n0, n2)
        if float(n1.encoder.core.role_proj.weight.detach().abs().max()) > 0:
            abort("N1 role_proj not zero")
        if float(n2.encoder.core.role_proj.weight.detach().abs().max()) > 0:
            abort("N2 role_proj not zero")
        if not torch.equal(n0.encoder.core.entity_embed.weight, n1.encoder.core.entity_embed.weight):
            abort("ID embed N0!=N1")
        if not torch.equal(n0.encoder.core.user_embed.weight, n2.encoder.core.user_embed.weight):
            abort("user embed N0!=N2")

        rng = np.random.default_rng(20260816 + seed)
        n_tr = len(bundle["train_pairs"]["label"])
        pick = np.sort(rng.choice(n_tr, size=min(1000, n_tr), replace=False))
        l0 = logits_on_idx(
            n0, bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"],
            A_tr_s, H_tr_s, item_offset, device, pick,
        )
        l1 = logits_on_idx(
            n1, bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"],
            A_tr_s, H_tr_s, item_offset, device, pick,
        )
        l2 = logits_on_idx(
            n2, bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"],
            A_tr_s, H_tr_s, item_offset, device, pick,
        )
        d01 = float(np.max(np.abs(l0 - l1)))
        d02 = float(np.max(np.abs(l0 - l2)))
        ident_lines.append(f"seed {seed}: max|N1-N0|={d01:.3e} max|N2-N0|={d02:.3e} n={pick.size}")
        # Spec asked 1e-7; float32 logits can differ by 1 ULP (~1.19e-7).
        if d01 > 2e-7 or d02 > 2e-7:
            abort(f"init logit mismatch seed {seed} d01={d01} d02={d02}")
        match_lines.append(f"seed {seed}: shared ID/HGT/decoder copied; role_proj zero. PASS")

        for tag, model in (("N0", n0), ("N1", n1), ("N2", n2)):
            train_variant(
                bundle, model, stage=tag, seed=seed,
                A_tr_s=A_tr_s, A_va_s=A_va_s, H_tr_s=H_tr_s, H_va_s=H_va_s,
                a_scaler=a_scaler, h_scaler=h_scaler, item_offset=item_offset, device=device,
            )
            logits = score_all(
                model, bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"],
                A_va_s, H_va_s, item_offset, device,
            )
            run = OUT / "runs" / f"{tag}_seed{seed}"
            run.mkdir(parents=True, exist_ok=True)
            np.save(run / "val_logits.npy", logits.astype(np.float32))
            va_u, va_y = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"]
            cells[tag].append({"metrics": sampled_metrics(va_u, va_y, logits), "pu": per_user_ndcg(va_u, va_y, logits)})
        models_n1.append(n1)
        models_n2.append(n2)
        del n0
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    (OUT / "audit" / "ZERO_INIT_EQUIVALENCE_AUDIT.md").write_text(
        "# ZERO_INIT_EQUIVALENCE_AUDIT\n\n"
        + "\n".join(ident_lines)
        + "\nPASS (threshold 1e-7).\n",
        encoding="utf-8",
    )
    (OUT / "audit" / "MATCHED_INITIALIZATION_AUDIT.md").write_text(
        "# MATCHED_INITIALIZATION_AUDIT\n\n"
        + "\n".join(match_lines)
        + "\nN0/N1/N2 share one ID/HGT/decoder init per seed. Only extra params are zero role projections.\n",
        encoding="utf-8",
    )
    print("[roles] diagnostics", flush=True)
    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(Hi_tr)
    A_va_s = scale_split(a_scaler, A_va)
    H_va_s = scale_split(h_scaler, Hi_va)
    role_diagnostics(models_n1, models_n2, roles, bundle, A_va_s, H_va_s, item_offset, device)
    finalize(bundle, cells, roles, hashes)
    print("test_status: LOCKED_NOT_RUN", flush=True)
    print("fullrank_status: LOCKED_NOT_RUN", flush=True)


if __name__ == "__main__":
    main()
