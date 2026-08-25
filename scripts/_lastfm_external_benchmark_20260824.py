#!/usr/bin/env python3
"""Helpers for the sealed external Last-FM* benchmark preparation.

This stage prepares TRAIN_EXTERNAL = upstream corrected train and keeps the
official corrected test split sealed. No function in this module evaluates the
sealed test.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_repos" / "IntentAwareRS"))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG  # noqa: E402
from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_artist_a11_residual_branch_v1 import scale_split  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    build_race_model,
    host_info,
    recipe_kwargs,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    git_commit,
    write_json,
)
from scripts.run_lastfm_final_clean_training_v1 import (  # noqa: E402
    a11_audit,
    architecture_audit,
)
from scripts.run_lastfm_true_final_joint_training_v1 import (  # noqa: E402
    seed_everything,
    train_epoch_true_joint,
    true_final_kw,
)
from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix  # noqa: E402
from src.lastfm_lp.binary.contingency_tables import build_cooccurrence  # noqa: E402
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices  # noqa: E402
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex  # noqa: E402
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_TOP_K  # noqa: E402
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import (  # noqa: E402
    CLEAN_V2_TABULAR_FEATURE_NAMES,
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import (  # noqa: E402
    load_data_and_typed_graph,
    user_item_to_nodes,
)
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.data.candidate_sampling import build_pair_table  # noqa: E402
from src.lastfm_lp.data.load_kgat_lastfm import item_popularity, load_lastfm  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout  # noqa: E402
from topn_baselines_neurals.Recommenders.EASE_R.EASE_R_Recommender import (  # noqa: E402
    EASE_R_Recommender,
)
from topn_baselines_neurals.Recommenders.GraphBased.P3alphaRecommender import (  # noqa: E402
    P3alphaRecommender,
)
from topn_baselines_neurals.Recommenders.GraphBased.RP3betaRecommender import (  # noqa: E402
    RP3betaRecommender,
)
from topn_baselines_neurals.Recommenders.KNN.ItemKNNCFRecommender import (  # noqa: E402
    ItemKNNCFRecommender,
)
from topn_baselines_neurals.Recommenders.KNN.UserKNNCFRecommender import (  # noqa: E402
    UserKNNCFRecommender,
)
from topn_baselines_neurals.Recommenders.NonPersonalizedRecommender import TopPop  # noqa: E402


SEEDS = (101, 202, 303)
FROZEN_EPOCHS = {101: 36, 202: 34, 303: 43}
OUT = ROOT / "LASTFM_EXTERNAL_BENCHMARK_20260824"
SPLITS_DIR = OUT / "splits"
FEATURES_DIR = OUT / "features"
MODELS_DIR = OUT / "models"
BASELINES_DIR = OUT / "baselines"
AUDIT_DIR = OUT / "audit"
REPORTS_DIR = ROOT / "reports"
EXT_COMMIT = "63ece4444659b9505c36058be219c2db951ea087"
REFERENCE_ITEMKNN = {"topK": 144, "similarity": "cosine", "shrink": 1000, "normalize": True}
REFERENCE_USERKNN = {"topK": 144, "similarity": "cosine", "shrink": 1000, "normalize": True}
REFERENCE_P3ALPHA = {"topK": 496, "alpha": 0.7681732734954694, "normalize_similarity": False, "implicit": False, "min_rating": 0}
REFERENCE_RP3BETA = {"topK": 350, "alpha": 0.7681732734954694, "beta": 0.4181395996963926, "normalize_similarity": True, "implicit": False, "min_rating": 0}
REFERENCE_EASER = {"topK": None, "l2_norm": 1e3, "normalize_matrix": False}

# IntentAwareRS still contains older numpy aliases in some recommenders.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    for p in (OUT, SPLITS_DIR, FEATURES_DIR, MODELS_DIR, BASELINES_DIR, AUDIT_DIR, REPORTS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_jsonable(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def dict_union(a: dict[int, set[int]], b: dict[int, set[int]]) -> dict[int, set[int]]:
    keys = sorted(set(a) | set(b))
    return {u: set(a.get(u, set())) | set(b.get(u, set())) for u in keys}


def pair_set(d: dict[int, set[int]]) -> set[tuple[int, int]]:
    return {(int(u), int(i)) for u, items in d.items() for i in items}


def save_user_sets(path: Path, d: dict[int, set[int]]) -> None:
    lines = []
    for u in sorted(d):
        items = sorted(int(i) for i in d[u])
        line = " ".join([str(int(u)), *[str(i) for i in items]])
        lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def empty_pair_table() -> dict[str, np.ndarray]:
    return {
        "user_id": np.zeros(0, dtype=np.int32),
        "item_id": np.zeros(0, dtype=np.int32),
        "label": np.zeros(0, dtype=np.int8),
    }


@dataclass
class ExternalBundle:
    cfg: dict[str, Any]
    data_path: Path
    train_external: dict[int, set[int]]
    sealed_test: dict[int, set[int]]
    popularity: np.ndarray
    item_kg_degree: np.ndarray
    train_pairs: dict[str, np.ndarray]
    eval_users_test: list[int]
    training_hash: str
    test_hash: str
    n_users: int
    n_items: int


def build_external_bundle() -> ExternalBundle:
    ensure_dirs()
    cfg = load_protocol_config(PROTOCOL_CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["outputs_root"] = str(OUT)
    cfg["paths"]["splits"] = str(SPLITS_DIR)
    cfg["paths"]["features"] = str(FEATURES_DIR)
    cfg["paths"]["audit"] = str(AUDIT_DIR)
    cfg["paths"]["manifest"] = str(AUDIT_DIR / "external_protocol_manifest.json")
    data = load_lastfm(cfg["data"]["path"])
    dev_model_train = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "model_train.txt")
    dev_valid = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "valid.txt")
    dev_test = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "test.txt")
    train_external = dict_union(dev_model_train, dev_valid)
    sealed_test = {u: set(items) for u, items in data.test_by_user.items()}

    training_hash = hash_jsonable({str(u): sorted(v) for u, v in train_external.items()})
    test_hash = hash_jsonable({str(u): sorted(v) for u, v in sealed_test.items()})
    popularity = item_popularity(train_external, data.n_items)
    eval_users_test = sorted(u for u, items in sealed_test.items() if items)

    return ExternalBundle(
        cfg=cfg,
        data_path=Path(cfg["data"]["path"]),
        train_external=train_external,
        sealed_test=sealed_test,
        popularity=popularity,
        item_kg_degree=data.item_kg_degree,
        train_pairs={},
        eval_users_test=eval_users_test,
        training_hash=training_hash,
        test_hash=test_hash,
        n_users=data.n_users,
        n_items=data.n_items,
    )


def write_manifest_inputs(bundle: ExternalBundle) -> dict[str, Any]:
    ensure_dirs()
    upstream_train = load_lastfm(bundle.data_path).train_by_user
    upstream_test = load_lastfm(bundle.data_path).test_by_user
    dev_model_train = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "model_train.txt")
    dev_valid = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "valid.txt")
    dev_test = load_user_sets(ROOT / "outputs" / "lastfm_star" / "splits" / "test.txt")
    save_user_sets(SPLITS_DIR / "train_external.txt", bundle.train_external)
    save_user_sets(SPLITS_DIR / "sealed_test.txt", bundle.sealed_test)
    (SPLITS_DIR / "eval_users_test.json").write_text(json.dumps({"eval_users": bundle.eval_users_test}, indent=2) + "\n", encoding="utf-8")

    checks = {
        "upstream_train_equals_model_train_plus_validation": pair_set(upstream_train) == (pair_set(dev_model_train) | pair_set(dev_valid)),
        "upstream_test_equals_sealed_test": pair_set(upstream_test) == pair_set(bundle.sealed_test),
        "model_train_cap_validation": len(pair_set(dev_model_train) & pair_set(dev_valid)),
        "model_train_cap_test": len(pair_set(dev_model_train) & pair_set(dev_test)),
        "validation_cap_test": len(pair_set(dev_valid) & pair_set(dev_test)),
    }
    counts = {
        "model_train": {"users": len(dev_model_train), "items": len({i for xs in dev_model_train.values() for i in xs}), "interactions": sum(len(v) for v in dev_model_train.values())},
        "validation": {"users": len(dev_valid), "items": len({i for xs in dev_valid.values() for i in xs}), "interactions": sum(len(v) for v in dev_valid.values())},
        "upstream_corrected_train": {"users": len(upstream_train), "items": len({i for xs in upstream_train.values() for i in xs}), "interactions": sum(len(v) for v in upstream_train.values())},
        "sealed_test": {"users": len(bundle.sealed_test), "items": len({i for xs in bundle.sealed_test.values() for i in xs}), "interactions": sum(len(v) for v in bundle.sealed_test.values())},
    }
    payload = {
        "generated": utc_now(),
        "training_hash": bundle.training_hash,
        "test_hash": bundle.test_hash,
        "counts": counts,
        "checks": checks,
        "source_data_path": str(bundle.data_path),
        "intentawarers_commit": EXT_COMMIT,
        "upstream_train_sha256": sha256_file(bundle.data_path / "train.txt"),
        "upstream_test_sha256": sha256_file(bundle.data_path / "test.txt"),
        "dev_model_train_sha256": sha256_file(ROOT / "outputs" / "lastfm_star" / "splits" / "model_train.txt"),
        "dev_valid_sha256": sha256_file(ROOT / "outputs" / "lastfm_star" / "splits" / "valid.txt"),
        "dev_test_sha256": sha256_file(ROOT / "outputs" / "lastfm_star" / "splits" / "test.txt"),
    }
    write_json(AUDIT_DIR / "external_train_manifest.json", payload)
    return payload


def build_train_pairs(bundle: ExternalBundle) -> dict[str, np.ndarray]:
    if bundle.train_pairs:
        return bundle.train_pairs
    train_pairs = build_pair_table(
        users=sorted(bundle.train_external),
        positives_by_user=bundle.train_external,
        history_by_user=bundle.train_external,
        n_items=bundle.n_items,
        n_neg_per_pos=4,
        max_positives_per_user=30,
        popularity=bundle.popularity,
        mode="random",
        seed=2026,
        split_name="external_fit_train",
    )
    bundle.train_pairs = train_pairs
    np.savez_compressed(FEATURES_DIR / "train_pairs.npz", **train_pairs)
    np.save(FEATURES_DIR / "popularity.npy", bundle.popularity)
    np.save(FEATURES_DIR / "item_kg_degree.npy", bundle.item_kg_degree)
    return train_pairs


def build_external_training_bundle(bundle: ExternalBundle) -> dict[str, Any]:
    train_pairs = build_train_pairs(bundle)
    cooc = build_cooccurrence(bundle.train_external, bundle.n_items, max_history_for_pairs=60, seed=2026)
    index = PairwiseStatsIndex(cooc, bundle.popularity, n_users=len(bundle.train_external), smoothing=float(bundle.cfg.get("hcr", {}).get("smoothing", 0.5)))
    prepared = {
        "cfg": bundle.cfg,
        "model_train": bundle.train_external,
        "popularity": bundle.popularity,
        "item_kg_degree": bundle.item_kg_degree,
        "index": index,
        "train_pairs": train_pairs,
        "val_pairs": empty_pair_table(),
        "test_pairs": empty_pair_table(),
        "eval_users": bundle.eval_users_test,
    }
    return prepared


def frozen_hgt_training_hist() -> dict[str, Any]:
    """Frozen C1 optimizer recipe from TRUE FINAL development (no retuning).

    Recovered from LASTFM_TRUE_FINAL/JOINT_TRAINING_V1 audits / TRUE_FINAL_CONFIG.yaml
    because the exploratory capacity-race cache is no longer on disk.
    """
    return {
        "lr0": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs_current": 15,
        "patience_current": 4,
        "dropout_hgt": 0.1,
        "batch_size": 4096,
        "optimizer": "Adam",
        "decoder": "LateFusionHead 265 → 128 LayerNorm GELU Dropout(0.2) → 64 GELU Dropout → 1",
        "leg_k2": "Linear(1,16) GELU Linear(16,1); last layer zero-init; trains jointly",
        "gradient_clipping": "NONE",
        "amp_precision": "fp32 (no AMP)",
        "loss": "BCEWithLogitsLoss pos_weight=n_neg/n_pos",
        "scheduler_current": "NONE",
        "source": "LASTFM_TRUE_FINAL/JOINT_TRAINING_V1/00_AUDIT/FINAL_TRAINING_CONFIG_AUDIT.json",
    }


def materialize_a5_h3_leg_train(bundle: dict[str, Any]) -> dict[str, Path]:
    out_dir = FEATURES_DIR / "materialized"
    a_dir = out_dir / "A_true"
    h_dir = out_dir / "H3_a11_top25"
    leg_dir = out_dir / "LEG_K2"
    for p in (a_dir, h_dir, leg_dir):
        p.mkdir(parents=True, exist_ok=True)

    train_pairs = bundle["train_pairs"]
    users = train_pairs["user_id"]
    items = train_pairs["item_id"]
    labels = train_pairs["label"]
    n = len(labels)
    if (
        (a_dir / "X_train.npy").exists()
        and (h_dir / "X_train.npy").exists()
        and (leg_dir / "LEG_K2_train.npy").exists()
        and np.load(a_dir / "X_train.npy", mmap_mode="r").shape[0] == n
        and np.load(h_dir / "X_train.npy", mmap_mode="r").shape[0] == n
        and np.load(leg_dir / "LEG_K2_train.npy", mmap_mode="r").shape[0] == n
    ):
        print("[external] reuse TRAIN_EXTERNAL A5/H3/LEG caches", flush=True)
        return {"a_dir": a_dir, "h_dir": h_dir, "leg_dir": leg_dir}

    cf = ensure_cross_fit(bundle)
    model_train = bundle["model_train"]
    pop = bundle["popularity"]
    kg = bundle["item_kg_degree"]

    A = np.lib.format.open_memmap(a_dir / "X_train.npy", mode="w+", dtype=np.float32, shape=(n, 5))
    H = np.lib.format.open_memmap(h_dir / "X_train.npy", mode="w+", dtype=np.float32, shape=(n, 3))
    L = np.lib.format.open_memmap(leg_dir / "LEG_K2_train.npy", mode="w+", dtype=np.float32, shape=(n, 1))
    size_cache: dict[int, np.ndarray] = {}
    chunk_size = 50_000
    for chunk_start in tqdm(range(0, n, chunk_size), desc="external A5/H3/LEG"):
        chunk_end = min(chunk_start + chunk_size, n)
        user_groups: dict[int, list[int]] = defaultdict(list)
        for r in range(chunk_start, chunk_end):
            user_groups[int(users[r])].append(r)
        for u, rows in user_groups.items():
            idx_cf = clean_v2_index_for(cf, int(u))
            fold = cf.user_to_fold.get(int(u), -1)
            if fold not in size_cache:
                size_cache[fold] = neighborhood_sizes(idx_cf)
            r_idx = np.asarray(rows, dtype=np.int64)
            A[r_idx] = vectorized_clean_v2_A_for_user(
                int(u),
                items[r_idx],
                model_train,
                pop,
                kg,
                idx_cf,
                n_x_sizes=size_cache[fold],
            )
            base = set(model_train.get(int(u), ()))
            for r in rows:
                i = int(items[r])
                y = int(labels[r])
                hist = set(base)
                if y == 1 and i in hist:
                    hist.remove(i)
                hist_arr = np.asarray(sorted(hist), dtype=np.int64)
                h3, l2 = h3_and_l2(hist_arr, np.asarray([i], dtype=np.int64), idx_cf)
                H[r] = h3[0]
                L[r] = l2[0]
        A.flush()
        H.flush()
        L.flush()
    with (leg_dir / "LEG_K2_scaler.pkl").open("wb") as f:
        pickle.dump(StandardScaler().fit(np.asarray(L)), f)
    write_json(a_dir / "meta.json", {"feature_names": CLEAN_V2_TABULAR_FEATURE_NAMES, "fit_population": "TRAIN_EXTERNAL_ONLY", "generated": utc_now()})
    write_json(h_dir / "meta.json", {"feature_names": ["mean:hcr_a11", "max:hcr_a11", "top3mean:hcr_a11"], "fit_population": "TRAIN_EXTERNAL_ONLY", "generated": utc_now()})
    write_json(leg_dir / "meta.json", {"formula": "L2 = mean P2(A11) over signed Top25", "fit_population": "TRAIN_EXTERNAL_ONLY", "generated": utc_now()})
    return {"a_dir": a_dir, "h_dir": h_dir, "leg_dir": leg_dir}


def h3_and_l2(hist: np.ndarray, cands: np.ndarray, index, *, max_history: int = CLEAN_V2_TOP_K) -> tuple[np.ndarray, np.ndarray]:
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    h3 = np.zeros((C, 3), dtype=np.float32)
    l2 = np.zeros((C, 1), dtype=np.float32)
    if hist.size == 0 or C == 0:
        return h3, l2
    n11 = index.cooccurrence_block(hist, cands)
    a11, _ = a11_energy_from_n11_matrix(n11, index.popularity[hist], index.popularity[cands], index.n_users)
    if hist.size <= max_history:
        a_sel = a11
    else:
        idx = deterministic_topk_indices(a11, hist, k=max_history)
        a_sel = np.take_along_axis(a11, idx, axis=0)
    h3 = pool_signed_a11_3d_matrix(a_sel)
    p2 = 0.5 * (3.0 * np.square(a_sel.astype(np.float64)) - 1.0)
    l2[:, 0] = p2.mean(axis=0).astype(np.float32)
    return h3, l2


def fit_true_final_refit(bundle: ExternalBundle) -> list[dict[str, Any]]:
    ensure_dirs()
    prepared = build_external_training_bundle(bundle)
    feat_dirs = materialize_a5_h3_leg_train(prepared)
    A_tr = np.load(feat_dirs["a_dir"] / "X_train.npy").astype(np.float32)
    H_tr = np.load(feat_dirs["h_dir"] / "X_train.npy").astype(np.float32)
    r_tr = np.load(feat_dirs["leg_dir"] / "LEG_K2_train.npy").astype(np.float32)
    if r_tr.ndim == 1:
        r_tr = r_tr[:, None]
    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(H_tr)
    A_tr_s = scale_split(a_scaler, A_tr)
    H_tr_s = scale_split(h_scaler, H_tr)
    with (AUDIT_DIR / "external_a_scaler.pkl").open("wb") as f:
        pickle.dump(a_scaler, f)
    with (AUDIT_DIR / "external_h_scaler.pkl").open("wb") as f:
        pickle.dump(h_scaler, f)
    with (AUDIT_DIR / "external_leg_scaler.pkl").open("wb") as f:
        pickle.dump(StandardScaler().fit(r_tr), f)

    graph = load_data_and_typed_graph(bundle.cfg, bundle.train_external, max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000))
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(str(bundle.cfg.get("models", {}).get("architecture", {}).get("device") or "auto"))
    hist = frozen_hgt_training_hist()
    write_json(AUDIT_DIR / "frozen_hgt_training_hist.json", hist)
    kw = true_final_kw(hist)
    arch_probe = build_race_model(prepared, graph, device, d=64, layers=2, heads=2)
    arch = architecture_audit(arch_probe, device)
    a11 = a11_audit(r_tr, r_tr[: min(len(r_tr), 1024)])
    del arch_probe
    empty_cache()
    if arch["ARCHITECTURE_STATUS"] != "FROZEN_REPRODUCED" or a11["A11_AUDIT_STATUS"] != "PASS":
        raise RuntimeError("Pre-fit architecture/A11 audit failed")

    u_all, i_all = user_item_to_nodes(prepared["train_pairs"]["user_id"], prepared["train_pairs"]["item_id"], item_offset)
    y_all = prepared["train_pairs"]["label"].astype(np.float32)
    n_train = len(y_all)
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = MODELS_DIR / f"TRUE_FINAL_REFIT_SEED{seed}"
        ckpt_path = seed_dir / "model.pt"
        meta_path = seed_dir / "meta.json"
        if ckpt_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": "TRUE_FINAL_REFIT",
                    "seed": seed,
                    "epoch_count": FROZEN_EPOCHS[seed],
                    "artifact_path": str(ckpt_path.relative_to(ROOT)),
                    "sha256": sha256_file(ckpt_path),
                    "training_hash": bundle.training_hash,
                    "timestamp": meta.get("timestamp", utc_now()),
                    "git_commit": meta.get("git_commit", git_commit()),
                    "hyperparameters": {
                        "lr0": kw["lr0"],
                        "weight_decay": float(hist["weight_decay"]),
                        "batch_size": int(bundle.cfg.get("models", {}).get("architecture", {}).get("batch_size", 4096)),
                    },
                    "reused": True,
                }
            )
            print(f"[external-refit] reuse seed={seed}", flush=True)
            continue
        seed_everything(seed)
        model = build_race_model(prepared, graph, device, d=64, layers=2, heads=2)
        opt = torch.optim.Adam(model.parameters(), lr=kw["lr0"], weight_decay=float(hist["weight_decay"]))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device))
        batch_size = int(bundle.cfg.get("models", {}).get("architecture", {}).get("batch_size", 4096))
        history: list[dict[str, Any]] = []
        t0 = time.time()
        for epoch in range(FROZEN_EPOCHS[seed]):
            model.train()
            perm = np.random.permutation(n_train)
            info = train_epoch_true_joint(
                model=model,
                opt=opt,
                loss_fn=loss_fn,
                u_all=u_all,
                i_all=i_all,
                y_all=y_all,
                A_tr_s=A_tr_s,
                H_tr_s=H_tr_s,
                r_tr=r_tr,
                device=device,
                perm=perm,
                batch_size=batch_size,
                collect_autograd=(epoch == 0),
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(info["epoch_mean_loss"]),
                    "hgt_forward": int(info["hgt_forward"]),
                    "hgt_backward": int(info["hgt_backward"]),
                    "n_opt_step": int(info["n_opt_step"]),
                    "sec": float(info.get("sec", 0.0)),
                }
            )
            print(f"[external-refit seed={seed}] epoch {epoch+1}/{FROZEN_EPOCHS[seed]} loss={history[-1]['loss']:.4f}", flush=True)
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        meta = {
            "seed": seed,
            "epoch_count": FROZEN_EPOCHS[seed],
            "architecture": "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL",
            "training_hash": bundle.training_hash,
            "timestamp": utc_now(),
            "git_commit": git_commit(),
            "fit_seconds": time.time() - t0,
            "history": history,
            "feature_dirs": {k: str(v) for k, v in feat_dirs.items()},
        }
        write_json(seed_dir / "meta.json", meta)
        rows.append(
            {
                "model": "TRUE_FINAL_REFIT",
                "seed": seed,
                "epoch_count": FROZEN_EPOCHS[seed],
                "artifact_path": str(ckpt_path.relative_to(ROOT)),
                "sha256": sha256_file(ckpt_path),
                "training_hash": bundle.training_hash,
                "timestamp": meta["timestamp"],
                "git_commit": meta["git_commit"],
                "hyperparameters": {
                    "lr0": kw["lr0"],
                    "weight_decay": float(hist["weight_decay"]),
                    "batch_size": batch_size,
                },
            }
        )
        del model
        empty_cache()
    write_json(AUDIT_DIR / "true_final_refit_artifacts.json", rows)
    return rows


def build_urm(train_external: dict[int, set[int]], n_users: int, n_items: int) -> sps.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, items in train_external.items():
        for i in items:
            rows.append(int(u))
            cols.append(int(i))
            data.append(1.0)
    return sps.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)


def fit_baselines(bundle: ExternalBundle, include_userknn: bool = True, include_easer: bool = True) -> list[dict[str, Any]]:
    ensure_dirs()
    urm = build_urm(bundle.train_external, bundle.n_users, bundle.n_items)
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, Any, dict[str, Any], str]] = [
        ("TopPop", TopPop, {}, "local"),
        ("ItemKNN", ItemKNNCFRecommender, REFERENCE_ITEMKNN, EXT_COMMIT),
        ("P3alpha", P3alphaRecommender, REFERENCE_P3ALPHA, EXT_COMMIT),
        ("RP3beta", RP3betaRecommender, REFERENCE_RP3BETA, EXT_COMMIT),
    ]
    if include_userknn:
        specs.append(("UserKNN", UserKNNCFRecommender, REFERENCE_USERKNN, EXT_COMMIT))
    if include_easer:
        specs.append(("EASE-R", EASE_R_Recommender, REFERENCE_EASER, EXT_COMMIT))
    for name, cls, hp, ext_commit in specs:
        model_dir = BASELINES_DIR / name.replace("-", "_")
        meta_path = model_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(meta)
            print(f"[baselines] reuse {name}", flush=True)
            continue
        t0 = time.time()
        rec = cls(urm, verbose=True) if name != "TopPop" else cls(urm)
        rec.fit(**hp)
        model_dir.mkdir(parents=True, exist_ok=True)
        save_name = name.replace("-", "_")
        try:
            rec.save_model(str(model_dir) + "/", file_name=save_name)
            artifact = next(model_dir.glob(f"{save_name}*"))
        except Exception:
            artifact = model_dir / f"{save_name}.pkl"
            with artifact.open("wb") as f:
                pickle.dump(rec, f)
        meta = {
            "model": name,
            "hyperparameters": hp,
            "training_hash": bundle.training_hash,
            "timestamp": utc_now(),
            "fit_seconds": time.time() - t0,
            "git_commit": git_commit(),
            "external_commit": ext_commit,
            "artifact_path": str(artifact.relative_to(ROOT)),
            "artifact_sha256": sha256_file(artifact),
        }
        write_json(model_dir / "meta.json", meta)
        rows.append(meta)
    write_json(AUDIT_DIR / "baseline_artifacts.json", rows)
    return rows


def upstream_eval_details() -> dict[str, Any]:
    return {
        "UPSTREAM_EVALUATOR_CLASS": "EvaluatorHoldout",
        "UPSTREAM_EVALUATOR_FILE": "external_repos/IntentAwareRS/topn_baselines_neurals/Evaluation/Evaluator.py",
        "UPSTREAM_CUTOFFS": [1, 5, 10, 20, 40, 50, 100],
        "UPSTREAM_TRAIN_MASKING": "exclude_seen=True passed to recommender.recommend/remove_seen_flag",
        "UPSTREAM_USER_FILTERING": "min_ratings_per_user=1; users with no test items skipped by Evaluator base class",
        "UPSTREAM_NDCG_IMPLEMENTATION": "external_repos/IntentAwareRS/topn_baselines_neurals/Evaluation/metrics.py::ndcg",
        "UPSTREAM_RECALL_IMPLEMENTATION": "external_repos/IntentAwareRS/topn_baselines_neurals/Evaluation/metrics.py::recall",
    }


class ScoreMatrixRecommender:
    RECOMMENDER_NAME = "ScoreMatrixAdapter"

    def __init__(self, URM_train: sps.csr_matrix, score_matrix: np.ndarray):
        self.URM_train = URM_train.tocsr()
        self.score_matrix = np.asarray(score_matrix, dtype=np.float64)
        self.items_to_ignore_ID = np.array([], dtype=np.int64)

    def get_URM_train(self):
        return self.URM_train

    def set_items_to_ignore(self, items):
        self.items_to_ignore_ID = np.asarray(items, dtype=np.int64)

    def reset_items_to_ignore(self):
        self.items_to_ignore_ID = np.array([], dtype=np.int64)

    def _remove_seen(self, user_id: int, scores: np.ndarray) -> np.ndarray:
        start, end = self.URM_train.indptr[user_id], self.URM_train.indptr[user_id + 1]
        seen = self.URM_train.indices[start:end]
        if seen.size:
            scores[seen] = -np.inf
        return scores

    def recommend(self, user_id_array, remove_seen_flag=True, cutoff=20, remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=True, items_to_compute=None):
        users = np.asarray(user_id_array, dtype=np.int64)
        score_batch = self.score_matrix[users].copy()
        if items_to_compute is not None:
            mask = np.full(score_batch.shape[1], True, dtype=bool)
            mask[np.asarray(items_to_compute, dtype=np.int64)] = False
            score_batch[:, mask] = -np.inf
        if remove_custom_items_flag and self.items_to_ignore_ID.size:
            score_batch[:, self.items_to_ignore_ID] = -np.inf
        ranked = []
        for row, u in zip(score_batch, users.tolist()):
            if remove_seen_flag:
                row = self._remove_seen(int(u), row)
            order = np.lexsort((np.arange(row.size, dtype=np.int64), -row))
            ranked.append(order[:cutoff])
        return ranked, score_batch


def verify_upstream_evaluator_reproduction(bundle: ExternalBundle) -> dict[str, Any]:
    """Non-sealed numerical check: EvaluatorHoldout == metrics.ndcg/recall on identical lists."""
    from topn_baselines_neurals.Evaluation.metrics import ndcg as upstream_ndcg
    from topn_baselines_neurals.Evaluation.metrics import recall as upstream_recall

    # Synthetic holdout only — never touch sealed TEST_EXTERNAL.
    _ = bundle.training_hash
    n_users, n_items = 8, 30
    train_rows = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7]
    train_cols = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8]
    test_rows = [0, 1, 2, 3, 4, 5, 6, 7]
    test_cols = [10, 11, 12, 13, 14, 15, 16, 17]
    urm_train = sps.csr_matrix((np.ones(len(train_rows)), (train_rows, train_cols)), shape=(n_users, n_items), dtype=np.float32)
    urm_valid = sps.csr_matrix((np.ones(len(test_rows)), (test_rows, test_cols)), shape=(n_users, n_items), dtype=np.float32)
    rng = np.random.default_rng(20260824)
    score_matrix = rng.normal(size=(n_users, n_items)).astype(np.float64)
    for u, i in zip(test_rows, test_cols):
        score_matrix[u, i] += 5.0
    adapter = ScoreMatrixRecommender(urm_train, score_matrix)
    cutoff = 20
    evaluator = EvaluatorHoldout(urm_valid, [cutoff], exclude_seen=True, verbose=False)
    df_holdout, _ = evaluator.evaluateRecommender(adapter)
    holdout_ndcg20 = float(df_holdout.loc[cutoff, "NDCG"])
    holdout_recall20 = float(df_holdout.loc[cutoff, "RECALL"])

    ranked, _ = adapter.recommend(np.arange(n_users), remove_seen_flag=True, cutoff=cutoff, return_scores=True)
    manual_ndcgs: list[float] = []
    manual_recalls: list[float] = []
    for u, rec in enumerate(ranked):
        start, end = urm_valid.indptr[u], urm_valid.indptr[u + 1]
        pos_items = urm_valid.indices[start:end]
        if pos_items.size == 0:
            continue
        is_relevant = np.in1d(rec, pos_items, assume_unique=True)
        manual_recalls.append(float(upstream_recall(is_relevant, pos_items)))
        manual_ndcgs.append(float(upstream_ndcg(rec, pos_items, relevance=np.ones_like(pos_items), at=cutoff)))
    manual_ndcg20 = float(np.mean(manual_ndcgs)) if manual_ndcgs else 0.0
    manual_recall20 = float(np.mean(manual_recalls)) if manual_recalls else 0.0
    abs_diff_ndcg20 = abs(holdout_ndcg20 - manual_ndcg20)
    abs_diff_recall20 = abs(holdout_recall20 - manual_recall20)
    passed = abs_diff_ndcg20 <= 1e-10 and abs_diff_recall20 <= 1e-10
    payload = {
        **upstream_eval_details(),
        "method": "EvaluatorHoldout vs metrics.ndcg/recall on identical ranked lists",
        "holdout_ndcg20": holdout_ndcg20,
        "manual_ndcg20": manual_ndcg20,
        "holdout_recall20": holdout_recall20,
        "manual_recall20": manual_recall20,
        "abs_diff_ndcg20": abs_diff_ndcg20,
        "abs_diff_recall20": abs_diff_recall20,
        "SHEHZAD_EVALUATOR_REPRODUCTION": "PASS" if passed else "FAIL",
        "timestamp": utc_now(),
        "sealed_test_accessed": False,
    }
    write_json(AUDIT_DIR / "upstream_evaluator_reproduction.json", payload)
    return payload


def run_leakage_and_freeze_audit() -> dict[str, Any]:
    """Gate FINAL_MODELS_FROZEN / READY_TO_UNSEAL without opening the sealed test."""
    ensure_dirs()
    manifest = json.loads((AUDIT_DIR / "external_train_manifest.json").read_text(encoding="utf-8"))
    baselines = json.loads((AUDIT_DIR / "baseline_artifacts.json").read_text(encoding="utf-8")) if (AUDIT_DIR / "baseline_artifacts.json").exists() else []
    refit = json.loads((AUDIT_DIR / "true_final_refit_artifacts.json").read_text(encoding="utf-8")) if (AUDIT_DIR / "true_final_refit_artifacts.json").exists() else []
    repro = json.loads((AUDIT_DIR / "upstream_evaluator_reproduction.json").read_text(encoding="utf-8")) if (AUDIT_DIR / "upstream_evaluator_reproduction.json").exists() else {}

    required_baselines = {"TopPop", "ItemKNN", "P3alpha", "RP3beta"}
    fitted_baselines = {str(r.get("model")) for r in baselines}
    baseline_ok = required_baselines.issubset(fitted_baselines)
    seeds_ok = {int(r["seed"]) for r in refit} == set(SEEDS) and all(
        int(r["epoch_count"]) == FROZEN_EPOCHS[int(r["seed"])] for r in refit
    )
    ckpts_ok = all((MODELS_DIR / f"TRUE_FINAL_REFIT_SEED{s}" / "model.pt").exists() for s in SEEDS)
    hashes_ok = all(str(r.get("sha256") or r.get("artifact_sha256") or "") for r in [*baselines, *refit])
    train_hash_ok = all(r.get("training_hash") == manifest["training_hash"] for r in [*baselines, *refit])
    overlap_ok = (
        int(manifest["checks"]["model_train_cap_validation"]) == 0
        and int(manifest["checks"]["model_train_cap_test"]) == 0
        and int(manifest["checks"]["validation_cap_test"]) == 0
        and bool(manifest["checks"]["upstream_train_equals_model_train_plus_validation"])
    )
    feat_ok = all(
        (FEATURES_DIR / "materialized" / name / "meta.json").exists()
        for name in ("A_true", "H3_a11_top25", "LEG_K2")
    )
    sealed_scoring_absent = not (AUDIT_DIR / "sealed_test_results.json").exists()
    repro_ok = repro.get("SHEHZAD_EVALUATOR_REPRODUCTION") == "PASS"

    checks = {
        "required_baselines_fitted": baseline_ok,
        "true_final_refit_seeds": seeds_ok and ckpts_ok,
        "artifact_hashes_present": hashes_ok,
        "training_hash_consistent": train_hash_ok,
        "split_overlap_zero": overlap_ok,
        "features_materialized_on_train_external": feat_ok,
        "sealed_test_not_scored": sealed_scoring_absent,
        "evaluator_reproduction": repro_ok,
        "ease_r_status": "NOT_INCLUDED" if "EASE-R" not in fitted_baselines else "INCLUDED",
        "userknn_status": "INCLUDED" if "UserKNN" in fitted_baselines else "NOT_INCLUDED",
    }
    all_pass = all(
        v is True
        for k, v in checks.items()
        if k not in {"ease_r_status", "userknn_status"}
    )
    payload = {
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "intentawarers_commit": EXT_COMMIT,
        "training_hash": manifest["training_hash"],
        "test_hash": manifest["test_hash"],
        "checks": checks,
        "FINAL_MODELS_FROZEN": "YES" if all_pass else "NO",
        "READY_TO_UNSEAL": "YES" if all_pass else "NO",
        "sealed_test_accessed": False,
        "baselines": baselines,
        "true_final_refit": refit,
        "upstream_evaluator": {**upstream_eval_details(), "reproduction": repro.get("SHEHZAD_EVALUATOR_REPRODUCTION")},
    }
    write_json(AUDIT_DIR / "final_models_freeze_gate.json", payload)
    return payload


def write_protocol_manifest(freeze: dict[str, Any]) -> Path:
    path = REPORTS_DIR / "LASTFM_FINAL_EXTERNAL_PROTOCOL_MANIFEST_20260824.md"
    up = upstream_eval_details()
    baselines = freeze.get("baselines") or []
    refit = freeze.get("true_final_refit") or []
    lines = [
        "# Last-FM* final external protocol manifest (2026-08-24)",
        "",
        "Status after fitting stage. **Sealed test not evaluated.**",
        "",
        f"- `FINAL_MODELS_FROZEN` = **{freeze['FINAL_MODELS_FROZEN']}**",
        f"- `READY_TO_UNSEAL` = **{freeze['READY_TO_UNSEAL']}**",
        f"- `sealed_test_accessed` = **NO**",
        f"- our git commit = `{freeze['git_commit']}`",
        f"- IntentAwareRS commit = `{freeze['intentawarers_commit']}`",
        f"- `TRAIN_EXTERNAL` hash = `{freeze['training_hash']}`",
        f"- sealed test hash (identity only) = `{freeze['test_hash']}`",
        "",
        "## Protocol history",
        "",
        "1. Development / design freeze (model_train → validation)",
        "2. Refit all models on TRAIN_EXTERNAL = model_train ∪ validation",
        "3. Freeze + hash artifacts",
        "4. *(not yet)* ONE sealed test evaluation",
        "",
        "## Upstream evaluator (publication comparison = Block A)",
        "",
        f"- `UPSTREAM_EVALUATOR_CLASS` = `{up['UPSTREAM_EVALUATOR_CLASS']}`",
        f"- `UPSTREAM_EVALUATOR_FILE` = `{up['UPSTREAM_EVALUATOR_FILE']}`",
        f"- `UPSTREAM_CUTOFFS` = `{up['UPSTREAM_CUTOFFS']}`",
        f"- `UPSTREAM_TRAIN_MASKING` = `{up['UPSTREAM_TRAIN_MASKING']}`",
        f"- `UPSTREAM_USER_FILTERING` = `{up['UPSTREAM_USER_FILTERING']}`",
        f"- `UPSTREAM_NDCG_IMPLEMENTATION` = `{up['UPSTREAM_NDCG_IMPLEMENTATION']}`",
        f"- `UPSTREAM_RECALL_IMPLEMENTATION` = `{up['UPSTREAM_RECALL_IMPLEMENTATION']}`",
        f"- reproduction = `{freeze['upstream_evaluator'].get('reproduction')}`",
        "",
        "Primary metrics for literature comparison: **NDCG@20**, **Recall@20** (plus any other metrics the upstream holdout naturally returns at the same cutoffs).",
        "",
        "Do **not** mix Block A (IntentAwareRS) with Block B (our frozen full-catalog evaluator) in one comparison column.",
        "",
        "## Fitted models",
        "",
        "### TRUE FINAL refit (frozen development epochs)",
        "",
    ]
    for r in refit:
        lines.append(
            f"- seed `{r['seed']}`: epochs=`{r['epoch_count']}` sha256=`{r.get('sha256')}` path=`{r.get('artifact_path')}`"
        )
    lines += ["", "### Baselines", ""]
    for r in baselines:
        lines.append(
            f"- `{r['model']}`: sha256=`{r.get('artifact_sha256')}` external_commit=`{r.get('external_commit')}` path=`{r.get('artifact_path')}`"
        )
    if freeze["checks"].get("ease_r_status") == "NOT_INCLUDED":
        lines.append("- `EASE-R`: **NOT INCLUDED** (dense inversion opt-in only via `LASTFM_FIT_EASER=1`)")
    lines += [
        "",
        "## Freeze gate checks",
        "",
    ]
    for k, v in freeze["checks"].items():
        lines.append(f"- `{k}` = `{v}`")
    lines += [
        "",
        "## Sealed test results",
        "",
        "_Empty by design. Do not run `LAST_FM_EXT_04` until explicit unseal approval._",
        "",
        "| Model | NDCG@20 (A) | Recall@20 (A) | notes |",
        "|---|---:|---:|---|",
        "| — | — | — | sealed |",
        "",
        "## Development ablation (reference only; not external test)",
        "",
        "From development protocol (model_train → validation), do not merge with sealed results:",
        "",
        "| Variant | NDCG@20 | Recall@20 | MRR |",
        "|---|---:|---:|---:|",
        "| HGT only | ~0.0093 | ~0.0169 | — |",
        "| +A5 | ~0.0640 | ~0.1128 | — |",
        "| +H3 | ~0.2321 | ~0.3647 | — |",
        "| TRUE FINAL | ~0.2764 | ~0.3768 | ~0.3534 |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
