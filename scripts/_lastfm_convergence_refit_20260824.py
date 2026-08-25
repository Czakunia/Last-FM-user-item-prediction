#!/usr/bin/env python3
"""POST_HOC_CONVERGENCE_CONTROLLED_REFIT helpers for Last-FM* TRUE FINAL.

Creates TRAIN_INNER / VAL_INNER inside TRAIN_EXTERNAL, retrains with inner-val
NDCG@20 model selection, and records full learning curves. Does not touch the
original external benchmark artefacts or sealed-test epoch selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    build_external_bundle,
    frozen_hgt_training_hist,
    h3_and_l2,
    hash_jsonable,
    pair_set,
    save_user_sets,
    sha256_file,
    utc_now,
)
from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG  # noqa: E402
from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_artist_a11_residual_branch_v1 import sampled_metrics, scale_split  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    MIN_IMPROVE,
    build_race_model,
    git_commit,
    host_info,
    score_model,
    write_json,
)
from scripts.run_lastfm_final_clean_training_v1 import a11_audit, architecture_audit  # noqa: E402
from scripts.run_lastfm_true_final_joint_training_v1 import (  # noqa: E402
    seed_everything,
    train_epoch_true_joint,
)
from src.lastfm_lp.binary.contingency_tables import build_cooccurrence  # noqa: E402
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
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes  # noqa: E402
from src.lastfm_lp.data.candidate_sampling import build_pair_table  # noqa: E402
from src.lastfm_lp.data.load_kgat_lastfm import item_popularity  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

EXPERIMENT_TAG = "POST_HOC_CONVERGENCE_CONTROLLED_REFIT"
OUT = ROOT / "LASTFM_EXTERNAL_CONVERGENCE_REFIT_20260824"
SPLITS_DIR = OUT / "splits"
FEATURES_DIR = OUT / "features"
MODELS_DIR = OUT / "models"
AUDIT_DIR = OUT / "audit"
REPORTS_DIR = OUT / "reports"
FIGURES_DIR = OUT / "figures"

INNER_SPLIT_SEED = 20260824
VALIDATION_RATIO = 0.1
MIN_TRAIN_ITEMS = 5
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 30
MIN_EPOCHS = 0
MIN_LR = 6.25e-5
LR0 = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_THRESHOLD = 1e-4
CHECKPOINT_EPOCHS = {36, 50, 75, 100, 125, 150, 200, 250, 300}
REPORT_EPOCHS = {20, 36, 50, 75, 100}
PAIR_SAMPLE_SEED = 2026
LEGACY_EPOCH = 36


def ensure_dirs() -> None:
    for p in (OUT, SPLITS_DIR, FEATURES_DIR, MODELS_DIR, AUDIT_DIR, REPORTS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)


def split_train_external(
    train_external: dict[int, set[int]],
    *,
    validation_ratio: float = VALIDATION_RATIO,
    min_train_items: int = MIN_TRAIN_ITEMS,
    seed: int = INNER_SPLIT_SEED,
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    train_inner: dict[int, set[int]] = {}
    val_inner: dict[int, set[int]] = {}
    skipped = 0
    for u, items in train_external.items():
        items_list = sorted(items)
        if len(items_list) < min_train_items:
            train_inner[u] = set(items_list)
            val_inner[u] = set()
            skipped += 1
            continue
        n_val = max(1, int(round(len(items_list) * validation_ratio)))
        n_val = min(n_val, len(items_list) - min_train_items + 1)
        n_val = max(1, n_val)
        chosen = set(rng.choice(items_list, size=n_val, replace=False).tolist())
        val_inner[u] = chosen
        train_inner[u] = set(items_list) - chosen
    meta = {
        "seed": seed,
        "validation_ratio": validation_ratio,
        "min_train_items": min_train_items,
        "users_without_val_holdout": skipped,
        "n_users": len(train_external),
        "n_train_inner_interactions": int(sum(len(v) for v in train_inner.values())),
        "n_val_inner_interactions": int(sum(len(v) for v in val_inner.values())),
        "n_train_external_interactions": int(sum(len(v) for v in train_external.values())),
    }
    return train_inner, val_inner, meta


def inner_split_manifest(
    train_external: dict[int, set[int]],
    train_inner: dict[int, set[int]],
    val_inner: dict[int, set[int]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    union_pairs = pair_set(train_inner) | pair_set(val_inner)
    external_pairs = pair_set(train_external)
    per_user_overlap = sum(len(train_inner.get(u, set()) & val_inner.get(u, set())) for u in train_external)
    payload = {
        "generated": utc_now(),
        "experiment_tag": EXPERIMENT_TAG,
        "inner_split_seed": INNER_SPLIT_SEED,
        "validation_ratio": VALIDATION_RATIO,
        "min_train_items": MIN_TRAIN_ITEMS,
        "train_external_hash": hash_jsonable({str(u): sorted(v) for u, v in train_external.items()}),
        "train_inner_hash": hash_jsonable({str(u): sorted(v) for u, v in train_inner.items()}),
        "val_inner_hash": hash_jsonable({str(u): sorted(v) for u, v in val_inner.items()}),
        "counts": {
            "users_train_external": len(train_external),
            "users_train_inner": len(train_inner),
            "users_val_inner_nonempty": sum(1 for v in val_inner.values() if v),
            "interactions_train_external": meta["n_train_external_interactions"],
            "interactions_train_inner": meta["n_train_inner_interactions"],
            "interactions_val_inner": meta["n_val_inner_interactions"],
        },
        "checks": {
            "union_equals_train_external": union_pairs == external_pairs,
            "per_user_train_val_disjoint": per_user_overlap == 0,
            "train_inner_subset_external": pair_set(train_inner).issubset(external_pairs),
            "val_inner_subset_external": pair_set(val_inner).issubset(external_pairs),
        },
        "meta": meta,
        "SEALED_TEST_NOT_USED_FOR_EPOCH_SELECTION": "YES",
    }
    return payload


def select_inner_eval_users(train_inner: dict[int, set[int]], val_inner: dict[int, set[int]]) -> list[int]:
    return sorted(
        u
        for u in train_inner
        if len(train_inner.get(u, ())) >= 1 and len(val_inner.get(u, ())) > 0
    )


def build_inner_pairs(
    cfg: dict[str, Any],
    train_inner: dict[int, set[int]],
    val_inner: dict[int, set[int]],
    *,
    n_items: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    eval_users = select_inner_eval_users(train_inner, val_inner)
    pop = item_popularity(train_inner, n_items)
    neg_mode = cfg["negatives"]["active_type"]
    train_pairs = build_pair_table(
        users=eval_users,
        positives_by_user=train_inner,
        history_by_user=train_inner,
        n_items=n_items,
        n_neg_per_pos=cfg["negatives"]["train_per_positive"],
        max_positives_per_user=30,
        popularity=pop,
        mode=neg_mode,
        seed=PAIR_SAMPLE_SEED,
        split_name="convergence_train_inner",
    )
    val_pairs = build_pair_table(
        users=eval_users,
        positives_by_user=val_inner,
        history_by_user=train_inner,
        n_items=n_items,
        n_neg_per_pos=cfg["negatives"]["validation_per_positive"],
        max_positives_per_user=cfg["evaluation"]["max_val_positives_per_user"],
        popularity=pop,
        mode=neg_mode,
        seed=PAIR_SAMPLE_SEED,
        split_name="convergence_val_inner",
    )
    return train_pairs, val_pairs, eval_users


def materialize_split_features(
    bundle: dict[str, Any],
    *,
    split: str,
    pairs: dict[str, np.ndarray],
    out_root: Path,
) -> Path:
    users = pairs["user_id"]
    items = pairs["item_id"]
    labels = pairs["label"]
    n = len(labels)
    if split == "train":
        a_dir = out_root / "A_true"
        h_dir = out_root / "H3_a11_top25"
        leg_dir = out_root / "LEG_K2"
        fit_tag = "TRAIN_INNER_ONLY"
    else:
        a_dir = out_root / "A_true"
        h_dir = out_root / "H3_a11_top25"
        leg_dir = out_root / "LEG_K2"
        fit_tag = "TRAIN_INNER_ONLY_EVAL_ON_VAL_INNER"
    for p in (a_dir, h_dir, leg_dir):
        p.mkdir(parents=True, exist_ok=True)

    x_a = a_dir / f"X_{split}.npy"
    x_h = h_dir / f"X_{split}.npy"
    x_l = leg_dir / f"LEG_K2_{split}.npy"
    if x_a.exists() and x_h.exists() and x_l.exists():
        if np.load(x_a, mmap_mode="r").shape[0] == n:
            return out_root

    cf = ensure_cross_fit(bundle)
    model_train = bundle["model_train"]
    pop = bundle["popularity"]
    kg = bundle["item_kg_degree"]
    A = np.lib.format.open_memmap(x_a, mode="w+", dtype=np.float32, shape=(n, 5))
    H = np.lib.format.open_memmap(x_h, mode="w+", dtype=np.float32, shape=(n, 3))
    L = np.lib.format.open_memmap(x_l, mode="w+", dtype=np.float32, shape=(n, 1))
    size_cache: dict[int, np.ndarray] = {}
    chunk_size = 50_000
    for chunk_start in tqdm(range(0, n, chunk_size), desc=f"materialize {split}"):
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
    if split == "train":
        with (leg_dir / "LEG_K2_scaler.pkl").open("wb") as f:
            pickle.dump(StandardScaler().fit(np.asarray(L)), f)
        write_json(a_dir / "meta.json", {"feature_names": CLEAN_V2_TABULAR_FEATURE_NAMES, "fit_population": fit_tag, "generated": utc_now()})
        write_json(h_dir / "meta.json", {"feature_names": ["mean:hcr_a11", "max:hcr_a11", "top3mean:hcr_a11"], "fit_population": fit_tag, "generated": utc_now()})
        write_json(leg_dir / "meta.json", {"formula": "L2 = mean P2(A11) over signed Top25", "fit_population": fit_tag, "generated": utc_now()})
    return out_root


def _cfg_for_convergence() -> dict[str, Any]:
    cfg = load_protocol_config(PROTOCOL_CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["paths"]["outputs_root"] = str(OUT)
    cfg["paths"]["splits"] = str(SPLITS_DIR)
    cfg["paths"]["features"] = str(FEATURES_DIR)
    cfg["paths"]["audit"] = str(AUDIT_DIR)
    cfg["paths"]["manifest"] = str(AUDIT_DIR / "convergence_protocol_manifest.json")
    return cfg


def build_convergence_bundle() -> dict[str, Any]:
    ensure_dirs()
    ext = build_external_bundle()
    cfg = _cfg_for_convergence()

    train_inner, val_inner, split_meta = split_train_external(ext.train_external)
    manifest = inner_split_manifest(ext.train_external, train_inner, val_inner, split_meta)
    save_user_sets(SPLITS_DIR / "train_external.txt", ext.train_external)
    save_user_sets(SPLITS_DIR / "train_inner.txt", train_inner)
    save_user_sets(SPLITS_DIR / "val_inner.txt", val_inner)
    write_json(SPLITS_DIR / "inner_split_manifest.json", manifest)
    write_json(AUDIT_DIR / "inner_split_manifest.json", manifest)

    train_pairs, val_pairs, eval_users = build_inner_pairs(cfg, train_inner, val_inner, n_items=ext.n_items)
    np.savez_compressed(FEATURES_DIR / "train_pairs.npz", **train_pairs)
    np.savez_compressed(FEATURES_DIR / "val_pairs.npz", **val_pairs)
    pop = item_popularity(train_inner, ext.n_items)
    np.save(FEATURES_DIR / "popularity.npy", pop)
    np.save(FEATURES_DIR / "item_kg_degree.npy", ext.item_kg_degree)
    (FEATURES_DIR / "eval_users_inner.json").write_text(json.dumps({"eval_users": eval_users}, indent=2) + "\n", encoding="utf-8")

    cooc = build_cooccurrence(train_inner, ext.n_items, max_history_for_pairs=60, seed=PAIR_SAMPLE_SEED)
    index = PairwiseStatsIndex(cooc, pop, n_users=len(train_inner), smoothing=float(cfg.get("hcr", {}).get("smoothing", 0.5)))
    prepared = {
        "cfg": cfg,
        "model_train": train_inner,
        "popularity": pop,
        "item_kg_degree": ext.item_kg_degree,
        "index": index,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": {"user_id": np.zeros(0, dtype=np.int32), "item_id": np.zeros(0, dtype=np.int32), "label": np.zeros(0, dtype=np.int8)},
        "eval_users": eval_users,
        "train_external_hash": manifest["train_external_hash"],
        "train_inner_hash": manifest["train_inner_hash"],
        "val_inner_hash": manifest["val_inner_hash"],
    }
    write_json(FEATURES_DIR / "bundle_meta.json", {
        "generated": utc_now(),
        "experiment_tag": EXPERIMENT_TAG,
        "n_train_pairs": int(len(train_pairs["label"])),
        "n_val_pairs": int(len(val_pairs["label"])),
        "n_eval_users": len(eval_users),
    })
    return prepared


def load_or_build_bundle() -> dict[str, Any]:
    manifest_path = AUDIT_DIR / "inner_split_manifest.json"
    bundle_meta = FEATURES_DIR / "bundle_meta.json"
    if manifest_path.exists() and bundle_meta.exists() and (FEATURES_DIR / "train_pairs.npz").exists():
        from src.lastfm_lp.data.build_splits import load_user_sets

        ext = build_external_bundle()
        cfg = _cfg_for_convergence()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_inner = load_user_sets(SPLITS_DIR / "train_inner.txt")
        val_inner = load_user_sets(SPLITS_DIR / "val_inner.txt")
        pop = np.load(FEATURES_DIR / "popularity.npy")
        item_kg_degree = np.load(FEATURES_DIR / "item_kg_degree.npy")
        train_pairs = dict(np.load(FEATURES_DIR / "train_pairs.npz"))
        val_pairs = dict(np.load(FEATURES_DIR / "val_pairs.npz"))
        eval_users = json.loads((FEATURES_DIR / "eval_users_inner.json").read_text(encoding="utf-8"))["eval_users"]
        cooc = build_cooccurrence(train_inner, ext.n_items, max_history_for_pairs=60, seed=PAIR_SAMPLE_SEED)
        index = PairwiseStatsIndex(cooc, pop, n_users=len(train_inner), smoothing=float(cfg.get("hcr", {}).get("smoothing", 0.5)))
        return {
            "cfg": cfg,
            "model_train": train_inner,
            "popularity": pop,
            "item_kg_degree": item_kg_degree,
            "index": index,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": {"user_id": np.zeros(0, dtype=np.int32), "item_id": np.zeros(0, dtype=np.int32), "label": np.zeros(0, dtype=np.int8)},
            "eval_users": eval_users,
            "train_external_hash": manifest["train_external_hash"],
            "train_inner_hash": manifest["train_inner_hash"],
            "val_inner_hash": manifest["val_inner_hash"],
        }
    return build_convergence_bundle()


def convergence_training_config() -> dict[str, Any]:
    hist = frozen_hgt_training_hist()
    return {
        "experiment_tag": EXPERIMENT_TAG,
        "optimizer": "Adam",
        "lr0": LR0,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_mode": "max",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_threshold": SCHEDULER_THRESHOLD,
        "min_lr": MIN_LR,
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_delta": MIN_IMPROVE,
        "checkpoint_metric": "VAL_INNER_NDCG@20",
        "batch_size": int(hist["batch_size"]),
        "loss": hist["loss"],
        "SEALED_TEST_NOT_USED_FOR_EPOCH_SELECTION": "YES",
        "source_hgt_hist": hist,
    }


def write_learning_curve(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fields = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in history:
            w.writerow(row)


def read_learning_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_resume_state(run_dir: Path, ckpt_dir: Path) -> dict[str, Any] | None:
    history = read_learning_curve(run_dir / "learning_curve.csv")
    if not history:
        return None
    last = history[-1]
    best_row = max(history, key=lambda r: float(r["val_inner_NDCG@20"]))
    last_ep1 = int(last["epoch_1based"])
    best_ep1 = int(best_row["epoch_1based"])
    resume_ckpt: Path | None = None
    for ep in sorted(CHECKPOINT_EPOCHS, reverse=True):
        if ep <= last_ep1:
            candidate = ckpt_dir / f"epoch_{ep:03d}.pt"
            if candidate.exists():
                resume_ckpt = candidate
                break
    if resume_ckpt is None and (ckpt_dir / "best.pt").exists():
        resume_ckpt = ckpt_dir / "best.pt"
    if resume_ckpt is None:
        return None
    best_pt = ckpt_dir / "best.pt"
    return {
        "history": history,
        "start_epoch": int(last["epoch"]) + 1,
        "best_epoch": int(best_row["epoch"]),
        "best_ndcg": float(best_row["val_inner_NDCG@20"]),
        "left": EARLY_STOP_PATIENCE - (last_ep1 - best_ep1),
        "resume_ckpt": resume_ckpt,
        "best_pt": best_pt if best_pt.exists() else resume_ckpt,
        "resume_lr": float(last["lr"]),
        "last_ep1": last_ep1,
        "best_ep1": best_ep1,
    }


def train_convergence_pilot(
    *,
    seed: int,
    bundle: dict[str, Any],
    run_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mat_root = FEATURES_DIR / "materialized"
    materialize_split_features(bundle, split="train", pairs=bundle["train_pairs"], out_root=mat_root)
    materialize_split_features(bundle, split="val", pairs=bundle["val_pairs"], out_root=mat_root)

    A_tr = np.load(mat_root / "A_true" / "X_train.npy").astype(np.float32)
    A_va = np.load(mat_root / "A_true" / "X_val.npy").astype(np.float32)
    H_tr = np.load(mat_root / "H3_a11_top25" / "X_train.npy").astype(np.float32)
    H_va = np.load(mat_root / "H3_a11_top25" / "X_val.npy").astype(np.float32)
    r_tr = np.load(mat_root / "LEG_K2" / "LEG_K2_train.npy").astype(np.float32)
    r_va = np.load(mat_root / "LEG_K2" / "LEG_K2_val.npy").astype(np.float32)
    if r_tr.ndim == 1:
        r_tr = r_tr[:, None]
        r_va = r_va[:, None]

    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(H_tr)
    A_tr_s = scale_split(a_scaler, A_tr)
    A_va_s = scale_split(a_scaler, A_va)
    H_tr_s = scale_split(h_scaler, H_tr)
    H_va_s = scale_split(h_scaler, H_va)

    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device_name = os.environ.get("LASTFM_TORCH_DEVICE", acfg.get("device") or "auto")
    device = resolve_torch_device(str(device_name))
    graph = load_data_and_typed_graph(
        bundle["cfg"],
        bundle["model_train"],
        max_kg_edges=acfg.get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    batch_size = int(acfg.get("batch_size", 4096))

    seed_everything(seed)
    model = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
    arch = architecture_audit(model, device)
    a11 = a11_audit(r_tr, r_va[: min(len(r_va), 1024)])
    if arch["ARCHITECTURE_STATUS"] != "FROZEN_REPRODUCED" or a11["A11_AUDIT_STATUS"] != "PASS":
        raise RuntimeError("Pre-train architecture/A11 audit failed")

    u_all, i_all = user_item_to_nodes(bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset)
    y_all = bundle["train_pairs"]["label"].astype(np.float32)
    n_train = len(y_all)
    va_u = bundle["val_pairs"]["user_id"]
    va_i = bundle["val_pairs"]["item_id"]
    va_y = bundle["val_pairs"]["label"]

    opt = torch.optim.Adam(model.parameters(), lr=LR0, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, threshold=SCHEDULER_THRESHOLD, min_lr=MIN_LR
    )
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device))

    tcfg = convergence_training_config()
    write_json(run_dir / "training_config.json", tcfg)
    write_json(AUDIT_DIR / "convergence_training_config.json", tcfg)

    history: list[dict[str, Any]] = []
    best_state = None
    last_state = None
    best_ndcg = -1.0
    best_epoch = -1
    left = EARLY_STOP_PATIENCE
    stop_reason = "MAX_EPOCH"
    t0 = time.time()
    start_epoch = 0
    resumed = False

    if resume:
        state = load_resume_state(run_dir, ckpt_dir)
        if state and state["start_epoch"] < MAX_EPOCHS:
            history = state["history"]
            start_epoch = int(state["start_epoch"])
            best_ndcg = float(state["best_ndcg"])
            best_epoch = int(state["best_epoch"])
            left = int(state["left"])
            resumed = True
            model.load_state_dict(torch.load(state["resume_ckpt"], map_location=device))
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in torch.load(state["best_pt"], map_location="cpu").items()
            }
            for pg in opt.param_groups:
                pg["lr"] = float(state["resume_lr"])
            print(
                f"[convergence seed={seed}] resume from epoch {start_epoch + 1}/{MAX_EPOCHS} "
                f"(logged through {state['last_ep1']}); best={state['best_ep1']} "
                f"NDCG@20={best_ndcg:.6f}; early_stop_left={left}; ckpt={state['resume_ckpt'].name}",
                flush=True,
            )
            if left <= 0:
                stop_reason = "EARLY_STOP"
                print(
                    f"[convergence seed={seed}] patience already exhausted on resume — finalizing",
                    flush=True,
                )
                if best_state:
                    model.load_state_dict(best_state)
                meta = {
                    "seed": seed,
                    "experiment_tag": EXPERIMENT_TAG,
                    "best_epoch": best_epoch,
                    "best_epoch_1based": best_epoch + 1 if best_epoch >= 0 else -1,
                    "best_val_inner_NDCG@20": best_ndcg,
                    "final_epoch": history[-1]["epoch"] if history else -1,
                    "final_epoch_1based": history[-1]["epoch_1based"] if history else -1,
                    "stop_reason": stop_reason,
                    "resumed": resumed,
                    "training_config": tcfg,
                    "device": str(device),
                    "host": host_info(),
                    "git_commit": git_commit(),
                    "generated": utc_now(),
                    "seconds": time.time() - t0,
                    "history": history,
                    "SEALED_TEST_NOT_USED_FOR_EPOCH_SELECTION": "YES",
                    "metric_note": "VAL_INNER sampled NDCG@20 (20 negs/pos); comparable to TRUE FINAL ~0.873, not sealed/full-catalog",
                }
                write_json(run_dir / "train_meta.json", meta)
                return meta

    for epoch in range(start_epoch, MAX_EPOCHS):
        te = time.time()
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
        empty_cache()
        scores = score_model(model, va_u, va_i, A_va_s, H_va_s, r_va, item_offset, device)
        sm = sampled_metrics(va_u, va_y, scores)
        ndcg = float(sm["NDCG@20"])
        recall = float(sm.get("Recall@20", float("nan")))
        lr_now = float(opt.param_groups[0]["lr"])
        sched.step(ndcg)
        row = {
            "epoch": epoch,
            "epoch_1based": epoch + 1,
            "train_loss": float(info["epoch_mean_loss"]),
            "val_inner_NDCG@20": ndcg,
            "val_inner_Recall@20": recall,
            "lr": lr_now,
            "sec": time.time() - te,
        }
        history.append(row)
        write_learning_curve(run_dir / "learning_curve.csv", history)
        print(
            f"[convergence seed={seed}] epoch {epoch + 1}/{MAX_EPOCHS} "
            f"loss={row['train_loss']:.4f} val_NDCG@20={ndcg:.4f} Recall@20={recall:.4f} lr={lr_now:.2e}",
            flush=True,
        )
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ep1 = epoch + 1
        if ep1 in CHECKPOINT_EPOCHS:
            torch.save(last_state, ckpt_dir / f"epoch_{ep1:03d}.pt")
        if ndcg > best_ndcg + MIN_IMPROVE:
            best_ndcg = ndcg
            best_epoch = epoch
            best_state = last_state
            left = EARLY_STOP_PATIENCE
            torch.save(best_state, ckpt_dir / "best.pt")
        elif epoch + 1 >= MIN_EPOCHS:
            left -= 1
            if left <= 0:
                stop_reason = "EARLY_STOP"
                break
    else:
        stop_reason = "MAX_EPOCH"

    if last_state:
        torch.save(last_state, ckpt_dir / "final.pt")
    if best_state:
        model.load_state_dict(best_state)

    meta = {
        "seed": seed,
        "experiment_tag": EXPERIMENT_TAG,
        "best_epoch": best_epoch,
        "best_epoch_1based": best_epoch + 1 if best_epoch >= 0 else -1,
        "best_val_inner_NDCG@20": best_ndcg,
        "final_epoch": history[-1]["epoch"] if history else -1,
        "final_epoch_1based": history[-1]["epoch_1based"] if history else -1,
        "stop_reason": stop_reason,
        "resumed": resumed,
        "training_config": tcfg,
        "device": str(device),
        "host": host_info(),
        "git_commit": git_commit(),
        "generated": utc_now(),
        "seconds": time.time() - t0,
        "history": history,
        "SEALED_TEST_NOT_USED_FOR_EPOCH_SELECTION": "YES",
        "metric_note": "VAL_INNER sampled NDCG@20 (20 negs/pos); comparable to TRUE FINAL ~0.873, not sealed/full-catalog",
    }
    write_json(run_dir / "train_meta.json", meta)
    return meta


def history_row_at_epoch(history: list[dict[str, Any]], epoch_1based: int) -> dict[str, Any] | None:
    for row in history:
        ep1 = int(row.get("epoch_1based", int(row["epoch"]) + 1))
        if ep1 == epoch_1based:
            return row
    return None


def normalize_history_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["epoch"] = int(row["epoch"])
    out["epoch_1based"] = int(row.get("epoch_1based", int(row["epoch"]) + 1))
    out["train_loss"] = float(row["train_loss"])
    out["val_inner_NDCG@20"] = float(row["val_inner_NDCG@20"])
    out["val_inner_Recall@20"] = float(row["val_inner_Recall@20"])
    out["lr"] = float(row["lr"])
    out["sec"] = float(row.get("sec", 0.0))
    return out


def generate_pilot_report(seed: int, run_dir: Path) -> Path:
    meta_path = run_dir / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    history = [normalize_history_row(h) for h in (meta.get("history") or [])]
    best_ep = int(meta.get("best_epoch_1based", -1))
    best_row = history_row_at_epoch(history, best_ep) if best_ep > 0 else None
    last_row = history[-1] if history else None
    ep36 = history_row_at_epoch(history, LEGACY_EPOCH)

    report_epochs = sorted(REPORT_EPOCHS | {best_ep, int(last_row["epoch_1based"]) if last_row else -1})
    table_rows = []
    for ep in report_epochs:
        if ep <= 0:
            continue
        row = history_row_at_epoch(history, ep)
        if row is None:
            continue
        table_rows.append({
            "epoch": ep,
            "train_loss": float(row["train_loss"]),
            "inner_val_NDCG@20": float(row["val_inner_NDCG@20"]),
            "inner_val_Recall@20": float(row["val_inner_Recall@20"]),
            "lr": float(row["lr"]),
        })

    ndcg36 = float(ep36["val_inner_NDCG@20"]) if ep36 else float("nan")
    best_ndcg = float(meta.get("best_val_inner_NDCG@20", float("nan")))
    preconvergence = "YES" if (best_ep > LEGACY_EPOCH or (last_row and float(last_row["val_inner_NDCG@20"]) > ndcg36 + MIN_IMPROVE and int(last_row["epoch_1based"]) > LEGACY_EPOCH)) else "NO"
    if best_ep > LEGACY_EPOCH:
        preconvergence = "YES"
    elif best_ep == LEGACY_EPOCH and abs(best_ndcg - ndcg36) < MIN_IMPROVE:
        preconvergence = "NO"
    elif best_ep < LEGACY_EPOCH:
        preconvergence = "NO"

    summary = {
        "experiment_tag": EXPERIMENT_TAG,
        "seed": seed,
        "BEST_EPOCH_101" if seed == 101 else f"BEST_EPOCH_{seed}": best_ep,
        f"BEST_INNER_NDCG20_{seed}": best_ndcg,
        f"NDCG20_AT_EPOCH36": ndcg36,
        f"TRAIN_LOSS_AT_EPOCH36": float(ep36["train_loss"]) if ep36 else float("nan"),
        f"TRAIN_LOSS_AT_BEST_EPOCH": float(best_row["train_loss"]) if best_row else float("nan"),
        "EPOCH_36_PRECONVERGENCE": preconvergence,
        "stop_reason": meta.get("stop_reason"),
        "final_epoch": meta.get("final_epoch_1based"),
        "table": table_rows,
        "SEALED_TEST_NOT_USED_FOR_EPOCH_SELECTION": "YES",
    }
    report_path = REPORTS_DIR / f"CONVERGENCE_PILOT_SEED{seed}_20260824.json"
    write_json(report_path, summary)

    md_lines = [
        f"# {EXPERIMENT_TAG} — seed {seed} convergence pilot",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Key outcomes",
        "",
        f"- **BEST_EPOCH_{seed}** = {best_ep}",
        f"- **BEST_INNER_NDCG20_{seed}** = {best_ndcg:.6f}",
        f"- **NDCG20_AT_EPOCH36** = {ndcg36:.6f}",
        f"- **TRAIN_LOSS_AT_EPOCH36** = {summary[f'TRAIN_LOSS_AT_EPOCH36']:.6f}",
        f"- **TRAIN_LOSS_AT_BEST_EPOCH** = {summary['TRAIN_LOSS_AT_BEST_EPOCH']:.6f}",
        f"- **EPOCH_36_PRECONVERGENCE** = {preconvergence}",
        "",
        "## Learning curve (selected epochs)",
        "",
        "| epoch | train loss | inner-val NDCG@20 | inner-val Recall@20 | LR |",
        "|------:|-----------:|------------------:|--------------------:|---:|",
    ]
    for row in table_rows:
        md_lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['inner_val_NDCG@20']:.6f} | "
            f"{row['inner_val_Recall@20']:.6f} | {row['lr']:.2e} |"
        )
    md_path = REPORTS_DIR / f"CONVERGENCE_PILOT_SEED{seed}_20260824.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if history:
        ep = [h["epoch_1based"] for h in history]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(ep, [h["train_loss"] for h in history], color="#1f77b4", lw=1.6)
        axes[0, 0].axvline(LEGACY_EPOCH, color="red", ls="--", lw=1, label=f"legacy stop @ {LEGACY_EPOCH}")
        axes[0, 0].set_title("Train loss vs epoch")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 0].legend()

        axes[0, 1].plot(ep, [h["val_inner_NDCG@20"] for h in history], color="#2ca02c", lw=1.6)
        axes[0, 1].axvline(LEGACY_EPOCH, color="red", ls="--", lw=1, label=f"legacy stop @ {LEGACY_EPOCH}")
        if best_ep > 0:
            axes[0, 1].axvline(best_ep, color="orange", ls=":", lw=1, label=f"best @ {best_ep}")
        axes[0, 1].set_title("VAL_INNER NDCG@20 vs epoch")
        axes[0, 1].set_xlabel("epoch")
        axes[0, 1].legend()

        axes[1, 0].plot(ep, [h["val_inner_Recall@20"] for h in history], color="#9467bd", lw=1.6)
        axes[1, 0].axvline(LEGACY_EPOCH, color="red", ls="--", lw=1)
        axes[1, 0].set_title("VAL_INNER Recall@20 vs epoch")
        axes[1, 0].set_xlabel("epoch")

        axes[1, 1].plot(ep, [h["lr"] for h in history], color="#ff7f0e", lw=1.6)
        axes[1, 1].set_title("Learning rate vs epoch")
        axes[1, 1].set_xlabel("epoch")
        axes[1, 1].set_yscale("log")

        fig.suptitle(f"{EXPERIMENT_TAG} — seed {seed}", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"convergence_pilot_seed{seed}_20260824.png", dpi=150)
        plt.close(fig)

        for name, key in [
            ("train_loss", "train_loss"),
            ("val_ndcg20", "val_inner_NDCG@20"),
            ("val_recall20", "val_inner_Recall@20"),
            ("lr", "lr"),
        ]:
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(ep, [h[key] for h in history], lw=1.6)
            ax1.axvline(LEGACY_EPOCH, color="red", ls="--", lw=1, label=f"legacy @ {LEGACY_EPOCH}")
            ax1.set_xlabel("epoch")
            ax1.set_title(name)
            if name == "lr":
                ax1.set_yscale("log")
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(FIGURES_DIR / f"convergence_pilot_seed{seed}_{name}_20260824.png", dpi=150)
            plt.close(fig1)

    return md_path
