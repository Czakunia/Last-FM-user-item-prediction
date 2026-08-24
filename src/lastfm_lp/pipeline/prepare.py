"""Prepare splits, candidates, co-occurrence index (shared by stages)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import load_npz, save_npz

from src.lastfm_lp.binary.contingency_tables import build_cooccurrence
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.config import load_protocol_config, verify_manifest
from src.lastfm_lp.data.audit import run_audit
from src.lastfm_lp.data.build_splits import (
    build_per_user_splits,
    save_splits,
    select_eval_users,
)
from src.lastfm_lp.data.candidate_sampling import build_pair_table
from src.lastfm_lp.data.load_kgat_lastfm import item_popularity, load_lastfm


def prepare_all(
    cfg: dict[str, Any] | None = None,
    *,
    run_audit_flag: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    cfg = cfg or load_protocol_config()
    if verify:
        verify_manifest(cfg)

    data = load_lastfm(cfg["data"]["path"])
    splits = build_per_user_splits(
        data,
        validation_ratio=cfg["split"]["validation_ratio"],
        min_train_items=cfg["split"]["min_train_items"],
        seed=cfg["split"]["seed"],
    )
    n_eval = cfg["evaluation"].get("n_eval_users", cfg["evaluation"].get("max_users"))
    if n_eval is not None:
        n_eval = int(n_eval)
    eval_users = select_eval_users(
        splits,
        n_eval_users=n_eval,
        seed=cfg["evaluation"]["seed"],
    )
    print(f"Eval users: {len(eval_users)} (n_eval_users={n_eval})")
    save_splits(splits, cfg["paths"]["splits"], eval_users)

    if run_audit_flag:
        run_audit(data, splits, eval_users, cfg["paths"]["audit"])

    pop = item_popularity(splits["model_train"], data.n_items)
    print("Building co-occurrence from model_train …")
    cooc = build_cooccurrence(
        splits["model_train"],
        data.n_items,
        max_history_for_pairs=60,
        seed=cfg["split"]["seed"],
    )
    index = PairwiseStatsIndex(
        cooc,
        pop,
        n_users=len(splits["model_train"]),
        smoothing=cfg["hcr"]["smoothing"],
    )

    neg_mode = cfg["negatives"]["active_type"]
    seed = cfg["evaluation"]["seed"]

    print("Sampling candidate tables …")
    train_pairs = build_pair_table(
        users=eval_users,
        positives_by_user=splits["model_train"],
        history_by_user=splits["model_train"],
        n_items=data.n_items,
        n_neg_per_pos=cfg["negatives"]["train_per_positive"],
        max_positives_per_user=30,
        popularity=pop,
        mode=neg_mode,
        seed=seed,
        split_name="fit_train",
    )
    val_pairs = build_pair_table(
        users=eval_users,
        positives_by_user=splits["valid"],
        history_by_user=splits["model_train"],
        n_items=data.n_items,
        n_neg_per_pos=cfg["negatives"]["validation_per_positive"],
        max_positives_per_user=cfg["evaluation"]["max_val_positives_per_user"],
        popularity=pop,
        mode=neg_mode,
        seed=seed,
        split_name="valid",
    )
    test_pairs = build_pair_table(
        users=eval_users,
        positives_by_user=splits["test"],
        history_by_user=splits["model_train"],
        n_items=data.n_items,
        n_neg_per_pos=cfg["negatives"]["test_per_positive"],
        max_positives_per_user=cfg["evaluation"]["max_test_positives_per_user"],
        popularity=pop,
        mode=neg_mode,
        seed=seed,
        split_name="test",
    )

    feat_dir = Path(cfg["paths"]["features"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feat_dir / "train_pairs.npz", **train_pairs)
    np.savez_compressed(feat_dir / "val_pairs.npz", **val_pairs)
    np.savez_compressed(feat_dir / "test_pairs.npz", **test_pairs)
    np.save(feat_dir / "popularity.npy", pop)
    np.save(feat_dir / "item_kg_degree.npy", data.item_kg_degree)
    (feat_dir / "eval_users.json").write_text(json.dumps(eval_users), encoding="utf-8")
    save_npz(feat_dir / "cooccurrence.npz", cooc)

    meta = {
        "n_train_pairs": int(len(train_pairs["label"])),
        "n_val_pairs": int(len(val_pairs["label"])),
        "n_test_pairs": int(len(test_pairs["label"])),
        "n_eval_users": len(eval_users),
        "cooc_nnz": int(cooc.nnz),
    }
    (feat_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Prepared:", meta)

    return {
        "cfg": cfg,
        "data": data,
        "splits": splits,
        "eval_users": eval_users,
        "popularity": pop,
        "index": index,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
    }


def load_prepared(
    cfg: dict[str, Any] | None = None,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    cfg = cfg or load_protocol_config()
    if verify:
        verify_manifest(cfg)
    feat_dir = Path(cfg["paths"]["features"])
    splits_dir = Path(cfg["paths"]["splits"])
    from src.lastfm_lp.data.build_splits import load_user_sets

    model_train = load_user_sets(splits_dir / "model_train.txt")
    pop = np.load(feat_dir / "popularity.npy")
    kg_deg = np.load(feat_dir / "item_kg_degree.npy")
    cooc = load_npz(feat_dir / "cooccurrence.npz")
    index = PairwiseStatsIndex(
        cooc, pop, n_users=len(model_train), smoothing=cfg["hcr"]["smoothing"]
    )

    def _load_pairs(name: str) -> dict[str, np.ndarray]:
        z = np.load(feat_dir / name)
        return {k: z[k] for k in z.files}

    return {
        "cfg": cfg,
        "model_train": model_train,
        "popularity": pop,
        "item_kg_degree": kg_deg,
        "index": index,
        "train_pairs": _load_pairs("train_pairs.npz"),
        "val_pairs": _load_pairs("val_pairs.npz"),
        "test_pairs": _load_pairs("test_pairs.npz"),
        "eval_users": json.loads((feat_dir / "eval_users.json").read_text(encoding="utf-8")),
    }
