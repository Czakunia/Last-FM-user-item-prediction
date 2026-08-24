"""Materialize feature matrices for Stage A/B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.lastfm_lp.binary.user_hcr_aggregation import aggregate_user_hcr
from src.lastfm_lp.features.tabular_pair_features import (
    build_tabular_pair_features,
    resolve_feature_names,
)


def feature_family(stage: str) -> str:
    if stage in {"A0", "A1", "A2"}:
        return "A"
    if stage in {"B1", "B2"}:
        return "B_classical"
    if stage in {"B0", "B3", "B4"}:
        return "B_hcr"
    raise KeyError(stage)


def _history_for_row(
    user_id: int,
    item_id: int,
    label: int,
    model_train: dict[int, set[int]],
    *,
    exclude_candidate_from_train_pos: bool,
) -> set[int]:
    hist = set(model_train.get(user_id, ()))
    if exclude_candidate_from_train_pos and label == 1 and item_id in hist:
        hist = hist - {item_id}
    return hist


def build_feature_matrix(
    pairs: dict[str, np.ndarray],
    model_train: dict[int, set[int]],
    item_statistics: dict[str, Any],
    kg_statistics: dict[str, Any],
    feature_names: list[str],
    *,
    pairwise_index=None,
    need_binary: bool = False,
    exclude_candidate_from_train_pos: bool = False,
    desc: str = "features",
) -> np.ndarray:
    n = len(pairs["label"])
    X = np.zeros((n, len(feature_names)), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    for r in tqdm(range(n), desc=desc, mininterval=2.0):
        u = int(pairs["user_id"][r])
        i = int(pairs["item_id"][r])
        y = int(pairs["label"][r])
        hist_map = {
            u: _history_for_row(
                u,
                i,
                y,
                model_train,
                exclude_candidate_from_train_pos=exclude_candidate_from_train_pos,
            )
        }
        feats = build_tabular_pair_features(
            u,
            i,
            hist_map,
            item_statistics,
            kg_statistics,
            pairwise_index=pairwise_index,
            include_binary=False,
        )
        if need_binary:
            feats.update(aggregate_user_hcr(hist_map[u], i, pairwise_index, max_history=25))
        for name in feature_names:
            X[r, name_to_idx[name]] = float(feats.get(name, 0.0))
    return X


def materialize_stage_features(bundle: dict[str, Any], stage: str) -> dict[str, Any]:
    cfg = bundle["cfg"]
    family = feature_family(stage)
    names = resolve_feature_names(stage)
    need_binary = family.startswith("B")
    item_statistics = {"popularity": bundle["popularity"]}
    kg_statistics = {"item_kg_degree": bundle["item_kg_degree"]}
    out_dir = Path(cfg["paths"]["features"]) / family
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "X_train.npy").exists():
        # may be a superset family; slice columns if needed
        cached_names = json.loads((out_dir / "feature_names.json").read_text(encoding="utf-8"))
        Xtr = np.load(out_dir / "X_train.npy")
        Xva = np.load(out_dir / "X_val.npy")
        Xte = np.load(out_dir / "X_test.npy")
        if cached_names != names:
            idx = [cached_names.index(n) for n in names]
            Xtr, Xva, Xte = Xtr[:, idx], Xva[:, idx], Xte[:, idx]
        return {
            "feature_names": names,
            "X_train": Xtr,
            "X_val": Xva,
            "X_test": Xte,
            "y_train": bundle["train_pairs"]["label"],
            "y_val": bundle["val_pairs"]["label"],
            "y_test": bundle["test_pairs"]["label"],
        }

    # For B_classical store full B_hcr once if building classical first — prefer full HCR family.
    build_names = names
    if family == "B_classical":
        # build full HCR feature set and cache under B_hcr for reuse
        from src.lastfm_lp.features.tabular_pair_features import BINARY_DEPENDENCE_FEATURES
        from src.lastfm_lp.features.tabular_pair_features import TABULAR_BASE_FEATURES

        build_names = list(TABULAR_BASE_FEATURES) + list(BINARY_DEPENDENCE_FEATURES)
        out_dir = Path(cfg["paths"]["features"]) / "B_hcr"
        out_dir.mkdir(parents=True, exist_ok=True)
        family = "B_hcr"
        need_binary = True
        if (out_dir / "X_train.npy").exists():
            cached_names = json.loads((out_dir / "feature_names.json").read_text(encoding="utf-8"))
            Xtr = np.load(out_dir / "X_train.npy")
            Xva = np.load(out_dir / "X_val.npy")
            Xte = np.load(out_dir / "X_test.npy")
            idx = [cached_names.index(n) for n in names]
            return {
                "feature_names": names,
                "X_train": Xtr[:, idx],
                "X_val": Xva[:, idx],
                "X_test": Xte[:, idx],
                "y_train": bundle["train_pairs"]["label"],
                "y_val": bundle["val_pairs"]["label"],
                "y_test": bundle["test_pairs"]["label"],
            }

    X_train = build_feature_matrix(
        bundle["train_pairs"],
        bundle["model_train"],
        item_statistics,
        kg_statistics,
        build_names,
        pairwise_index=bundle["index"],
        need_binary=need_binary,
        exclude_candidate_from_train_pos=True,
        desc=f"{family} train",
    )
    X_val = build_feature_matrix(
        bundle["val_pairs"],
        bundle["model_train"],
        item_statistics,
        kg_statistics,
        build_names,
        pairwise_index=bundle["index"],
        need_binary=need_binary,
        exclude_candidate_from_train_pos=False,
        desc=f"{family} val",
    )
    X_test = build_feature_matrix(
        bundle["test_pairs"],
        bundle["model_train"],
        item_statistics,
        kg_statistics,
        build_names,
        pairwise_index=bundle["index"],
        need_binary=need_binary,
        exclude_candidate_from_train_pos=False,
        desc=f"{family} test",
    )
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "X_test.npy", X_test)
    (out_dir / "feature_names.json").write_text(json.dumps(build_names, indent=2), encoding="utf-8")

    if build_names != names:
        idx = [build_names.index(n) for n in names]
        X_train, X_val, X_test = X_train[:, idx], X_val[:, idx], X_test[:, idx]

    return {
        "feature_names": names,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": bundle["train_pairs"]["label"],
        "y_val": bundle["val_pairs"]["label"],
        "y_test": bundle["test_pairs"]["label"],
    }
