"""Materialize KG-HCR aggregated side features for Stage F1/F2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.lastfm_lp.binary.kg_hcr_aggregation import KG_HCR_AGG_FEATURES, aggregate_kg_hcr
from src.lastfm_lp.data.load_kgat_lastfm import load_lastfm
from src.lastfm_lp.kg.path_index import KGPathIndex


def materialize_kg_hcr_sides(
    bundle: dict[str, Any],
    *,
    mask_no_path: bool,
    tag: str,
    max_history: int = 25,
) -> dict[str, np.ndarray]:
    cfg = bundle["cfg"]
    out_dir = Path(cfg["paths"]["features"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    names_path = out_dir / "feature_names.json"
    if (out_dir / "X_train.npy").exists() and names_path.exists():
        return {
            "names": json.loads(names_path.read_text(encoding="utf-8")),
            "train": np.load(out_dir / "X_train.npy"),
            "val": np.load(out_dir / "X_val.npy"),
            "test": np.load(out_dir / "X_test.npy"),
        }

    data = load_lastfm(cfg["data"]["path"])
    kg_index = KGPathIndex(data.kg, data.n_entities, data.n_relations)
    hcr_index = bundle["index"]
    model_train = bundle["model_train"]
    names = list(KG_HCR_AGG_FEATURES)

    def build(pairs: dict[str, np.ndarray], desc: str) -> np.ndarray:
        n = len(pairs["label"])
        X = np.zeros((n, len(names)), dtype=np.float32)
        for r in tqdm(range(n), desc=desc, mininterval=2.0):
            u = int(pairs["user_id"][r])
            i = int(pairs["item_id"][r])
            y = int(pairs["label"][r])
            hist = set(model_train.get(u, ()))
            if y == 1 and i in hist:
                hist = hist - {i}
            feats = aggregate_kg_hcr(
                hist,
                i,
                hcr_index,
                kg_index,
                max_history=max_history,
                mask_no_path=mask_no_path,
            )
            X[r] = np.array([feats[k] for k in names], dtype=np.float32)
        return X

    X_train = build(bundle["train_pairs"], f"{tag} train")
    X_val = build(bundle["val_pairs"], f"{tag} val")
    X_test = build(bundle["test_pairs"], f"{tag} test")
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "X_test.npy", X_test)
    names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    # dump relation schema used
    (out_dir / "relations.json").write_text(
        json.dumps(data.relation_names, indent=2), encoding="utf-8"
    )
    return {"names": names, "train": X_train, "val": X_val, "test": X_test}
