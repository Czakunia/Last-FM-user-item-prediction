"""Materialize Stage-P path×H2 feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.lastfm_lp.data.load_kgat_lastfm import load_lastfm
from src.lastfm_lp.features.path_h2_aggregation import (
    aggregate_path_h2,
    h2_split_feature_names,
    inter_feature_names,
    struct_feature_names,
)
from src.lastfm_lp.kg.path_index import KGPathIndex


def materialize_path_h2(
    bundle: dict[str, Any],
    *,
    max_history: int = 25,
    tag: str = "stage_p",
    force: bool = False,
) -> dict[str, Any]:
    cfg = bundle["cfg"]
    out_root = Path(cfg["paths"].get("features", "outputs/lastfm/features")) / tag
    struct_dir = out_root / "P_struct"
    inter_dir = out_root / "C_inter"
    split_dir = out_root / "H2_kg_split"
    for d in (struct_dir, inter_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    sn, inn, hn = struct_feature_names(), inter_feature_names(), h2_split_feature_names()
    ready = (
        not force
        and (struct_dir / "X_train.npy").exists()
        and (inter_dir / "X_train.npy").exists()
        and (split_dir / "X_train.npy").exists()
    )
    if ready:
        return {
            "struct_names": sn,
            "inter_names": inn,
            "h2_split_names": hn,
            "struct": {
                "train": np.load(struct_dir / "X_train.npy"),
                "val": np.load(struct_dir / "X_val.npy"),
                "test": np.load(struct_dir / "X_test.npy"),
            },
            "inter": {
                "train": np.load(inter_dir / "X_train.npy"),
                "val": np.load(inter_dir / "X_val.npy"),
                "test": np.load(inter_dir / "X_test.npy"),
            },
            "h2_split": {
                "train": np.load(split_dir / "X_train.npy"),
                "val": np.load(split_dir / "X_val.npy"),
                "test": np.load(split_dir / "X_test.npy"),
            },
            "out_root": out_root,
        }

    data = load_lastfm(cfg["data"]["path"])
    kg_index = KGPathIndex(data.kg, data.n_entities, data.n_relations)
    hcr_index = bundle["index"]
    model_train = bundle["model_train"]

    def build(pairs: dict[str, np.ndarray], desc: str):
        n = len(pairs["label"])
        Xs = np.zeros((n, len(sn)), dtype=np.float32)
        Xi = np.zeros((n, len(inn)), dtype=np.float32)
        Xh = np.zeros((n, len(hn)), dtype=np.float32)
        for r in tqdm(range(n), desc=desc, mininterval=2.0):
            u = int(pairs["user_id"][r])
            i = int(pairs["item_id"][r])
            y = int(pairs["label"][r])
            hist = set(model_train.get(u, ()))
            if y == 1 and i in hist:
                hist = hist - {i}
            agg = aggregate_path_h2(
                hist, i, hcr_index, kg_index, max_history=max_history
            )
            Xs[r] = agg["struct"]
            Xi[r] = agg["inter"]
            Xh[r] = agg["h2_split"]
        return Xs, Xi, Xh

    def save(d: Path, names: list[str], tr, va, te):
        np.save(d / "X_train.npy", tr)
        np.save(d / "X_val.npy", va)
        np.save(d / "X_test.npy", te)
        (d / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")

    print(f"Materializing Stage-P path×H2 caches under {out_root}")
    tr_s, tr_i, tr_h = build(bundle["train_pairs"], f"{tag} train")
    va_s, va_i, va_h = build(bundle["val_pairs"], f"{tag} val")
    te_s, te_i, te_h = build(bundle["test_pairs"], f"{tag} test")
    save(struct_dir, sn, tr_s, va_s, te_s)
    save(inter_dir, inn, tr_i, va_i, te_i)
    save(split_dir, hn, tr_h, va_h, te_h)
    (out_root / "relations.json").write_text(
        json.dumps(data.relation_names, indent=2), encoding="utf-8"
    )
    return {
        "struct_names": sn,
        "inter_names": inn,
        "h2_split_names": hn,
        "struct": {"train": tr_s, "val": va_s, "test": te_s},
        "inter": {"train": tr_i, "val": va_i, "test": te_i},
        "h2_split": {"train": tr_h, "val": va_h, "test": te_h},
        "out_root": out_root,
    }
