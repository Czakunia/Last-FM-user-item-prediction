"""Frozen H2 HCR feature contract (LASTFM_HCR_H2_FEATURE_CONTRACT_V1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = ROOT / "configs/contracts/LASTFM_HCR_H2_FEATURE_CONTRACT_V1.json"

ShuffleMode = Literal["none", "global", "popularity_stratified"]


def load_h2_contract(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONTRACT
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _reorder_from_cache(
    X: np.ndarray,
    cache_names: list[str],
    contract: dict[str, Any],
) -> np.ndarray:
    cmap = contract["hcr_block"]["cache_column_map"]
    order = contract["hcr_block"]["order"]
    idx = [cache_names.index(cmap[name]) for name in order]
    return X[:, idx].astype(np.float32)


def load_base_and_hcr(
    cfg: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load frozen Stage-A tabular + H2 HCR matrices (train-normalized later)."""

    contract = contract or load_h2_contract(cfg.get("feature_contract"))
    feat_root = Path(cfg["paths"]["features"])

    # Base A
    a_dir = feat_root / "A"
    a_names = json.loads((a_dir / "feature_names.json").read_text(encoding="utf-8"))
    want_a = contract["base_block"]["order"]
    a_idx = [a_names.index(n) for n in want_a]
    A_train = np.load(a_dir / "X_train.npy")[:, a_idx].astype(np.float32)
    A_val = np.load(a_dir / "X_val.npy")[:, a_idx].astype(np.float32)
    A_test = np.load(a_dir / "X_test.npy")[:, a_idx].astype(np.float32)

    # H2 cache (from Stage H)
    h_dir = Path(cfg["paths"].get("hcr_h2_cache", feat_root / "stage_h" / "H2_a11_energy"))
    if not h_dir.is_absolute():
        h_dir = ROOT / h_dir
    h_names = json.loads((h_dir / "feature_names.json").read_text(encoding="utf-8"))
    H_train = _reorder_from_cache(np.load(h_dir / "X_train.npy"), h_names, contract)
    H_val = _reorder_from_cache(np.load(h_dir / "X_val.npy"), h_names, contract)
    H_test = _reorder_from_cache(np.load(h_dir / "X_test.npy"), h_names, contract)

    return {
        "contract": contract,
        "base_names": want_a,
        "hcr_names": list(contract["hcr_block"]["order"]),
        "A_train": A_train,
        "A_val": A_val,
        "A_test": A_test,
        "H_train": H_train,
        "H_val": H_val,
        "H_test": H_test,
    }


def shuffle_hcr_block(
    H: np.ndarray,
    *,
    mode: ShuffleMode,
    popularity: np.ndarray | None,
    item_ids: np.ndarray | None,
    seed: int,
    n_strata: int = 10,
) -> np.ndarray:
    """Shuffle whole HCR rows (never column-wise)."""

    if mode == "none":
        return H
    rng = np.random.default_rng(seed)
    out = H.copy()
    n = len(out)
    if mode == "global":
        perm = rng.permutation(n)
        return out[perm]
    if mode != "popularity_stratified":
        raise ValueError(mode)
    if popularity is None or item_ids is None:
        raise ValueError("popularity_stratified shuffle requires popularity and item_ids")
    pops = popularity[item_ids.astype(np.int64)].astype(np.float64)
    # decile edges on this split
    edges = np.quantile(pops, np.linspace(0, 1, n_strata + 1))
    edges[0] -= 1.0
    edges[-1] += 1.0
    buckets = np.digitize(pops, edges) - 1
    for b in range(n_strata):
        idx = np.where(buckets == b)[0]
        if len(idx) < 2:
            continue
        out[idx] = out[rng.permutation(idx)]
    return out
