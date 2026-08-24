"""Stage P: flat KG-path descriptors × orthonormal a11/energy (no attention/gate)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
from src.lastfm_lp.binary.user_hcr_aggregation import _truncate_history
from src.lastfm_lp.kg.path_index import KGPathIndex

# Pair-level structural channels (counts already log1p where needed)
STRUCT_PAIR_CHANNELS = [
    "kg_connected",
    "kg_shortest_path",
    "log1p_path_count_len2",
    "log1p_path_count_len3",
    "log1p_shared_entity",
    "kg_relation_overlap",
]

# a11 × path (signed) and energy × path (non-negative)
A11_INTER_CHANNELS = [
    "a11_x_connected",
    "a11_x_path_len2",
    "a11_x_path_len3",
    "a11_x_shared_entity",
    "a11_x_relation_overlap",
]
ENERGY_INTER_CHANNELS = [
    "energy_x_connected",
    "energy_x_path_len2",
    "energy_x_path_len3",
    "energy_x_shared_entity",
    "energy_x_relation_overlap",
]

POOL4 = ("mean", "max", "top3", "wmean")
POOL5 = ("mean", "max", "top3", "wmean", "min")  # min for signed


def _pool4(X: np.ndarray, w: np.ndarray, top_k: int = 3) -> np.ndarray:
    """X: (n,d) → 4d [mean,max,top3,wmean]."""
    d = X.shape[1]
    out = np.zeros(4 * d, dtype=np.float32)
    if len(X) == 0:
        return out
    out[0:d] = X.mean(axis=0)
    out[d : 2 * d] = X.max(axis=0)
    key = X[:, 0]
    order = np.argsort(-key)
    k = min(int(top_k), len(order))
    out[2 * d : 3 * d] = X[order[:k]].mean(axis=0)
    wsum = float(w.sum())
    if wsum > 0:
        out[3 * d : 4 * d] = (X * w[:, None]).sum(axis=0) / wsum
    else:
        out[3 * d : 4 * d] = out[0:d]
    return out


def _pool5_signed(X: np.ndarray, w: np.ndarray, top_k: int = 3) -> np.ndarray:
    """Signed channels: mean/max/top3/wmean/min → 5d."""
    d = X.shape[1]
    out = np.zeros(5 * d, dtype=np.float32)
    if len(X) == 0:
        return out
    out[0:d] = X.mean(axis=0)
    out[d : 2 * d] = X.max(axis=0)
    key = np.abs(X[:, 0])
    order = np.argsort(-key)
    k = min(int(top_k), len(order))
    out[2 * d : 3 * d] = X[order[:k]].mean(axis=0)
    wsum = float(w.sum())
    if wsum > 0:
        out[3 * d : 4 * d] = (X * w[:, None]).sum(axis=0) / wsum
    else:
        out[3 * d : 4 * d] = out[0:d]
    out[4 * d : 5 * d] = X.min(axis=0)
    return out


def struct_feature_names() -> list[str]:
    names = [f"{ch}_{p}" for p in POOL4 for ch in STRUCT_PAIR_CHANNELS]
    names.append("path_coverage")
    return names


def inter_feature_names() -> list[str]:
    names = [f"{ch}_{p}" for p in POOL5 for ch in A11_INTER_CHANNELS]
    names += [f"{ch}_{p}" for p in POOL4 for ch in ENERGY_INTER_CHANNELS]
    names += ["a11_on_path_mean", "a11_off_path_mean", "energy_on_path_mean", "energy_off_path_mean"]
    return names


def h2_split_feature_names() -> list[str]:
    """H2-style pools of a11/energy separately on KG-supported vs unsupported history."""
    bases = ["hcr_a11", "hcr_energy"]
    names = []
    for side in ("kg_plus", "kg_minus"):
        for p in POOL4:
            for b in bases:
                names.append(f"{side}_{p}_{b}")
        # signed min for a11 on each side
        names.append(f"{side}_min_hcr_a11")
    return names


def _pair_row(
    j: int,
    i: int,
    hcr_index: PairwiseStatsIndex,
    kg_index: KGPathIndex,
    *,
    decouple_rng: np.random.Generator | None = None,
    decouple_a11: float | None = None,
    decouple_energy: float | None = None,
) -> dict[str, float]:
    m = hcr_index.pair(j, i)
    kg = kg_index.pair_features(j, i)
    if decouple_rng is not None and decouple_a11 is not None:
        a11 = float(decouple_a11)
        energy = float(decouple_energy if decouple_energy is not None else a11 * a11)
    else:
        a11 = float(m.hcr_a11)
        energy = float(m.hcr_energy)
    connected = float(kg["kg_connected"])
    lp2 = float(kg["log1p_path_count_len2"])
    lp3 = float(kg["log1p_path_count_len3"])
    shared = float(np.log1p(kg["kg_shared_entity_count"]))
    rel = float(kg["kg_relation_overlap"])
    support = float(m.n11)
    return {
        "kg_connected": connected,
        "kg_shortest_path": float(kg["kg_shortest_path"]),
        "log1p_path_count_len2": lp2,
        "log1p_path_count_len3": lp3,
        "log1p_shared_entity": shared,
        "kg_relation_overlap": rel,
        "a11": a11,
        "energy": energy,
        "support": support,
        "a11_x_connected": a11 * connected,
        "a11_x_path_len2": a11 * lp2,
        "a11_x_path_len3": a11 * lp3,
        "a11_x_shared_entity": a11 * shared,
        "a11_x_relation_overlap": a11 * rel,
        "energy_x_connected": energy * connected,
        "energy_x_path_len2": energy * lp2,
        "energy_x_path_len3": energy * lp3,
        "energy_x_shared_entity": energy * shared,
        "energy_x_relation_overlap": energy * rel,
    }


def aggregate_path_h2(
    history_items: Iterable[int],
    candidate_item: int,
    hcr_index: PairwiseStatsIndex,
    kg_index: KGPathIndex,
    *,
    max_history: int = 25,
    decouple_rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Return struct / inter / h2_split vectors for one (user-history, candidate)."""
    hist = _truncate_history(history_items, max_history)
    sn = struct_feature_names()
    inn = inter_feature_names()
    hn = h2_split_feature_names()
    if not hist:
        return {
            "struct": np.zeros(len(sn), dtype=np.float32),
            "inter": np.zeros(len(inn), dtype=np.float32),
            "h2_split": np.zeros(len(hn), dtype=np.float32),
        }

    # Optional HCR–path decoupling: keep path of (j,i), swap a11 from random other history pair
    rows = []
    raw_a11 = []
    raw_e = []
    for j in hist:
        m = hcr_index.pair(int(j), int(candidate_item))
        raw_a11.append(float(m.hcr_a11))
        raw_e.append(float(m.hcr_energy))
    if decouple_rng is not None and len(hist) > 1:
        perm = decouple_rng.permutation(len(hist))
        for t, j in enumerate(hist):
            rows.append(
                _pair_row(
                    int(j),
                    int(candidate_item),
                    hcr_index,
                    kg_index,
                    decouple_rng=decouple_rng,
                    decouple_a11=raw_a11[perm[t]],
                    decouple_energy=raw_e[perm[t]],
                )
            )
    else:
        for j in hist:
            rows.append(_pair_row(int(j), int(candidate_item), hcr_index, kg_index))

    w = np.log1p(np.array([r["support"] for r in rows], dtype=np.float64))
    connected = np.array([r["kg_connected"] for r in rows], dtype=np.float64)

    S = np.array([[r[c] for c in STRUCT_PAIR_CHANNELS] for r in rows], dtype=np.float64)
    struct = np.concatenate([_pool4(S, w), np.array([connected.mean()], dtype=np.float32)])

    A = np.array([[r[c] for c in A11_INTER_CHANNELS] for r in rows], dtype=np.float64)
    E = np.array([[r[c] for c in ENERGY_INTER_CHANNELS] for r in rows], dtype=np.float64)
    a11s = np.array([r["a11"] for r in rows], dtype=np.float64)
    energies = np.array([r["energy"] for r in rows], dtype=np.float64)
    on = connected > 0.5
    off = ~on
    inter = np.concatenate(
        [
            _pool5_signed(A, w),
            _pool4(E, w),
            np.array(
                [
                    float(a11s[on].mean()) if on.any() else 0.0,
                    float(a11s[off].mean()) if off.any() else 0.0,
                    float(energies[on].mean()) if on.any() else 0.0,
                    float(energies[off].mean()) if off.any() else 0.0,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)

    # H2 split: pool [a11, energy] on kg+ / kg-
    def _side(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            # 4*2 + 1 min_a11
            return np.zeros(9, dtype=np.float32)
        X = np.stack([a11s[mask], energies[mask]], axis=1)
        ww = w[mask]
        pooled = _pool4(X, ww)  # 8
        return np.concatenate([pooled, np.array([float(X[:, 0].min())], dtype=np.float32)])

    h2_split = np.concatenate([_side(on), _side(off)]).astype(np.float32)

    assert struct.shape[0] == len(sn)
    assert inter.shape[0] == len(inn)
    assert h2_split.shape[0] == len(hn)
    return {"struct": struct.astype(np.float32), "inter": inter, "h2_split": h2_split}
