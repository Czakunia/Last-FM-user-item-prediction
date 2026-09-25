#!/usr/bin/env python3
"""Lightweight TRAIN_EXTERNAL R3 feature materialization (no torch).

Writes artifacts/external_refit/features/{A,H,L}_train.npy + scalers + DATA_READY.flag.
Safe to re-run; resumes only when DATA_READY is absent.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ART = EXP / "artifacts"
OUT = ART / "external_refit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "src"))

from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    build_external_bundle,
    empty_pair_table,
    utc_now,
    write_json,
)
from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix  # noqa: E402
from src.lastfm_lp.binary.contingency_tables import build_cooccurrence  # noqa: E402
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex  # noqa: E402
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices  # noqa: E402
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_TOP_K  # noqa: E402
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import (  # noqa: E402
    CLEAN_V2_TABULAR_FEATURE_NAMES,
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402


def h3_and_l2(hist: np.ndarray, cands: np.ndarray, index, *, max_history: int = CLEAN_V2_TOP_K):
    cands = np.asarray(cands, dtype=np.int64).reshape(-1)
    C = int(cands.size)
    h3 = np.zeros((C, 3), dtype=np.float32)
    l2 = np.zeros((C, 1), dtype=np.float32)
    if hist.size == 0 or C == 0:
        return h3, l2
    n11 = index.cooccurrence_block(hist, cands)
    a11, _ = a11_energy_from_n11_matrix(
        n11, index.popularity[hist], index.popularity[cands], index.n_users
    )
    if hist.size <= max_history:
        a_sel = a11
    else:
        idx = deterministic_topk_indices(a11, hist, k=max_history)
        a_sel = np.take_along_axis(a11, idx, axis=0)
    h3 = pool_signed_a11_3d_matrix(a_sel)
    p2 = 0.5 * (3.0 * np.square(a_sel.astype(np.float64)) - 1.0)
    l2[:, 0] = p2.mean(axis=0).astype(np.float32)
    return h3, l2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs_path = OUT / "hardneg_train_pairs.npz"
    feat_dir = OUT / "features"
    ready = OUT / "DATA_READY.flag"
    if ready.exists() and (feat_dir / "a_scaler.pkl").exists():
        print("[tfhn-ext-feat] already ready — skip", flush=True)
        return
    if not pairs_path.exists():
        raise RuntimeError(f"missing {pairs_path}; run sealed external sampling first")

    print("[tfhn-ext-feat] load bundle + pairs…", flush=True)
    bundle = build_external_bundle()
    te = bundle.train_external
    pairs = np.load(pairs_path)
    users = pairs["user_id"]
    items = pairs["item_id"]
    labels = pairs["label"]
    n = int(len(labels))
    print(f"[tfhn-ext-feat] n_pairs={n}", flush=True)

    idx_path = OUT / "te_pairwise_index.pkl"
    if idx_path.exists():
        print("[tfhn-ext-feat] load te_pairwise_index.pkl", flush=True)
        with idx_path.open("rb") as f:
            index = pickle.load(f)["index"]
    else:
        print("[tfhn-ext-feat] build cooc+index…", flush=True)
        cooc = build_cooccurrence(te, bundle.n_items, max_history_for_pairs=60, seed=2026)
        index = PairwiseStatsIndex(
            cooc,
            bundle.popularity,
            n_users=len(te),
            smoothing=float(bundle.cfg.get("hcr", {}).get("smoothing", 0.5)),
        )
        with idx_path.open("wb") as f:
            pickle.dump(
                {"index": index, "training_hash": bundle.training_hash},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    prepared = {
        "cfg": bundle.cfg,
        "model_train": te,
        "popularity": bundle.popularity,
        "item_kg_degree": bundle.item_kg_degree,
        "index": index,
        "train_pairs": {"user_id": users, "item_id": items, "label": labels},
        "val_pairs": empty_pair_table(),
        "test_pairs": empty_pair_table(),
        "eval_users": bundle.eval_users_test,
    }
    print("[tfhn-ext-feat] ensure_cross_fit…", flush=True)
    cf = ensure_cross_fit(prepared)

    feat_dir.mkdir(parents=True, exist_ok=True)
    # rewrite memmaps from scratch (partial files from OOM kills are unsafe)
    for name in ("A_train.npy", "H_train.npy", "L_train.npy"):
        p = feat_dir / name
        if p.exists():
            p.unlink()

    A = np.lib.format.open_memmap(feat_dir / "A_train.npy", mode="w+", dtype=np.float32, shape=(n, 5))
    H = np.lib.format.open_memmap(feat_dir / "H_train.npy", mode="w+", dtype=np.float32, shape=(n, 3))
    L = np.lib.format.open_memmap(feat_dir / "L_train.npy", mode="w+", dtype=np.float32, shape=(n, 1))
    pop_b = bundle.popularity
    kg = bundle.item_kg_degree
    size_cache: dict[int, np.ndarray] = {}
    chunk = 20000
    for start in tqdm(range(0, n, chunk), desc="ext-features"):
        end = min(start + chunk, n)
        groups: dict[int, list[int]] = defaultdict(list)
        for r in range(start, end):
            groups[int(users[r])].append(r)
        for u, rows in groups.items():
            idx_cf = clean_v2_index_for(cf, int(u))
            fold = cf.user_to_fold.get(int(u), -1)
            if fold not in size_cache:
                size_cache[fold] = neighborhood_sizes(idx_cf)
            r_idx = np.asarray(rows, dtype=np.int64)
            A[r_idx] = vectorized_clean_v2_A_for_user(
                int(u), items[r_idx], te, pop_b, kg, idx_cf, n_x_sizes=size_cache[fold]
            )
            base = set(te.get(int(u), ()))
            neg_rows = [r for r in rows if int(labels[r]) == 0]
            pos_rows = [r for r in rows if int(labels[r]) == 1]
            hist_full = np.asarray(sorted(base), dtype=np.int64)
            if neg_rows and hist_full.size:
                r_idx_n = np.asarray(neg_rows, dtype=np.int64)
                h3, l2 = h3_and_l2(hist_full, items[r_idx_n], idx_cf)
                H[r_idx_n] = h3
                L[r_idx_n] = l2
            for r in pos_rows:
                i = int(items[r])
                hist = set(base)
                hist.discard(i)
                hist_arr = np.asarray(sorted(hist), dtype=np.int64)
                h3, l2 = h3_and_l2(hist_arr, np.asarray([i], dtype=np.int64), idx_cf)
                H[r] = h3[0]
                L[r] = l2[0]
        A.flush()
        H.flush()
        L.flush()

    print("[tfhn-ext-feat] fit scalers (chunked)…", flush=True)
    a_sc = StandardScaler()
    h_sc = StandardScaler()
    l_sc = StandardScaler()
    for start in tqdm(range(0, n, 200_000), desc="scaler-partial"):
        end = min(start + 200_000, n)
        a_sc.partial_fit(np.asarray(A[start:end]))
        h_sc.partial_fit(np.asarray(H[start:end]))
        l_sc.partial_fit(np.asarray(L[start:end]))
    for name, sc in (("a_scaler", a_sc), ("h_scaler", h_sc), ("l_scaler", l_sc)):
        with (feat_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(sc, f)

    write_json(
        feat_dir / "meta.json",
        {
            "fit_population": "TRAIN_EXTERNAL",
            "n_pairs": n,
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
            "sampler": "R3",
            "A_names": CLEAN_V2_TABULAR_FEATURE_NAMES,
            "training_hash": bundle.training_hash,
            "generated": utc_now(),
        },
    )
    ready.write_text("READY\n")
    print("[tfhn-ext-feat] DATA_READY", flush=True)


if __name__ == "__main__":
    main()
