#!/usr/bin/env python3
"""Build full-scale R3 hardneg train pairs + A5/H3/LEG features for TRUE FINAL.

Uses the SAME positives as outputs/lastfm_star/features/train_pairs.npz (~500k),
replacing random negatives with R3 mixture (2 hard / 1 pop / 1 random).

Env:
  TFHN_MAX_POS=0          # 0 = all positives from train_pairs
  TFHN_SEED=303           # sampler RNG (pairs frozen once; train seeds separate)
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_rank_v2" / "src"))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG  # noqa: E402
from scripts.run_lastfm_noleak_fullrank_validation_v1 import h3_and_l2  # noqa: E402
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import (  # noqa: E402
    CLEAN_V2_TABULAR_FEATURE_NAMES,
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402

from rankv2.pos_hard_negatives import PositiveLocalHardNegativeSampler  # noqa: E402
from tfhn.paths import ART, NEIGHBORS_SRC, REP, SPLITS_DIR  # noqa: E402


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    max_pos = int(os.environ.get("TFHN_MAX_POS", "0"))
    n_neg = 4
    seed = int(os.environ.get("TFHN_SEED", "303"))

    neigh_dst = ART / "item_a11_neighbors.npz"
    if not neigh_dst.exists():
        assert NEIGHBORS_SRC.exists(), f"missing neighbors {NEIGHBORS_SRC}"
        shutil.copy2(NEIGHBORS_SRC, neigh_dst)
        print(f"[tfhn-data] copied neighbors → {neigh_dst}", flush=True)

    nn = np.load(neigh_dst)
    neighbors, scores, pop = nn["neighbors"], nn["scores"], nn["pop"]

    print("[tfhn-data] load bundle + original train positives…", flush=True)
    cfg = load_protocol_config(PROTOCOL_CONFIG)
    bundle = load_prepared(cfg, verify=False)
    mt = load_user_sets(SPLITS_DIR / "model_train.txt")
    bundle["model_train"] = mt
    n_items = int(len(pop))
    rng = np.random.default_rng(seed)
    samp = PositiveLocalHardNegativeSampler(
        neighbors=neighbors,
        neighbor_scores=scores,
        item_pop=pop,
        model_train=mt,
        n_items=n_items,
        rng=rng,
        band_lo=0.05,
        band_hi=0.30,
    )

    tp = bundle["train_pairs"]
    y = tp["label"]
    pos_mask = y > 0.5
    pos_u = tp["user_id"][pos_mask].astype(np.int64)
    pos_i = tp["item_id"][pos_mask].astype(np.int64)
    if max_pos > 0 and len(pos_u) > max_pos:
        idx = rng.choice(len(pos_u), size=max_pos, replace=False)
        pos_u, pos_i = pos_u[idx], pos_i[idx]
    print(f"[tfhn-data] positives={len(pos_u)} building hard-neg table…", flush=True)

    users, items, labels = [], [], []
    for u, i in tqdm(zip(pos_u.tolist(), pos_i.tolist()), total=len(pos_u), desc="sample-negs"):
        users.append(u)
        items.append(i)
        labels.append(1)
        out = samp.sample(u, i, n_neg=n_neg)
        for j in out["items"]:
            users.append(u)
            items.append(int(j))
            labels.append(0)
            if int(j) in mt.get(u, set()):
                raise RuntimeError(f"leak neg {j} in hist u={u}")

    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    pairs_path = ART / "hardneg_train_pairs.npz"
    np.savez_compressed(pairs_path, user_id=users, item_id=items, label=labels)
    np.savez_compressed(
        ART / "val_pairs.npz",
        user_id=bundle["val_pairs"]["user_id"],
        item_id=bundle["val_pairs"]["item_id"],
        label=bundle["val_pairs"]["label"],
    )
    print(
        f"[tfhn-data] pairs={len(labels)} pos={int((labels == 1).sum())} "
        f"neg={int((labels == 0).sum())}",
        flush=True,
    )

    print("[tfhn-data] materialize A5/H3/LEG…", flush=True)
    cf = ensure_cross_fit(bundle)
    pop_b = bundle["popularity"]
    kg = bundle["item_kg_degree"]
    n = len(labels)
    feat_dir = ART / "features"
    feat_dir.mkdir(exist_ok=True)
    A = np.lib.format.open_memmap(feat_dir / "A_train.npy", mode="w+", dtype=np.float32, shape=(n, 5))
    H = np.lib.format.open_memmap(feat_dir / "H_train.npy", mode="w+", dtype=np.float32, shape=(n, 3))
    L = np.lib.format.open_memmap(feat_dir / "L_train.npy", mode="w+", dtype=np.float32, shape=(n, 1))
    size_cache: dict[int, np.ndarray] = {}
    chunk = 20000
    for start in tqdm(range(0, n, chunk), desc="features"):
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
                int(u), items[r_idx], mt, pop_b, kg, idx_cf, n_x_sizes=size_cache[fold]
            )
            base = set(mt.get(int(u), ()))
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

    a_scaler = StandardScaler().fit(np.asarray(A))
    h_scaler = StandardScaler().fit(np.asarray(H))
    l_scaler = StandardScaler().fit(np.asarray(L))
    for name, sc in (("a_scaler", a_scaler), ("h_scaler", h_scaler), ("l_scaler", l_scaler)):
        with (feat_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(sc, f)

    meta = {
        "max_pos": max_pos,
        "n_pairs": int(n),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "n_neg_per_pos": n_neg,
        "sampler": "R3_positive_local_A11_semi_hard_5_30_plus_pop_random",
        "band": [0.05, 0.30],
        "mix": {"hard_local": 2, "pop": 1, "random": 1},
        "seed": seed,
        "positives_source": "bundle.train_pairs (TRUE FINAL)",
        "A_names": CLEAN_V2_TABULAR_FEATURE_NAMES,
        "architecture": "HGT+A5+H3+LEG from-scratch target",
        "external": "LOCKED",
    }
    (feat_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (ART / "DATA_READY.flag").write_text("READY\n")
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
