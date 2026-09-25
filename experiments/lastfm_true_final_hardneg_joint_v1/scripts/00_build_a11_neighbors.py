#!/usr/bin/env python3
"""Rebuild item_a11_neighbors.npz (original file was deleted with lastfm_rank_v2).

Shared A11 top-100 from model_train co-occurrence, signed phi (= a11).
This is the pool Graph50 called ``A11_top100_shared``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1" / "src"))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG  # noqa: E402
from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix  # noqa: E402
from src.lastfm_lp.binary.contingency_tables import (  # noqa: E402
    build_cooccurrence,
    cooc_row_arrays,
)
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from tfhn.paths import ART, NEIGHBORS_SRC, SPLITS_DIR  # noqa: E402

TOP_N = int(os.environ.get("TFHN_A11_TOP_N", "100"))
COOC_SEED = int(os.environ.get("TFHN_COOC_SEED", "2026"))
MAX_HIST = int(os.environ.get("TFHN_COOC_MAX_HIST", "60"))


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    cfg = load_protocol_config(PROTOCOL_CONFIG)
    bundle = load_prepared(cfg, verify=False)
    mt = load_user_sets(SPLITS_DIR / "model_train.txt")
    pop = np.asarray(bundle["popularity"], dtype=np.int32).reshape(-1)
    n_items = int(len(pop))
    n_users = int(len(mt))
    print(
        f"[a11-nn] n_users={n_users} n_items={n_items} top_n={TOP_N} "
        f"max_hist={MAX_HIST} seed={COOC_SEED}",
        flush=True,
    )
    cooc = build_cooccurrence(mt, n_items, max_history_for_pairs=MAX_HIST, seed=COOC_SEED)
    neighbors = np.full((n_items, TOP_N), -1, dtype=np.int32)
    scores = np.zeros((n_items, TOP_N), dtype=np.float32)
    n_empty = 0
    for i in range(n_items):
        cols, data = cooc_row_arrays(cooc, i)
        if cols.size == 0:
            n_empty += 1
            continue
        keep = cols != i
        cols = cols[keep]
        data = data[keep]
        if cols.size == 0:
            n_empty += 1
            continue
        a11, _ = a11_energy_from_n11_matrix(
            np.asarray(data, dtype=np.float64).reshape(1, -1),
            np.asarray([pop[i]], dtype=np.int32),
            pop[cols],
            n_users,
        )
        a11 = np.asarray(a11, dtype=np.float64).reshape(-1)
        order = np.argsort(-a11, kind="stable")[:TOP_N]
        n_keep = int(order.size)
        neighbors[i, :n_keep] = cols[order].astype(np.int32)
        scores[i, :n_keep] = a11[order].astype(np.float32)
        if (i + 1) % 5000 == 0:
            print(f"[a11-nn] scored {i + 1}/{n_items}", flush=True)

    out = ART / "item_a11_neighbors.npz"
    payload = dict(
        neighbors=neighbors,
        scores=scores,
        pop=pop,
        top_n=np.asarray([TOP_N], dtype=np.int32),
    )
    np.savez_compressed(out, **payload)
    if out.resolve() != NEIGHBORS_SRC.resolve():
        NEIGHBORS_SRC.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(NEIGHBORS_SRC, **payload)
    meta = {
        "path": str(out),
        "n_items": n_items,
        "n_users": n_users,
        "top_n": TOP_N,
        "n_empty": n_empty,
        "mean_degree": float((neighbors >= 0).sum(axis=1).mean()),
        "cooc_seed": COOC_SEED,
        "max_history_for_pairs": MAX_HIST,
        "score": "signed_a11_phi",
        "reconstructed": True,
        "original_sha256": "a28f73afbeb308c96aa2a636a5fb1ff6883de9f08bc32408735ebf64042cd158",
        "note": "original lastfm_rank_v2 neighbors file was deleted; this is a rebuild",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
