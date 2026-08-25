#!/usr/bin/env python3
"""Sealed eval for POST_HOC convergence checkpoint (seed 101 @ best epoch 59).

Does NOT overwrite original epoch-36 sealed results under LASTFM_EXTERNAL_BENCHMARK_20260824/.
Writes only to LASTFM_EXTERNAL_CONVERGENCE_REFIT_20260824/sealed_eval_seed101_epoch59/.

Requires CONFIRM_UNSEAL_LASTFM_TEST=YES (same guard as EXT_04).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if os.environ.get("CONFIRM_UNSEAL_LASTFM_TEST") != "YES":
    raise RuntimeError(
        "Sealed Last-FM* test evaluation is disabled. "
        "Set CONFIRM_UNSEAL_LASTFM_TEST=YES only after protocol approval."
    )

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_repos" / "IntentAwareRS"))

from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    AUDIT_DIR,
    build_external_bundle,
    build_external_training_bundle,
    build_urm,
    git_commit,
    sha256_file,
    utc_now,
    write_json,
)
from scripts._lastfm_sealed_test_eval_20260824 import (  # noqa: E402
    TrueFinalOnTheFlyRecommender,
    holdout_df_to_dict,
    primary_from_holdout,
    score_user_catalog_external,
)
from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import build_race_model  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import neighborhood_sizes  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout  # noqa: E402

SEED = 101
EPOCH = 59
CKPT = (
    ROOT
    / "LASTFM_EXTERNAL_CONVERGENCE_REFIT_20260824"
    / "models"
    / f"FULL_TRAIN_SEED{SEED}_EPOCH{EPOCH}"
    / "model.pt"
)
OUT_DIR = (
    ROOT
    / "LASTFM_EXTERNAL_CONVERGENCE_REFIT_20260824"
    / f"sealed_eval_seed{SEED}_epoch{EPOCH}"
)
UPSTREAM_CUTOFFS = [1, 5, 10, 20, 40, 50, 100]


def main() -> None:
    if not CKPT.exists():
        raise FileNotFoundError(f"missing convergence checkpoint: {CKPT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a_path = OUT_DIR / "block_a.json"
    b_path = OUT_DIR / "block_b.json"
    if a_path.exists() and b_path.exists():
        a_res = json.loads(a_path.read_text(encoding="utf-8"))
        b_res = json.loads(b_path.read_text(encoding="utf-8"))
        print(f"[conv-sealed] reuse A NDCG@20={a_res['primary']['NDCG@20']:.6f}", flush=True)
        print(f"[conv-sealed] reuse B NDCG@20={b_res['NDCG@20']:.6f}", flush=True)
        return

    print(f"[conv-sealed] seed={SEED} epoch={EPOCH} ckpt={CKPT}", flush=True)
    print(f"[conv-sealed] sha256={sha256_file(CKPT)}", flush=True)
    print("[conv-sealed] does NOT touch LASTFM_EXTERNAL_BENCHMARK sealed artifacts", flush=True)

    bundle = build_external_bundle()
    prepared = build_external_training_bundle(bundle)
    urm_train = build_urm(bundle.train_external, bundle.n_users, bundle.n_items)
    urm_test = build_urm(bundle.sealed_test, bundle.n_users, bundle.n_items)

    # Prefer convergence scalers if present; else external audit scalers.
    seed_dir = CKPT.parent
    a_scaler_path = seed_dir / "a_scaler.pkl"
    h_scaler_path = seed_dir / "h_scaler.pkl"
    if not a_scaler_path.exists():
        a_scaler_path = AUDIT_DIR / "external_a_scaler.pkl"
        h_scaler_path = AUDIT_DIR / "external_h_scaler.pkl"
    with a_scaler_path.open("rb") as f:
        a_scaler = pickle.load(f)
    with h_scaler_path.open("rb") as f:
        h_scaler = pickle.load(f)

    cf = ensure_cross_fit(prepared)
    graph = load_data_and_typed_graph(
        bundle.cfg,
        bundle.train_external,
        max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(
        str(bundle.cfg.get("models", {}).get("architecture", {}).get("device") or "auto")
    )
    size_cache: dict[int, np.ndarray] = {}

    def n_x_sizes_for(u: int) -> np.ndarray:
        fold = int(cf.user_to_fold[int(u)])
        if fold not in size_cache:
            size_cache[fold] = neighborhood_sizes(cf.fold_indices[fold])
        return size_cache[fold]

    t0 = time.time()
    model = build_race_model(prepared, graph, device, d=64, layers=2, heads=2)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    z = model.node_z()
    ctx = {
        "device": device,
        "model": model,
        "z": z,
        "a_mean": a_scaler.mean_.astype(np.float32),
        "a_scale": a_scaler.scale_.astype(np.float32),
        "h_mean": h_scaler.mean_.astype(np.float32),
        "h_scale": h_scaler.scale_.astype(np.float32),
    }

    def score_fn(u: int, hist: set[int], _ctx=ctx):
        return score_user_catalog_external(
            u,
            n_items=bundle.n_items,
            item_offset=item_offset,
            hist=hist,
            prepared=prepared,
            cf=cf,
            ctx=_ctx,
            n_x_sizes=n_x_sizes_for(u),
        )

    block_b_acc: dict[str, float] = {}
    block_b_n = [0]
    adapter = TrueFinalOnTheFlyRecommender(
        urm_train,
        score_fn=score_fn,
        train_external=bundle.train_external,
        sealed_test=bundle.sealed_test,
        block_b_acc=block_b_acc,
        block_b_n=block_b_n,
    )
    evaluator = EvaluatorHoldout(urm_test, UPSTREAM_CUTOFFS, exclude_seen=True, verbose=True)
    df, raw = evaluator.evaluateRecommender(adapter)
    a_res = {
        "seed": SEED,
        "epoch_count": EPOCH,
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "artifact_sha256": sha256_file(CKPT),
        "primary": primary_from_holdout(df),
        "all_cutoffs": holdout_df_to_dict(df),
        "raw_string": str(raw),
        "seconds": time.time() - t0,
        "NOTE": "POST_HOC convergence checkpoint; original epoch-36 sealed results unchanged",
        "timestamp": utc_now(),
        "git_commit": git_commit(),
    }
    write_json(a_path, a_res)
    n_u = max(block_b_n[0], 1)
    b_res = {
        "seed": SEED,
        "epoch_count": EPOCH,
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "NDCG@5": block_b_acc.get("NDCG@5", 0.0) / n_u,
        "NDCG@10": block_b_acc.get("NDCG@10", 0.0) / n_u,
        "NDCG@20": block_b_acc.get("NDCG@20", 0.0) / n_u,
        "Recall@20": block_b_acc.get("Recall@20", 0.0) / n_u,
        "MRR": block_b_acc.get("MRR", 0.0) / n_u,
        "HitRate@20": block_b_acc.get("HitRate@20", 0.0) / n_u,
        "n_users": block_b_n[0],
        "seconds": time.time() - t0,
        "timestamp": utc_now(),
    }
    write_json(b_path, b_res)

    # Compare vs original epoch-36 sealed seed 101 if present
    legacy_a = (
        ROOT
        / "LASTFM_EXTERNAL_BENCHMARK_20260824"
        / "sealed_test_results"
        / "raw"
        / "block_a_TRUE_FINAL_seed101.json"
    )
    compare = {"epoch59": a_res["primary"]}
    if legacy_a.exists():
        leg = json.loads(legacy_a.read_text(encoding="utf-8"))
        compare["epoch36_legacy"] = leg.get("primary", leg)
        compare["delta_NDCG@20"] = float(a_res["primary"]["NDCG@20"]) - float(
            compare["epoch36_legacy"]["NDCG@20"]
        )
        compare["delta_RECALL@20"] = float(a_res["primary"]["RECALL@20"]) - float(
            compare["epoch36_legacy"]["RECALL@20"]
        )
    write_json(OUT_DIR / "compare_vs_epoch36.json", compare)

    print(
        f"[conv-sealed] Block A NDCG@20={a_res['primary']['NDCG@20']:.6f} "
        f"Recall@20={a_res['primary']['RECALL@20']:.6f}",
        flush=True,
    )
    print(
        f"[conv-sealed] Block B NDCG@20={b_res['NDCG@20']:.6f} "
        f"Recall@20={b_res['Recall@20']:.6f} MRR={b_res['MRR']:.6f}",
        flush=True,
    )
    if "delta_NDCG@20" in compare:
        print(
            f"[conv-sealed] Δ vs epoch36 sealed: NDCG@20={compare['delta_NDCG@20']:+.6f} "
            f"Recall@20={compare['delta_RECALL@20']:+.6f}",
            flush=True,
        )
    print(f"[conv-sealed] wrote {OUT_DIR}", flush=True)
    empty_cache()


if __name__ == "__main__":
    main()
