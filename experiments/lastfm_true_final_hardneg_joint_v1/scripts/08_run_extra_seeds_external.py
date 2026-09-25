#!/usr/bin/env python3
"""EXTRA sealed external for seeds 404/505 — NOT part of primary hardneg claim.

Requires FULLCAT.json for 404/505 (DEV selection). Writes under:
  LASTFM_TRUE_FINAL/JOINT_HARDNEG_R3_FROM_SCRATCH_V1/08_EXTRA_SEEDS_404_505/
Does not touch 07_EXTERNAL / primary mean / easy TRUE FINAL.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ART = EXP / "artifacts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "src"))

from tfhn.paths import JOINT_OUT  # noqa: E402
from tfhn.sealed_external import (  # noqa: E402
    EASY_EXT_MEAN_NDCG20,
    ensure_hardneg_external_data,
    refit_seed,
    run_sealed_eval,
    scale,
    subsample_hardneg_groups,
    EXT_MAX_POS,
    N_NEG,
    SAMPLER_SEED,
)
from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    build_external_bundle,
    utc_now,
    write_json,
)
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402
import numpy as np  # noqa: E402
import pickle  # noqa: E402
from tqdm import tqdm  # noqa: E402


EXTRA_SEEDS = (404, 505)
OUT = JOINT_OUT / "08_EXTRA_SEEDS_404_505"


def main() -> None:
    if os.environ.get("CONFIRM_UNSEAL_LASTFM_TEST") != "YES":
        raise RuntimeError("Set CONFIRM_UNSEAL_LASTFM_TEST=YES")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EXTRA_ONLY_NOT_PRIMARY_CLAIM.txt").write_text(
        "Seeds 404/505 are post-unseal exploratory extras.\n"
        "They MUST NOT be mixed into the primary hardneg external mean (101/202/303).\n"
        f"generated={utc_now()}\n",
        encoding="utf-8",
    )

    selected = {}
    for seed in EXTRA_SEEDS:
        fc_path = ART / f"seed{seed}" / "FULLCAT.json"
        if not fc_path.exists():
            raise RuntimeError(f"missing {fc_path} — run DEV FULLCAT for seed {seed} first")
        fc = json.loads(fc_path.read_text())
        selected[seed] = int(fc["selected_epoch"])
        write_json(OUT / f"dev_fullcat_seed{seed}.json", fc)
    print(f"[tfhn-extra] selected_epochs={selected}", flush=True)

    # Temporarily point sealed_external.ext_root via monkeypatch of run outputs
    import tfhn.sealed_external as se

    def extra_root() -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        return OUT

    se.ext_root = extra_root  # type: ignore[assignment]

    bundle = build_external_bundle()
    write_json(
        OUT / "external_bundle_meta.json",
        {
            "training_hash": bundle.training_hash,
            "test_hash": bundle.test_hash,
            "EXTRA_SEEDS": list(EXTRA_SEEDS),
            "PRIMARY_CLAIM": False,
            "NOTE": "Exploratory extras after primary 101/202/303 unseal",
        },
    )

    data = ensure_hardneg_external_data(bundle)
    pairs_npz = np.load(data["pairs_path"])
    users_f, items_f, labels_f, row_idx = subsample_hardneg_groups(
        pairs_npz["user_id"],
        pairs_npz["item_id"],
        pairs_npz["label"],
        max_pos=EXT_MAX_POS,
        seed=SAMPLER_SEED,
    )
    pairs = {"user_id": users_f, "item_id": items_f, "label": labels_f}
    feat_dir = data["feat_dir"]
    A = np.load(feat_dir / "A_train.npy", mmap_mode="r")
    H = np.load(feat_dir / "H_train.npy", mmap_mode="r")
    L = np.load(feat_dir / "L_train.npy", mmap_mode="r")
    with (feat_dir / "a_scaler.pkl").open("rb") as f:
        a_sc = pickle.load(f)
    with (feat_dir / "h_scaler.pkl").open("rb") as f:
        h_sc = pickle.load(f)
    with (feat_dir / "l_scaler.pkl").open("rb") as f:
        l_sc = pickle.load(f)

    n = int(len(labels_f))
    tag = f"pos{n // (1 + N_NEG)}"
    A_s_path = feat_dir / f"A_train_scaled_{tag}.npy"
    H_s_path = feat_dir / f"H_train_scaled_{tag}.npy"
    L_s_path = feat_dir / f"L_train_scaled_{tag}.npy"
    if not (A_s_path.exists() and H_s_path.exists() and L_s_path.exists()):
        A_s = np.lib.format.open_memmap(A_s_path, mode="w+", dtype=np.float32, shape=(n, 5))
        H_s = np.lib.format.open_memmap(H_s_path, mode="w+", dtype=np.float32, shape=(n, 3))
        L_s = np.lib.format.open_memmap(L_s_path, mode="w+", dtype=np.float32, shape=(n, 1))
        for start in tqdm(range(0, n, 200_000), desc="scale-features-extra"):
            end = min(start + 200_000, n)
            src = row_idx[start:end]
            A_s[start:end] = scale(np.asarray(A[src]), a_sc)
            H_s[start:end] = scale(np.asarray(H[src]), h_sc)
            L_s[start:end] = scale(np.asarray(L[src]), l_sc)
        A_s.flush()
        H_s.flush()
        L_s.flush()
        del A_s, H_s, L_s
    A_s = np.load(A_s_path, mmap_mode="r")
    H_s = np.load(H_s_path, mmap_mode="r")
    L_s = np.load(L_s_path, mmap_mode="r")

    graph = load_data_and_typed_graph(
        bundle.cfg,
        bundle.train_external,
        max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(os.environ.get("TFHN_DEVICE", "cpu"))

    # Patch run_sealed_eval seed loop by temporarily replacing SEEDS usage —
    # implement local eval call with selected only.
    refit_metas = []
    for seed in EXTRA_SEEDS:
        n_epochs = int(selected[seed]) + 1
        meta = refit_seed(
            seed=seed,
            n_epochs=n_epochs,
            bundle=bundle,
            graph=graph,
            item_offset=item_offset,
            device=device,
            pairs=pairs,
            A_s=A_s,
            H_s=H_s,
            L_s=L_s,
            out_dir=OUT,
        )
        meta["EXTRA_ONLY"] = True
        meta["PRIMARY_CLAIM"] = False
        refit_metas.append(meta)
        write_json(OUT / f"HARDNEG_REFIT_SEED{seed}" / "meta.json", meta)
    write_json(OUT / "refit_artifacts.json", refit_metas)

    # run_sealed_eval hardcodes (101,202,303) — call a thin wrapper
    payload = _eval_extra(bundle, selected, refit_metas, feat_dir, OUT)
    a = payload["block_a"]["true_final_hardneg_mean"]
    print(
        f"EXTRA_404_505 = OK mean NDCG@20={a['NDCG@20']:.6f} "
        f"(NOT primary; primary remains ~0.201) REPLACE=DEFERRED",
        flush=True,
    )


def _eval_extra(bundle, selected, refit_metas, feat_dir, out_dir):
    """Copy of sealed eval loop but for EXTRA_SEEDS only."""
    import time
    import scipy.sparse as sps
    import torch
    from scripts._lastfm_external_benchmark_20260824 import build_urm, sha256_file, EXT_COMMIT, git_commit
    from scripts._lastfm_sealed_test_eval_20260824 import (
        UPSTREAM_CUTOFFS,
        TrueFinalOnTheFlyRecommender,
        holdout_df_to_dict,
        mean_metrics,
        primary_from_holdout,
    )
    from scripts.hgt_aggregation_common import empty_cache
    from scripts.run_final_hgt_capacity_convergence_race_v1 import build_race_model
    from scripts.run_lastfm_noleak_fullrank_validation_v1 import score_user_catalog
    from src.lastfm_lp.binary.contingency_tables import build_cooccurrence
    from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex
    from src.lastfm_lp.clean_v2.tabular_true import neighborhood_sizes
    from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit
    from scripts._lastfm_external_benchmark_20260824 import empty_pair_table
    from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "SEALED_TEST_EXECUTED.lock"
    if lock.exists() and (out_dir / "sealed_test_results.json").exists():
        return json.loads((out_dir / "sealed_test_results.json").read_text())

    urm_train = build_urm(bundle.train_external, bundle.n_users, bundle.n_items)
    urm_test = build_urm(bundle.sealed_test, bundle.n_users, bundle.n_items)
    with (feat_dir / "a_scaler.pkl").open("rb") as f:
        a_sc = pickle.load(f)
    with (feat_dir / "h_scaler.pkl").open("rb") as f:
        h_sc = pickle.load(f)
    with (feat_dir / "l_scaler.pkl").open("rb") as f:
        l_sc = pickle.load(f)

    idx_path = ART / "external_refit" / "te_pairwise_index.pkl"
    with idx_path.open("rb") as f:
        index = pickle.load(f)["index"]
    pos_u = np.asarray([u for u, xs in bundle.train_external.items() for _ in xs], dtype=np.int64)
    pos_i = np.asarray([i for u, xs in bundle.train_external.items() for i in xs], dtype=np.int64)
    prepared_cf = {
        "cfg": bundle.cfg,
        "model_train": bundle.train_external,
        "popularity": bundle.popularity,
        "item_kg_degree": bundle.item_kg_degree,
        "index": index,
        "train_pairs": {
            "user_id": pos_u,
            "item_id": pos_i,
            "label": np.ones(len(pos_u), dtype=np.int8),
        },
        "val_pairs": empty_pair_table(),
        "test_pairs": empty_pair_table(),
        "eval_users": bundle.eval_users_test,
    }
    cf = ensure_cross_fit(prepared_cf)
    graph = load_data_and_typed_graph(
        bundle.cfg,
        bundle.train_external,
        max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(os.environ.get("TFHN_DEVICE", "cpu"))
    size_cache: dict = {}

    def n_x_sizes_for(u: int):
        fold = int(cf.user_to_fold[int(u)])
        if fold not in size_cache:
            size_cache[fold] = neighborhood_sizes(cf.fold_indices[fold])
        return size_cache[fold]

    tf_a_seeds = []
    tf_b_seeds = []
    t_all = time.time()
    for seed in EXTRA_SEEDS:
        a_path = raw_dir / f"block_a_HARDNEG_seed{seed}.json"
        b_path = raw_dir / f"block_b_HARDNEG_seed{seed}.json"
        if a_path.exists() and b_path.exists():
            tf_a_seeds.append(json.loads(a_path.read_text()))
            tf_b_seeds.append(json.loads(b_path.read_text()))
            continue
        n_epochs = int(selected[seed]) + 1
        ckpt = out_dir / f"HARDNEG_REFIT_SEED{seed}" / "model.pt"
        print(f"[tfhn-extra] sealed eval seed={seed} n_epochs={n_epochs} …", flush=True)
        t0 = time.time()
        model = build_race_model(prepared_cf, graph, device, d=64, layers=2, heads=2)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        z = model.node_z()
        ctx = {
            "device": device,
            "model": model,
            "z": z,
            "a_mean": a_sc.mean_.astype(np.float32),
            "a_scale": a_sc.scale_.astype(np.float32),
            "h_mean": h_sc.mean_.astype(np.float32),
            "h_scale": h_sc.scale_.astype(np.float32),
            "l2_mean": l_sc.mean_.astype(np.float32),
            "l2_scale": l_sc.scale_.astype(np.float32),
        }

        def score_fn(u, hist, _ctx=ctx):
            return score_user_catalog(
                u,
                n_items=bundle.n_items,
                item_offset=item_offset,
                hist=hist,
                bundle=prepared_cf,
                cf=cf,
                ctx=_ctx,
                n_x_sizes=n_x_sizes_for(u),
            )

        block_b_acc: dict = {}
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
            "seed": seed,
            "selected_epoch": selected[seed],
            "n_epochs": n_epochs,
            "primary": primary_from_holdout(df),
            "all_cutoffs": holdout_df_to_dict(df),
            "artifact_sha256": sha256_file(ckpt),
            "seconds": time.time() - t0,
            "EXTRA_ONLY": True,
        }
        write_json(a_path, a_res)
        n_u = max(block_b_n[0], 1)
        b_res = {
            "seed": seed,
            "selected_epoch": selected[seed],
            "n_epochs": n_epochs,
            "NDCG@5": block_b_acc.get("NDCG@5", 0.0) / n_u,
            "NDCG@10": block_b_acc.get("NDCG@10", 0.0) / n_u,
            "NDCG@20": block_b_acc.get("NDCG@20", 0.0) / n_u,
            "Recall@20": block_b_acc.get("Recall@20", 0.0) / n_u,
            "MRR": block_b_acc.get("MRR", 0.0) / n_u,
            "HitRate@20": block_b_acc.get("HitRate@20", 0.0) / n_u,
            "n_users": block_b_n[0],
            "seconds": time.time() - t0,
            "EXTRA_ONLY": True,
        }
        write_json(b_path, b_res)
        tf_a_seeds.append(a_res)
        tf_b_seeds.append(b_res)
        print(
            f"[tfhn-extra] seed={seed} A NDCG@20={a_res['primary']['NDCG@20']:.6f} "
            f"(EXTRA only)",
            flush=True,
        )
        del model, adapter, ctx, z
        empty_cache()

    tf_a_mean = mean_metrics([r["primary"] for r in tf_a_seeds], ["NDCG@20", "RECALL@20"])
    tf_b_mean = mean_metrics(
        tf_b_seeds, ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@20", "MRR", "HitRate@20"]
    )
    payload = {
        "name": "TRUE_FINAL_HARDNEG_EXTRA_SEEDS_404_505",
        "PRIMARY_CLAIM": False,
        "EXTRA_ONLY": True,
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "intentawarers_commit": EXT_COMMIT,
        "training_hash": bundle.training_hash,
        "test_hash": bundle.test_hash,
        "easy_TRUE_FINAL_external_mean_NDCG@20": EASY_EXT_MEAN_NDCG20,
        "primary_hardneg_mean_NDCG@20": 0.200906,
        "delta_vs_easy_ext_mean": float(tf_a_mean["NDCG@20"] - EASY_EXT_MEAN_NDCG20),
        "block_a": {
            "true_final_hardneg_per_seed": [
                {
                    "seed": r["seed"],
                    "selected_epoch": r["selected_epoch"],
                    "primary": r["primary"],
                }
                for r in tf_a_seeds
            ],
            "true_final_hardneg_mean": tf_a_mean,
        },
        "block_b": {
            "true_final_hardneg_per_seed": tf_b_seeds,
            "true_final_hardneg_mean": tf_b_mean,
        },
        "refit": refit_metas,
        "elapsed_seconds": time.time() - t_all,
    }
    results_path = out_dir / "sealed_test_results.json"
    write_json(results_path, payload)
    payload["results_sha256"] = sha256_file(results_path)
    write_json(results_path, payload)
    lock.write_text(f"EXECUTED_EXTRA {utc_now()}\n")
    (out_dir / "EXTRA_REPORT.md").write_text(
        f"""# EXTRA hardneg seeds 404/505 (NOT primary)

Primary claim remains **101/202/303 mean = 0.2009**.

| Seed | NDCG@20 |
|---:|---:|
"""
        + "\n".join(
            f"| {r['seed']} | {r['primary']['NDCG@20']:.6f} |"
            for r in payload["block_a"]["true_final_hardneg_per_seed"]
        )
        + f"\n| **extra mean** | **{tf_a_mean['NDCG@20']:.6f}** |\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    main()
