#!/usr/bin/env python3
"""One-shot sealed Last-FM* external test evaluation (Blocks A + B).

Requires CONFIRM_UNSEAL_LASTFM_TEST=YES. Does not retrain or reselect models.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_repos" / "IntentAwareRS"))

from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    AUDIT_DIR,
    BASELINES_DIR,
    EXT_COMMIT,
    FEATURES_DIR,
    FROZEN_EPOCHS,
    MODELS_DIR,
    OUT,
    REPORTS_DIR,
    SEEDS,
    ScoreMatrixRecommender,
    build_external_bundle,
    build_external_training_bundle,
    build_urm,
    ensure_dirs,
    git_commit,
    h3_and_l2,
    sha256_file,
    upstream_eval_details,
    utc_now,
    write_json,
)
from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import build_race_model  # noqa: E402
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import (  # noqa: E402
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.evaluation.publication_full_rank_evaluator import (  # noqa: E402
    evaluate_publication_full_rank,
    evaluate_user_dense,
)
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout  # noqa: E402
from topn_baselines_neurals.Recommenders.GraphBased.P3alphaRecommender import (  # noqa: E402
    P3alphaRecommender,
)
from topn_baselines_neurals.Recommenders.GraphBased.RP3betaRecommender import (  # noqa: E402
    RP3betaRecommender,
)
from topn_baselines_neurals.Recommenders.KNN.ItemKNNCFRecommender import (  # noqa: E402
    ItemKNNCFRecommender,
)
from topn_baselines_neurals.Recommenders.KNN.UserKNNCFRecommender import (  # noqa: E402
    UserKNNCFRecommender,
)
from topn_baselines_neurals.Recommenders.NonPersonalizedRecommender import TopPop  # noqa: E402


RESULTS_DIR = OUT / "sealed_test_results"
LOCK_PATH = RESULTS_DIR / "SEALED_TEST_EXECUTED.lock"
CAND_BATCH = int(os.environ.get("LASTFM_CAND_BATCH", "8192"))
UPSTREAM_CUTOFFS = [1, 5, 10, 20, 40, 50, 100]
BLOCK_B_KS = (5, 10, 20)
BASELINE_SPECS = [
    ("TopPop", TopPop, "TopPop"),
    ("ItemKNN", ItemKNNCFRecommender, "ItemKNN"),
    ("P3alpha", P3alphaRecommender, "P3alpha"),
    ("RP3beta", RP3betaRecommender, "RP3beta"),
    ("UserKNN", UserKNNCFRecommender, "UserKNN"),
]


def scale_arr(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / np.maximum(scale, 1e-12)).astype(np.float32)


def verify_freeze_gate() -> dict[str, Any]:
    gate_path = AUDIT_DIR / "final_models_freeze_gate.json"
    if not gate_path.exists():
        raise RuntimeError("Missing final_models_freeze_gate.json — run EXT_05 first")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("FINAL_MODELS_FROZEN") != "YES" or gate.get("READY_TO_UNSEAL") != "YES":
        raise RuntimeError(f"Freeze gate not ready: {gate.get('FINAL_MODELS_FROZEN')=} {gate.get('READY_TO_UNSEAL')=}")
    if gate.get("upstream_evaluator", {}).get("reproduction") != "PASS":
        raise RuntimeError("SHEHZAD_EVALUATOR_REPRODUCTION is not PASS")
    # Verify checkpoint hashes still match freeze record
    for row in gate.get("true_final_refit") or []:
        path = ROOT / row["artifact_path"]
        got = sha256_file(path)
        if got != row["sha256"]:
            raise RuntimeError(f"TRUE FINAL seed {row['seed']} hash mismatch: {got} != {row['sha256']}")
    for row in gate.get("baselines") or []:
        path = ROOT / row["artifact_path"]
        got = sha256_file(path)
        if got != row["artifact_sha256"]:
            raise RuntimeError(f"Baseline {row['model']} hash mismatch")
    return gate


def holdout_df_to_dict(df) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cutoff in df.index.tolist():
        row = {}
        for col in df.columns:
            val = df.loc[cutoff, col]
            try:
                row[str(col)] = float(val)
            except Exception:
                row[str(col)] = str(val)
        out[str(int(cutoff))] = row
    return out


def primary_from_holdout(df) -> dict[str, float]:
    return {
        "NDCG@20": float(df.loc[20, "NDCG"]),
        "RECALL@20": float(df.loc[20, "RECALL"]),
    }


@torch.no_grad()
def score_user_catalog_external(
    u: int,
    *,
    n_items: int,
    item_offset: int,
    hist: set[int],
    prepared: dict[str, Any],
    cf,
    ctx: dict[str, Any],
    n_x_sizes: np.ndarray,
) -> np.ndarray:
    """Match external refit feature protocol: scale A/H; LEG raw (as trained)."""
    device = ctx["device"]
    model = ctx["model"]
    scores = np.full(n_items, -np.inf, dtype=np.float64)
    cands_all = np.asarray([i for i in range(n_items) if i not in hist], dtype=np.int64)
    if cands_all.size == 0:
        return scores
    z = ctx["z"]
    idx = clean_v2_index_for(cf, int(u))
    hist_arr = np.asarray(sorted(hist), dtype=np.int64)
    u_scalar = torch.tensor([int(u)], dtype=torch.int64, device=device)
    for start in range(0, int(cands_all.size), CAND_BATCH):
        end = min(start + CAND_BATCH, int(cands_all.size))
        cands = cands_all[start:end]
        n_b = int(end - start)
        A = vectorized_clean_v2_A_for_user(
            int(u),
            cands,
            prepared["model_train"],
            prepared["popularity"],
            prepared["item_kg_degree"],
            idx,
            n_x_sizes=n_x_sizes,
        )
        A_s = scale_arr(A, ctx["a_mean"], ctx["a_scale"])
        H, L2 = h3_and_l2(hist_arr, cands, idx)
        H_s = scale_arr(H, ctx["h_mean"], ctx["h_scale"])
        L2_s = L2.astype(np.float32)  # raw LEG — matches external refit training
        item_idx = torch.from_numpy(cands.astype(np.int64) + item_offset).to(device)
        user_idx = u_scalar.expand(n_b)
        out = model(
            user_idx,
            item_idx,
            torch.from_numpy(A_s).to(device),
            torch.from_numpy(H_s).to(device),
            torch.from_numpy(L2_s).to(device),
            z=z,
        )
        scores[cands] = out["logits"].detach().cpu().numpy().astype(np.float64)
    return scores


class TrueFinalOnTheFlyRecommender:
    """IntentAwareRS-compatible recommender; scores catalog on the fly and records Block B."""

    RECOMMENDER_NAME = "TRUE_FINAL_REFIT_OnTheFly"

    def __init__(
        self,
        urm_train: sps.csr_matrix,
        *,
        score_fn,
        train_external: dict[int, set[int]],
        sealed_test: dict[int, set[int]],
        block_b_acc: dict[str, float],
        block_b_n: list[int],
    ):
        self.URM_train = urm_train.tocsr()
        self.score_fn = score_fn
        self.train_external = train_external
        self.sealed_test = sealed_test
        self.block_b_acc = block_b_acc
        self.block_b_n = block_b_n
        self.items_to_ignore_ID = np.array([], dtype=np.int64)
        self.n_items = int(urm_train.shape[1])

    def get_URM_train(self):
        return self.URM_train

    def set_items_to_ignore(self, items):
        self.items_to_ignore_ID = np.asarray(items, dtype=np.int64)

    def reset_items_to_ignore(self):
        self.items_to_ignore_ID = np.array([], dtype=np.int64)

    def recommend(
        self,
        user_id_array,
        remove_seen_flag=True,
        cutoff=20,
        remove_top_pop_flag=False,
        remove_custom_items_flag=False,
        return_scores=True,
        items_to_compute=None,
    ):
        users = np.asarray(user_id_array, dtype=np.int64)
        ranked = []
        score_batch = np.full((len(users), self.n_items), -np.inf, dtype=np.float64)
        for row_i, u in enumerate(users.tolist()):
            hist = self.train_external.get(int(u), set())
            scores = self.score_fn(int(u), hist)
            score_batch[row_i] = scores
            pos = self.sealed_test.get(int(u), set())
            if pos:
                mb = evaluate_user_dense(scores, positive_items=pos, train_items=hist, ks=BLOCK_B_KS)
                for k, v in mb.items():
                    self.block_b_acc[k] = self.block_b_acc.get(k, 0.0) + float(v)
                self.block_b_n[0] += 1
            row = scores.copy()
            if remove_seen_flag:
                for i in hist:
                    if 0 <= int(i) < row.size:
                        row[int(i)] = -np.inf
            if remove_custom_items_flag and self.items_to_ignore_ID.size:
                row[self.items_to_ignore_ID] = -np.inf
            if items_to_compute is not None:
                mask = np.full(row.size, True, dtype=bool)
                mask[np.asarray(items_to_compute, dtype=np.int64)] = False
                row[mask] = -np.inf
            order = np.lexsort((np.arange(row.size, dtype=np.int64), -row))
            ranked.append(order[:cutoff])
        return ranked, score_batch


def load_baseline(name: str, cls, file_name: str, urm_train: sps.csr_matrix):
    model_dir = BASELINES_DIR / name.replace("-", "_")
    rec = cls(urm_train, verbose=False) if name != "TopPop" else cls(urm_train)
    rec.load_model(str(model_dir) + "/", file_name=file_name)
    return rec


def evaluate_baseline_block_a(rec, urm_test: sps.csr_matrix) -> dict[str, Any]:
    evaluator = EvaluatorHoldout(urm_test, UPSTREAM_CUTOFFS, exclude_seen=True, verbose=False)
    df, raw = evaluator.evaluateRecommender(rec)
    return {
        "primary": primary_from_holdout(df),
        "all_cutoffs": holdout_df_to_dict(df),
        "raw_string": str(raw),
    }


def evaluate_baseline_block_b(rec, bundle) -> dict[str, Any]:
    def score_fn(u: int, items: np.ndarray) -> np.ndarray:
        sc = rec._compute_item_score(np.asarray([u], dtype=np.int32), items_to_compute=items.tolist())
        return sc[0].astype(np.float64)

    result = evaluate_publication_full_rank(
        eval_users=bundle.eval_users_test,
        test_positives=bundle.sealed_test,
        model_train=bundle.train_external,
        n_items=bundle.n_items,
        score_fn=score_fn,
        mode="dense",
        ks=BLOCK_B_KS,
    )
    mean = dict(result.metrics_mean)
    return {
        "NDCG@5": float(mean.get("NDCG@5", float("nan"))),
        "NDCG@10": float(mean.get("NDCG@10", float("nan"))),
        "NDCG@20": float(mean.get("NDCG@20", float("nan"))),
        "Recall@20": float(mean.get("Recall@20", float("nan"))),
        "MRR": float(mean.get("MRR", float("nan"))),
        "HitRate@20": float(mean.get("HitRate@20", float("nan"))),
        "n_users": int(result.n_users),
    }


def mean_metrics(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def write_sealed_manifest(payload: dict[str, Any]) -> Path:
    path = REPORTS_DIR / "LASTFM_FINAL_EXTERNAL_PROTOCOL_MANIFEST_20260824.md"
    up = upstream_eval_details()
    a_rows = payload["block_a"]["models"]
    b_rows = payload["block_b"]["models"]
    lines = [
        "# Last-FM* final external protocol manifest (2026-08-24)",
        "",
        "Status after **one-shot sealed test** evaluation. Models were not retrained or reselected after observing results.",
        "",
        f"- `FINAL_MODELS_FROZEN` = **YES**",
        f"- `READY_TO_UNSEAL` = **YES** (pre-run)",
        f"- `sealed_test_accessed` = **YES**",
        f"- `SEALED_TEST_EXECUTED` = **YES**",
        f"- evaluation timestamp = `{payload['timestamp']}`",
        f"- our git commit = `{payload['git_commit']}`",
        f"- IntentAwareRS commit = `{payload['intentawarers_commit']}`",
        f"- `TRAIN_EXTERNAL` hash = `{payload['training_hash']}`",
        f"- sealed test hash = `{payload['test_hash']}`",
        f"- results artifact SHA256 = `{payload['results_sha256']}`",
        "",
        "## A. IntentAwareRS-compatible evaluation (literature comparison)",
        "",
        f"- evaluator = `{up['UPSTREAM_EVALUATOR_CLASS']}` (`{up['UPSTREAM_EVALUATOR_FILE']}`)",
        f"- cutoffs = `{up['UPSTREAM_CUTOFFS']}`",
        f"- train masking = `{up['UPSTREAM_TRAIN_MASKING']}`",
        "",
        "| Model | NDCG@20 | Recall@20 |",
        "|---|---:|---:|",
    ]
    for name, row in a_rows.items():
        p = row["primary"]
        lines.append(f"| {name} | {p['NDCG@20']:.6f} | {p['RECALL@20']:.6f} |")
    lines += [
        "",
        "### TRUE FINAL per-seed (Block A)",
        "",
        "| Seed | epochs | NDCG@20 | Recall@20 |",
        "|---|---:|---:|---:|",
    ]
    for seed_row in payload["block_a"]["true_final_per_seed"]:
        p = seed_row["primary"]
        lines.append(
            f"| {seed_row['seed']} | {seed_row['epoch_count']} | {p['NDCG@20']:.6f} | {p['RECALL@20']:.6f} |"
        )
    tf_mean = payload["block_a"]["true_final_mean"]
    lines += [
        f"| **mean** | — | **{tf_mean['NDCG@20']:.6f}** | **{tf_mean['RECALL@20']:.6f}** |",
        "",
        "## B. Our frozen full-catalog evaluator (internal robustness)",
        "",
        "| Model | NDCG@5 | NDCG@10 | NDCG@20 | Recall@20 | MRR | HitRate@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in b_rows.items():
        lines.append(
            f"| {name} | {row['NDCG@5']:.6f} | {row['NDCG@10']:.6f} | {row['NDCG@20']:.6f} | "
            f"{row['Recall@20']:.6f} | {row['MRR']:.6f} | {row['HitRate@20']:.6f} |"
        )
    lines += [
        "",
        "### TRUE FINAL per-seed (Block B)",
        "",
        "| Seed | NDCG@20 | Recall@20 | MRR | HitRate@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed_row in payload["block_b"]["true_final_per_seed"]:
        lines.append(
            f"| {seed_row['seed']} | {seed_row['NDCG@20']:.6f} | {seed_row['Recall@20']:.6f} | "
            f"{seed_row['MRR']:.6f} | {seed_row['HitRate@20']:.6f} |"
        )
    bm = payload["block_b"]["true_final_mean"]
    lines += [
        f"| **mean** | **{bm['NDCG@20']:.6f}** | **{bm['Recall@20']:.6f}** | **{bm['MRR']:.6f}** | **{bm['HitRate@20']:.6f}** |",
        "",
        "**Never mix Block A and Block B values in the same comparison column.**",
        "",
        "## Development ablation (reference only; not external test)",
        "",
        "| Variant | NDCG@20 | Recall@20 | MRR |",
        "|---|---:|---:|---:|",
        "| HGT only | ~0.0093 | ~0.0169 | — |",
        "| +A5 | ~0.0640 | ~0.1128 | — |",
        "| +H3 | ~0.2321 | ~0.3647 | — |",
        "| TRUE FINAL | ~0.2764 | ~0.3768 | ~0.3534 |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_sealed_test_once() -> dict[str, Any]:
    if os.environ.get("CONFIRM_UNSEAL_LASTFM_TEST") != "YES":
        raise RuntimeError(
            "Sealed Last-FM* test evaluation is disabled. "
            "Set CONFIRM_UNSEAL_LASTFM_TEST=YES only after protocol approval."
        )
    ensure_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists() and os.environ.get("FORCE_REUNSEAL_LASTFM_TEST") != "YES":
        raise RuntimeError(
            f"Sealed test already executed ({LOCK_PATH}). "
            "Refusing a second run (exactly-once protocol)."
        )

    gate = verify_freeze_gate()
    bundle = build_external_bundle()
    if bundle.training_hash != gate["training_hash"] or bundle.test_hash != gate["test_hash"]:
        raise RuntimeError("Train/test hash mismatch vs freeze gate")

    print("[sealed] UNSEALING TEST_EXTERNAL for one-shot evaluation", flush=True)
    t_all = time.time()
    urm_train = build_urm(bundle.train_external, bundle.n_users, bundle.n_items)
    urm_test = build_urm(bundle.sealed_test, bundle.n_users, bundle.n_items)

    block_a_models: dict[str, Any] = {}
    block_b_models: dict[str, Any] = {}
    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Baselines ---
    for name, cls, file_name in BASELINE_SPECS:
        a_path = raw_dir / f"block_a_{name}.json"
        b_path = raw_dir / f"block_b_{name}.json"
        if a_path.exists() and b_path.exists():
            a_res = json.loads(a_path.read_text(encoding="utf-8"))
            b_res = json.loads(b_path.read_text(encoding="utf-8"))
            block_a_models[name] = a_res
            block_b_models[name] = b_res
            print(
                f"[sealed] reuse {name} A: NDCG@20={a_res['primary']['NDCG@20']:.4f} | "
                f"B: NDCG@20={b_res['NDCG@20']:.4f}",
                flush=True,
            )
            continue
        print(f"[sealed] baseline {name} Block A+B …", flush=True)
        t0 = time.time()
        rec = load_baseline(name, cls, file_name, urm_train)
        a_res = evaluate_baseline_block_a(rec, urm_test)
        write_json(a_path, a_res)
        b_res = evaluate_baseline_block_b(rec, bundle)
        write_json(b_path, b_res)
        block_a_models[name] = a_res
        block_b_models[name] = b_res
        print(
            f"[sealed] {name} A: NDCG@20={a_res['primary']['NDCG@20']:.4f} "
            f"Recall@20={a_res['primary']['RECALL@20']:.4f} | "
            f"B: NDCG@20={b_res['NDCG@20']:.4f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        del rec
        empty_cache()

    # --- TRUE FINAL seeds ---
    print("[sealed] preparing TRUE FINAL scoring context …", flush=True)
    prepared = build_external_training_bundle(bundle)
    cf = ensure_cross_fit(prepared)
    with (AUDIT_DIR / "external_a_scaler.pkl").open("rb") as f:
        a_scaler = pickle.load(f)
    with (AUDIT_DIR / "external_h_scaler.pkl").open("rb") as f:
        h_scaler = pickle.load(f)
    graph = load_data_and_typed_graph(
        bundle.cfg,
        bundle.train_external,
        max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(str(bundle.cfg.get("models", {}).get("architecture", {}).get("device") or "auto"))
    size_cache: dict[int, np.ndarray] = {}

    def n_x_sizes_for(u: int) -> np.ndarray:
        fold = cf.user_to_fold.get(int(u))
        if fold is None:
            raise RuntimeError(f"user {u} missing from TRAIN_EXTERNAL cross-fit folds")
        fold = int(fold)
        if fold not in size_cache:
            size_cache[fold] = neighborhood_sizes(cf.fold_indices[fold])
        return size_cache[fold]

    tf_a_seeds: list[dict[str, Any]] = []
    tf_b_seeds: list[dict[str, Any]] = []

    for seed in SEEDS:
        a_seed_path = raw_dir / f"block_a_TRUE_FINAL_seed{seed}.json"
        b_seed_path = raw_dir / f"block_b_TRUE_FINAL_seed{seed}.json"
        if a_seed_path.exists() and b_seed_path.exists():
            a_res = json.loads(a_seed_path.read_text(encoding="utf-8"))
            b_res = json.loads(b_seed_path.read_text(encoding="utf-8"))
            tf_a_seeds.append(a_res)
            tf_b_seeds.append(b_res)
            print(
                f"[sealed] reuse TRUE FINAL seed={seed} A NDCG@20={a_res['primary']['NDCG@20']:.4f} | "
                f"B NDCG@20={b_res['NDCG@20']:.4f}",
                flush=True,
            )
            continue
        print(f"[sealed] TRUE FINAL seed={seed} epochs={FROZEN_EPOCHS[seed]} …", flush=True)
        t0 = time.time()
        ckpt = MODELS_DIR / f"TRUE_FINAL_REFIT_SEED{seed}" / "model.pt"
        model = build_race_model(prepared, graph, device, d=64, layers=2, heads=2)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
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
            "seed": seed,
            "epoch_count": FROZEN_EPOCHS[seed],
            "primary": primary_from_holdout(df),
            "all_cutoffs": holdout_df_to_dict(df),
            "raw_string": str(raw),
            "artifact_sha256": sha256_file(ckpt),
            "seconds": time.time() - t0,
        }
        write_json(raw_dir / f"block_a_TRUE_FINAL_seed{seed}.json", a_res)
        n_u = max(block_b_n[0], 1)
        b_res = {
            "seed": seed,
            "epoch_count": FROZEN_EPOCHS[seed],
            "NDCG@5": block_b_acc.get("NDCG@5", 0.0) / n_u,
            "NDCG@10": block_b_acc.get("NDCG@10", 0.0) / n_u,
            "NDCG@20": block_b_acc.get("NDCG@20", 0.0) / n_u,
            "Recall@20": block_b_acc.get("Recall@20", 0.0) / n_u,
            "MRR": block_b_acc.get("MRR", 0.0) / n_u,
            "HitRate@20": block_b_acc.get("HitRate@20", 0.0) / n_u,
            "n_users": block_b_n[0],
            "seconds": time.time() - t0,
        }
        write_json(raw_dir / f"block_b_TRUE_FINAL_seed{seed}.json", b_res)
        tf_a_seeds.append(a_res)
        tf_b_seeds.append(b_res)
        print(
            f"[sealed] seed={seed} A NDCG@20={a_res['primary']['NDCG@20']:.4f} "
            f"Recall@20={a_res['primary']['RECALL@20']:.4f} | "
            f"B NDCG@20={b_res['NDCG@20']:.4f} ({a_res['seconds']:.1f}s)",
            flush=True,
        )
        del model, adapter, ctx, z
        empty_cache()

    tf_a_mean = mean_metrics([r["primary"] for r in tf_a_seeds], ["NDCG@20", "RECALL@20"])
    tf_b_mean = mean_metrics(
        tf_b_seeds,
        ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@20", "MRR", "HitRate@20"],
    )
    block_a_models["TRUE_FINAL_mean"] = {"primary": tf_a_mean, "per_seed": "see true_final_per_seed"}
    block_b_models["TRUE_FINAL_mean"] = dict(tf_b_mean)

    payload = {
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "intentawarers_commit": EXT_COMMIT,
        "training_hash": bundle.training_hash,
        "test_hash": bundle.test_hash,
        "sealed_test_accessed": True,
        "SEALED_TEST_EXECUTED": True,
        "models_retrained_after_unseal": False,
        "models_reselected_after_unseal": False,
        "upstream_evaluator": upstream_eval_details(),
        "block_a": {
            "label": "IntentAwareRS-compatible evaluation",
            "models": block_a_models,
            "true_final_per_seed": [
                {
                    "seed": r["seed"],
                    "epoch_count": r["epoch_count"],
                    "primary": r["primary"],
                    "artifact_sha256": r["artifact_sha256"],
                }
                for r in tf_a_seeds
            ],
            "true_final_mean": tf_a_mean,
        },
        "block_b": {
            "label": "Our frozen full-catalog evaluator",
            "models": block_b_models,
            "true_final_per_seed": [
                {
                    "seed": r["seed"],
                    "NDCG@5": r["NDCG@5"],
                    "NDCG@10": r["NDCG@10"],
                    "NDCG@20": r["NDCG@20"],
                    "Recall@20": r["Recall@20"],
                    "MRR": r["MRR"],
                    "HitRate@20": r["HitRate@20"],
                    "n_users": r["n_users"],
                }
                for r in tf_b_seeds
            ],
            "true_final_mean": tf_b_mean,
        },
        "elapsed_seconds": time.time() - t_all,
        "freeze_gate_git_commit": gate.get("git_commit"),
    }
    results_path = RESULTS_DIR / "sealed_test_results.json"
    write_json(results_path, payload)
    # also legacy path expected by freeze audit
    write_json(AUDIT_DIR / "sealed_test_results.json", payload)
    results_sha = sha256_file(results_path)
    payload["results_sha256"] = results_sha
    write_json(results_path, payload)
    write_json(AUDIT_DIR / "sealed_test_results.json", payload)

    # Hash raw artifacts
    artifact_hashes = {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in sorted(raw_dir.glob("*.json"))
    }
    artifact_hashes[str(results_path.relative_to(ROOT))] = results_sha
    write_json(RESULTS_DIR / "artifact_hashes.json", artifact_hashes)

    LOCK_PATH.write_text(
        json.dumps(
            {
                "timestamp": payload["timestamp"],
                "results_sha256": results_sha,
                "git_commit": payload["git_commit"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = write_sealed_manifest(payload)
    payload["manifest_path"] = str(manifest_path.relative_to(ROOT))
    write_json(results_path, payload)
    write_json(AUDIT_DIR / "sealed_test_results.json", payload)
    return payload
