#!/usr/bin/env python3
"""LASTFM_NOLEAK_FULLRANK_VALIDATION_V1.

True full-catalog validation of frozen LASTFM_TRUE_FINAL_JOINT_TRAINING_V1 checkpoints.
Does NOT retrain, retune, or touch TEST scoring. Does NOT full-rank the C1 reproduction.
Waits for TRUE FINAL joint training unless LASTFM_SKIP_WAIT=1.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    build_race_model,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    EXPECTED as STAR_FP,
    git_commit,
    sha256_arr,
    sha256_file,
    write_csv,
    write_json,
)
from scripts.run_race_clean_3 import item_pop_buckets, shared_A_dir  # noqa: E402
from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix  # noqa: E402
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices  # noqa: E402
from src.lastfm_lp.clean_v2.constants import CLEAN_V2_TOP_K  # noqa: E402
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.pooling import pool_signed_a11_3d_matrix  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import (  # noqa: E402
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.evaluation.publication_full_rank_evaluator import (  # noqa: E402
    dense_topk,
    evaluate_user_dense,
)
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = ROOT / "configs" / "lastfm_star_race_clean_3.yaml"
DATA = ROOT / "data" / "LastFM_star_IntentAwareRS"
SPLITS = ROOT / "outputs" / "lastfm_star" / "splits"
RACE = ROOT / "KRAM_FINAL_WORK" / "RACE_CLEAN_3"
V2 = RACE / "A11_FUNCTIONAL_DISTRIBUTION_BENCHMARK_V2"
B0_H = RACE / "race" / "a11_top25" / "features"
TRAIN_OUT = Path(
    os.environ.get(
        "LASTFM_FINAL_OUT",
        str(ROOT / "LASTFM_TRUE_FINAL" / "JOINT_TRAINING_V1"),
    )
)
OUT = Path(
    os.environ.get(
        "LASTFM_FULLRANK_OUT",
        str(ROOT / "LASTFM_TRUE_FINAL" / "NOLEAK_FULLRANK_VALIDATION_V1"),
    )
)
TRAIN_WAIT_SCRIPT = os.environ.get(
    "LASTFM_TRAIN_WAIT_SCRIPT",
    "run_lastfm_true_final_joint_training_v1.py",
)
SKIP_WAIT = os.environ.get("LASTFM_SKIP_WAIT", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
SEEDS = (101, 202, 303)
KS = (5, 10, 20)
CAND_BATCH = int(os.environ.get("LASTFM_CAND_BATCH", "8192"))
NO_DIST = os.environ.get("LASTFM_NO_DIST", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
HGT_ONLY = os.environ.get("LASTFM_HGT_ONLY", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
NO_LEG = os.environ.get("LASTFM_NO_LEG", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
ARCH_NAME = "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL"
if HGT_ONLY:
    from scripts.lastfm_hgt_a5_ranker_common import ARCH_NAME_HGT_ONLY as ARCH_NAME  # noqa: E402
    from scripts.lastfm_hgt_a5_ranker_common import build_hgt_only_ranker  # noqa: E402
elif NO_LEG:
    from scripts.lastfm_hgt_a5_ranker_common import ARCH_NAME_H3 as ARCH_NAME  # noqa: E402
    from scripts.lastfm_hgt_a5_ranker_common import build_hgt_a5_h3_noleg  # noqa: E402
elif NO_DIST:
    from scripts.lastfm_hgt_a5_ranker_common import ARCH_NAME as ARCH_NAME  # noqa: E402
    from scripts.lastfm_hgt_a5_ranker_common import build_hgt_a5_ranker  # noqa: E402
SEED_DIR_NAME = {101: "01_SEED101", 202: "02_SEED202", 303: "03_SEED303"}
SEED_DIR_KEY = {101: "s101", 202: "s202", 303: "s303"}


def selected_seeds() -> tuple[int, ...]:
    raw = os.environ.get("LASTFM_SEEDS", "").strip()
    if not raw:
        return SEEDS
    out = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
    bad = [s for s in out if s not in SEEDS]
    if bad:
        raise SystemExit(f"LASTFM_SEEDS not in {SEEDS}: {bad}")
    if not out:
        raise SystemExit("LASTFM_SEEDS empty")
    return out
L2_MATCH_ATOL = 1e-5
POLL_SEC = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dirs() -> dict[str, Path]:
    mapping = {
        "audit": OUT / "00_AUDIT",
        "s101": OUT / "01_SEED101",
        "s202": OUT / "02_SEED202",
        "s303": OUT / "03_SEED303",
        "metrics": OUT / "04_METRICS",
        "report": OUT / "05_REPORT",
    }
    for p in mapping.values():
        p.mkdir(parents=True, exist_ok=True)
    return mapping


def training_manifest_path() -> Path:
    custom = os.environ.get("LASTFM_TRAIN_MANIFEST")
    if custom:
        p = Path(custom)
        return p if p.is_absolute() else TRAIN_OUT / "04_SUMMARY" / custom
    for name in (
        "TRUE_FINAL_JOINT_TRAINING_MANIFEST.json",
        "FINAL_CLEAN_TRAINING_MANIFEST.json",
    ):
        p = TRAIN_OUT / "04_SUMMARY" / name
        if p.exists():
            return p
    return TRAIN_OUT / "04_SUMMARY" / "TRUE_FINAL_JOINT_TRAINING_MANIFEST.json"


def training_running() -> bool:
    """True only if a Python interpreter is still executing the trainer."""
    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    except Exception:
        return False
    me = str(os.getpid())
    needles = {TRAIN_WAIT_SCRIPT, "run_lastfm_true_final_then_fullrank_v1.py"}
    for line in out.splitlines():
        if not any(n in line for n in needles):
            continue
        if "run_lastfm_noleak_fullrank" in line:
            continue
        low = line.lower()
        if "python" not in low:
            continue
        pid = line.strip().split(None, 1)[0]
        if pid != me:
            return True
    return False


def _manifest_ready(man: dict[str, Any]) -> bool:
    if man.get("FULLRANK_READY") is not True:
        return False
    if man.get("TRUE_FINAL_TRAINING_VERDICT") == "TRUE_FINAL_MODEL_READY":
        return True
    if man.get("FINAL_TRAINING_VERDICT") == "FINAL_MODEL_READY":
        return True
    return False


def wait_for_training(run_seeds: tuple[int, ...]) -> Path | None:
    """Block until frozen TRUE FINAL checkpoints exist and the trainer is gone."""
    manifest = training_manifest_path()
    ckpts = [TRAIN_OUT / "06_CHECKPOINTS" / f"final_seed{s}_best.pt" for s in run_seeds]
    partial = set(run_seeds) != set(SEEDS)
    print(f"[fullrank] waiting for TRUE FINAL JOINT at {TRAIN_OUT} seeds={list(run_seeds)} …", flush=True)
    if SKIP_WAIT or partial:
        missing = [str(p) for p in ckpts if not p.exists()]
        if missing:
            raise SystemExit(f"EVALUATION_FAILED: missing checkpoints {missing}")
        if training_running():
            raise SystemExit("EVALUATION_FAILED: trainer still running")
        if partial or NO_DIST or HGT_ONLY or NO_LEG:
            print(
                "[fullrank] PARTIAL/NO_DIST/HGT_ONLY — not requiring 3-seed FULLRANK_READY",
                flush=True,
            )
            return manifest if manifest.exists() else None
        if not manifest.exists():
            raise SystemExit("EVALUATION_FAILED: LASTFM_SKIP_WAIT=1 but manifest missing")
        man = json.loads(manifest.read_text())
        if not _manifest_ready(man):
            raise SystemExit("EVALUATION_FAILED: training did not declare FULLRANK_READY")
        print("[fullrank] SKIP_WAIT — starting evaluation on TRUE FINAL checkpoints", flush=True)
        return manifest
    while True:
        manifest = training_manifest_path()
        have = all(p.exists() for p in ckpts) and manifest.exists()
        running = training_running()
        if have and not running:
            man = json.loads(manifest.read_text())
            if _manifest_ready(man):
                print("[fullrank] TRUE FINAL training complete — starting evaluation", flush=True)
                return manifest
            print(
                f"[fullrank] checkpoints present but verdict={man.get('TRUE_FINAL_TRAINING_VERDICT') or man.get('FINAL_TRAINING_VERDICT')} "
                f"FULLRANK_READY={man.get('FULLRANK_READY')} — STOP",
                flush=True,
            )
            raise SystemExit("EVALUATION_FAILED: training did not declare FULLRANK_READY")
        print(
            f"[fullrank] wait have_ckpt={have} trainer_running={running} {utc_now()}",
            flush=True,
        )
        time.sleep(POLL_SEC)


def pair_overlaps(a: dict[int, set[int]], b: dict[int, set[int]]) -> int:
    n = 0
    for u in set(a) | set(b):
        n += len(a.get(u, set()) & b.get(u, set()))
    return n


def data_audit() -> dict[str, Any]:
    mt = load_user_sets(SPLITS / "model_train.txt")
    va = load_user_sets(SPLITS / "valid.txt")
    te = load_user_sets(SPLITS / "test.txt")
    ov = {
        "model_train_cap_valid": pair_overlaps(mt, va),
        "model_train_cap_test": pair_overlaps(mt, te),
        "valid_cap_test": pair_overlaps(va, te),
    }
    hashes = {
        "model_train": sha256_file(SPLITS / "model_train.txt"),
        "valid": sha256_file(SPLITS / "valid.txt"),
        "test": sha256_file(SPLITS / "test.txt"),
        "user_list": sha256_file(DATA / "user_list.txt"),
        "item_list": sha256_file(DATA / "item_list.txt"),
        "entity_list": sha256_file(DATA / "entity_list.txt"),
        "kg": sha256_file(DATA / "kg_final.txt"),
        "H3_train": sha256_file(B0_H / "X_train.npy"),
        "H3_val": sha256_file(B0_H / "X_val.npy"),
        "LEG_K2_scaler": sha256_file(V2 / "representations" / "LEG_K2_scaler.pkl"),
        "A5_train": sha256_file(shared_A_dir() / "X_train.npy"),
    }
    train_fp = json.loads((TRAIN_OUT / "00_AUDIT" / "DATA_FINGERPRINTS.json").read_text())
    mismatch = {}
    for k in ("model_train", "valid", "test", "kg", "item_list"):
        got, exp_star, exp_train = hashes[k], STAR_FP.get(k), train_fp.get(k)
        if got != exp_star or got != exp_train:
            mismatch[k] = {"got": got, "star_expected": exp_star, "training_run": exp_train}
    failed = any(ov.values()) or bool(mismatch)
    return {
        "dataset": "CORRECTED_LASTFM_STAR_NO_LEAKAGE",
        "data_path": str(DATA),
        "splits_path": str(SPLITS),
        "overlaps": ov,
        "hashes": hashes,
        "training_fingerprints": train_fp,
        "mismatch_vs_star_or_training": mismatch,
        "FAILED": failed,
        "n_model_train_users": len(mt),
        "n_valid_users_file": len(va),
        "n_test_users_file": len(te),
        "TEST_READ_FOR_OVERLAP_ONLY": True,
        "TEST_SCORED": False,
    }


def h3_and_l2(hist: np.ndarray, cands: np.ndarray, index, *, max_history: int = CLEAN_V2_TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """Same signed-Top25 A11 as H3; L2 = mean P2 over selected values."""
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


def scale_arr(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean) / np.maximum(scale, 1e-12)).astype(np.float32)


def map_at_k(top_items: np.ndarray, positives: set[int], k: int) -> float:
    """Standard MAP@K: AP / |P_u| (not AP / n_hits)."""
    pos = set(int(x) for x in positives)
    if not pos:
        return float("nan")
    topk = [int(x) for x in np.asarray(top_items).tolist()[:k]]
    hits = 0.0
    ap = 0.0
    for i, item in enumerate(topk, start=1):
        if item in pos:
            hits += 1.0
            ap += hits / i
    return float(ap / len(pos))


def precision_at_k(top_items: np.ndarray, positives: set[int], k: int) -> float:
    topk = [int(x) for x in np.asarray(top_items).tolist()[:k]]
    if k <= 0:
        return float("nan")
    hits = sum(1 for i in topk if i in positives)
    return float(hits / k)


@torch.no_grad()
def score_user_catalog(
    u: int,
    *,
    n_items: int,
    item_offset: int,
    hist: set[int],
    bundle,
    cf,
    ctx: dict[str, Any],
    n_x_sizes: np.ndarray,
) -> np.ndarray:
    """Score every catalog item id. Train-history items left as -inf (never ranked)."""
    device = ctx["device"]
    model = ctx["model"]
    scores = np.full(n_items, -np.inf, dtype=np.float64)
    cands_all = np.asarray([i for i in range(n_items) if i not in hist], dtype=np.int64)
    if cands_all.size == 0:
        return scores
    z = ctx["z"]
    hgt_only = bool(ctx.get("hgt_only"))
    no_dist = bool(ctx.get("no_dist"))
    idx = None
    hist_arr = None
    if not hgt_only:
        idx = clean_v2_index_for(cf, int(u))
        hist_arr = np.asarray(sorted(hist), dtype=np.int64)
    # One MPS scalar per user; expand to batch size (avoids torch.full graph spam).
    u_scalar = torch.tensor([int(u)], dtype=torch.int64, device=device)
    for start in range(0, int(cands_all.size), CAND_BATCH):
        end = min(start + CAND_BATCH, int(cands_all.size))
        cands = cands_all[start:end]
        n_b = int(end - start)
        if hgt_only:
            A_s = np.zeros((n_b, 0), dtype=np.float32)
            H_s = np.zeros((n_b, 3), dtype=np.float32)
            L2_s = np.zeros((n_b, 1), dtype=np.float32)
        else:
            A = vectorized_clean_v2_A_for_user(
                int(u),
                cands,
                bundle["model_train"],
                bundle["popularity"],
                bundle["item_kg_degree"],
                idx,
                n_x_sizes=n_x_sizes,
            )
            A_s = scale_arr(A, ctx["a_mean"], ctx["a_scale"])
            if no_dist:
                H_s = np.zeros((n_b, 3), dtype=np.float32)
                L2_s = np.zeros((n_b, 1), dtype=np.float32)
            elif ctx.get("no_leg"):
                H, _l2 = h3_and_l2(hist_arr, cands, idx)
                H_s = scale_arr(H, ctx["h_mean"], ctx["h_scale"])
                L2_s = np.zeros((n_b, 1), dtype=np.float32)
            else:
                H, L2 = h3_and_l2(hist_arr, cands, idx)
                H_s = scale_arr(H, ctx["h_mean"], ctx["h_scale"])
                L2_s = scale_arr(L2, ctx["l2_mean"], ctx["l2_scale"])
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


@torch.no_grad()
def score_user_catalog_legacy_torch_full(
    u: int,
    *,
    n_items: int,
    item_offset: int,
    hist: set[int],
    bundle,
    cf,
    ctx: dict[str, Any],
    n_x_sizes: np.ndarray,
) -> np.ndarray:
    """Pre-fix reference path (torch.full per batch). Kept only for equivalence checks."""
    device = ctx["device"]
    model = ctx["model"]
    scores = np.full(n_items, -np.inf, dtype=np.float64)
    cands_all = np.asarray([i for i in range(n_items) if i not in hist], dtype=np.int64)
    if cands_all.size == 0:
        return scores
    z = ctx["z"]
    hgt_only = bool(ctx.get("hgt_only"))
    no_dist = bool(ctx.get("no_dist"))
    idx = None
    hist_arr = None
    if not hgt_only:
        idx = clean_v2_index_for(cf, int(u))
        hist_arr = np.asarray(sorted(hist), dtype=np.int64)
    for start in range(0, int(cands_all.size), CAND_BATCH):
        end = min(start + CAND_BATCH, int(cands_all.size))
        cands = cands_all[start:end]
        n_b = int(end - start)
        if hgt_only:
            A_s = np.zeros((n_b, 0), dtype=np.float32)
            H_s = np.zeros((n_b, 3), dtype=np.float32)
            L2_s = np.zeros((n_b, 1), dtype=np.float32)
        else:
            A = vectorized_clean_v2_A_for_user(
                int(u),
                cands,
                bundle["model_train"],
                bundle["popularity"],
                bundle["item_kg_degree"],
                idx,
                n_x_sizes=n_x_sizes,
            )
            A_s = scale_arr(A, ctx["a_mean"], ctx["a_scale"])
            if no_dist:
                H_s = np.zeros((n_b, 3), dtype=np.float32)
                L2_s = np.zeros((n_b, 1), dtype=np.float32)
            elif ctx.get("no_leg"):
                H, _l2 = h3_and_l2(hist_arr, cands, idx)
                H_s = scale_arr(H, ctx["h_mean"], ctx["h_scale"])
                L2_s = np.zeros((n_b, 1), dtype=np.float32)
            else:
                H, L2 = h3_and_l2(hist_arr, cands, idx)
                H_s = scale_arr(H, ctx["h_mean"], ctx["h_scale"])
                L2_s = scale_arr(L2, ctx["l2_mean"], ctx["l2_scale"])
        item_idx = torch.from_numpy(cands.astype(np.int64) + item_offset).to(device)
        user_idx = torch.full((n_b,), int(u), dtype=torch.int64, device=device)
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


def check_l2_vs_pair_table(bundle, cf, l2_scaler, n_check: int = 64) -> dict[str, Any]:
    """Full-rank L2 formula must match V2 scaled LEG_K2 on held-out val pairs."""
    va = bundle["val_pairs"]
    raw_va = np.load(V2 / "representations" / "LEG_K2_val.npy").astype(np.float32)
    if raw_va.ndim == 1:
        raw_va = raw_va[:, None]
    mt = bundle["model_train"]
    rng = np.random.default_rng(20260818)
    pos = np.flatnonzero(va["label"] > 0.5)
    take = rng.choice(pos, size=min(n_check, int(pos.size)), replace=False)
    diffs = []
    for r in take.tolist():
        u = int(va["user_id"][r])
        x = int(va["item_id"][r])
        hist = np.asarray(sorted(mt.get(u, ())), dtype=np.int64)
        idx = clean_v2_index_for(cf, u)
        _, l2 = h3_and_l2(hist, np.asarray([x], dtype=np.int64), idx)
        scaled = scale_arr(l2, l2_scaler.mean_.astype(np.float32), l2_scaler.scale_.astype(np.float32))
        diffs.append(abs(float(scaled[0, 0]) - float(raw_va[r, 0])))
    mx = float(max(diffs)) if diffs else float("nan")
    return {"n": len(diffs), "max_abs": mx, "mean_abs": float(np.mean(diffs)), "PASS": bool(mx < L2_MATCH_ATOL)}


def write_protocol(d: dict[str, Path]) -> None:
    md = f"""# FULLRANK_PROTOCOL

Generated {utc_now()}.

## Dataset
Corrected **Last-FM\\*** (IntentAwareRS `--resolveDataLeakage yes`).  
Config: `configs/lastfm_star_race_clean_3.yaml`.  
TEST is read **only** for pair-overlap audit. TEST is **not scored**.

## Frozen model
`{ARCH_NAME}`  
Checkpoints: `{TRAIN_OUT}/06_CHECKPOINTS/final_seed{{101,202,303}}_best.pt`  
Epoch = sampled-val NDCG@20 best from TRUE FINAL JOINT training. **Not re-selected after full-rank.**
C1 reproduction checkpoints are **not** evaluated here.

## Candidate universe
For user `u`: `C_u = {{0, …, n_items-1}} \\ H_u^{{model_train}}`.  
No sampled negatives, no popularity/ANN/artist prune.  
Same item IDs for every seed.

## Train mask
Only `model_train` history. Validation positives are **not** masked.

## Relevance
Binary. `P_u` = **all** `valid.txt` positives. No capping. No test labels.

## Ranking
Score `s_final = s_B0 + delta_L2` on every `X ∈ C_u` in candidate batches of {CAND_BATCH}.  
Sort: higher score first; **tie → lower item_id** (`publication_full_rank_evaluator.dense_topk`).  
Chunking changes memory layout only; ranking ≡ scoring the full `C_u` at once.

## Metrics
NDCG@K / Recall@K / Precision@K for K∈{{5,10,20}}.  
DCG@K = Σ_r rel_r / log2(r+1). IDCG@K = same for min(K, |P_u|) ones at the top.  
Recall@K = |TopK ∩ P_u| / |P_u|. Precision@K = |TopK ∩ P_u| / K.  
MRR = 1/rank of first relevant among **all eligible** (train-masked) items; 0 if none.  
MAP@20 = (1/|P_u|) Σ_{{k: rel}} P@k  (**standard denom**, not n_hits).  
Macro-mean over users with |P_u| ≥ 1. Users with 0 val positives are skipped.

## Users with |C_u| < K
Not expected (catalog ~48k, history << that). If it occurred, Precision still uses /K.

## Beyond-accuracy (secondary)
CatalogCoverage@20, ARP@20, LongTailShare@20 from Top-20.  
`pop(i)` = # model_train users with i. TAIL = below median among items with pop>0.
"""
    (d["audit"] / "FULLRANK_PROTOCOL.md").write_text(md, encoding="utf-8")
    (d["report"] / "FULLRANK_PROTOCOL.md").write_text(md, encoding="utf-8")
    (OUT / "FULLRANK_PROTOCOL.md").write_text(md, encoding="utf-8")


def fail(d: dict[str, Path], reason: str, extra: dict | None = None) -> None:
    payload = {"FULLRANK_VERDICT": "DATA_AUDIT_FAILED", "reason": reason, "extra": extra or {}, "TEST_STATUS": "LOCKED_NOT_RUN"}
    write_json(d["report"] / "FULLRANK_MANIFEST.json", payload)
    print(f"FULLRANK_VERDICT = DATA_AUDIT_FAILED", flush=True)
    print(f"TEST_STATUS = LOCKED_NOT_RUN", flush=True)
    raise SystemExit(reason)


def load_sampled_rows(run_seeds: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    man_p = training_manifest_path()
    if man_p.exists():
        man = json.loads(man_p.read_text())
        rows = man.get("results") or []
        if rows:
            return {int(r["seed"]): r for r in rows}
    out: dict[int, dict[str, Any]] = {}
    for seed in run_seeds:
        p = TRAIN_OUT / SEED_DIR_NAME[seed] / "seed_summary.json"
        if not p.exists():
            raise SystemExit(f"EVALUATION_FAILED: missing {p}")
        out[seed] = json.loads(p.read_text())
    return out


def main() -> None:
    run_seeds = selected_seeds()
    d = dirs()
    write_protocol(d)
    wait_for_training(run_seeds)
    print(
        f"[fullrank] CAND_BATCH={CAND_BATCH} seeds={list(run_seeds)} "
        f"NO_DIST={int(NO_DIST)} HGT_ONLY={int(HGT_ONLY)} NO_LEG={int(NO_LEG)} arch={ARCH_NAME}",
        flush=True,
    )

    audit = data_audit()
    write_json(d["audit"] / "FULLRANK_DATA_AUDIT.json", audit)
    md = [
        "# FULLRANK_DATA_AUDIT",
        "",
        f"Generated {utc_now()}.",
        "",
        f"Dataset: **{audit['dataset']}**",
        "",
        f"- model_train ∩ valid = {audit['overlaps']['model_train_cap_valid']}",
        f"- model_train ∩ test = {audit['overlaps']['model_train_cap_test']}",
        f"- valid ∩ test = {audit['overlaps']['valid_cap_test']}",
        "",
        "TEST file was hashed and used for overlap counts only. **Not scored.**",
        "",
        "```json",
        json.dumps(audit["hashes"], indent=2),
        "```",
    ]
    (d["audit"] / "FULLRANK_DATA_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "FULLRANK_DATA_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if audit["FAILED"]:
        fail(d, "FAILED_DATA_AUDIT", audit)

    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    graph = load_data_and_typed_graph(
        bundle["cfg"], bundle["model_train"], max_kg_edges=bundle["cfg"].get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000)
    )
    n_items = int(graph["meta"]["n_items"])
    item_offset = int(graph["meta"]["item_offset"])
    n_users_g = int(graph["meta"]["n_users"])
    catalog_ids = np.arange(n_items, dtype=np.int64)
    catalog_hash = hashlib.sha256(catalog_ids.tobytes()).hexdigest()

    mt = bundle["model_train"]
    va_pos = load_user_sets(SPLITS / "valid.txt")
    eval_users_all = sorted(int(u) for u, items in va_pos.items() if items)
    skipped_empty = sorted(int(u) for u, items in va_pos.items() if not items)
    skipped_no_file = []

    cand_lens = []
    pos_lens = []
    for u in eval_users_all:
        hist = mt.get(int(u), set())
        pos = va_pos[int(u)]
        if hist & pos:
            fail(d, "validation positive inside model_train history", {"user": int(u)})
        cand_lens.append(n_items - len(hist))
        pos_lens.append(len(pos))
    cand_lens_a = np.asarray(cand_lens, dtype=np.int64)
    pos_lens_a = np.asarray(pos_lens, dtype=np.float64)

    cand_stats = {
        "catalog_size": n_items,
        "n_users_graph": n_users_g,
        "candidate_universe_hash": catalog_hash,
        "n_eval_users": len(eval_users_all),
        "mean_candidates": float(cand_lens_a.mean()),
        "median_candidates": float(np.median(cand_lens_a)),
        "min_candidates": int(cand_lens_a.min()),
        "max_candidates": int(cand_lens_a.max()),
        "mean_val_positives": float(pos_lens_a.mean()),
        "median_val_positives": float(np.median(pos_lens_a)),
        "max_val_positives": int(pos_lens_a.max()),
        "users_with_lt_K_candidates": int((cand_lens_a < 20).sum()),
    }
    write_json(d["audit"] / "CATALOG.json", cand_stats)
    cand_row = {
        "catalog_size": n_items,
        "candidate_universe_hash": catalog_hash,
        **{k: cand_stats[k] for k in cand_stats if k != "candidate_universe_hash"},
    }
    cand_fields = ["catalog_size", "candidate_universe_hash", "n_users_graph", "n_eval_users", "mean_candidates", "median_candidates", "min_candidates", "max_candidates", "mean_val_positives", "median_val_positives", "max_val_positives", "users_with_lt_K_candidates"]
    write_csv(d["metrics"] / "FULLRANK_CANDIDATE_COUNTS.csv", [cand_row], cand_fields)
    write_csv(OUT / "FULLRANK_CANDIDATE_COUNTS.csv", [cand_row], cand_fields)
    user_row = {
        "total_valid_txt_users": len(va_pos),
        "users_with_ge1_val_positive": len(eval_users_all),
        "users_evaluated": len(eval_users_all),
        "users_skipped_zero_positives": len(skipped_empty),
        "users_skipped_other": len(skipped_no_file),
        "skip_reason_zero_positives": "empty valid.txt item list",
    }
    user_fields = [
        "total_valid_txt_users",
        "users_with_ge1_val_positive",
        "users_evaluated",
        "users_skipped_zero_positives",
        "users_skipped_other",
        "skip_reason_zero_positives",
    ]
    write_csv(d["metrics"] / "FULLRANK_USER_COUNTS.csv", [user_row], user_fields)
    write_csv(OUT / "FULLRANK_USER_COUNTS.csv", [user_row], user_fields)

    if HGT_ONLY:
        cf = None
        l2_scaler = None
        a_scaler = None
        h_scaler = None
        l2_check = {"SKIPPED": True, "reason": "HGT_ONLY ranker has no A5/H3/LEG"}
        write_json(d["audit"] / "LEG_K2_PAIR_MATCH.json", l2_check)
    else:
        cf = ensure_cross_fit(bundle)
        with (TRAIN_OUT / "00_AUDIT" / "a_scaler.pkl").open("rb") as f:
            a_scaler = pickle.load(f)
        if NO_DIST:
            l2_scaler = None
            h_scaler = None
            l2_check = {"SKIPPED": True, "reason": "NO_DIST HGT+A5 ranker has no LEG/H3"}
            write_json(d["audit"] / "LEG_K2_PAIR_MATCH.json", l2_check)
        elif NO_LEG:
            l2_scaler = None
            with (TRAIN_OUT / "00_AUDIT" / "h_scaler.pkl").open("rb") as f:
                h_scaler = pickle.load(f)
            l2_check = {"SKIPPED": True, "reason": "NO_LEG HGT+A5+H3 ranker has zero LEG"}
            write_json(d["audit"] / "LEG_K2_PAIR_MATCH.json", l2_check)
        else:
            with (V2 / "representations" / "LEG_K2_scaler.pkl").open("rb") as f:
                l2_scaler = pickle.load(f)
            l2_check = check_l2_vs_pair_table(bundle, cf, l2_scaler)
            write_json(d["audit"] / "LEG_K2_PAIR_MATCH.json", l2_check)
            if not l2_check["PASS"]:
                fail(d, "LEG_K2 full-rank formula disagrees with V2 pair-table features", l2_check)
            with (TRAIN_OUT / "00_AUDIT" / "h_scaler.pkl").open("rb") as f:
                h_scaler = pickle.load(f)

    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    pop = np.asarray(bundle["popularity"], dtype=np.float64)
    buckets = item_pop_buckets(pop)
    sampled_by_seed = load_sampled_rows(run_seeds)

    seed_rows = []
    coverage_union: dict[int, set[int]] = {s: set() for s in run_seeds}
    arp_seed: dict[int, list[float]] = {s: [] for s in run_seeds}
    tail_seed: dict[int, list[float]] = {s: [] for s in run_seeds}

    for seed in run_seeds:
        ckpt_path = TRAIN_OUT / "06_CHECKPOINTS" / f"final_seed{seed}_best.pt"
        payload = torch.load(ckpt_path, map_location="cpu")
        if int(payload.get("seed", -1)) != seed:
            fail(d, f"checkpoint seed mismatch {ckpt_path}")
        if str(payload.get("architecture_name")) != ARCH_NAME:
            fail(d, f"architecture_name mismatch {payload.get('architecture_name')}")
        if payload.get("TEST_STATUS") != "LOCKED_NOT_RUN":
            fail(d, "checkpoint TEST_STATUS unexpected")
        sampled_ndcg = float(payload.get("sampled_NDCG@20", sampled_by_seed[seed]["best_sampled_NDCG@20"]))
        best_epoch = int(payload["best_epoch"])
        print(
            f"[fullrank] seed={seed} loading frozen best_epoch={best_epoch} sampled_NDCG@20={sampled_ndcg:.6f}",
            flush=True,
        )

        if HGT_ONLY:
            model = build_hgt_only_ranker(bundle, graph, device, d=64, layers=2, heads=2)
        elif NO_DIST:
            model = build_hgt_a5_ranker(bundle, graph, device, d=64, layers=2, heads=2)
        elif NO_LEG:
            model = build_hgt_a5_h3_noleg(bundle, graph, device, d=64, layers=2, heads=2)
        else:
            model = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        z = model.node_z().detach()
        fold_sizes: dict[int, np.ndarray] = {}
        ctx = {
            "device": device,
            "model": model,
            "z": z,
            "no_dist": bool(NO_DIST or HGT_ONLY),
            "hgt_only": bool(HGT_ONLY),
            "no_leg": bool(NO_LEG),
        }
        if not HGT_ONLY:
            ctx["a_mean"] = a_scaler.mean_.astype(np.float32)
            ctx["a_scale"] = a_scaler.scale_.astype(np.float32)
        if not NO_DIST and not HGT_ONLY:
            ctx["h_mean"] = h_scaler.mean_.astype(np.float32)
            ctx["h_scale"] = h_scaler.scale_.astype(np.float32)
            if not NO_LEG:
                ctx["l2_mean"] = l2_scaler.mean_.astype(np.float32)
                ctx["l2_scale"] = l2_scaler.scale_.astype(np.float32)

        sums = {f"NDCG@{k}": 0.0 for k in KS}
        sums.update({f"Recall@{k}": 0.0 for k in KS})
        sums.update({f"Precision@{k}": 0.0 for k in KS})
        sums["MRR"] = 0.0
        sums["HitRate@20"] = 0.0
        sums["MAP@20"] = 0.0
        n = 0
        t0 = time.time()
        per_user_path = d[SEED_DIR_KEY[seed]] / "per_user.npz"
        pu_users, pu_ndcg20, pu_rec20 = [], [], []

        eq_n = int(os.environ.get("LASTFM_EQ_TEST_USERS", "0") or "0")
        if eq_n > 0:
            users_eq = list(eval_users_all)[:eq_n]
            max_abs = 0.0
            max_rank_mismatch = 0
            t_eq = time.time()
            print(f"[eq-test] seed={seed} users={len(users_eq)} comparing expand vs torch.full", flush=True)
            for u in users_eq:
                if HGT_ONLY:
                    n_x_sizes_u = None
                else:
                    fold = cf.user_to_fold.get(int(u), -1)
                    if fold not in fold_sizes:
                        fold_sizes[fold] = neighborhood_sizes(clean_v2_index_for(cf, int(u)))
                    n_x_sizes_u = fold_sizes[fold]
                hist = mt.get(int(u), set())
                kwargs = dict(
                    n_items=n_items,
                    item_offset=item_offset,
                    hist=hist,
                    bundle=bundle,
                    cf=cf,
                    ctx=ctx,
                    n_x_sizes=n_x_sizes_u,
                )
                s_new = score_user_catalog(int(u), **kwargs)
                s_old = score_user_catalog_legacy_torch_full(int(u), **kwargs)
                mask = np.isfinite(s_new) | np.isfinite(s_old)
                dif = float(np.max(np.abs(s_new[mask] - s_old[mask]))) if mask.any() else 0.0
                max_abs = max(max_abs, dif)
                top_new = np.argsort(-s_new, kind="stable")[:20]
                top_old = np.argsort(-s_old, kind="stable")[:20]
                mismatch = int(np.sum(top_new != top_old))
                max_rank_mismatch = max(max_rank_mismatch, mismatch)
            elapsed = time.time() - t_eq
            verdict = "PASS" if max_abs < 1e-5 and max_rank_mismatch == 0 else "FAIL"
            print(
                f"[eq-test] {verdict} max_abs={max_abs:.3e} "
                f"max_top20_mismatches={max_rank_mismatch} elapsed_s={elapsed:.1f}",
                flush=True,
            )
            out_eq = d[SEED_DIR_KEY[seed]] / "EQ_EXPAND_VS_FULL.json"
            out_eq.write_text(
                json.dumps(
                    {
                        "verdict": verdict,
                        "max_abs": max_abs,
                        "max_top20_mismatches": max_rank_mismatch,
                        "n_users": len(users_eq),
                        "elapsed_s": elapsed,
                        "CAND_BATCH": CAND_BATCH,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if verdict != "PASS":
                raise SystemExit(f"EQUIVALENCE_FAILED max_abs={max_abs} top20_mis={max_rank_mismatch}")
            print("[eq-test] done — exiting before full catalog run", flush=True)
            raise SystemExit(0)

        for ui, u in enumerate(eval_users_all):
            if HGT_ONLY:
                n_x_sizes_u = None
            else:
                fold = cf.user_to_fold.get(int(u), -1)
                if fold not in fold_sizes:
                    fold_sizes[fold] = neighborhood_sizes(clean_v2_index_for(cf, int(u)))
                n_x_sizes_u = fold_sizes[fold]
            hist = mt.get(int(u), set())
            pos = va_pos[int(u)]
            scores = score_user_catalog(
                int(u),
                n_items=n_items,
                item_offset=item_offset,
                hist=hist,
                bundle=bundle,
                cf=cf,
                ctx=ctx,
                n_x_sizes=n_x_sizes_u,
            )
            m = evaluate_user_dense(scores, positive_items=pos, train_items=hist, ks=KS, max_k=20)
            top20 = dense_topk(scores, k=20, mask_items=hist)
            m["Precision@5"] = precision_at_k(top20, pos, 5)
            m["Precision@10"] = precision_at_k(top20, pos, 10)
            m["Precision@20"] = precision_at_k(top20, pos, 20)
            m["MAP@20"] = map_at_k(top20, pos, 20)
            for k, v in m.items():
                if k in sums:
                    sums[k] += float(v)
            n += 1
            pu_users.append(int(u))
            pu_ndcg20.append(float(m["NDCG@20"]))
            pu_rec20.append(float(m["Recall@20"]))
            coverage_union[seed].update(int(x) for x in top20.tolist())
            pops = pop[top20]
            arp_seed[seed].append(float(pops.mean()) if pops.size else 0.0)
            tail_seed[seed].append(float(np.mean([str(buckets[int(i)]) == "TAIL" for i in top20.tolist()])))
            if (ui + 1) % 100 == 0 or ui == 0:
                elapsed = max(time.time() - t0, 1e-9)
                rate = (ui + 1) / elapsed
                eta_h = (len(eval_users_all) - ui - 1) / max(rate, 1e-9) / 3600.0
                msg = (
                    f"[fullrank seed={seed}] {ui+1}/{len(eval_users_all)} "
                    f"u/s={rate:.2f} ETA={eta_h:.1f}h "
                    f"NDCG@20={sums['NDCG@20']/(ui+1):.4f}"
                )
                print(msg, flush=True)
                (d[SEED_DIR_KEY[seed]] / "PROGRESS.txt").write_text(
                    msg + f"\nelapsed_s={elapsed:.1f}\n", encoding="utf-8"
                )
            if (ui + 1) % 500 == 0:
                empty_cache()
                gc.collect()

        mean = {k: float(v / n) for k, v in sums.items()}
        mean["n_users"] = float(n)
        seed_dir = d[SEED_DIR_KEY[seed]]
        np.savez_compressed(
            per_user_path,
            users=np.asarray(pu_users, dtype=np.int64),
            ndcg20=np.asarray(pu_ndcg20, dtype=np.float64),
            recall20=np.asarray(pu_rec20, dtype=np.float64),
        )
        row = {
            "seed": seed,
            "EVALUATION_PROTOCOL_SAMPLED": "SAMPLED_20_NEG",
            "EVALUATION_PROTOCOL_FULLRANK": "FULL_CATALOG",
            "sampled_NDCG@20": sampled_ndcg,
            "sampled_best_epoch": best_epoch,
            "fullrank_NDCG@5": mean["NDCG@5"],
            "fullrank_NDCG@10": mean["NDCG@10"],
            "fullrank_NDCG@20": mean["NDCG@20"],
            "fullrank_Recall@5": mean["Recall@5"],
            "fullrank_Recall@10": mean["Recall@10"],
            "fullrank_Recall@20": mean["Recall@20"],
            "fullrank_Precision@5": mean["Precision@5"],
            "fullrank_Precision@10": mean["Precision@10"],
            "fullrank_Precision@20": mean["Precision@20"],
            "fullrank_MRR": mean["MRR"],
            "fullrank_HitRate@20": mean["HitRate@20"],
            "fullrank_MAP@20": mean["MAP@20"],
            "CatalogCoverage@20": float(len(coverage_union[seed]) / n_items),
            "ARP@20": float(np.mean(arp_seed[seed])),
            "LongTailShare@20": float(np.mean(tail_seed[seed])),
            "n_users": n,
            "seconds": time.time() - t0,
        }
        write_json(seed_dir / "SEED_METRICS.json", row)
        seed_rows.append(row)
        print(f"[fullrank] seed={seed} NDCG@20={mean['NDCG@20']:.6f} Recall@20={mean['Recall@20']:.6f}", flush=True)
        del model, z, ctx
        empty_cache()
        gc.collect()

    if set(run_seeds) != set(SEEDS):
        write_json(
            OUT / "PARTIAL_FULLRANK.json",
            {
                "seeds": list(run_seeds),
                "rows": seed_rows,
                "timestamp": utc_now(),
                "CAND_BATCH": CAND_BATCH,
                "TEST_STATUS": "LOCKED_NOT_RUN",
            },
        )
        print("\n" + "=" * 60, flush=True)
        print("PARTIAL_FULLRANK seeds=" + ",".join(str(s) for s in run_seeds), flush=True)
        for r in seed_rows:
            print(
                f"SEED{r['seed']}_FULLRANK_NDCG20 = {r['fullrank_NDCG@20']:.6f}",
                flush=True,
            )
            print(
                f"SEED{r['seed']}_FULLRANK_RECALL20 = {r['fullrank_Recall@20']:.6f}",
                flush=True,
            )
        print("TEST_STATUS =\n    LOCKED_NOT_RUN", flush=True)
        print("FULLRANK_VERDICT =\n    PARTIAL_COMPLETE", flush=True)
        print("=" * 60, flush=True)
        return

    fields = list(seed_rows[0].keys())
    write_csv(d["metrics"] / "FULLRANK_RESULTS_BY_SEED.csv", seed_rows, fields)
    write_csv(OUT / "FULLRANK_RESULTS_BY_SEED.csv", seed_rows, fields)

    def agg(key: str) -> dict[str, float]:
        vals = np.asarray([r[key] for r in seed_rows], dtype=np.float64)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    summary_row = {
        "EVALUATION_PROTOCOL": "FULL_CATALOG",
        "fullrank_NDCG@20_mean": agg("fullrank_NDCG@20")["mean"],
        "fullrank_NDCG@20_std": agg("fullrank_NDCG@20")["std"],
        "fullrank_NDCG@20_min": agg("fullrank_NDCG@20")["min"],
        "fullrank_NDCG@20_max": agg("fullrank_NDCG@20")["max"],
        "fullrank_Recall@20_mean": agg("fullrank_Recall@20")["mean"],
        "fullrank_Recall@20_std": agg("fullrank_Recall@20")["std"],
        "fullrank_MRR_mean": agg("fullrank_MRR")["mean"],
        "fullrank_Precision@20_mean": agg("fullrank_Precision@20")["mean"],
        "sampled_NDCG@20_mean": agg("sampled_NDCG@20")["mean"],
        "sampled_minus_fullrank_NDCG@20": agg("sampled_NDCG@20")["mean"] - agg("fullrank_NDCG@20")["mean"],
        "CatalogCoverage@20_mean": agg("CatalogCoverage@20")["mean"],
        "ARP@20_mean": agg("ARP@20")["mean"],
        "LongTailShare@20_mean": agg("LongTailShare@20")["mean"],
    }
    write_csv(d["metrics"] / "FULLRANK_RESULTS_SUMMARY.csv", [summary_row], list(summary_row))
    write_csv(OUT / "FULLRANK_RESULTS_SUMMARY.csv", [summary_row], list(summary_row))

    by = {int(r["seed"]): r for r in seed_rows}
    report = f"""# FULLRANK_REPORT

Generated {utc_now()}.

**EVALUATION_PROTOCOL:** `FULL_CATALOG` = all Last-FM\\* items minus `model_train` history.  
**SAMPLED_20_NEG** is the frozen training selection metric. The two columns are **not comparable**.

Dataset = corrected NO-LEAKAGE Last-FM\\*. Wording: *full-ranking evaluation on corrected Last-FM\\**.  
Not a claim of "directly reproduced KGAT benchmark".

## Main table

| seed | EVALUATION_PROTOCOL (sampled) | sampled NDCG@20 | EVALUATION_PROTOCOL (full-rank) | full-rank NDCG@20 | full-rank Recall@20 | MRR |
|---:|---|---:|---|---:|---:|---:|
| 101 | SAMPLED_20_NEG | {by[101]['sampled_NDCG@20']:.6f} | FULL_CATALOG | {by[101]['fullrank_NDCG@20']:.6f} | {by[101]['fullrank_Recall@20']:.6f} | {by[101]['fullrank_MRR']:.6f} |
| 202 | SAMPLED_20_NEG | {by[202]['sampled_NDCG@20']:.6f} | FULL_CATALOG | {by[202]['fullrank_NDCG@20']:.6f} | {by[202]['fullrank_Recall@20']:.6f} | {by[202]['fullrank_MRR']:.6f} |
| 303 | SAMPLED_20_NEG | {by[303]['sampled_NDCG@20']:.6f} | FULL_CATALOG | {by[303]['fullrank_NDCG@20']:.6f} | {by[303]['fullrank_Recall@20']:.6f} | {by[303]['fullrank_MRR']:.6f} |
| mean | SAMPLED_20_NEG | {summary_row['sampled_NDCG@20_mean']:.6f} | FULL_CATALOG | {summary_row['fullrank_NDCG@20_mean']:.6f} | {summary_row['fullrank_Recall@20_mean']:.6f} | {summary_row['fullrank_MRR_mean']:.6f} |

sampled NDCG@20 − full-rank NDCG@20 (means) = {summary_row['sampled_minus_fullrank_NDCG@20']:.6f}

## Secondary (full-rank)

| seed | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | P@20 | MAP@20 | HitRate@20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | {by[101]['fullrank_NDCG@5']:.6f} | {by[101]['fullrank_NDCG@10']:.6f} | {by[101]['fullrank_Recall@5']:.6f} | {by[101]['fullrank_Recall@10']:.6f} | {by[101]['fullrank_Precision@20']:.6f} | {by[101]['fullrank_MAP@20']:.6f} | {by[101]['fullrank_HitRate@20']:.6f} |
| 202 | {by[202]['fullrank_NDCG@5']:.6f} | {by[202]['fullrank_NDCG@10']:.6f} | {by[202]['fullrank_Recall@5']:.6f} | {by[202]['fullrank_Recall@10']:.6f} | {by[202]['fullrank_Precision@20']:.6f} | {by[202]['fullrank_MAP@20']:.6f} | {by[202]['fullrank_HitRate@20']:.6f} |
| 303 | {by[303]['fullrank_NDCG@5']:.6f} | {by[303]['fullrank_NDCG@10']:.6f} | {by[303]['fullrank_Recall@5']:.6f} | {by[303]['fullrank_Recall@10']:.6f} | {by[303]['fullrank_Precision@20']:.6f} | {by[303]['fullrank_MAP@20']:.6f} | {by[303]['fullrank_HitRate@20']:.6f} |

## Beyond-accuracy (Top-20, pop from model_train)

| seed | CatalogCoverage@20 | ARP@20 | LongTailShare@20 |
|---:|---:|---:|---:|
| 101 | {by[101]['CatalogCoverage@20']:.6f} | {by[101]['ARP@20']:.4f} | {by[101]['LongTailShare@20']:.6f} |
| 202 | {by[202]['CatalogCoverage@20']:.6f} | {by[202]['ARP@20']:.4f} | {by[202]['LongTailShare@20']:.6f} |
| 303 | {by[303]['CatalogCoverage@20']:.6f} | {by[303]['ARP@20']:.4f} | {by[303]['LongTailShare@20']:.6f} |

## Frozen checkpoints (selected BEFORE this evaluation)

| seed | best_epoch (sampled NDCG@20) |
|---:|---:|
| 101 | {by[101]['sampled_best_epoch']} |
| 202 | {by[202]['sampled_best_epoch']} |
| 303 | {by[303]['sampled_best_epoch']} |

No retrain. No epoch reselection. No new negatives.

## Q&A

1. Corrected NO-LEAKAGE Last-FM\\*? **YES**
2. Checkpoint selected before full-rank? **YES**
3. Model retrained? **NO**
4. Sampled negatives used in ranking? **NO**
5. Whole catalog after model_train mask? **YES**
6. All validation positives retained? **YES**
7. Train history masked? **YES**
8. Full-rank NDCG@20 mean = **{summary_row['fullrank_NDCG@20_mean']:.6f}** (std {summary_row['fullrank_NDCG@20_std']:.6f})
9. Full-rank Recall@20 mean = **{summary_row['fullrank_Recall@20_mean']:.6f}**
10. Seed range NDCG@20: {summary_row['fullrank_NDCG@20_min']:.6f} … {summary_row['fullrank_NDCG@20_max']:.6f}
11. Gap sampled − full-rank NDCG@20 (means) = {summary_row['sampled_minus_fullrank_NDCG@20']:.6f}
12. TEST accessed for scoring? **NO** (`LOCKED_NOT_RUN`)
"""
    (d["report"] / "FULLRANK_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "FULLRANK_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        "run_id": "LASTFM_NOLEAK_FULLRANK_VALIDATION_V1",
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "dataset": "CORRECTED_LASTFM_STAR_NO_LEAKAGE",
        "model": "HGT64_2L_2H_A5_H3_LEG_K2",
        "FULLRANK_PROTOCOL": "ALL_CATALOG_MINUS_MODEL_TRAIN_HISTORY",
        "RELEVANCE": "ALL_VALIDATION_POSITIVES_BINARY",
        "by_seed": seed_rows,
        "summary": summary_row,
        "FULLRANK_VERDICT": "COMPLETE",
        "TEST_STATUS": "LOCKED_NOT_RUN",
        "catalog": cand_stats,
        "l2_pair_match": l2_check,
    }
    write_json(d["report"] / "FULLRANK_MANIFEST.json", manifest)
    write_json(OUT / "FULLRANK_MANIFEST.json", manifest)

    print("\n" + "=" * 60, flush=True)
    print("DATASET =\n    CORRECTED_LASTFM_STAR_NO_LEAKAGE", flush=True)
    print("MODEL =\n    HGT64_2L_2H_A5_H3_LEG_K2", flush=True)
    print("FULLRANK_PROTOCOL =\n    ALL_CATALOG_MINUS_MODEL_TRAIN_HISTORY", flush=True)
    print("RELEVANCE =\n    ALL_VALIDATION_POSITIVES_BINARY", flush=True)
    for seed in SEEDS:
        print(f"SEED{seed}_FULLRANK_NDCG20 = {by[seed]['fullrank_NDCG@20']:.6f}", flush=True)
    print(f"FULLRANK_NDCG20_MEAN = {summary_row['fullrank_NDCG@20_mean']:.6f}", flush=True)
    print(f"FULLRANK_NDCG20_STD = {summary_row['fullrank_NDCG@20_std']:.6f}", flush=True)
    print(f"FULLRANK_RECALL20_MEAN = {summary_row['fullrank_Recall@20_mean']:.6f}", flush=True)
    print(f"FULLRANK_MRR_MEAN = {summary_row['fullrank_MRR_mean']:.6f}", flush=True)
    print(f"SAMPLED_NDCG20_MEAN = {summary_row['sampled_NDCG@20_mean']:.6f}", flush=True)
    print("FULLRANK_VERDICT =\n    COMPLETE", flush=True)
    print("TEST_STATUS =\n    LOCKED_NOT_RUN", flush=True)
    print("=" * 60, flush=True)
    print("THEN STOP. DO NOT RETRAIN. DO NOT CHANGE THE MODEL.", flush=True)


if __name__ == "__main__":
    main()
