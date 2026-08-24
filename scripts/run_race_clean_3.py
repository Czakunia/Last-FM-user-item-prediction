#!/usr/bin/env python3
"""RACE CLEAN 3 — Last-FM* + CLEAN formulas: true Tabular A + 3-D A11 pool + 265-D MLP.

Outputs under KRAM_FINAL_WORK/RACE_CLEAN_3/ — does NOT touch V1 measure_race / CLEAN_V2 / outputs/lastfm_full.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_publication_our_hgt_fullrank import train_and_save  # noqa: E402
from src.lastfm_lp.clean_v2.constants import (  # noqa: E402
    CLEAN_V2_FUSION_DIM,
    CLEAN_V2_HCR_DIM,
    CLEAN_V2_TOP_K,
)
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.routing import (  # noqa: E402
    CLEAN_V2_ROUTING_POLICIES,
    CleanV2RoutingPolicy,
    aggregate_clean_v2_a11_pool,
)
from src.lastfm_lp.clean_v2.tabular_true import (  # noqa: E402
    CLEAN_V2_TABULAR_FEATURE_NAMES,
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.evaluation.publication_full_rank_evaluator import (  # noqa: E402
    dense_topk,
    metrics_from_topk,
)
from src.lastfm_lp.models.encoders import HGTGraphEncoder  # noqa: E402
from src.lastfm_lp.models.fusion import LateFusionHead, RecommendationModel  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

OUT = ROOT / "KRAM_FINAL_WORK" / "RACE_CLEAN_3"
SPLITS = ROOT / "outputs" / "lastfm_star" / "splits"
SEEDS = (101, 202, 303)
N_ITEMS = 48123
KS = (10, 20, 50)
SHARD = 500
ACT_ORDER = ("VERY_LIGHT", "LIGHT", "MEDIUM", "HEAVY", "VERY_HEAVY")
POP_ORDER = ("HEAD", "MID", "TAIL")
STOP_FILE = OUT / "STOP_FULLRANK"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def policy_dirs(policy: str) -> dict[str, Path]:
    base = OUT / "race" / policy
    return {
        "features": base / "features",
        "ckpt_root": base / "checkpoints",
        "fullval": base / "fullval",
        "fulltest": base / "fulltest",
    }


def shared_A_dir() -> Path:
    return OUT / "features" / "A_true"


def materialize_A_true(bundle: dict[str, Any]) -> Path:
    out_dir = shared_A_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = CLEAN_V2_TABULAR_FEATURE_NAMES
    complete = out_dir / "COMPLETE"
    pair_map = {
        "train": bundle["train_pairs"],
        "val": bundle["val_pairs"],
    }
    if complete.exists() and all(
        (out_dir / f"X_{s}.npy").exists()
        and np.load(out_dir / f"X_{s}.npy", mmap_mode="r").shape
        == (len(pair_map[s]["label"]), 5)
        for s in pair_map
    ):
        print("[race_clean_3] skip A_true: COMPLETE", flush=True)
        return out_dir

    cf = ensure_cross_fit(bundle)
    model_train = bundle["model_train"]
    pop = bundle["popularity"]
    kg = bundle["item_kg_degree"]
    # cache |N_X| per fold index
    size_cache: dict[int, np.ndarray] = {}
    chunk_size = 20_000
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")

    for split, pairs in pair_map.items():
        n = len(pairs["label"])
        xpath = out_dir / f"X_{split}.npy"
        if xpath.exists() and np.load(xpath, mmap_mode="r").shape == (n, 5):
            print(f"[race_clean_3] skip A {split}", flush=True)
            continue
        X = np.lib.format.open_memmap(xpath, mode="w+", dtype=np.float32, shape=(n, 5))
        users, items = pairs["user_id"], pairs["item_id"]
        pbar = tqdm(total=n, desc=f"A_true:{split}", mininterval=2.0)
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            user_groups: dict[int, list[int]] = defaultdict(list)
            for r in range(chunk_start, chunk_end):
                user_groups[int(users[r])].append(r)
            for u, rows in user_groups.items():
                idx = clean_v2_index_for(cf, int(u))
                fold = cf.user_to_fold.get(int(u), -1)
                if fold not in size_cache:
                    print(f"[race_clean_3] precompute |N_X| fold={fold} …", flush=True)
                    size_cache[fold] = neighborhood_sizes(idx)
                r_idx = np.asarray(rows, dtype=np.int64)
                X[r_idx] = vectorized_clean_v2_A_for_user(
                    int(u),
                    items[r_idx],
                    model_train,
                    pop,
                    kg,
                    idx,
                    n_x_sizes=size_cache[fold],
                )
            X.flush()
            pbar.update(chunk_end - chunk_start)
        pbar.close()
        del X
    complete.write_text("ok\n", encoding="utf-8")
    write_json(
        out_dir / "meta.json",
        {
            "created": utc_now(),
            "feature_names": names,
            "note": "CLEAN V2 true Jaccard/Cosine vs N_X; leave-one-fold index",
        },
    )
    print(f"[race_clean_3] A_true → {out_dir}", flush=True)
    return out_dir


def materialize_H(bundle: dict[str, Any], policy: CleanV2RoutingPolicy) -> Path:
    dirs = policy_dirs(policy)
    out_dir = dirs["features"]
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["mean:hcr_a11", "max:hcr_a11", "top3mean:hcr_a11"]
    complete = out_dir / "COMPLETE"
    pair_map = {"train": bundle["train_pairs"], "val": bundle["val_pairs"]}
    if complete.exists() and all(
        (out_dir / f"X_{s}.npy").exists()
        and np.load(out_dir / f"X_{s}.npy", mmap_mode="r").shape
        == (len(pair_map[s]["label"]), CLEAN_V2_HCR_DIM)
        for s in pair_map
    ):
        print(f"[race_clean_3] skip H {policy}: COMPLETE", flush=True)
        return out_dir

    cf = ensure_cross_fit(bundle)
    model_train = bundle["model_train"]
    chunk_size = 50_000
    (out_dir / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    for split, pairs in pair_map.items():
        n = len(pairs["label"])
        xpath = out_dir / f"X_{split}.npy"
        if xpath.exists() and np.load(xpath, mmap_mode="r").shape == (n, CLEAN_V2_HCR_DIM):
            print(f"[race_clean_3] skip H {policy} {split}", flush=True)
            continue
        X = np.lib.format.open_memmap(
            xpath, mode="w+", dtype=np.float32, shape=(n, CLEAN_V2_HCR_DIM)
        )
        users, items, labels = pairs["user_id"], pairs["item_id"], pairs["label"]
        pbar = tqdm(total=n, desc=f"H:{policy}:{split}", mininterval=2.0)
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            user_groups: dict[int, list[int]] = defaultdict(list)
            for r in range(chunk_start, chunk_end):
                user_groups[int(users[r])].append(r)
            for u, rows in user_groups.items():
                index = clean_v2_index_for(cf, int(u))
                base = set(model_train.get(u, ()))
                normal_rows: list[int] = []
                removal: dict[int, list[int]] = defaultdict(list)
                for r in rows:
                    i = int(items[r])
                    y = int(labels[r])
                    if y == 1 and i in base:
                        removal[i].append(r)
                    else:
                        normal_rows.append(r)
                if normal_rows:
                    r_idx = np.asarray(normal_rows, dtype=np.int64)
                    feats = aggregate_clean_v2_a11_pool(
                        base,
                        items[r_idx],
                        index,
                        policy=policy,
                        max_history=CLEAN_V2_TOP_K,
                    )
                    X[r_idx] = feats
                for cand_i, rlist in removal.items():
                    hist = base - {int(cand_i)}
                    feat = aggregate_clean_v2_a11_pool(
                        hist,
                        np.asarray([cand_i], dtype=np.int64),
                        index,
                        policy=policy,
                        max_history=CLEAN_V2_TOP_K,
                    )[0]
                    for r in rlist:
                        X[r] = feat
            X.flush()
            pbar.update(chunk_end - chunk_start)
        pbar.close()
        del X
    complete.write_text("ok\n", encoding="utf-8")
    write_json(
        out_dir / "SELECTION_META.json",
        {
            "policy": policy,
            "max_history": CLEAN_V2_TOP_K,
            "hcr_dim": CLEAN_V2_HCR_DIM,
            "self_exclusion": True,
            "downstream": "signed_a11_mean_max_top3mean",
            "created": utc_now(),
        },
    )
    print(f"[race_clean_3] H {policy} → {out_dir}", flush=True)
    return out_dir


def train_policy_seed(bundle: dict[str, Any], policy: str, seed: int) -> Path:
    dirs = policy_dirs(policy)
    hdir = dirs["features"]
    ckpt_root = dirs["ckpt_root"]
    stage = f"RACE_CLEAN_3_{policy}"
    ckpt = ckpt_root / stage / f"seed_{seed}"
    if (ckpt / "model.pt").exists():
        print(f"[race_clean_3] reuse ckpt {ckpt}", flush=True)
        return ckpt
    materialize_A_true(bundle)
    if not (hdir / "COMPLETE").exists():
        materialize_H(bundle, policy)  # type: ignore[arg-type]
    return train_and_save(
        bundle,
        stage=stage,
        use_a11=True,
        seed=seed,
        hcr_feature_dir=hdir,
        a_feature_dir=shared_A_dir(),
        ckpt_root=ckpt_root,
    )


def build_ctx(bundle: dict[str, Any], ckpt: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    device = resolve_torch_device("cpu")
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    meta_t = json.loads((ckpt / "train_meta.json").read_text())
    with (ckpt / "a_scaler.pkl").open("rb") as f:
        a_scaler = pickle.load(f)
    with (ckpt / "h_scaler.pkl").open("rb") as f:
        h_scaler = pickle.load(f)
    graph = load_data_and_typed_graph(
        cfg, bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
    edge_index_dict = {
        k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0
    }
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    encoder = HGTGraphEncoder(
        graph["meta"]["n_users"],
        graph["meta"]["n_entities"],
        metadata,
        edge_index_dict,
        embed_dim=64,
        n_layers=2,
        heads=2,
        dropout=0.1,
    ).to(device)
    hcr_dim = int(meta_t["hcr_dim"])
    assert hcr_dim == CLEAN_V2_HCR_DIM, hcr_dim
    fusion = LateFusionHead(
        graph_dim=encoder.context_dim,
        base_feature_dim=5,
        hcr_feature_dim=hcr_dim,
        hidden_dim=128,
        dropout=0.2,
        use_hcr=True,
        activation="gelu",
    ).to(device)
    assert encoder.context_dim + 1 + 5 + hcr_dim == CLEAN_V2_FUSION_DIM
    model = RecommendationModel(encoder, fusion).to(device)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        zemb = model.encoder.encode_all()
        item_offset = graph["meta"]["item_offset"]
        zi_all = zemb[item_offset : item_offset + N_ITEMS].detach()
    return {
        "device": device,
        "model": model,
        "zemb": zemb,
        "zi_all": zi_all,
        "a_mean": a_scaler.mean_.astype(np.float32),
        "a_scale": a_scaler.scale_.astype(np.float32),
        "h_mean": h_scaler.mean_.astype(np.float32),
        "h_scale": h_scaler.scale_.astype(np.float32),
        "n_x_sizes": None,
    }


@torch.no_grad()
def score_user_policy(u, bundle, ctx, cf, policy: str, n_x_sizes: np.ndarray) -> np.ndarray:
    device = ctx["device"]
    model = ctx["model"]
    pop = bundle["popularity"]
    kg_deg = bundle["item_kg_degree"]
    model_train = bundle["model_train"]
    all_items = np.arange(N_ITEMS, dtype=np.int64)
    idx = clean_v2_index_for(cf, int(u))
    hist = model_train.get(int(u), set())
    A = vectorized_clean_v2_A_for_user(
        int(u), all_items, model_train, pop, kg_deg, idx, n_x_sizes=n_x_sizes
    )
    A_s = ((A - ctx["a_mean"]) / np.maximum(ctx["a_scale"], 1e-12)).astype(np.float32)
    H = aggregate_clean_v2_a11_pool(
        hist,
        all_items,
        idx,
        policy=policy,  # type: ignore[arg-type]
        max_history=CLEAN_V2_TOP_K,
    )
    H_s = ((H - ctx["h_mean"]) / np.maximum(ctx["h_scale"], 1e-12)).astype(np.float32)
    zu = ctx["zemb"][int(u)]
    zi_all = ctx["zi_all"]
    graph_score = (zi_all * zu).sum(dim=-1)
    scores = np.empty(N_ITEMS, dtype=np.float64)
    for start in range(0, N_ITEMS, 16384):
        end = min(start + 16384, N_ITEMS)
        zi = zi_all[start:end]
        gctx = torch.cat(
            [zu.expand(end - start, -1), zi, zu * zi, (zu - zi).abs()], dim=-1
        )
        logits = model.fusion_head(
            graph_context=gctx,
            graph_score=graph_score[start:end],
            base_features=torch.from_numpy(A_s[start:end]).to(device),
            hcr_features=torch.from_numpy(H_s[start:end]).to(device),
        )
        scores[start:end] = logits.detach().cpu().numpy()
    return scores


def activity_bucket(h: int) -> str:
    if h <= 10:
        return "VERY_LIGHT"
    if h <= 25:
        return "LIGHT"
    if h <= 50:
        return "MEDIUM"
    if h <= 100:
        return "HEAVY"
    return "VERY_HEAVY"


def item_pop_buckets(popularity: np.ndarray) -> np.ndarray:
    """HEAD ≥ p90, MID p50–p90, TAIL < p50 among items with pop>0 (Phase 9)."""
    pop = np.asarray(popularity, dtype=np.float64)
    perc = np.zeros(len(pop), dtype=np.float64)
    pos = pop > 0
    if pos.any():
        x = pop[pos]
        ranks = np.empty_like(x)
        order = np.argsort(x, kind="mergesort")
        ranks[order] = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        perc[pos] = ranks
    bucket = np.array(["TAIL"] * len(pop), dtype=object)
    bucket[perc >= 0.90] = "HEAD"
    bucket[(perc >= 0.50) & (perc < 0.90)] = "MID"
    return bucket


def majority_pop_bucket(items: set[int], item_bucket: np.ndarray) -> str | None:
    if not items:
        return None
    from collections import Counter

    c = Counter(str(item_bucket[i]) for i in items if 0 <= int(i) < len(item_bucket))
    if not c:
        return None
    return str(c.most_common(1)[0][0])


def precision_map_at_k(top_items: np.ndarray, positives: set[int], k: int) -> tuple[float, float]:
    topk = [int(x) for x in np.asarray(top_items).tolist()[:k]]
    if not topk:
        return 0.0, 0.0
    hits = [1.0 if i in positives else 0.0 for i in topk]
    prec = float(sum(hits) / k)
    if sum(hits) == 0:
        return prec, 0.0
    cum = 0.0
    ap = 0.0
    for i, h in enumerate(hits, start=1):
        cum += h
        if h > 0:
            ap += cum / i
    return prec, float(ap / sum(hits))


def eval_policy_seed(
    bundle, policy: str, seed: int, *, split: str = "val"
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError(split)
    dirs = policy_dirs(policy)
    ckpt = dirs["ckpt_root"] / f"RACE_CLEAN_3_{policy}" / f"seed_{seed}"
    if not (ckpt / "model.pt").exists():
        raise SystemExit(f"missing ckpt {ckpt}")
    out_dir = dirs["fullval" if split == "val" else "fulltest"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_name = "VAL_METRICS.json" if split == "val" else "TEST_METRICS.json"
    metrics_path = out_dir / metrics_name
    cache_path = out_dir / "per_user_cache.npz"
    if metrics_path.exists() and (out_dir / "per_user.npz").exists():
        return json.loads(metrics_path.read_text())

    pos_file = "valid.txt" if split == "val" else "test.txt"
    val_pos = load_user_sets(SPLITS / pos_file)
    users = sorted(u for u in val_pos if val_pos[u])
    model_train = bundle["model_train"]
    cf = ensure_cross_fit(bundle)
    ctx = build_ctx(bundle, ckpt)

    # |N_X| from full_index is wrong for CLEAN — use per-fold cache lazily
    fold_sizes: dict[int, np.ndarray] = {}

    done: dict[int, dict[str, float]] = {}
    if cache_path.exists():
        z = np.load(cache_path)
        for i, u in enumerate(z["users"].tolist()):
            done[int(u)] = {
                "NDCG@10": float(z["ndcg10"][i]),
                "NDCG@20": float(z["ndcg20"][i]),
                "NDCG@50": float(z["ndcg50"][i]),
                "Recall@10": float(z["recall10"][i]),
                "Recall@20": float(z["recall20"][i]),
                "Recall@50": float(z["recall50"][i]),
                "Precision@20": float(z["precision20"][i]),
                "HitRate@20": float(z["hr20"][i]),
                "MRR": float(z["mrr"][i]),
                "MAP@20": float(z["map20"][i]),
                "hist_len": float(z["hist_len"][i]),
            }
        print(f"[clean_v2 {policy} seed={seed}] resume cache n={len(done)}", flush=True)

    remaining = [u for u in users if u not in done]
    t0 = time.time()
    processed = 0
    for u in remaining:
        fold = cf.user_to_fold.get(int(u), -1)
        if fold not in fold_sizes:
            print(f"[clean_v2 eval] |N_X| fold={fold}", flush=True)
            fold_sizes[fold] = neighborhood_sizes(clean_v2_index_for(cf, int(u)))
        mask = model_train.get(int(u), set())
        scores = score_user_policy(u, bundle, ctx, cf, policy, fold_sizes[fold])
        top = dense_topk(scores, k=100, mask_items=mask)
        m = metrics_from_topk(top, val_pos[int(u)], ks=KS, full_rank_order_for_mrr=top)
        p20, map20 = precision_map_at_k(top, val_pos[int(u)], 20)
        done[int(u)] = {
            "NDCG@10": float(m["NDCG@10"]),
            "NDCG@20": float(m["NDCG@20"]),
            "NDCG@50": float(m["NDCG@50"]),
            "Recall@10": float(m["Recall@10"]),
            "Recall@20": float(m["Recall@20"]),
            "Recall@50": float(m["Recall@50"]),
            "Precision@20": float(p20),
            "HitRate@20": float(m["HitRate@20"]),
            "MRR": float(m["MRR"]),
            "MAP@20": float(map20),
            "hist_len": float(len(mask)),
        }
        processed += 1
        if processed % SHARD == 0 or processed == len(remaining):
            users_a = np.asarray(sorted(done.keys()), dtype=np.int64)
            np.savez_compressed(
                cache_path,
                users=users_a,
                ndcg10=np.asarray([done[u]["NDCG@10"] for u in users_a], dtype=np.float64),
                ndcg20=np.asarray([done[u]["NDCG@20"] for u in users_a], dtype=np.float64),
                ndcg50=np.asarray([done[u]["NDCG@50"] for u in users_a], dtype=np.float64),
                recall10=np.asarray([done[u]["Recall@10"] for u in users_a], dtype=np.float64),
                recall20=np.asarray([done[u]["Recall@20"] for u in users_a], dtype=np.float64),
                recall50=np.asarray([done[u]["Recall@50"] for u in users_a], dtype=np.float64),
                precision20=np.asarray([done[u]["Precision@20"] for u in users_a], dtype=np.float64),
                hr20=np.asarray([done[u]["HitRate@20"] for u in users_a], dtype=np.float64),
                mrr=np.asarray([done[u]["MRR"] for u in users_a], dtype=np.float64),
                map20=np.asarray([done[u]["MAP@20"] for u in users_a], dtype=np.float64),
                hist_len=np.asarray([done[u]["hist_len"] for u in users_a], dtype=np.float64),
            )
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1e-9)
            mean20 = float(np.mean([done[u]["NDCG@20"] for u in users_a]))
            print(
                f"[clean_v2 {policy} seed={seed}] {len(done)}/{len(users)} "
                f"u/s={rate:.2f} NDCG@20={mean20:.4f}",
                flush=True,
            )

    users_a = np.asarray(sorted(done.keys()), dtype=np.int64)
    metrics = {
        "policy": policy,
        "seed": seed,
        "split": split,
        "n_users": float(len(users_a)),
        "NDCG@10": float(np.mean([done[u]["NDCG@10"] for u in users_a])),
        "NDCG@20": float(np.mean([done[u]["NDCG@20"] for u in users_a])),
        "NDCG@50": float(np.mean([done[u]["NDCG@50"] for u in users_a])),
        "Recall@20": float(np.mean([done[u]["Recall@20"] for u in users_a])),
        "Precision@20": float(np.mean([done[u]["Precision@20"] for u in users_a])),
        "HitRate@20": float(np.mean([done[u]["HitRate@20"] for u in users_a])),
        "MRR": float(np.mean([done[u]["MRR"] for u in users_a])),
        "MAP@20": float(np.mean([done[u]["MAP@20"] for u in users_a])),
        "created": utc_now(),
        "fusion_dim": CLEAN_V2_FUSION_DIM,
        "hcr_dim": CLEAN_V2_HCR_DIM,
    }
    # activity segments
    by_act: dict[str, list[float]] = defaultdict(list)
    for u in users_a:
        by_act[activity_bucket(int(done[u]["hist_len"]))].append(done[u]["NDCG@20"])
    metrics["by_activity"] = {
        k: {"n": len(v), "NDCG@20": float(np.mean(v)) if v else float("nan")}
        for k, v in by_act.items()
    }
    item_bucket = item_pop_buckets(np.asarray(bundle["popularity"]))
    by_pop: dict[str, list[float]] = defaultdict(list)
    for u in users_a:
        b = majority_pop_bucket(val_pos.get(int(u), set()), item_bucket)
        if b:
            by_pop[b].append(done[u]["NDCG@20"])
    metrics["by_target_popularity"] = {
        k: {"n": len(v), "NDCG@20": float(np.mean(v)) if v else float("nan")}
        for k, v in by_pop.items()
    }
    write_json(metrics_path, metrics)
    np.savez_compressed(
        out_dir / "per_user.npz",
        users=users_a,
        ndcg20=np.asarray([done[u]["NDCG@20"] for u in users_a], dtype=np.float64),
        hist_len=np.asarray([done[u]["hist_len"] for u in users_a], dtype=np.float64),
    )
    print(f"[race_clean_3] DONE {policy} seed={seed} NDCG@20={metrics['NDCG@20']:.6f}", flush=True)
    return metrics


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


def _segment_from_per_user(
    bundle: dict[str, Any],
    policy: str,
    seed: int,
    *,
    split: str,
) -> dict[str, Any] | None:
    dirs = policy_dirs(policy)
    root = dirs["fullval" if split == "val" else "fulltest"] / f"seed_{seed}"
    pu = root / "per_user.npz"
    if not pu.exists():
        return None
    z = np.load(pu)
    users = [int(u) for u in z["users"].tolist()]
    ndcg = {u: float(n) for u, n in zip(users, z["ndcg20"].tolist())}
    hist = {u: int(h) for u, h in zip(users, z["hist_len"].tolist())}
    pos_file = "valid.txt" if split == "val" else "test.txt"
    pos = load_user_sets(SPLITS / pos_file)
    item_bucket = item_pop_buckets(np.asarray(bundle["popularity"]))
    by_act: dict[str, list[float]] = defaultdict(list)
    by_pop: dict[str, list[float]] = defaultdict(list)
    for u in users:
        by_act[activity_bucket(hist[u])].append(ndcg[u])
        b = majority_pop_bucket(pos.get(u, set()), item_bucket)
        if b:
            by_pop[b].append(ndcg[u])
    return {
        "policy": policy,
        "seed": seed,
        "split": split,
        "n_users": len(users),
        "NDCG@20": float(np.mean(list(ndcg.values()))),
        "by_activity": {
            k: {"n": len(v), "NDCG@20": float(np.mean(v))} for k, v in by_act.items()
        },
        "by_target_popularity": {
            k: {"n": len(v), "NDCG@20": float(np.mean(v))} for k, v in by_pop.items()
        },
    }


def finalize(bundle: dict[str, Any] | None = None, *, split: str = "val") -> None:
    rows = []
    seg_act: dict[tuple[str, str], list[float]] = defaultdict(list)
    seg_pop: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_act: dict[tuple[str, str], int] = {}
    n_pop: dict[tuple[str, str], int] = {}
    for policy in CLEAN_V2_ROUTING_POLICIES:
        vals = []
        for seed in SEEDS:
            mp = (
                policy_dirs(policy)["fullval" if split == "val" else "fulltest"]
                / f"seed_{seed}"
                / ("VAL_METRICS.json" if split == "val" else "TEST_METRICS.json")
            )
            if not mp.exists():
                print(f"[race_clean_3] missing {mp}", flush=True)
                return
            m = json.loads(mp.read_text())
            if bundle is not None and (
                "by_target_popularity" not in m or "by_activity" not in m
            ):
                rebuilt = _segment_from_per_user(bundle, policy, seed, split=split)
                if rebuilt:
                    m.setdefault("by_activity", rebuilt["by_activity"])
                    m.setdefault("by_target_popularity", rebuilt["by_target_popularity"])
            vals.append(m["NDCG@20"])
            rows.append({"policy": policy, "seed": seed, "NDCG@20": m["NDCG@20"]})
            for b, st in (m.get("by_activity") or {}).items():
                seg_act[(policy, b)].append(float(st["NDCG@20"]))
                n_act[(policy, b)] = int(st.get("n") or 0)
            for b, st in (m.get("by_target_popularity") or {}).items():
                seg_pop[(policy, b)].append(float(st["NDCG@20"]))
                n_pop[(policy, b)] = int(st.get("n") or 0)
        mean, std = _mean_std(vals)
        rows.append({"policy": policy, "seed": "mean", "NDCG@20": mean, "std": std})
        print(f"[race_clean_3] {policy} mean NDCG@20={mean:.6f} ± {std:.6f}", flush=True)
    summary = sorted(
        [r for r in rows if r.get("seed") == "mean"],
        key=lambda r: -float(r["NDCG@20"]),
    )
    act_table = []
    for policy in CLEAN_V2_ROUTING_POLICIES:
        row = {"policy": policy}
        for b in ACT_ORDER:
            xs = seg_act.get((policy, b), [])
            mu, sd = _mean_std(xs)
            row[b] = mu
            row[f"{b}_std"] = sd
            row[f"{b}_n"] = n_act.get((policy, b), 0)
        act_table.append(row)
    pop_table = []
    for policy in CLEAN_V2_ROUTING_POLICIES:
        row = {"policy": policy}
        for b in POP_ORDER:
            xs = seg_pop.get((policy, b), [])
            mu, sd = _mean_std(xs)
            row[b] = mu
            row[f"{b}_std"] = sd
            row[f"{b}_n"] = n_pop.get((policy, b), 0)
        pop_table.append(row)
    payload = {
        "split": split,
        "primary": "full-rank NDCG@20",
        "architecture": "BASE HGT L=2 heads=2 batch=4096 MLP 128, 265-D, A11 pool 3-D",
        "rows": rows,
        "leaderboard": summary,
        "by_activity": act_table,
        "by_target_popularity": pop_table,
        "note": (
            "Activity = |model_train| buckets. Target popularity = majority HEAD/MID/TAIL "
            "of the user's val/test positives (user-level NDCG, not item-conditional)."
        ),
    }
    tag = "VAL" if split == "val" else "TEST"
    write_json(OUT / "results" / f"RACE_CLEAN_3_{tag}_SUMMARY.json", payload)
    lines = [
        f"# RACE CLEAN 3 — full-rank {split} NDCG@20",
        "",
        "Architecture frozen: BASE HGT (L=2, heads=2, batch=4096) + MLP 128, 265-D, A11 3-D pool.",
        "Routing is the only variable. Sampled NDCG is diagnostic, not the winner.",
        "",
        "## Overall",
        "",
        "| Rank | Policy | mean NDCG@20 | std |",
        "|---:|---|---:|---:|",
    ]
    for i, r in enumerate(summary, start=1):
        lines.append(
            f"| {i} | {r['policy']} | {float(r['NDCG@20']):.6f} | {float(r.get('std') or 0):.6f} |"
        )
    lines += [
        "",
        "## Activity segments (mean over seeds)",
        "",
        "| Policy | VL ≤10 | L ≤25 | M ≤50 | H ≤100 | VH >100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in act_table:
        lines.append(
            "| {policy} | {VERY_LIGHT:.4f} | {LIGHT:.4f} | {MEDIUM:.4f} | {HEAVY:.4f} | {VERY_HEAVY:.4f} |".format(
                **{k: (row[k] if isinstance(row[k], str) else float(row[k])) for k in ["policy", *ACT_ORDER]}
            )
        )
    lines += [
        "",
        "## Target-popularity segments (majority of positives; mean over seeds)",
        "",
        "| Policy | HEAD ≥p90 | MID p50–p90 | TAIL <p50 |",
        "|---|---:|---:|---:|",
    ]
    for row in pop_table:
        lines.append(
            "| {policy} | {HEAD:.4f} | {MID:.4f} | {TAIL:.4f} |".format(
                **{k: (row[k] if isinstance(row[k], str) else float(row[k])) for k in ["policy", *POP_ORDER]}
            )
        )
    (OUT / "results" / f"RACE_CLEAN_3_{tag}_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if split == "val":
        (OUT / "FINALIZE_DONE.txt").write_text(
            f"finalize OK\n{OUT / 'results' / 'RACE_CLEAN_3_VAL_SUMMARY.json'}\n",
            encoding="utf-8",
        )
    print(f"[race_clean_3] finalize {split} OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "materialize-A",
            "materialize-H",
            "train",
            "eval",
            "eval-test",
            "status",
            "finalize",
            "run-all",
            "final",
        ],
    )
    ap.add_argument("--policies", type=str, default=",".join(CLEAN_V2_ROUTING_POLICIES))
    ap.add_argument("--seeds", type=str, default="101,202,303")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = ap.parse_args()
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    cfg = load_protocol_config(str(ROOT / "configs/lastfm_star_race_clean_3.yaml"))
    bundle = load_prepared(cfg, verify=False)
    bundle["cfg"] = cfg

    if args.cmd == "materialize-A":
        materialize_A_true(bundle)
        return
    if args.cmd == "materialize-H":
        for p in policies:
            materialize_H(bundle, p)  # type: ignore[arg-type]
        return
    if args.cmd == "train":
        materialize_A_true(bundle)
        for p in policies:
            materialize_H(bundle, p)  # type: ignore[arg-type]
            for s in seeds:
                train_policy_seed(bundle, p, s)
        return
    if args.cmd == "eval":
        for p in policies:
            for s in seeds:
                eval_policy_seed(bundle, p, s, split=args.split)
        return
    if args.cmd == "eval-test":
        for p in policies:
            for s in seeds:
                eval_policy_seed(bundle, p, s, split="test")
        finalize(bundle, split="test")
        return
    if args.cmd == "finalize":
        finalize(bundle, split=args.split)
        return
    if args.cmd == "status":
        for p in policies:
            for s in seeds:
                ck = policy_dirs(p)["ckpt_root"] / f"RACE_CLEAN_3_{p}" / f"seed_{s}" / "model.pt"
                ev = policy_dirs(p)["fullval"] / f"seed_{s}" / "VAL_METRICS.json"
                te = policy_dirs(p)["fulltest"] / f"seed_{s}" / "TEST_METRICS.json"
                print(f"{p} seed={s} train={ck.exists()} val={ev.exists()} test={te.exists()}")
        print(f"A_true={(shared_A_dir() / 'COMPLETE').exists()}")
        return
    if args.cmd == "run-all":
        materialize_A_true(bundle)
        for p in policies:
            materialize_H(bundle, p)  # type: ignore[arg-type]
            for s in seeds:
                train_policy_seed(bundle, p, s)
            for s in seeds:
                eval_policy_seed(bundle, p, s, split="val")
        finalize(bundle, split="val")
        return
    if args.cmd == "final":
        # Honest freeze: reuse existing BASE ckpts; full-rank val all routing policies;
        # write overall + segment tables. TEST is a separate eval-test after winner lock.
        print("[race_clean_3] FINAL val — 1 worker, skip existing ckpts/evals", flush=True)
        materialize_A_true(bundle)
        for p in policies:
            materialize_H(bundle, p)  # type: ignore[arg-type]
            for s in seeds:
                train_policy_seed(bundle, p, s)
        for p in policies:
            for s in seeds:
                print(f"[race_clean_3] full-rank VAL {p} seed={s}", flush=True)
                eval_policy_seed(bundle, p, s, split="val")
        finalize(bundle, split="val")
        print("[race_clean_3] FINAL val STOP — test only after winner lock.", flush=True)
        return


if __name__ == "__main__":
    main()
