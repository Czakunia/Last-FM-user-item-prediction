"""Hardneg sealed external: R3+BCE+BPR refit on TRAIN_EXTERNAL, one-shot test.

Writes only under LASTFM_TRUE_FINAL/JOINT_HARDNEG_R3_FROM_SCRATCH_V1/07_EXTERNAL/.
Does not overwrite easy TRUE FINAL / LASTFM_EXTERNAL_BENCHMARK artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[2]
ART = EXP / "artifacts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_rank_v2" / "src"))
sys.path.insert(0, str(ROOT / "external_repos" / "IntentAwareRS"))

from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    EXT_COMMIT,
    build_external_bundle,
    build_urm,
    empty_pair_table,
    git_commit,
    sha256_file,
    utc_now,
    write_json,
)
from src.lastfm_lp.binary.contingency_tables import build_cooccurrence  # noqa: E402
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex  # noqa: E402
from scripts._lastfm_sealed_test_eval_20260824 import (  # noqa: E402
    UPSTREAM_CUTOFFS,
    TrueFinalOnTheFlyRecommender,
    holdout_df_to_dict,
    mean_metrics,
    primary_from_holdout,
)
from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import build_race_model  # noqa: E402
from scripts.run_lastfm_noleak_fullrank_validation_v1 import h3_and_l2, score_user_catalog  # noqa: E402
from scripts.run_lastfm_true_final_joint_training_v1 import (  # noqa: E402
    chunk_weighted_bce,
    user_item_to_nodes,
    wrap_encode_all,
)
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import (  # noqa: E402
    CLEAN_V2_TABULAR_FEATURE_NAMES,
    neighborhood_sizes,
    vectorized_clean_v2_A_for_user,
)
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout  # noqa: E402

from rankv2.pos_hard_negatives import PositiveLocalHardNegativeSampler  # noqa: E402
from tfhn.paths import ARCH, JOINT_OUT, NEIGHBORS_SRC  # noqa: E402

LAMBDA_BPR = 0.5
LR = 1e-3
WD = 1e-4
BATCH_BCE = 4096
BATCH_BPR = 1024
N_NEG = 4
EASY_EXT_MEAN_NDCG20 = 0.171778
SAMPLER_SEED = 303  # freeze R3 table once (train seeds separate)
# Cap positives for TE refit to ~DEV scale (avoids Jetsam/OOM on full 1.2M×5 table).
# Still R3 on TRAIN_EXTERNAL; set TFHN_EXT_MAX_POS=0 for full table.
EXT_MAX_POS = int(os.environ.get("TFHN_EXT_MAX_POS", "500676"))


def subsample_hardneg_groups(
    users: np.ndarray,
    items: np.ndarray,
    labels: np.ndarray,
    *,
    max_pos: int,
    seed: int = 303,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep first-max_pos positive groups (each group = 1 pos + N_NEG negs). Returns row index too."""
    assert len(users) % (1 + N_NEG) == 0
    n_pos = len(users) // (1 + N_NEG)
    if max_pos <= 0 or n_pos <= max_pos:
        idx = np.arange(len(users), dtype=np.int64)
        return users, items, labels, idx
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n_pos, size=max_pos, replace=False))
    rows = []
    for g in keep:
        base = int(g) * (1 + N_NEG)
        rows.extend(range(base, base + 1 + N_NEG))
    rows_a = np.asarray(rows, dtype=np.int64)
    return users[rows_a], items[rows_a], labels[rows_a], rows_a


def ext_root() -> Path:
    p = JOINT_OUT / "07_EXTERNAL"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    p = ART / "external_refit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def scale(x, scaler):
    return ((x - scaler.mean_) / np.maximum(scaler.scale_, 1e-12)).astype(np.float32)


def build_triplets(users, items, labels, n_neg=N_NEG):
    assert len(users) % (1 + n_neg) == 0
    n_pos = len(users) // (1 + n_neg)
    u_t, i_pos, i_neg, idx_pos, idx_neg = [], [], [], [], []
    for g in range(n_pos):
        base = g * (1 + n_neg)
        for k in range(1, 1 + n_neg):
            u_t.append(int(users[base]))
            i_pos.append(int(items[base]))
            i_neg.append(int(items[base + k]))
            idx_pos.append(base)
            idx_neg.append(base + k)
    return (
        np.asarray(u_t, dtype=np.int64),
        np.asarray(i_pos, dtype=np.int64),
        np.asarray(i_neg, dtype=np.int64),
        np.asarray(idx_pos, dtype=np.int64),
        np.asarray(idx_neg, dtype=np.int64),
    )


def _idx_to_device(arr, b, device):
    if isinstance(arr, torch.Tensor):
        return arr[b].to(device)
    return torch.as_tensor(arr[b], device=device)


def train_epoch_bce_bpr_joint(
    *,
    model,
    opt,
    loss_fn,
    u_all,
    i_all,
    y_all,
    A_s,
    H_s,
    L_s,
    u_pos_n,
    i_pos_n,
    i_neg_n,
    idx_pos,
    idx_neg,
    device,
    perm_bce,
    perm_bpr,
    batch_bce,
    batch_bpr,
):
    model.train()
    n_bce = int(len(perm_bce))
    n_bpr = int(len(perm_bpr))
    fwd_c = [0]
    orig_encode = wrap_encode_all(model, fwd_c)
    opt.zero_grad()
    z = model.node_z()
    z_proxy = z.detach().requires_grad_(True)

    mean_bce = 0.0
    for start in range(0, n_bce, batch_bce):
        idx = perm_bce[start : start + batch_bce]
        n_b, _w, loss_item = chunk_weighted_bce(
            model=model,
            loss_fn=loss_fn,
            z_used=z_proxy,
            u_all=u_all,
            i_all=i_all,
            y_all=y_all,
            A_tr_s=A_s,
            H_tr_s=H_s,
            r_tr=L_s,
            idx=idx,
            n_train=n_bce,
            device=device,
            retain_graph=False,
        )
        mean_bce += loss_item * n_b
    mean_bce /= float(max(n_bce, 1))

    mean_bpr = 0.0
    for start in range(0, n_bpr, batch_bpr):
        b = perm_bpr[start : start + batch_bpr]
        n_b = int(len(b))
        up = _idx_to_device(u_pos_n, b, device)
        ip = _idx_to_device(i_pos_n, b, device)
        inn = _idx_to_device(i_neg_n, b, device)
        Ap = torch.from_numpy(A_s[idx_pos[b]]).to(device)
        Hp = torch.from_numpy(H_s[idx_pos[b]]).to(device)
        Lp = torch.from_numpy(L_s[idx_pos[b]]).to(device)
        An = torch.from_numpy(A_s[idx_neg[b]]).to(device)
        Hn = torch.from_numpy(H_s[idx_neg[b]]).to(device)
        Ln = torch.from_numpy(L_s[idx_neg[b]]).to(device)
        sp = model(up, ip, Ap, Hp, Lp, z=z_proxy)["logits"]
        sn = model(up, inn, An, Hn, Ln, z=z_proxy)["logits"]
        loss = -F.logsigmoid(sp - sn).mean() * LAMBDA_BPR * (n_b / float(n_bpr))
        loss.backward()
        mean_bpr += float(loss.item()) / max(LAMBDA_BPR, 1e-12) * n_b
    mean_bpr /= float(max(n_bpr, 1))

    if z_proxy.grad is None:
        raise RuntimeError("z_proxy.grad is None after BCE+BPR chunks")
    z.backward(z_proxy.grad)
    opt.step()
    model.encoder.encode_all = orig_encode  # type: ignore[method-assign]
    del z, z_proxy
    return {
        "bce_loss": float(mean_bce),
        "bpr_loss": float(mean_bpr),
        "train_loss": float(mean_bce + LAMBDA_BPR * mean_bpr),
        "hgt_forward": int(fwd_c[0]),
    }


def load_selected_epochs(manifest: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in manifest["per_seed"]:
        out[int(row["seed"])] = int(row["selected_epoch"])
    return out


def ensure_hardneg_external_data(bundle) -> dict[str, Any]:
    """Build R3 pairs + A5/H3/LEG on TRAIN_EXTERNAL (cached).

    Feature materialization runs in a separate lightweight process (07a) to avoid
    OOM when torch/HGT are already imported in this module.
    """
    import subprocess

    d = data_dir()
    pairs_path = d / "hardneg_train_pairs.npz"
    feat_dir = d / "features"
    ready = d / "DATA_READY.flag"
    if ready.exists() and pairs_path.exists() and (feat_dir / "a_scaler.pkl").exists():
        print("[tfhn-ext] reuse TRAIN_EXTERNAL hardneg data cache", flush=True)
        return {"pairs_path": pairs_path, "feat_dir": feat_dir}

    neigh = ART / "item_a11_neighbors.npz"
    if not neigh.exists():
        assert NEIGHBORS_SRC.exists(), f"missing {NEIGHBORS_SRC}"
        import shutil

        shutil.copy2(NEIGHBORS_SRC, neigh)

    te = bundle.train_external
    if not pairs_path.exists():
        nn = np.load(neigh)
        neighbors, scores, pop = nn["neighbors"], nn["scores"], nn["pop"]
        n_items = int(bundle.n_items)
        rng = np.random.default_rng(SAMPLER_SEED)
        samp = PositiveLocalHardNegativeSampler(
            neighbors=neighbors,
            neighbor_scores=scores,
            item_pop=pop,
            model_train=te,
            n_items=n_items,
            rng=rng,
            band_lo=0.05,
            band_hi=0.30,
        )
        pos_u, pos_i = [], []
        for u in sorted(te):
            for i in sorted(te[u]):
                pos_u.append(int(u))
                pos_i.append(int(i))
        print(f"[tfhn-ext] TRAIN_EXTERNAL positives={len(pos_u)} sampling R3 negs…", flush=True)
        users_l, items_l, labels_l = [], [], []
        for u, i in tqdm(zip(pos_u, pos_i), total=len(pos_u), desc="ext-sample-negs"):
            users_l.append(u)
            items_l.append(i)
            labels_l.append(1)
            out = samp.sample(u, i, n_neg=N_NEG)
            for j in out["items"]:
                users_l.append(u)
                items_l.append(int(j))
                labels_l.append(0)
                if int(j) in te.get(u, set()):
                    raise RuntimeError(f"leak neg {j} in hist u={u}")
        users = np.asarray(users_l, dtype=np.int64)
        items = np.asarray(items_l, dtype=np.int64)
        labels = np.asarray(labels_l, dtype=np.int8)
        np.savez_compressed(pairs_path, user_id=users, item_id=items, label=labels)
    else:
        print(f"[tfhn-ext] reuse R3 pairs cache {pairs_path.name}", flush=True)

    script = EXP / "scripts" / "07a_materialize_external_features.py"
    print("[tfhn-ext] spawn lightweight 07a feature materializer…", flush=True)
    subprocess.check_call([sys.executable, "-u", str(script)], cwd=str(ROOT))
    if not ready.exists():
        raise RuntimeError("07a finished without DATA_READY.flag")
    return {"pairs_path": pairs_path, "feat_dir": feat_dir}


def refit_seed(
    *,
    seed: int,
    n_epochs: int,
    bundle,
    graph,
    item_offset,
    device,
    pairs,
    A_s,
    H_s,
    L_s,
    out_dir: Path,
) -> dict[str, Any]:
    seed_dir = out_dir / f"HARDNEG_REFIT_SEED{seed}"
    ckpt_path = seed_dir / "model.pt"
    meta_path = seed_dir / "meta.json"
    if ckpt_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if int(meta.get("n_epochs", -1)) == n_epochs:
            print(f"[tfhn-ext] reuse refit seed={seed} n_epochs={n_epochs}", flush=True)
            return meta

    torch.manual_seed(seed)
    np.random.seed(seed)
    prepared = {
        "cfg": bundle.cfg,
        "model_train": bundle.train_external,
        "popularity": bundle.popularity,
        "item_kg_degree": bundle.item_kg_degree,
        "train_pairs": pairs,
        "val_pairs": {"user_id": np.zeros(0, np.int64), "item_id": np.zeros(0, np.int64), "label": np.zeros(0, np.int8)},
        "test_pairs": {"user_id": np.zeros(0, np.int64), "item_id": np.zeros(0, np.int64), "label": np.zeros(0, np.int8)},
        "eval_users": bundle.eval_users_test,
    }
    model = build_race_model(prepared, graph, device, d=64, layers=2, heads=2)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    u_all, i_all = user_item_to_nodes(pairs["user_id"], pairs["item_id"], item_offset)
    y_all = pairs["label"].astype(np.float32)
    u_t, i_p, i_n, idx_p, idx_n = build_triplets(pairs["user_id"], pairs["item_id"], pairs["label"])
    u_pos_n, i_pos_n = user_item_to_nodes(u_t, i_p, item_offset)
    _, i_neg_n = user_item_to_nodes(u_t, i_n, item_offset)
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )

    history = []
    t0 = time.time()
    print(
        f"[tfhn-ext] REFIT seed={seed} n_epochs={n_epochs} "
        f"(selected_epoch={n_epochs - 1}) λ_BPR={LAMBDA_BPR}",
        flush=True,
    )
    for epoch in range(n_epochs):
        te = time.time()
        print(f"[tfhn-ext s{seed}] starting ep {epoch}/{n_epochs - 1} …", flush=True)
        info = train_epoch_bce_bpr_joint(
            model=model,
            opt=opt,
            loss_fn=loss_fn,
            u_all=u_all,
            i_all=i_all,
            y_all=y_all,
            A_s=A_s,
            H_s=H_s,
            L_s=L_s,
            u_pos_n=u_pos_n,
            i_pos_n=i_pos_n,
            i_neg_n=i_neg_n,
            idx_pos=idx_p,
            idx_neg=idx_n,
            device=device,
            perm_bce=np.random.permutation(len(y_all)),
            perm_bpr=np.random.permutation(len(u_t)),
            batch_bce=BATCH_BCE,
            batch_bpr=BATCH_BPR,
        )
        if int(info["hgt_forward"]) != 1:
            raise RuntimeError(f"expected 1 HGT forward/epoch, got {info['hgt_forward']}")
        empty_cache()
        row = {
            "epoch": epoch,
            "bce_loss": info["bce_loss"],
            "bpr_loss": info["bpr_loss"],
            "train_loss": info["train_loss"],
            "sec": time.time() - te,
        }
        history.append(row)
        print(
            f"[tfhn-ext s{seed}] ep {epoch}/{n_epochs - 1} "
            f"loss={row['train_loss']:.4f} ({row['sec']:.1f}s)",
            flush=True,
        )

    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    meta = {
        "seed": seed,
        "selected_epoch": n_epochs - 1,
        "n_epochs": n_epochs,
        "architecture": ARCH,
        "loss": "BCE + 0.5*BPR",
        "negatives": "R3",
        "training_hash": bundle.training_hash,
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "fit_seconds": time.time() - t0,
        "history": history,
        "ckpt_sha256": sha256_file(ckpt_path),
        "REPLACE_TRUE_FINAL": "DEFERRED",
    }
    write_json(meta_path, meta)
    del model
    empty_cache()
    return meta


def run_sealed_eval(
    *,
    bundle,
    selected: dict[int, int],
    refit_metas: list[dict],
    feat_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "SEALED_TEST_EXECUTED.lock"
    if lock.exists() and (out_dir / "sealed_test_results.json").exists():
        print("[tfhn-ext] sealed already executed — reuse", flush=True)
        return json.loads((out_dir / "sealed_test_results.json").read_text())

    urm_train = build_urm(bundle.train_external, bundle.n_users, bundle.n_items)
    urm_test = build_urm(bundle.sealed_test, bundle.n_users, bundle.n_items)

    with (feat_dir / "a_scaler.pkl").open("rb") as f:
        a_sc = pickle.load(f)
    with (feat_dir / "h_scaler.pkl").open("rb") as f:
        h_sc = pickle.load(f)
    with (feat_dir / "l_scaler.pkl").open("rb") as f:
        l_sc = pickle.load(f)

    prepared = {
        "cfg": bundle.cfg,
        "model_train": bundle.train_external,
        "popularity": bundle.popularity,
        "item_kg_degree": bundle.item_kg_degree,
        "train_pairs": {},
        "val_pairs": {},
        "test_pairs": {},
        "eval_users": bundle.eval_users_test,
    }
    # ensure_cross_fit needs pairwise index + train structure
    cooc = build_cooccurrence(bundle.train_external, bundle.n_items, max_history_for_pairs=60, seed=2026)
    index = PairwiseStatsIndex(
        cooc,
        bundle.popularity,
        n_users=len(bundle.train_external),
        smoothing=float(bundle.cfg.get("hcr", {}).get("smoothing", 0.5)),
    )
    pos_u = np.asarray([u for u, xs in bundle.train_external.items() for _ in xs], dtype=np.int64)
    pos_i = np.asarray([i for u, xs in bundle.train_external.items() for i in xs], dtype=np.int64)
    prepared_cf = {
        **prepared,
        "index": index,
        "train_pairs": {
            "user_id": pos_u,
            "item_id": pos_i,
            "label": np.ones(len(pos_u), dtype=np.int8),
        },
        "val_pairs": empty_pair_table(),
        "test_pairs": empty_pair_table(),
    }
    cf = ensure_cross_fit(prepared_cf)
    graph = load_data_and_typed_graph(
        bundle.cfg,
        bundle.train_external,
        max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(os.environ.get("TFHN_DEVICE", "cpu"))
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
    t_all = time.time()

    for seed in (101, 202, 303):
        a_path = raw_dir / f"block_a_HARDNEG_seed{seed}.json"
        b_path = raw_dir / f"block_b_HARDNEG_seed{seed}.json"
        if a_path.exists() and b_path.exists():
            a_res = json.loads(a_path.read_text())
            b_res = json.loads(b_path.read_text())
            tf_a_seeds.append(a_res)
            tf_b_seeds.append(b_res)
            print(
                f"[tfhn-ext] reuse sealed seed={seed} A NDCG@20={a_res['primary']['NDCG@20']:.4f}",
                flush=True,
            )
            continue

        n_epochs = int(selected[seed]) + 1
        ckpt = out_dir / f"HARDNEG_REFIT_SEED{seed}" / "model.pt"
        print(f"[tfhn-ext] sealed eval seed={seed} n_epochs={n_epochs} …", flush=True)
        t0 = time.time()
        model = build_race_model(prepared_cf, graph, device, d=64, layers=2, heads=2)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
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

        def score_fn(u: int, hist: set[int], _ctx=ctx):
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
            "selected_epoch": selected[seed],
            "n_epochs": n_epochs,
            "primary": primary_from_holdout(df),
            "all_cutoffs": holdout_df_to_dict(df),
            "raw_string": str(raw),
            "artifact_sha256": sha256_file(ckpt),
            "seconds": time.time() - t0,
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
        }
        write_json(b_path, b_res)
        tf_a_seeds.append(a_res)
        tf_b_seeds.append(b_res)
        print(
            f"[tfhn-ext] seed={seed} A NDCG@20={a_res['primary']['NDCG@20']:.6f} "
            f"Recall@20={a_res['primary']['RECALL@20']:.6f} | "
            f"B NDCG@20={b_res['NDCG@20']:.6f} ({a_res['seconds']:.1f}s)",
            flush=True,
        )
        del model, adapter, ctx, z
        empty_cache()

    tf_a_mean = mean_metrics([r["primary"] for r in tf_a_seeds], ["NDCG@20", "RECALL@20"])
    tf_b_mean = mean_metrics(
        tf_b_seeds,
        ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@20", "MRR", "HitRate@20"],
    )
    payload = {
        "name": "TRUE_FINAL_HARDNEG_SEALED_EXTERNAL",
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "intentawarers_commit": EXT_COMMIT,
        "training_hash": bundle.training_hash,
        "test_hash": bundle.test_hash,
        "sealed_test_accessed": True,
        "SEALED_TEST_EXECUTED": True,
        "REPLACE_TRUE_FINAL": "DEFERRED",
        "models_retrained_after_unseal": False,
        "models_reselected_after_unseal": False,
        "easy_TRUE_FINAL_external_mean_NDCG@20": EASY_EXT_MEAN_NDCG20,
        "delta_vs_easy_ext_mean": float(tf_a_mean["NDCG@20"] - EASY_EXT_MEAN_NDCG20),
        "refit": refit_metas,
        "block_a": {
            "label": "IntentAwareRS-compatible evaluation (hardneg only; baselines frozen from prior sealed run)",
            "true_final_hardneg_per_seed": [
                {
                    "seed": r["seed"],
                    "selected_epoch": r["selected_epoch"],
                    "n_epochs": r["n_epochs"],
                    "primary": r["primary"],
                    "artifact_sha256": r["artifact_sha256"],
                }
                for r in tf_a_seeds
            ],
            "true_final_hardneg_mean": tf_a_mean,
        },
        "block_b": {
            "label": "Our frozen full-catalog evaluator",
            "true_final_hardneg_per_seed": [
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
            "true_final_hardneg_mean": tf_b_mean,
        },
        "elapsed_seconds": time.time() - t_all,
    }
    results_path = out_dir / "sealed_test_results.json"
    write_json(results_path, payload)
    payload["results_sha256"] = sha256_file(results_path)
    write_json(results_path, payload)
    lock.write_text(f"EXECUTED {utc_now()}\n")
    # unlock official TEST_LOCKED marker in hardneg tree only
    (JOINT_OUT / "TEST_UNLOCKED_HARDNEG_ONLY").write_text(
        "Sealed hardneg external executed once. Official easy TRUE FINAL untouched.\n"
    )
    return payload


def run_hardneg_sealed_external(manifest: dict) -> dict[str, Any]:
    out = ext_root()
    selected = load_selected_epochs(manifest)
    print(f"[tfhn-ext] selected_epochs={selected}", flush=True)

    bundle = build_external_bundle()
    write_json(
        out / "external_bundle_meta.json",
        {
            "training_hash": bundle.training_hash,
            "test_hash": bundle.test_hash,
            "n_users": bundle.n_users,
            "n_items": bundle.n_items,
            "intentawarers_commit": EXT_COMMIT,
        },
    )

    data = ensure_hardneg_external_data(bundle)
    pairs_npz = np.load(data["pairs_path"])
    users_f = pairs_npz["user_id"]
    items_f = pairs_npz["item_id"]
    labels_f = pairs_npz["label"]
    users_f, items_f, labels_f, row_idx = subsample_hardneg_groups(
        users_f, items_f, labels_f, max_pos=EXT_MAX_POS, seed=SAMPLER_SEED
    )
    pairs = {"user_id": users_f, "item_id": items_f, "label": labels_f}
    print(
        f"[tfhn-ext] train table n_pairs={len(labels_f)} n_pos={int((labels_f == 1).sum())} "
        f"(EXT_MAX_POS={EXT_MAX_POS})",
        flush=True,
    )
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
    print("[tfhn-ext] scaling features → memmap (subsampled rows)…", flush=True)
    n = int(len(labels_f))
    tag = f"pos{n // (1 + N_NEG)}"
    A_s_path = feat_dir / f"A_train_scaled_{tag}.npy"
    H_s_path = feat_dir / f"H_train_scaled_{tag}.npy"
    L_s_path = feat_dir / f"L_train_scaled_{tag}.npy"
    if not (A_s_path.exists() and H_s_path.exists() and L_s_path.exists()):
        A_s = np.lib.format.open_memmap(A_s_path, mode="w+", dtype=np.float32, shape=(n, 5))
        H_s = np.lib.format.open_memmap(H_s_path, mode="w+", dtype=np.float32, shape=(n, 3))
        L_s = np.lib.format.open_memmap(L_s_path, mode="w+", dtype=np.float32, shape=(n, 1))
        for start in tqdm(range(0, n, 200_000), desc="scale-features"):
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
    # training indexes numpy arrays; convert views per-batch already via torch.from_numpy

    graph = load_data_and_typed_graph(
        bundle.cfg,
        bundle.train_external,
        max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(os.environ.get("TFHN_DEVICE", "cpu"))

    refit_metas = []
    for seed in (101, 202, 303):
        n_epochs = int(selected[seed]) + 1  # match DEV ckpt epoch_{selected}
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
            out_dir=out,
        )
        meta["EXT_MAX_POS"] = EXT_MAX_POS
        meta["n_train_pairs"] = int(len(labels_f))
        refit_metas.append(meta)
        write_json(out / f"HARDNEG_REFIT_SEED{seed}" / "meta.json", meta)
    write_json(out / "refit_artifacts.json", refit_metas)

    payload = run_sealed_eval(
        bundle=bundle,
        selected=selected,
        refit_metas=refit_metas,
        feat_dir=feat_dir,
        out_dir=out,
    )

    report = out / "HARDNEG_EXTERNAL_REPORT.md"
    a = payload["block_a"]["true_final_hardneg_mean"]
    b = payload["block_b"]["true_final_hardneg_mean"]
    report.write_text(
        f"""# TRUE FINAL hardneg — sealed external

- Status: **EXECUTED ONCE** · `REPLACE_TRUE_FINAL=DEFERRED`
- TRAIN_EXTERNAL hash: `{bundle.training_hash[:16]}…`
- Easy TRUE FINAL external mean NDCG@20: **{EASY_EXT_MEAN_NDCG20:.6f}**
- Hardneg mean NDCG@20 (Block A): **{a['NDCG@20']:.6f}** (Δ={payload['delta_vs_easy_ext_mean']:+.6f})
- Hardneg mean NDCG@20 (Block B): **{b['NDCG@20']:.6f}**

## Per seed (Block A)

| Seed | selected_epoch | NDCG@20 | Recall@20 |
|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {r['seed']} | {r['selected_epoch']} | {r['primary']['NDCG@20']:.6f} | {r['primary']['RECALL@20']:.6f} |"
            for r in payload["block_a"]["true_final_hardneg_per_seed"]
        )
        + "\n",
        encoding="utf-8",
    )
    print(report.read_text(), flush=True)
    return payload
