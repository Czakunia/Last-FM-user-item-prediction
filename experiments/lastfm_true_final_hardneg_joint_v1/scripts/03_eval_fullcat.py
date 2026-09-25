#!/usr/bin/env python3
"""SCREEN_3K (+ eval-time early stop) + FULLCAT on protocol candidate set.

Rule B candidates per seed (default):
  - top K distinct epochs by SCREEN_3K NDCG@20 (default K=2: best + 2nd)
  - no mandatory init / ep1–3 extras

SCREEN_3K early stop (eval only; training checkpoints already exist):
  after min_epoch trained epochs, stop if the last `patience` consecutive
  epoch-to-epoch NDCG@20 gains are all < min_delta.

Scientific freeze = highest FULLCAT DEV NDCG@20 among those candidates.

Env:
  TFHN_SEED=303
  TFHN_SKIP_CACHED=1
  TFHN_SCREEN_EARLY_STOP=1
  TFHN_SCREEN_MIN_EPOCH=5
  TFHN_SCREEN_PATIENCE=3
  TFHN_SCREEN_MIN_DELTA=0.002
  TFHN_FULLCAT_TOP_K=2
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_true_final_hardneg_joint_v1" / "src"))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import build_race_model  # noqa: E402
from scripts.run_lastfm_noleak_fullrank_validation_v1 import score_user_catalog  # noqa: E402
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.tabular_true import neighborhood_sizes  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

from tfhn.ckpt_io import build_payload, copy_scalers_to_audit, save_final_seed_ckpt  # noqa: E402
from tfhn.paths import (  # noqa: E402
    ARCH,
    ART,
    B0_MEAN,
    B0_SEED303,
    JOINT_OUT,
    REP,
    SPLITS_DIR,
    resolve_hgt_dims,
)

HGT_D, HGT_LAYERS, HGT_HEADS = resolve_hgt_dims()


class ZeroLEG(torch.nn.Module):
    """Inference-only LEG ablation; contributes exactly zero to every logit."""

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.zeros(r.shape[0], dtype=r.dtype, device=r.device)


def _dcg(rels, k):
    rels = rels[:k]
    if not rels.size:
        return 0.0
    return float(np.sum(rels / np.log2(np.arange(2, rels.size + 2))))


def metrics(scores, gt, ks=(5, 10, 20)):
    max_k = max(ks)
    top = np.argpartition(-scores, max_k)[:max_k]
    top = top[np.argsort(-scores[top])]
    rels = np.array([1.0 if int(i) in gt else 0.0 for i in top])
    n_gt = float(len(gt))
    out = {}
    for k in ks:
        hits = rels[:k]
        idcg = _dcg(np.ones(min(int(n_gt), k)), k)
        out[f"NDCG@{k}"] = (_dcg(hits, k) / idcg) if idcg else 0.0
        out[f"Recall@{k}"] = float(hits.sum()) / n_gt if n_gt else 0.0
    rank = next((r for r, i in enumerate(top, 1) if int(i) in gt), None)
    out["MRR"] = (1.0 / rank) if rank else 0.0
    return out


def load_ckpt_path(ckpt_dir: Path, ep) -> Path:
    if ep == "init":
        return ckpt_dir / "epoch_init.pt"
    return ckpt_dir / f"epoch_{int(ep):03d}.pt"


def list_epochs(ckpt_dir: Path) -> list:
    eps = ["init"]
    for p in sorted(ckpt_dir.glob("epoch_*.pt")):
        if p.stem == "epoch_init":
            continue
        suf = p.stem.split("_", 1)[1]
        if suf.isdigit():
            eps.append(int(suf))
    return eps


def candidate_epochs(screen_rows: list[dict], top_k: int = 2) -> list:
    """Rule B: top-K trained epochs by SCREEN_3K NDCG@20 (default best + 2nd)."""
    ranked = sorted(
        [r for r in screen_rows if r["epoch"] != "init"],
        key=lambda r: r["NDCG@20"],
        reverse=True,
    )
    out = []
    for r in ranked[: max(1, int(top_k))]:
        e = r["epoch"]
        if e not in out:
            out.append(e)
    return out


def screen_plateau_stop(
    trained_rows: list[dict],
    *,
    min_epoch: int,
    patience: int,
    min_delta: float,
) -> bool:
    """True if last `patience` consecutive epoch-to-epoch gains are all < min_delta."""
    if patience <= 0 or len(trained_rows) < max(min_epoch + 1, patience + 1):
        return False
    last = trained_rows[-1]["epoch"]
    if not isinstance(last, int) or last < min_epoch:
        return False
    gains = []
    for i in range(len(trained_rows) - patience, len(trained_rows)):
        prev = float(trained_rows[i - 1]["NDCG@20"])
        cur = float(trained_rows[i]["NDCG@20"])
        gains.append(cur - prev)
    return all(g < min_delta for g in gains)


def eval_ckpt(ckpt_path, users, mt, va, bundle, cf, graph, item_offset, n_items, ctx_base, cache_dir, skip_cached):
    device = ctx_base["device"]
    model = build_race_model(
        bundle, graph, device, d=HGT_D, layers=HGT_LAYERS, heads=HGT_HEADS
    )
    payload = torch.load(ckpt_path, map_location="cpu")
    assert payload["architecture_name"] == ARCH
    model.load_state_dict(payload["model_state_dict"])
    if ctx_base.get("inference_no_leg"):
        model.leg = ZeroLEG().to(device)
    model.eval()
    z = model.node_z().detach()
    ctx = dict(ctx_base)
    ctx["model"] = model
    ctx["z"] = z
    size_cache: dict[int, np.ndarray] = {}
    acc = {k: [] for k in ("NDCG@5", "NDCG@10", "NDCG@20", "Recall@20", "MRR")}
    tag = ckpt_path.stem
    for u in tqdm(users, desc=f"eval-{tag}"):
        u = int(u)
        hist = set(mt.get(u, ()))
        gt = set(va.get(u, ()))
        if not gt:
            continue
        cache_path = cache_dir / f"{tag}_u{u}.npy"
        if skip_cached and cache_path.exists():
            scores = np.load(cache_path).astype(np.float64)
        else:
            fold = cf.user_to_fold.get(u, -1)
            if fold not in size_cache:
                size_cache[fold] = neighborhood_sizes(clean_v2_index_for(cf, u))
            scores = score_user_catalog(
                u,
                n_items=n_items,
                item_offset=item_offset,
                hist=hist,
                bundle=bundle,
                cf=cf,
                ctx=ctx,
                n_x_sizes=size_cache[fold],
            )
            np.save(cache_path, scores.astype(np.float32))
        m = metrics(scores, gt)
        for k in acc:
            acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    seed = int(os.environ.get("TFHN_SEED", "303"))
    inference_no_leg = os.environ.get("TFHN_INFERENCE_NO_LEG", "0") == "1"
    skip_cached = os.environ.get("TFHN_SKIP_CACHED", "1") == "1"
    screen_es = os.environ.get("TFHN_SCREEN_EARLY_STOP", "1") == "1"
    screen_min_epoch = int(os.environ.get("TFHN_SCREEN_MIN_EPOCH", "5"))
    screen_patience = int(os.environ.get("TFHN_SCREEN_PATIENCE", "3"))
    screen_min_delta = float(os.environ.get("TFHN_SCREEN_MIN_DELTA", "0.002"))
    fullcat_top_k = int(os.environ.get("TFHN_FULLCAT_TOP_K", "2"))
    seed_dir = ART / f"seed{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    assert (seed_dir / "TRAIN_DONE.flag").exists(), f"train seed {seed} first"
    # Prefer local rebuild when iCloud leaves train_summary.json dataless/hanging.
    _summary_path = seed_dir / "train_summary_local.json"
    if not _summary_path.exists():
        _summary_path = seed_dir / "train_summary.json"
    summary = json.loads(_summary_path.read_text())
    history_by_ep = {int(h["epoch"]): h for h in summary.get("history", [])}

    screen_meta_path = ART / "SCREEN_3K_USERS_local.json"
    if not screen_meta_path.exists():
        screen_meta_path = ART / "SCREEN_3K_USERS.json"
    screen_npy_path = ART / "SCREEN_3K_USERS_local.npy"
    if not screen_npy_path.exists():
        screen_npy_path = ART / "SCREEN_3K_USERS.npy"
    if not screen_meta_path.exists():
        import subprocess

        subprocess.check_call(
            [sys.executable, "-u", str(Path(__file__).resolve().parent / "00_freeze_screen3k_users.py")]
        )
        screen_meta_path = ART / "SCREEN_3K_USERS.json"
        screen_npy_path = ART / "SCREEN_3K_USERS.npy"
    screen_meta = json.loads(screen_meta_path.read_text())
    users_screen = np.load(screen_npy_path)
    assert len(users_screen) == int(screen_meta["n"])

    feat_dir = ART / "features"
    with (feat_dir / "a_scaler.pkl").open("rb") as f:
        a_sc = pickle.load(f)
    with (feat_dir / "h_scaler.pkl").open("rb") as f:
        h_sc = pickle.load(f)
    with (feat_dir / "l_scaler.pkl").open("rb") as f:
        l_sc = pickle.load(f)

    cfg = load_protocol_config(PROTOCOL_CONFIG)
    bundle = load_prepared(cfg, verify=False)
    mt = load_user_sets(SPLITS_DIR / "model_train.txt")
    va = load_user_sets(SPLITS_DIR / "valid.txt")
    bundle["model_train"] = mt
    graph = load_data_and_typed_graph(
        bundle["cfg"],
        mt,
        max_kg_edges=bundle["cfg"].get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    n_items = int(graph["meta"]["n_items"])
    item_offset = int(graph["meta"]["item_offset"])
    cf = ensure_cross_fit(bundle)
    device = resolve_torch_device(os.environ.get("TFHN_DEVICE", "cpu"))
    ctx_base = {
        "device": device,
        "no_dist": False,
        "hgt_only": False,
        "no_leg": inference_no_leg,
        "inference_no_leg": inference_no_leg,
        "a_mean": a_sc.mean_.astype(np.float32),
        "a_scale": a_sc.scale_.astype(np.float32),
        "h_mean": h_sc.mean_.astype(np.float32),
        "h_scale": h_sc.scale_.astype(np.float32),
        "l2_mean": l_sc.mean_.astype(np.float32),
        "l2_scale": l_sc.scale_.astype(np.float32),
    }

    users_full = np.array(sorted(u for u, items in va.items() if items), dtype=np.int64)
    epochs = list_epochs(ckpt_dir)
    only_epochs = os.environ.get("TFHN_EPOCHS", "").strip()
    if only_epochs:
        requested = {int(x) for x in only_epochs.split(",") if x.strip()}
        epochs = [ep for ep in epochs if ep != "init" and int(ep) in requested]
    cache_tag = "_noleg_inference" if inference_no_leg else ""
    screen_cache = seed_dir / f"screen3k_cache_{screen_meta['sha256'][:12]}{cache_tag}"
    screen_cache.mkdir(parents=True, exist_ok=True)

    screen_rows = []
    screen_stop_meta = {
        "enabled": screen_es,
        "min_epoch": screen_min_epoch,
        "patience": screen_patience,
        "min_delta": screen_min_delta,
        "stopped": False,
        "stopped_after_epoch": None,
        "epochs_skipped": [],
    }
    t0 = time.time()
    print(
        f"[tfhn-eval] seed={seed} SCREEN_3K n={len(users_screen)} "
        f"sha={screen_meta['sha256'][:12]} epochs={epochs} "
        f"early_stop={screen_es} min_epoch={screen_min_epoch} "
        f"patience={screen_patience} min_delta={screen_min_delta}",
        flush=True,
    )
    for ep in epochs:
        path = load_ckpt_path(ckpt_dir, ep)
        if not path.exists():
            continue
        m = eval_ckpt(
            path, users_screen, mt, va, bundle, cf, graph, item_offset, n_items, ctx_base, screen_cache, skip_cached
        )
        loss = None
        if ep == "init":
            loss = None
        elif int(ep) in history_by_ep:
            loss = history_by_ep[int(ep)].get("train_loss")
        row = {
            "epoch": ep,
            "NDCG@20": m["NDCG@20"],
            "Recall@20": m["Recall@20"],
            "MRR": m["MRR"],
            "train_loss": loss,
            "n_users": int(len(users_screen)),
            "screen_users_sha256": screen_meta["sha256"],
        }
        screen_rows.append(row)
        print(
            f"[tfhn-eval] SCREEN_3K ep={ep} NDCG@20={m['NDCG@20']:.6f} "
            f"R@20={m['Recall@20']:.6f} MRR={m['MRR']:.6f}",
            flush=True,
        )
        trained = [r for r in screen_rows if isinstance(r["epoch"], int)]
        if screen_es and screen_plateau_stop(
            trained,
            min_epoch=screen_min_epoch,
            patience=screen_patience,
            min_delta=screen_min_delta,
        ):
            remaining = [e for e in epochs[epochs.index(ep) + 1 :] if e != "init"]
            screen_stop_meta["stopped"] = True
            screen_stop_meta["stopped_after_epoch"] = ep
            screen_stop_meta["epochs_skipped"] = remaining
            print(
                f"[tfhn-eval] SCREEN_3K early_stop after ep={ep} "
                f"(last {screen_patience} gains < {screen_min_delta}); "
                f"skip={remaining}",
                flush=True,
            )
            break

    (seed_dir / "SCREEN_3K_CURVE.json").write_text(
        json.dumps({"rows": screen_rows, "early_stop": screen_stop_meta}, indent=2) + "\n"
    )
    # Phase-A hparam: SCREEN only (skip FULLCAT)
    if os.environ.get("TFHN_SCREEN_ONLY", "0") == "1":
        trained = [r for r in screen_rows if isinstance(r.get("epoch"), int)]
        best = max(trained, key=lambda r: r["NDCG@20"]) if trained else None
        phase = {
            "seed": seed,
            "SCREEN_ONLY": True,
            "best_epoch": None if best is None else best["epoch"],
            "best_NDCG@20": None if best is None else best["NDCG@20"],
            "screen_early_stop": screen_stop_meta,
            "rows": screen_rows,
        }
        (seed_dir / "PHASE_A_SCREEN.json").write_text(json.dumps(phase, indent=2) + "\n")
        print(
            f"[tfhn-eval] SCREEN_ONLY done seed={seed} best_ep={phase['best_epoch']} "
            f"NDCG@20={phase['best_NDCG@20']}",
            flush=True,
        )
        return

    cands = candidate_epochs(screen_rows, top_k=fullcat_top_k)
    print(
        f"[tfhn-eval] FULLCAT candidates={cands} (rule=B top_k={fullcat_top_k})",
        flush=True,
    )
    (seed_dir / "FULLCAT_CANDIDATES.json").write_text(
        json.dumps(
            {
                "candidates": cands,
                "rule": "B_topK_screen",
                "top_k": fullcat_top_k,
                "screen_early_stop": screen_stop_meta,
            },
            indent=2,
        )
        + "\n"
    )

    full_cache = seed_dir / f"fullcat_cache{cache_tag}"
    full_cache.mkdir(parents=True, exist_ok=True)
    full_rows = []
    for ep in cands:
        path = load_ckpt_path(ckpt_dir, ep)
        m = eval_ckpt(
            path, users_full, mt, va, bundle, cf, graph, item_offset, n_items, ctx_base, full_cache, skip_cached
        )
        m["epoch"] = ep
        m["n_users"] = int(len(users_full))
        m["delta_vs_B0_seed303"] = float(m["NDCG@20"] - B0_SEED303["NDCG@20"])
        m["delta_vs_B0_mean"] = float(m["NDCG@20"] - B0_MEAN["NDCG@20"])
        screen_hit = next(r for r in screen_rows if r["epoch"] == ep)
        m["screen3k_NDCG@20"] = screen_hit["NDCG@20"]
        full_rows.append(m)
        print(
            f"[tfhn-eval] FULLCAT ep={ep} NDCG@20={m['NDCG@20']:.6f} "
            f"ΔB0_303={m['delta_vs_B0_seed303']:+.6f}",
            flush=True,
        )

    full_rows.sort(key=lambda r: r["NDCG@20"], reverse=True)
    best = full_rows[0]
    if os.environ.get("TFHN_METRICS_ONLY", "0") == "1":
        out = {
            "seed": seed,
            "screen3k": screen_rows,
            "fullcat_candidates": cands,
            "fullcat": full_rows,
            "selected_epoch": best["epoch"],
            "NDCG@20": best["NDCG@20"],
            "Recall@20": best["Recall@20"],
            "MRR": best["MRR"],
            "screen3k_NDCG@20": best["screen3k_NDCG@20"],
            "seconds": time.time() - t0,
            "metrics_only": True,
            "EXTERNAL_SEEN": False,
        }
        out_path = seed_dir / "FULLCAT.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[tfhn-eval] metrics-only result → {out_path}", flush=True)
        return
    if inference_no_leg:
        out = {
            "seed": seed,
            "ablation": "same trained checkpoint; exact zero LEG contribution at inference",
            "screen3k": screen_rows,
            "fullcat_candidates": cands,
            "fullcat": full_rows,
            "selected_epoch": best["epoch"],
            "NDCG@20": best["NDCG@20"],
            "Recall@20": best["Recall@20"],
            "MRR": best["MRR"],
            "baseline_with_LEG_NDCG@20": 0.28256615785544326 if seed == 303 else None,
            "delta_vs_same_R3_with_LEG": (
                best["NDCG@20"] - 0.28256615785544326 if seed == 303 else None
            ),
            "seconds": time.time() - t0,
            "EXTERNAL_SEEN": False,
        }
        out_path = seed_dir / "FULLCAT_NOLEG_INFERENCE.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        (REP / f"FULLCAT_NOLEG_INFERENCE_SEED{seed}.md").write_text(
            f"# R3 inference-only LEG ablation, seed {seed}\n\n"
            f"- checkpoint epoch: **{best['epoch']}**\n"
            f"- NDCG@20: **{best['NDCG@20']:.6f}**\n"
            f"- Recall@20: **{best['Recall@20']:.6f}**\n"
            f"- MRR: **{best['MRR']:.6f}**\n"
            f"- delta vs same R3 checkpoint with LEG: "
            f"**{out['delta_vs_same_R3_with_LEG']:+.6f}**\n"
        )
        print(f"[tfhn-eval] NOLEG inference result → {out_path}", flush=True)
        return
    # freeze scientific best
    src = load_ckpt_path(ckpt_dir, best["epoch"])
    raw = torch.load(src, map_location="cpu")
    frozen_name = f"TRUE_FINAL_HARDNEG_seed{seed}_FROZEN.pt"
    torch.save(raw, seed_dir / frozen_name)
    torch.save(raw, seed_dir / "model_fullcat_best.pt")
    official = build_payload(
        state=raw["model_state_dict"],
        seed=seed,
        best_epoch=best["epoch"],
        sampled_ndcg=float(summary.get("best_sampled_NDCG@20", float("nan"))),
        extra={
            "selection": "fullcat_DEV_NDCG@20",
            "selection_rule": f"argmax FULLCAT DEV NDCG@20 among SCREEN_3K top-{fullcat_top_k} (rule B)",
            "fullcat_NDCG@20": float(best["NDCG@20"]),
            "fullcat_Recall@20": float(best["Recall@20"]),
            "fullcat_MRR": float(best["MRR"]),
            "screen3k_NDCG@20": float(best["screen3k_NDCG@20"]),
            "delta_vs_B0_seed303": float(best["delta_vs_B0_seed303"]),
            "delta_vs_B0_mean": float(best["delta_vs_B0_mean"]),
            "FULLRANK_STATUS": "DONE_DEV_FULLCAT",
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "EXTERNAL_STATUS": "LOCKED_NOT_RUN",
            "EXTERNAL_SEEN": False,
            "frozen_name": frozen_name,
            "ckpt_sha256": file_sha256(src),
        },
    )
    copy_scalers_to_audit(feat_dir)
    ckpt_path = save_final_seed_ckpt(seed, official)
    # also write named frozen copy under JOINT
    torch.save(official, JOINT_OUT / "06_CHECKPOINTS" / frozen_name)

    out = {
        "seed": seed,
        "screen3k": screen_rows,
        "fullcat_candidates": cands,
        "fullcat": full_rows,
        "selected_epoch": best["epoch"],
        "NDCG@20": best["NDCG@20"],
        "Recall@20": best["Recall@20"],
        "MRR": best["MRR"],
        "screen3k_NDCG@20": best["screen3k_NDCG@20"],
        "delta_vs_B0_seed303": best["delta_vs_B0_seed303"],
        "delta_vs_B0_mean": best["delta_vs_B0_mean"],
        "official_ckpt": str(ckpt_path),
        "frozen_name": frozen_name,
        "joint_out": str(JOINT_OUT),
        "seconds": time.time() - t0,
        "EXTERNAL_SEEN": False,
        "screen_early_stop": screen_stop_meta,
    }
    (seed_dir / "FULLCAT.json").write_text(json.dumps(out, indent=2) + "\n")
    (REP / f"FULLCAT_SEED{seed}.md").write_text(
        f"# TRUE FINAL hardneg seed {seed}\n\n"
        f"Selected (FULLCAT): **{best['epoch']}**\n\n"
        f"| metric | value |\n|---|---:|\n"
        f"| FULLCAT NDCG@20 | {best['NDCG@20']:.6f} |\n"
        f"| SCREEN_3K NDCG@20 | {best['screen3k_NDCG@20']:.6f} |\n"
        f"| Recall@20 | {best['Recall@20']:.6f} |\n"
        f"| MRR | {best['MRR']:.6f} |\n"
        f"| Δ vs 0.2719 | {best['delta_vs_B0_seed303']:+.6f} |\n"
        f"| Δ vs ~0.276 | {best['delta_vs_B0_mean']:+.6f} |\n"
    )
    print(
        f"[tfhn-eval] seed={seed} FROZEN ep={best['epoch']} "
        f"FULLCAT={best['NDCG@20']:.6f} → {ckpt_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
