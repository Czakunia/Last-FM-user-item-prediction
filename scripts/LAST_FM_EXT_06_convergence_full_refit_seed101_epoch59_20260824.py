#!/usr/bin/env python3
"""POST_HOC convergence: full TRAIN_EXTERNAL refit for seed 101 at best inner epoch.

Pilot (VAL_INNER sampled NDCG@20) selected best_epoch=59 (not 89 — 89 was only early-stop).
This script retrains TRUE FINAL on full TRAIN_EXTERNAL for exactly 59 epochs and stores
the checkpoint in a NEW directory so the original frozen epoch-36 sealed artifact is
not overwritten.

After this finishes, run sealed evaluation for this seed only via:
  LASTFM_CONVERGENCE_CKPT=... .venv/bin/python scripts/LAST_FM_EXT_06b_sealed_eval_convergence_seed101_20260824.py
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

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    AUDIT_DIR,
    build_external_bundle,
    build_external_training_bundle,
    empty_cache,
    frozen_hgt_training_hist,
    git_commit,
    materialize_a5_h3_leg_train,
    sha256_file,
    true_final_kw,
    utc_now,
    write_json,
)
from scripts.run_artist_a11_residual_branch_v1 import scale_split  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import build_race_model  # noqa: E402
from scripts.run_lastfm_final_clean_training_v1 import a11_audit, architecture_audit  # noqa: E402
from scripts.run_lastfm_true_final_joint_training_v1 import (  # noqa: E402
    seed_everything,
    train_epoch_true_joint,
)
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

# Best VAL_INNER epoch from POST_HOC pilot seed 101 (sampled NDCG@20).
BEST_EPOCH = 59
SEED = 101
OUT = ROOT / "LASTFM_EXTERNAL_CONVERGENCE_REFIT_20260824"
SEED_DIR = OUT / "models" / f"FULL_TRAIN_SEED{SEED}_EPOCH{BEST_EPOCH}"


def main() -> None:
    print(
        f"[conv-full] seed={SEED} fixed_epochs={BEST_EPOCH} "
        f"(pilot best; early-stop was 89 — not used)",
        flush=True,
    )
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SEED_DIR / "model.pt"
    if ckpt_path.exists() and (SEED_DIR / "meta.json").exists():
        meta = json.loads((SEED_DIR / "meta.json").read_text(encoding="utf-8"))
        print(f"[conv-full] reuse existing {ckpt_path} epochs={meta.get('epoch_count')}", flush=True)
        print(f"[conv-full] sha256={sha256_file(ckpt_path)}", flush=True)
        return

    bundle = build_external_bundle()
    prepared = build_external_training_bundle(bundle)
    feat_dirs = materialize_a5_h3_leg_train(prepared)
    A_tr = np.load(feat_dirs["a_dir"] / "X_train.npy").astype(np.float32)
    H_tr = np.load(feat_dirs["h_dir"] / "X_train.npy").astype(np.float32)
    r_tr = np.load(feat_dirs["leg_dir"] / "LEG_K2_train.npy").astype(np.float32)
    if r_tr.ndim == 1:
        r_tr = r_tr[:, None]
    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(H_tr)
    A_tr_s = scale_split(a_scaler, A_tr)
    H_tr_s = scale_split(h_scaler, H_tr)
    with (SEED_DIR / "a_scaler.pkl").open("wb") as f:
        pickle.dump(a_scaler, f)
    with (SEED_DIR / "h_scaler.pkl").open("wb") as f:
        pickle.dump(h_scaler, f)

    graph = load_data_and_typed_graph(
        bundle.cfg,
        bundle.train_external,
        max_kg_edges=bundle.cfg.get("models", {}).get("architecture", {}).get("max_kg_edges", 250_000),
    )
    item_offset = int(graph["meta"]["item_offset"])
    device = resolve_torch_device(
        str(bundle.cfg.get("models", {}).get("architecture", {}).get("device") or "auto")
    )
    hist = frozen_hgt_training_hist()
    kw = true_final_kw(hist)
    probe = build_race_model(prepared, graph, device, d=64, layers=2, heads=2)
    arch = architecture_audit(probe, device)
    a11 = a11_audit(r_tr, r_tr[: min(len(r_tr), 1024)])
    del probe
    empty_cache()
    if arch["ARCHITECTURE_STATUS"] != "FROZEN_REPRODUCED" or a11["A11_AUDIT_STATUS"] != "PASS":
        raise RuntimeError("Pre-fit architecture/A11 audit failed")

    u_all, i_all = user_item_to_nodes(
        prepared["train_pairs"]["user_id"], prepared["train_pairs"]["item_id"], item_offset
    )
    y_all = prepared["train_pairs"]["label"].astype(np.float32)
    n_train = len(y_all)
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())

    seed_everything(SEED)
    model = build_race_model(prepared, graph, device, d=64, layers=2, heads=2)
    opt = torch.optim.Adam(model.parameters(), lr=kw["lr0"], weight_decay=float(hist["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )
    batch_size = int(bundle.cfg.get("models", {}).get("architecture", {}).get("batch_size", 4096))
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(BEST_EPOCH):
        model.train()
        perm = np.random.permutation(n_train)
        info = train_epoch_true_joint(
            model=model,
            opt=opt,
            loss_fn=loss_fn,
            u_all=u_all,
            i_all=i_all,
            y_all=y_all,
            A_tr_s=A_tr_s,
            H_tr_s=H_tr_s,
            r_tr=r_tr,
            device=device,
            perm=perm,
            batch_size=batch_size,
            collect_autograd=(epoch == 0),
        )
        history.append(
            {
                "epoch": epoch,
                "epoch_1based": epoch + 1,
                "loss": float(info["epoch_mean_loss"]),
                "sec": float(info.get("sec", 0.0)),
            }
        )
        print(
            f"[conv-full seed={SEED}] epoch {epoch + 1}/{BEST_EPOCH} loss={history[-1]['loss']:.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), ckpt_path)
    meta = {
        "experiment_tag": "POST_HOC_CONVERGENCE_CONTROLLED_REFIT",
        "seed": SEED,
        "epoch_count": BEST_EPOCH,
        "pilot_best_epoch": BEST_EPOCH,
        "pilot_early_stop_epoch": 89,
        "NOTE": "89 was early-stop only; checkpoint selected at best VAL_INNER epoch 59",
        "architecture": "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL",
        "training_hash": bundle.training_hash,
        "does_not_overwrite_frozen_epoch36": True,
        "frozen_epoch36_path": "LASTFM_EXTERNAL_BENCHMARK_20260824/models/TRUE_FINAL_REFIT_SEED101/model.pt",
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "fit_seconds": time.time() - t0,
        "history": history,
        "sha256": sha256_file(ckpt_path),
        "SEALED_TEST_NOT_EVALUATED_YET": True,
    }
    write_json(SEED_DIR / "meta.json", meta)
    write_json(OUT / "audit" / f"full_train_seed{SEED}_epoch{BEST_EPOCH}.json", meta)
    print(f"[conv-full] saved {ckpt_path}", flush=True)
    print(f"[conv-full] sha256={meta['sha256']}", flush=True)
    print("[conv-full] DONE — next: sealed eval for this checkpoint only", flush=True)


if __name__ == "__main__":
    main()
