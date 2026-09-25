#!/usr/bin/env python3
"""Restore official R3 from-scratch trainer (original script was wiped).

BCE + 0.5 BPR, d=64 L=2 H=2, static R3 table, 20 epochs, seed from TFHN_SEED.
Writes epoch_init.pt + epoch_XXX.pt under $TFHN_ART/seed{seed}/checkpoints.
Does not touch the official seed303 JSON tree unless TFHN_ART points there.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "lastfm_rank_v2" / "src"))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    build_race_model,
)
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.data.build_splits import load_user_sets  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402
from tfhn.ckpt_io import build_payload  # noqa: E402
from tfhn.paths import ARCH, ART, SPLITS_DIR, resolve_hgt_dims  # noqa: E402
from tfhn.sealed_external import (  # noqa: E402
    BATCH_BCE,
    BATCH_BPR,
    LAMBDA_BPR,
    LR,
    WD,
    build_triplets,
    train_epoch_bce_bpr_joint,
)
from scripts.run_lastfm_true_final_joint_training_v1 import (  # noqa: E402
    user_item_to_nodes,
)

SEED = int(os.environ.get("TFHN_SEED", "303"))
MAX_EPOCHS = int(os.environ.get("TFHN_MAX_EPOCHS", "20"))
HGT_D, HGT_LAYERS, HGT_HEADS = resolve_hgt_dims()


def scale(arr: np.ndarray, scaler) -> np.ndarray:
    return ((np.asarray(arr) - scaler.mean_) / np.maximum(scaler.scale_, 1e-12)).astype(
        np.float32
    )


def save_epoch(path: Path, model, epoch, sampled: float) -> None:
    payload = build_payload(
        state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        seed=SEED,
        best_epoch=epoch,
        sampled_ndcg=float(sampled),
        extra={"epoch": epoch, "rebuild": True},
    )
    torch.save(payload, path)


def main() -> None:
    seed_dir = ART / f"seed{SEED}"
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = ART / "hardneg_train_pairs.npz"
    feat_dir = ART / "features"
    if not pairs_path.exists() or not (feat_dir / "A_train.npy").exists():
        raise SystemExit(f"missing R3 table under {ART}; run 01_build_hardneg_data.py")

    device = resolve_torch_device(os.environ.get("TFHN_DEVICE", "auto"))
    print(f"[tfhn-train] seed={SEED} device={device} epochs={MAX_EPOCHS} art={ART}", flush=True)

    pairs = np.load(pairs_path)
    users = np.asarray(pairs["user_id"], dtype=np.int64)
    items = np.asarray(pairs["item_id"], dtype=np.int64)
    labels = np.asarray(pairs["label"], dtype=np.int8)
    with (feat_dir / "a_scaler.pkl").open("rb") as handle:
        a_sc = pickle.load(handle)
    with (feat_dir / "h_scaler.pkl").open("rb") as handle:
        h_sc = pickle.load(handle)
    with (feat_dir / "l_scaler.pkl").open("rb") as handle:
        l_sc = pickle.load(handle)
    A_s = scale(np.load(feat_dir / "A_train.npy", mmap_mode="r"), a_sc)
    H_s = scale(np.load(feat_dir / "H_train.npy", mmap_mode="r"), h_sc)
    L_s = scale(np.load(feat_dir / "L_train.npy", mmap_mode="r"), l_sc)

    cfg = load_protocol_config(PROTOCOL_CONFIG)
    bundle = load_prepared(cfg, verify=False)
    mt = load_user_sets(SPLITS_DIR / "model_train.txt")
    bundle["model_train"] = mt
    graph = load_data_and_typed_graph(
        bundle["cfg"],
        mt,
        max_kg_edges=bundle["cfg"].get("models", {}).get("architecture", {}).get(
            "max_kg_edges", 250_000
        ),
    )
    item_offset = int(graph["meta"]["item_offset"])

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_race_model(bundle, graph, device, d=HGT_D, layers=HGT_LAYERS, heads=HGT_HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    u_all, i_all = user_item_to_nodes(users, items, item_offset)
    y_all = labels.astype(np.float32)
    u_t, i_p, i_n, idx_p, idx_n = build_triplets(users, items, labels)
    u_pos_n, i_pos_n = user_item_to_nodes(u_t, i_p, item_offset)
    _, i_neg_n = user_item_to_nodes(u_t, i_n, item_offset)
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )

    save_epoch(ckpt_dir / "epoch_init.pt", model, "init", float("nan"))
    history = []
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        te = time.time()
        print(f"[tfhn-train] ep {epoch}/{MAX_EPOCHS - 1} …", flush=True)
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
        row = {
            "epoch": epoch,
            "bce_loss": info["bce_loss"],
            "bpr_loss": info["bpr_loss"],
            "train_loss": info["train_loss"],
            "lambda_bpr": LAMBDA_BPR,
            "hgt_forward": info["hgt_forward"],
            "sampled_NDCG@20": None,
            "lr": LR,
            "sec": time.time() - te,
        }
        history.append(row)
        save_epoch(ckpt_dir / f"epoch_{epoch:03d}.pt", model, epoch, float("nan"))
        print(
            f"[tfhn-train] ep {epoch} loss={row['train_loss']:.4f} ({row['sec']:.1f}s)",
            flush=True,
        )

    summary = {
        "seed": SEED,
        "MAX_EPOCHS": MAX_EPOCHS,
        "early_stop": False,
        "best_sampled_epoch": None,
        "best_sampled_NDCG@20": float("nan"),
        "n_epochs_run": MAX_EPOCHS,
        "seconds": time.time() - t0,
        "lambda_bpr": LAMBDA_BPR,
        "from_scratch": True,
        "rebuild": True,
        "architecture": ARCH,
        "history": history,
    }
    (seed_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (seed_dir / "train_summary_local.json").write_text(json.dumps(summary, indent=2) + "\n")
    (seed_dir / "FROM_SCRATCH_PROOF.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "PRETRAINED_CHECKPOINT": "NONE",
                "RESUME_CHECKPOINT": "NONE",
                "INIT_MODE": "FRESH_RANDOM",
                "rebuild": True,
            },
            indent=2,
        )
        + "\n"
    )
    (seed_dir / "TRAIN_DONE.flag").write_text("DONE\n")
    print(f"[tfhn-train] done → {seed_dir} ({summary['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
