#!/usr/bin/env python3
"""LASTFM_JOINT_HGT_A5_RANKER_V1

Control: same TRUE FINAL joint HGT gradient (all train pairs, one HGT fwd/bwd
per epoch) but **no 4th branch** — no H3 A11-distribution pool, no LEG residual.

Architecture: HGT 64 / 2L / 2H → pair256 + graph_dot + A5 → LateFusion (262-D).
Seeds 101/202/303. TEST locked. Does not overwrite TRUE FINAL checkpoints.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_paths_20260824 import (  # noqa: E402
    PROTOCOL_CONFIG,
    leg_k2_dir,
    materialized_root,
)

from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.lastfm_hgt_a5_ranker_common import (  # noqa: E402
    ARCH_NAME as ARCH_NAME_A5,
    ARCH_NAME_H3,
    ARCH_NAME_HGT_ONLY,
    DECODER_INPUT_DIM as DECODER_INPUT_DIM_A5,
    DECODER_INPUT_DIM_H3,
    DECODER_INPUT_DIM_HGT_ONLY,
    build_hgt_a5_h3_noleg,
    build_hgt_a5_ranker,
    build_hgt_only_ranker,
)
from scripts.run_a11_distributional_representation_series_v1 import (  # noqa: E402
    cell_from_scores,
)
from scripts.run_artist_a11_residual_branch_v1 import sampled_metrics, scale_split  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    host_info,
    mps_bytes,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    abort,
    check_overlaps,
    git_commit,
    verify_fingerprints,
    write_json,
)
from scripts.run_lastfm_final_clean_training_v1 import (  # noqa: E402
    convergence_flag,
    recover_c1_recipe,
)
from scripts.run_lastfm_true_final_joint_training_v1 import (  # noqa: E402
    SEEDS,
    from_scratch_audit,
    selected_seeds,
    state_fingerprint,
    train_one_true_joint,
    utc_now,
    wait_for_c1_gpu,
)
from scripts.run_race_clean_3 import shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = PROTOCOL_CONFIG
RACE = materialized_root()
B0_H = RACE / "race" / "a11_top25" / "features"
LAYER = os.environ.get("LASTFM_LAYER", "a5").strip().lower()
if LAYER not in {"a5", "hgt", "h3"}:
    raise SystemExit(f"LASTFM_LAYER must be a5|hgt|h3, got {LAYER!r}")
HGT_ONLY = LAYER == "hgt"
USE_H3 = LAYER == "h3"
if LAYER == "hgt":
    ARCH_NAME = ARCH_NAME_HGT_ONLY
    DECODER_INPUT_DIM = DECODER_INPUT_DIM_HGT_ONLY
    _DEFAULT_OUT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_ONLY_RANKER_V1"
    _OUT_ENV = "LASTFM_HGT_ONLY_OUT"
elif LAYER == "h3":
    ARCH_NAME = ARCH_NAME_H3
    DECODER_INPUT_DIM = DECODER_INPUT_DIM_H3
    _DEFAULT_OUT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_A5_H3_NOLEG_V1"
    _OUT_ENV = "LASTFM_HGT_H3_OUT"
else:
    ARCH_NAME = ARCH_NAME_A5
    DECODER_INPUT_DIM = DECODER_INPUT_DIM_A5
    _DEFAULT_OUT = ROOT / "LASTFM_TRUE_FINAL" / "JOINT_HGT_A5_RANKER_V1"
    _OUT_ENV = "LASTFM_HGT_A5_OUT"
OUT = Path(os.environ.get(_OUT_ENV, str(_DEFAULT_OUT)))


def dirs() -> dict[str, Path]:
    mapping = {
        "audit": OUT / "00_AUDIT",
        "s101": OUT / "01_SEED101",
        "s202": OUT / "02_SEED202",
        "s303": OUT / "03_SEED303",
        "ckpts": OUT / "06_CHECKPOINTS",
    }
    for p in mapping.values():
        p.mkdir(parents=True, exist_ok=True)
    return mapping


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = dirs()
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n", encoding="utf-8")
    run_seeds = selected_seeds()
    print(f"[hgt-layer] JOINT layer={LAYER} start", flush=True)
    print(f"[hgt-layer] arch={ARCH_NAME} decoder_in={DECODER_INPUT_DIM}", flush=True)
    print(f"[hgt-layer] seeds={list(run_seeds)} hgt_grad=ALL_TRAIN", flush=True)
    wait_for_c1_gpu()

    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    check_overlaps()
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    host = host_info()
    recipe, hist = recover_c1_recipe(bundle)
    write_json(d["audit"] / "DATA_FINGERPRINTS.json", hashes)
    write_json(d["audit"] / "HOST.json", host)
    write_json(
        d["audit"] / "CONTROL_SPEC.json",
        {
            "architecture_name": ARCH_NAME,
            "decoder_input_dim": DECODER_INPUT_DIM,
            "use_hcr_H3": bool(USE_H3),
            "use_LEG": False,
            "use_A5": not HGT_ONLY,
            "hgt": "64/2L/2H drop 0.1",
            "table": "none" if HGT_ONLY else "A5 CLEAN V2",
            "ranker": (
                f"LateFusionHead {DECODER_INPUT_DIM}→128 LN GELU Drop0.2 →64 GELU Drop →1"
            ),
            "hgt_grad_scope": "ALL_TRAIN",
            "optimizer": "Adam lr=1e-3 wd=1e-4 ReduceLROnPlateau C1 recipe",
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "does_not_overwrite_true_final": True,
        },
    )

    n_tr = int(len(bundle["train_pairs"]["label"]))
    n_va = int(len(bundle["val_pairs"]["label"]))
    r_tr = np.zeros((n_tr, 1), dtype=np.float32)
    r_va = np.zeros((n_va, 1), dtype=np.float32)
    if HGT_ONLY:
        A_tr_s = np.zeros((n_tr, 0), dtype=np.float32)
        A_va_s = np.zeros((n_va, 0), dtype=np.float32)
        H_tr_s = np.zeros((n_tr, 3), dtype=np.float32)
        H_va_s = np.zeros((n_va, 3), dtype=np.float32)
    else:
        a_dir = shared_A_dir()
        A_tr = np.load(a_dir / "X_train.npy").astype(np.float32)
        A_va = np.load(a_dir / "X_val.npy").astype(np.float32)
        a_scaler = StandardScaler().fit(A_tr)
        A_tr_s, A_va_s = scale_split(a_scaler, A_tr), scale_split(a_scaler, A_va)
        with (d["audit"] / "a_scaler.pkl").open("wb") as f:
            pickle.dump(a_scaler, f)
        if USE_H3:
            H_tr = np.load(B0_H / "X_train.npy").astype(np.float32)
            H_va = np.load(B0_H / "X_val.npy").astype(np.float32)
            h_scaler = StandardScaler().fit(H_tr)
            H_tr_s, H_va_s = scale_split(h_scaler, H_tr), scale_split(h_scaler, H_va)
            with (d["audit"] / "h_scaler.pkl").open("wb") as f:
                pickle.dump(h_scaler, f)
        else:
            H_tr_s = np.zeros((n_tr, 3), dtype=np.float32)
            H_va_s = np.zeros((n_va, 3), dtype=np.float32)

    graph = load_data_and_typed_graph(
        bundle["cfg"], bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
    item_offset = int(graph["meta"]["item_offset"])
    va_u, va_y = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"]

    seed_dirs = {101: d["s101"], 202: d["s202"], 303: d["s303"]}
    rows: list[dict[str, Any]] = []
    commit = git_commit()

    for seed in run_seeds:
        ckpt_done = d["ckpts"] / f"final_seed{seed}_best.pt"
        summary_done = seed_dirs[seed] / "seed_summary.json"
        if ckpt_done.exists() and summary_done.exists():
            row = json.loads(summary_done.read_text())
            rows.append(row)
            print(
                f"[hgt-layer] skip seed={seed} already trained sampled={row.get('best_sampled_NDCG@20')}",
                flush=True,
            )
            continue
        print(f"[hgt-layer] FROM_SCRATCH seed={seed} layer={LAYER}", flush=True)
        run_dir = seed_dirs[seed] / "run"
        if run_dir.exists():
            for p in run_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass
        if LAYER == "hgt":
            builder = build_hgt_only_ranker
        elif LAYER == "h3":
            builder = build_hgt_a5_h3_noleg
        else:
            builder = build_hgt_a5_ranker
        model = builder(bundle, graph, device, d=64, layers=2, heads=2)
        n_in = int(model.fusion_head.net[0].in_features)
        if n_in != DECODER_INPUT_DIM:
            abort(f"decoder in_features={n_in} expected {DECODER_INPUT_DIM}")
        if sum(p.numel() for p in model.leg.parameters()) != 0:
            abort("ZeroLeg must have 0 parameters")
        init_fp = state_fingerprint(model)
        scratch = from_scratch_audit(model, seed, init_fp)
        scratch["LEG_K2"] = "ABSENT_ZERO_STUB"
        scratch["H3"] = "PRESENT" if USE_H3 else "ABSENT_use_hcr=False"
        scratch["A5"] = "ABSENT" if HGT_ONLY else "PRESENT"
        write_json(seed_dirs[seed] / "FROM_SCRATCH_AUDIT.json", scratch)
        print(f"[hgt-layer] seed={seed} fp={init_fp} decoder_in={n_in}", flush=True)
        out = train_one_true_joint(
            model=model,
            bundle=bundle,
            A_tr_s=A_tr_s,
            A_va_s=A_va_s,
            H_tr_s=H_tr_s,
            H_va_s=H_va_s,
            r_tr=r_tr,
            r_va=r_va,
            item_offset=item_offset,
            device=device,
            seed=seed,
            ckpt=run_dir,
            rec=hist,
        )
        meta = out["meta"]
        meta["leg_grad_scope"] = "ABSENT"
        meta["h3_in_decoder"] = bool(USE_H3)
        cell = cell_from_scores(va_u, va_y, out["logits"])
        m = cell["metrics"]
        conv = convergence_flag(meta)
        hist_list = meta.get("history") or []
        best_h = next((h for h in hist_list if int(h["epoch"]) == int(meta["best_epoch"])), None)
        row = {
            "seed": seed,
            "architecture_name": ARCH_NAME,
            "best_epoch": int(meta["best_epoch"]),
            "stopping_epoch": int(meta["final_epoch"]),
            "best_sampled_NDCG@20": float(meta["best_val_NDCG@20_sampled"]),
            "final_sampled_NDCG@20": float(m["NDCG@20"]),
            "MRR": float(m.get("MRR", float("nan"))),
            "Recall@20": float(m.get("Recall@20", float("nan"))),
            "HitRate@20": float(m.get("HitRate@20", m.get("HR@20", float("nan")))),
            "lr_at_best": float(best_h["lr"]) if best_h else float("nan"),
            "stop_reason": meta.get("stop_reason"),
            "CONVERGENCE_WARNING": bool(meta.get("CONVERGENCE_WARNING")),
            "convergence": conv,
            "seconds": float(meta.get("seconds", float("nan"))),
            "peak_mem": mps_bytes(),
            "init_state_fingerprint": init_fp,
            "hgt_grad_scope": "ALL_TRAIN",
            "use_H3": bool(USE_H3),
            "use_LEG": False,
            "use_A5": not HGT_ONLY,
            "layer": LAYER,
        }
        rows.append(row)
        write_json(seed_dirs[seed] / "seed_summary.json", row)
        write_json(seed_dirs[seed] / "train_meta.json", meta)
        ckpt_path = d["ckpts"] / f"final_seed{seed}_best.pt"
        payload = {
            "model_state_dict": torch.load(run_dir / "model.pt", map_location="cpu"),
            "seed": seed,
            "best_epoch": int(meta["best_epoch"]),
            "sampled_NDCG@20": float(meta["best_val_NDCG@20_sampled"]),
            "config_hash": recipe["config_hash"],
            "dataset_split_ids": hashes,
            "git_commit": commit,
            "architecture_name": ARCH_NAME,
            "timestamp": utc_now(),
            "training": (
                "JOINT_HGT_ONLY_RANKER"
                if HGT_ONLY
                else "JOINT_HGT_A5_H3_NOLEG"
                if USE_H3
                else "JOINT_HGT_A5_RANKER"
            ),
            "hgt_grad_scope": "ALL_TRAIN",
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "FULLRANK_STATUS": "LOCKED_NOT_RUN",
            "init_state_fingerprint": init_fp,
            "loaded_c1_checkpoint": False,
            "loaded_b0_checkpoint": False,
            "loaded_true_final_checkpoint": False,
        }
        torch.save(payload, ckpt_path)
        print(
            f"[hgt-layer] seed={seed} best_ep={row['best_epoch']} "
            f"sampled_NDCG={row['best_sampled_NDCG@20']:.6f} conv={conv}",
            flush=True,
        )
        del model
        empty_cache()
        gc.collect()

    write_json(
        OUT / "SEED_SUMMARY.json",
        {"seeds": list(run_seeds), "rows": rows, "timestamp": utc_now(), "arch": ARCH_NAME},
    )
    print("[hgt-layer] DONE seeds=" + ",".join(str(s) for s in run_seeds) + f" layer={LAYER}", flush=True)
    print("TEST_STATUS =\n    LOCKED_NOT_RUN", flush=True)


if __name__ == "__main__":
    main()
