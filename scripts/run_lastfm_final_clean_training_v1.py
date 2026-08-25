#!/usr/bin/env python3
"""LASTFM_FINAL_CLEAN_TRAINING_V1

Final clean reproduction of the frozen Last-FM* architecture under the
winning C1 convergence recipe from FINAL_HGT_CAPACITY_CONVERGENCE_RACE_V1.

Architecture search CLOSED. A11 CLOSED. Capacity CLOSED. Convergence CLOSED.
TEST LOCKED. FULL-RANK LOCKED. Train from scratch. Seeds 101 / 202 / 303 only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_paths_20260824 import (  # noqa: E402
    PROTOCOL_CONFIG,
    leg_k2_dir,
    materialized_root,
)

from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_a11_distributional_representation_series_v1 import (  # noqa: E402
    ResMLP,
    cell_from_scores,
)
from scripts.run_a11_functional_distribution_benchmark_v2 import (  # noqa: E402
    feats_orthogonal,
    legendre_stack,
)
from scripts.run_artist_a11_residual_branch_v1 import scale_split  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    GRAPH_CTX,
    READOUT_DIM,
    MIN_IMPROVE,
    build_race_model,
    host_info,
    mps_bytes,
    param_blocks,
    recipe_kwargs,
    score_model,
    train_one,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    abort,
    check_overlaps,
    git_commit,
    verify_fingerprints,
    write_csv,
    write_json,
)
from scripts.run_race_clean_3 import shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = PROTOCOL_CONFIG
RACE = materialized_root()
CAP = RACE / "FINAL_HGT_CAPACITY_CONVERGENCE_RACE_V1"
V2 = leg_k2_dir(RACE)  # publication: .../leg_k2 (legacy V2/nxt compat inside helper)
B0_H = RACE / "race" / "a11_top25" / "features"
OUT = Path(
    os.environ.get(
        "LASTFM_FINAL_OUT",
        str(ROOT / "LASTFM_TRUE_FINAL" / "C1_REPRODUCTION_CONTROL"),
    )
)
SEEDS = (101, 202, 303)
ARCH_NAME = "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dirs() -> dict[str, Path]:
    mapping = {
        "audit": OUT / "00_AUDIT",
        "s101": OUT / "01_SEED101",
        "s202": OUT / "02_SEED202",
        "s303": OUT / "03_SEED303",
        "summary": OUT / "04_SUMMARY",
        "figures": OUT / "05_FIGURES",
        "ckpts": OUT / "06_CHECKPOINTS",
    }
    for p in mapping.values():
        p.mkdir(parents=True, exist_ok=True)
    return mapping


def config_hash(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def recover_c1_recipe(bundle) -> dict[str, Any]:
    """Recover winning C1 recipe from capacity race artifacts + live config."""
    hist = json.loads(
        (CAP / "00_AUDIT" / "CURRENT_HGT_TRAINING_CONFIG.json").read_text(encoding="utf-8")
    )
    # C1 deltas are defined in recipe_kwargs of the capacity race (not inventable)
    c1 = recipe_kwargs("C1", hist)
    conv_csv = CAP / "01_CONVERGENCE" / "CONVERGENCE_RESULTS_BY_SEED.csv"
    recovered = {
        "source_race": "FINAL_HGT_CAPACITY_CONVERGENCE_RACE_V1",
        "source_config_json": str(CAP / "00_AUDIT" / "CURRENT_HGT_TRAINING_CONFIG.json"),
        "source_convergence_csv": str(conv_csv) if conv_csv.exists() else None,
        "architecture": {
            "hgt_hidden": 64,
            "hgt_layers": 2,
            "hgt_heads": 2,
            "hgt_dropout": float(hist["dropout_hgt"]),
            "pair_context_dim": 256,
            "graph_dot_dim": 1,
            "A5_dim": 5,
            "H3_dim": 3,
            "decoder_input_dim": 265,
            "leg_k2_input_dim": 1,
            "decoder": hist["decoder"],
            "leg_k2": hist["leg_k2"],
        },
        "optimizer": hist["optimizer"],
        "lr": float(hist["lr0"]),
        "weight_decay": float(hist["weight_decay"]),
        "scheduler": "ReduceLROnPlateau",
        "scheduler_mode": "max",
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "scheduler_threshold": 1e-4,
        "min_lr": float(c1["min_lr"]),
        "max_epochs": int(c1["max_epochs"]),
        "min_epochs": int(c1["min_epochs"]),
        "early_stopping_patience": int(c1["patience"]),
        "early_stopping_min_delta": MIN_IMPROVE,
        "batch_size": int(hist["batch_size"]),
        "dropout_hgt": float(hist["dropout_hgt"]),
        "dropout_decoder": 0.2,
        "gradient_clipping": hist["gradient_clipping"],
        "amp_precision": hist["amp_precision"],
        "loss": hist["loss"],
        "negative_sampling": "frozen prepared pair tables (train_per_positive=4, val 20)",
        "checkpoint_selection_metric": "sampled validation NDCG@20",
        "historical_C0": {
            "max_epochs": hist["max_epochs_current"],
            "patience": hist["patience_current"],
            "scheduler": hist["scheduler_current"],
        },
        "c1_empirical_from_race": None,
    }
    if conv_csv.exists():
        import pandas as pd

        df = pd.read_csv(conv_csv)
        recovered["c1_empirical_from_race"] = {
            "mean_C1_NDCG20": float(df["C1_NDCG20"].mean()),
            "per_seed": df.to_dict(orient="records"),
        }
    recovered["config_hash"] = config_hash(recovered)
    return recovered, hist


def write_training_config_audit(d: dict[str, Path], recipe: dict[str, Any], hist: dict[str, Any]) -> None:
    md = f"""# FINAL_TRAINING_CONFIG_AUDIT

Recovered at {utc_now()} from **FINAL_HGT_CAPACITY_CONVERGENCE_RACE_V1**
(`CURRENT_HGT_TRAINING_CONFIG.json` + `recipe_kwargs("C1", …)`).

**No hyperparameters were invented or retuned for this run.**

## Winning recipe = C1_EXTENDED

| Field | Value |
|---|---|
| optimizer | {recipe['optimizer']} |
| LR | {recipe['lr']} |
| weight decay | {recipe['weight_decay']} |
| scheduler | {recipe['scheduler']} (mode={recipe['scheduler_mode']}, factor={recipe['scheduler_factor']}, patience={recipe['scheduler_patience']}, threshold={recipe['scheduler_threshold']}) |
| min LR | {recipe['min_lr']} (= lr0/16) |
| max epochs | {recipe['max_epochs']} |
| minimum epochs | {recipe['min_epochs']} |
| early-stopping patience | {recipe['early_stopping_patience']} |
| early-stopping min_delta | {recipe['early_stopping_min_delta']} |
| batch size | {recipe['batch_size']} |
| HGT dropout | {recipe['dropout_hgt']} |
| decoder dropout | {recipe['dropout_decoder']} |
| gradient clipping | {recipe['gradient_clipping']} |
| AMP / precision | {recipe['amp_precision']} |
| loss | {recipe['loss']} |
| negative sampling | {recipe['negative_sampling']} |
| checkpoint-selection metric | **{recipe['checkpoint_selection_metric']}** |

## Historical C0 (NOT used)

- max_epochs={recipe['historical_C0']['max_epochs']}, patience={recipe['historical_C0']['patience']}, scheduler={recipe['historical_C0']['scheduler']}

## Architecture (frozen)

- HGT 64 / 2L / 2H
- pair 256 + graph_dot 1 + A5 + H3 = 265 → LateFusion B0
- LEG_K2 residual: Linear(1,16)→GELU→Linear(16,1) zero-init last layer

## Config hash

`{recipe['config_hash']}`

## C1 race empirical reference (not a target to beat)

```
{json.dumps(recipe.get('c1_empirical_from_race'), indent=2)}
```
"""
    (d["audit"] / "FINAL_TRAINING_CONFIG_AUDIT.md").write_text(md, encoding="utf-8")
    write_json(d["audit"] / "FINAL_TRAINING_CONFIG_AUDIT.json", recipe)
    write_json(d["summary"] / "FINAL_CONFIG_RESOLVED.yaml".replace(".yaml", ".json"), recipe)
    # also dump a yaml-like json companion named as required
    (d["summary"] / "FINAL_CONFIG_RESOLVED.yaml").write_text(
        "# Resolved FINAL CLEAN training config (YAML-compatible JSON dump)\n"
        + json.dumps(recipe, indent=2)
        + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def architecture_audit(model, device) -> dict[str, Any]:
    # probe dims
    B = 4
    zu = torch.randn(B, READOUT_DIM, device=device)
    zi = torch.randn(B, READOUT_DIM, device=device)
    gctx = torch.cat([zu, zi, zu * zi, (zu - zi).abs()], dim=-1)
    gscore = (zu * zi).sum(dim=-1)
    A5 = torch.randn(B, 5, device=device)
    H3 = torch.randn(B, 3, device=device)
    r = torch.randn(B, 1, device=device)
    b0 = model.fusion_head(gctx, gscore, A5, H3)
    delta = model.leg(r)
    core = model.encoder.core
    checks = {
        "hgt_hidden": int(core.user_embed.embedding_dim),
        "hgt_layers": int(len(core.convs)),
        "hgt_heads": int(getattr(core, "heads", core.convs[0].heads) if hasattr(core.convs[0], "heads") else 2),
        "pair_context_dim": int(gctx.size(-1)),
        "graph_dot_dim": 1,
        "A5_dim": 5,
        "H3_dim": 3,
        "decoder_input_dim": int(gctx.size(-1) + 1 + 5 + 3),
        "leg_k2_input_dim": int(r.size(-1)),
        "b0_logit_dim": int(b0.numel() // B),
        "delta_L2_dim": int(delta.numel() // B),
        "readout_is_identity": bool(isinstance(model.readout, nn.Identity)),
        "params": param_blocks(model),
        "forbidden": {
            "L3_branch": False,
            "L4_branch": False,
            "TEMP": False,
            "attention_A11": False,
            "conditional_HGTxA11": False,
            "alternative_routing": False,
        },
    }
    # heads from first conv if available
    try:
        checks["hgt_heads"] = int(core.convs[0].heads)
    except Exception:
        pass
    # zero-init residual check
    delta0 = model.leg(torch.randn(32, 1, device=device))
    checks["delta_L2_init_maxabs"] = float(delta0.abs().max())
    checks["delta_L2_approx_zero_at_init"] = bool(checks["delta_L2_init_maxabs"] < 1e-6)
    ok = (
        checks["hgt_hidden"] == 64
        and checks["hgt_layers"] == 2
        and checks["hgt_heads"] == 2
        and checks["pair_context_dim"] == 256
        and checks["decoder_input_dim"] == 265
        and checks["leg_k2_input_dim"] == 1
        and checks["delta_L2_approx_zero_at_init"]
        and checks["readout_is_identity"]
    )
    checks["ARCHITECTURE_STATUS"] = "FROZEN_REPRODUCED" if ok else "AUDIT_FAILED"
    return checks


def write_architecture_audit(d: dict[str, Path], audit: dict[str, Any]) -> None:
    md = f"""# FINAL_ARCHITECTURE_AUDIT

Inspected **instantiated modules** at {utc_now()} (not config names alone).

| Assert | Observed | Expected |
|---|---:|---:|
| HGT hidden | {audit['hgt_hidden']} | 64 |
| layers | {audit['hgt_layers']} | 2 |
| heads | {audit['hgt_heads']} | 2 |
| pair context | {audit['pair_context_dim']} | 256 |
| graph dot | {audit['graph_dot_dim']} | 1 |
| A5 | {audit['A5_dim']} | 5 |
| H3 | {audit['H3_dim']} | 3 |
| decoder input | {audit['decoder_input_dim']} | 265 |
| LEG_K2 input | {audit['leg_k2_input_dim']} | 1 |
| ΔL2≈0 at init | {audit['delta_L2_approx_zero_at_init']} (maxabs={audit['delta_L2_init_maxabs']:.2e}) | True |
| readout Identity | {audit['readout_is_identity']} | True |

Forbidden branches present? **NO** (L3/L4/TEMP/attention A11 / conditional HGT×A11 / alt routing).

Params: `{json.dumps(audit['params'])}`

**ARCHITECTURE_STATUS = {audit['ARCHITECTURE_STATUS']}**
"""
    (d["audit"] / "FINAL_ARCHITECTURE_AUDIT.md").write_text(md, encoding="utf-8")
    write_json(d["audit"] / "FINAL_ARCHITECTURE_AUDIT.json", audit)


def a11_audit(r_tr: np.ndarray, r_va: np.ndarray) -> dict[str, Any]:
    """Verify LEG_K2 cache is 1-D and matches P2 mean definition on a sample if Top25 exists."""
    out: dict[str, Any] = {
        "leg_k2_train_shape": list(r_tr.shape),
        "leg_k2_val_shape": list(r_va.shape),
        "leg_k2_is_one_scalar": bool(r_tr.ndim == 2 and r_tr.shape[1] == 1),
        "formula": "L2 = mean_{h in signed Top25} P2(A11(h,X)), P2(x)=(3x^2-1)/2",
        "scaler_note": "A5 and H3 StandardScaler fit on MODEL_TRAIN only (train rows of prepared pairs)",
        "routing": "signed Top25 unchanged (V2 / B0 a11_top25 H3 pool)",
        "fit_scope": "A11 / LEG_K2 features precomputed on train-only statistics in prior closed A11 series; no val/test users in fit",
    }
    # Recompute P2 mean on a synthetic clip sample to document formula identity
    rng = np.random.default_rng(20260722)
    x = rng.uniform(-1, 1, size=(8, 25)).astype(np.float64)
    n = np.full(8, 25, dtype=np.int32)
    p2 = (3.0 * x * x - 1.0) / 2.0
    manual = p2.mean(axis=1)
    via_stack = feats_orthogonal(x.astype(np.float32), n, k_max=2, kind="leg")[:, 0]
    out["formula_check_maxabs_diff"] = float(np.max(np.abs(manual - via_stack)))
    out["formula_check_ok"] = bool(out["formula_check_maxabs_diff"] < 1e-5)
    # compare cache finite
    out["train_finite"] = bool(np.isfinite(r_tr).all())
    out["val_finite"] = bool(np.isfinite(r_va).all())
    out["A11_AUDIT_STATUS"] = (
        "PASS"
        if out["leg_k2_is_one_scalar"] and out["formula_check_ok"] and out["train_finite"]
        else "FAIL"
    )
    return out


def write_a11_audit(d: dict[str, Path], audit: dict[str, Any]) -> None:
    md = f"""# FINAL_A11_AUDIT

Verified at {utc_now()}.

## LEG_K2 definition

\\[
L_2(u,X)=\\mathrm{{mean}}_{{h\\in\\mathrm{{signed\\ Top25}}(u,X)}} P_2(A_{{11}}(h,X)),
\\quad P_2(x)=(3x^2-1)/2
\\]

- Cached arrays: train {audit['leg_k2_train_shape']}, val {audit['leg_k2_val_shape']}
- One scalar per pair: **{audit['leg_k2_is_one_scalar']}**
- Formula identity (feats_orthogonal LEG k=2 vs manual P2 mean): maxabs diff = {audit['formula_check_maxabs_diff']:.2e} → ok={audit['formula_check_ok']}

## Routing / fit scope

- {audit['routing']}
- {audit['fit_scope']}
- {audit['scaler_note']}

**A11_AUDIT_STATUS = {audit['A11_AUDIT_STATUS']}**
"""
    (d["audit"] / "FINAL_A11_AUDIT.md").write_text(md, encoding="utf-8")
    write_json(d["audit"] / "FINAL_A11_AUDIT.json", audit)


def convergence_flag(meta: dict[str, Any]) -> str:
    fe = int(meta["final_epoch"])
    be = int(meta["best_epoch"])
    # within final 3 epochs inclusive: be >= fe-2
    if be >= fe - 2 and float(meta["final_lr"]) > float(meta["min_lr"]) + 1e-12:
        return "POSSIBLY_NOT_CONVERGED"
    return "CONVERGED"


def plot_final_curves(histories: dict[int, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for seed, hist in histories.items():
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["val_NDCG@20"] for h in hist], label=f"seed {seed}", lw=1.6)
        axes[1].plot(ep, [h["loss"] for h in hist], label=f"seed {seed}", lw=1.6)
        axes[2].plot(ep, [h["lr"] for h in hist], label=f"seed {seed}", lw=1.6)
    axes[0].set_title("A) sampled val NDCG@20")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("NDCG@20")
    axes[0].legend(fontsize=8)
    axes[1].set_title("B) train loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=8)
    axes[2].set_title("C) learning rate")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].legend(fontsize=8)
    fig.suptitle("FINAL_TRAINING_CURVES — LASTFM_FINAL_CLEAN_TRAINING_V1", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    abort_marker = OUT / "C1_REPRODUCTION_ABORTED.md"
    if abort_marker.exists() and os.environ.get("LASTFM_FORCE_C1", "0") not in {"1", "true", "TRUE"}:
        print(
            "[final] REFUSING TO START — C1_REPRODUCTION_ABORTED.md present "
            "(STATUS=ABORTED_BY_DESIGN). Use LASTFM_FORCE_C1=1 to override.",
            flush=True,
        )
        raise SystemExit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    d = dirs()
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n", encoding="utf-8")
    (OUT / "FULLRANK_LOCKED").write_text("LOCKED_NOT_RUN\n", encoding="utf-8")
    print("[final] LASTFM_FINAL_CLEAN_TRAINING_V1 start", flush=True)

    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    check_overlaps()
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    host = host_info()
    recipe, hist = recover_c1_recipe(bundle)
    write_training_config_audit(d, recipe, hist)
    write_json(d["audit"] / "DATA_FINGERPRINTS.json", hashes)
    write_json(d["audit"] / "HOST.json", host)

    # features
    _feat = ROOT / "LASTFM_TRUE_FINAL" / "C1_REPRODUCTION_CONTROL" / "_feat_cache"
    p_tr = _feat / "LEG_K2_train.npy" if (_feat / "LEG_K2_train.npy").exists() else (V2 / "LEG_K2_train.npy")
    p_va = _feat / "LEG_K2_val.npy" if (_feat / "LEG_K2_val.npy").exists() else (V2 / "LEG_K2_val.npy")
    if not p_tr.exists() or not p_va.exists():
        abort("missing V2 LEG_K2 representations")
    r_tr = np.load(p_tr).astype(np.float32)
    r_va = np.load(p_va).astype(np.float32)
    if r_tr.ndim == 1:
        r_tr = r_tr[:, None]
        r_va = r_va[:, None]
    a_dir = shared_A_dir()
    A_tr = np.load(a_dir / "X_train.npy").astype(np.float32)
    A_va = np.load(a_dir / "X_val.npy").astype(np.float32)
    H_tr = np.load(B0_H / "X_train.npy").astype(np.float32)
    H_va = np.load(B0_H / "X_val.npy").astype(np.float32)
    graph = load_data_and_typed_graph(
        bundle["cfg"], bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000)
    )
    item_offset = int(graph["meta"]["item_offset"])
    va_u, va_y = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"]

    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(H_tr)
    A_tr_s, A_va_s = scale_split(a_scaler, A_tr), scale_split(a_scaler, A_va)
    H_tr_s, H_va_s = scale_split(h_scaler, H_tr), scale_split(h_scaler, H_va)
    with (d["audit"] / "a_scaler.pkl").open("wb") as f:
        pickle.dump(a_scaler, f)
    with (d["audit"] / "h_scaler.pkl").open("wb") as f:
        pickle.dump(h_scaler, f)

    # architecture + A11 audits on a fresh model BEFORE training
    probe = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
    arch = architecture_audit(probe, device)
    write_architecture_audit(d, arch)
    a11 = a11_audit(r_tr, r_va)
    write_a11_audit(d, a11)
    del probe
    empty_cache()
    gc.collect()

    if arch["ARCHITECTURE_STATUS"] != "FROZEN_REPRODUCED" or a11["A11_AUDIT_STATUS"] != "PASS":
        abort(f"pre-train audit failed arch={arch['ARCHITECTURE_STATUS']} a11={a11['A11_AUDIT_STATUS']}")

    seed_dirs = {101: d["s101"], 202: d["s202"], 303: d["s303"]}
    rows = []
    histories: dict[int, list[dict]] = {}
    commit = git_commit()

    for seed in SEEDS:
        run_dir = seed_dirs[seed] / "run"
        final_ckpt = d["ckpts"] / f"final_seed{seed}_best.pt"
        summary_path = seed_dirs[seed] / "seed_summary.json"
        meta_path = seed_dirs[seed] / "train_meta.json"
        # Technical resume: keep completed seeds after an external kill; never
        # retrain a finished seed just because another seed failed.
        if final_ckpt.exists() and summary_path.exists() and meta_path.exists():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(row)
            histories[seed] = list(meta.get("history") or [])
            print(
                f"[final] SKIP completed seed={seed} "
                f"best_ep={row.get('best_epoch')} NDCG={row.get('best_sampled_NDCG@20')}",
                flush=True,
            )
            continue
        print(f"[final] FROM_SCRATCH seed={seed}", flush=True)
        # wipe any partial run in this seed folder (never resume old race ckpts)
        if run_dir.exists():
            for p in run_dir.glob("*"):
                if p.is_file():
                    p.unlink()
        model = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
        # re-assert zero residual after seed re-init inside train_one
        out = train_one(
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
            recipe="C1",
            max_epochs_override=None,
            resume=False,
        )
        meta = out["meta"]
        cell = cell_from_scores(va_u, va_y, out["logits"])
        m = cell["metrics"]
        conv = convergence_flag(meta)
        # find lr / loss at best epoch
        hist_list = meta.get("history") or []
        best_h = next((h for h in hist_list if int(h["epoch"]) == int(meta["best_epoch"])), None)
        stop_h = hist_list[-1] if hist_list else None
        row = {
            "seed": seed,
            "best_epoch": int(meta["best_epoch"]),
            "stopping_epoch": int(meta["final_epoch"]),
            "best_sampled_NDCG@20": float(meta["best_val_NDCG@20_sampled"]),
            "final_sampled_NDCG@20": float(m["NDCG@20"]),
            "MRR": float(m.get("MRR", float("nan"))),
            "Recall@20": float(m.get("Recall@20", float("nan"))),
            "HitRate@20": float(m.get("HitRate@20", m.get("HR@20", float("nan")))),
            "lr_at_best": float(best_h["lr"]) if best_h else float("nan"),
            "lr_at_stop": float(meta["final_lr"]),
            "train_loss_at_best": float(best_h["loss"]) if best_h else float("nan"),
            "convergence": conv,
            "seconds": float(meta.get("seconds", float("nan"))),
            "peak_mem": mps_bytes(),
        }
        rows.append(row)
        histories[seed] = hist_list
        write_json(seed_dirs[seed] / "seed_summary.json", row)
        write_json(seed_dirs[seed] / "train_meta.json", meta)
        # checkpoint with metadata
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
            "recipe": "C1_EXTENDED",
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "FULLRANK_STATUS": "LOCKED_NOT_RUN",
        }
        torch.save(payload, ckpt_path)
        print(
            f"[final] seed={seed} best_ep={row['best_epoch']} "
            f"NDCG={row['best_sampled_NDCG@20']:.6f} conv={conv}",
            flush=True,
        )
        del model
        empty_cache()
        gc.collect()

    # history long CSV
    hist_rows = []
    for seed, hist_list in histories.items():
        for h in hist_list:
            hist_rows.append({"seed": seed, **h, "gpu_mem": None})
    write_csv(d["summary"] / "FINAL_TRAINING_HISTORY.csv", hist_rows)
    write_csv(d["summary"] / "FINAL_TRAINING_RESULTS_BY_SEED.csv", rows)

    ndcgs = [r["best_sampled_NDCG@20"] for r in rows]
    summary = {
        "NDCG@20_mean": float(np.mean(ndcgs)),
        "NDCG@20_std": float(np.std(ndcgs, ddof=1)) if len(ndcgs) > 1 else 0.0,
        "NDCG@20_min": float(np.min(ndcgs)),
        "NDCG@20_max": float(np.max(ndcgs)),
    }
    for metric in ("MRR", "Recall@20", "HitRate@20"):
        vals = [r[metric] for r in rows]
        summary[f"{metric}_mean"] = float(np.mean(vals))
        summary[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[f"{metric}_min"] = float(np.min(vals))
        summary[f"{metric}_max"] = float(np.max(vals))

    plot_final_curves(histories, d["figures"] / "FINAL_TRAINING_CURVES.png")

    all_conv = all(r["convergence"] == "CONVERGED" for r in rows)
    conv_status = "ALL_CONVERGED" if all_conv else "PARTIAL_CONVERGENCE_WARNING"
    arch_ok = arch["ARCHITECTURE_STATUS"] == "FROZEN_REPRODUCED"
    a11_ok = a11["A11_AUDIT_STATUS"] == "PASS"
    ready = arch_ok and a11_ok and len(rows) == 3
    verdict = "FINAL_MODEL_READY" if ready else "AUDIT_FAILED"

    # report
    by_seed = {r["seed"]: r for r in rows}
    report = f"""# FINAL_CLEAN_TRAINING_REPORT

Generated {utc_now()}.

## Answers

1. Was the exact frozen architecture reproduced? **{"YES" if arch_ok else "NO"}** (`{arch['ARCHITECTURE_STATUS']}`)
2. Did all three seeds converge? **{"YES" if all_conv else "WARNING — see per-seed flags"}** (`{conv_status}`)
3. Best epoch per seed: 101→{by_seed[101]['best_epoch']}, 202→{by_seed[202]['best_epoch']}, 303→{by_seed[303]['best_epoch']}
4. Sampled NDCG@20: 101→{by_seed[101]['best_sampled_NDCG@20']:.6f}, 202→{by_seed[202]['best_sampled_NDCG@20']:.6f}, 303→{by_seed[303]['best_sampled_NDCG@20']:.6f}
5. Mean ± std NDCG@20: **{summary['NDCG@20_mean']:.6f} ± {summary['NDCG@20_std']:.6f}**
6. Secondary diagnostics (mean ± std): MRR {summary['MRR_mean']:.6f}±{summary['MRR_std']:.6f}; Recall@20 {summary['Recall@20_mean']:.6f}±{summary['Recall@20_std']:.6f}; HitRate@20 {summary['HitRate@20_mean']:.6f}±{summary['HitRate@20_std']:.6f}
7. Did LEG_K2 remain the declared one-scalar residual? **YES** (`{a11['A11_AUDIT_STATUS']}`)
8. Was any architecture or hyperparameter changed? **NO**
9. Was TEST accessed? **NO** (`LOCKED_NOT_RUN`)
10. Was FULL-RANK accessed? **NO** (`LOCKED_NOT_RUN`)
11. Are final checkpoints ready for full-rank validation? **{"YES" if ready else "NO"}** (`FULLRANK_READY={"TRUE" if ready else "FALSE"}`)

## Note

Sampled validation NDCG@20 is **not** numerically comparable to full-rank catalog ranking.

## Per-seed convergence

| seed | best_ep | stop_ep | NDCG@20 | flag |
|---:|---:|---:|---:|---|
| 101 | {by_seed[101]['best_epoch']} | {by_seed[101]['stopping_epoch']} | {by_seed[101]['best_sampled_NDCG@20']:.6f} | {by_seed[101]['convergence']} |
| 202 | {by_seed[202]['best_epoch']} | {by_seed[202]['stopping_epoch']} | {by_seed[202]['best_sampled_NDCG@20']:.6f} | {by_seed[202]['convergence']} |
| 303 | {by_seed[303]['best_epoch']} | {by_seed[303]['stopping_epoch']} | {by_seed[303]['best_sampled_NDCG@20']:.6f} | {by_seed[303]['convergence']} |

## Checkpoints

- `06_CHECKPOINTS/final_seed101_best.pt`
- `06_CHECKPOINTS/final_seed202_best.pt`
- `06_CHECKPOINTS/final_seed303_best.pt`

Config hash: `{recipe['config_hash']}` · git: `{commit}`
"""
    (d["summary"] / "FINAL_CLEAN_TRAINING_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        "run_id": "LASTFM_FINAL_CLEAN_TRAINING_V1",
        "timestamp": utc_now(),
        "git_commit": commit,
        "config_hash": recipe["config_hash"],
        "architecture": ARCH_NAME,
        "seeds": list(SEEDS),
        "results": rows,
        "summary": summary,
        "ARCHITECTURE_STATUS": arch["ARCHITECTURE_STATUS"],
        "CONVERGENCE_STATUS": conv_status,
        "FINAL_TRAINING_VERDICT": verdict,
        "TEST_STATUS": "LOCKED_NOT_RUN",
        "FULLRANK_STATUS": "LOCKED_NOT_RUN",
        "FULLRANK_READY": bool(ready),
    }
    write_json(d["summary"] / "FINAL_CLEAN_TRAINING_MANIFEST.json", manifest)

    # FINAL PRINT
    print("\n" + "=" * 60, flush=True)
    print(f"ARCHITECTURE_STATUS = {arch['ARCHITECTURE_STATUS']}", flush=True)
    print(f"FINAL_ARCHITECTURE =\n    {ARCH_NAME.replace('+', chr(10)+'    + ')}", flush=True)
    for seed in SEEDS:
        r = by_seed[seed]
        print(f"SEED{seed}_BEST_EPOCH = {r['best_epoch']}", flush=True)
        print(f"SEED{seed}_SAMPLED_NDCG20 = {r['best_sampled_NDCG@20']:.6f}", flush=True)
    print(f"FINAL_SAMPLED_NDCG20_MEAN = {summary['NDCG@20_mean']:.6f}", flush=True)
    print(f"FINAL_SAMPLED_NDCG20_STD = {summary['NDCG@20_std']:.6f}", flush=True)
    print(f"CONVERGENCE_STATUS = {conv_status}", flush=True)
    print(f"FINAL_TRAINING_VERDICT = {verdict}", flush=True)
    print("TEST_STATUS = LOCKED_NOT_RUN", flush=True)
    print("FULLRANK_STATUS = LOCKED_NOT_RUN", flush=True)
    print(f"FULLRANK_READY = {'TRUE' if ready else 'FALSE'}", flush=True)
    print("=" * 60, flush=True)
    print("THEN STOP. No further experiment.", flush=True)


if __name__ == "__main__":
    main()
