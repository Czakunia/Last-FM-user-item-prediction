#!/usr/bin/env python3
"""A11_FUNCTIONAL_DISTRIBUTION_BENCHMARK_V2.

Frozen-B0 residual benchmark of functions of the existing signed-Top25 A11
distribution. No joint HGT, no alternate routing, no attention over embeddings.
"""

from __future__ import annotations

import gc
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
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

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
    THETA0,
    assert_zero,
    cell_from_scores,
    hist_bucket,
    mag,
    mask_from_n,
    train_frozen_branch,
)
from scripts.run_artist_a11_residual_branch_v1 import (  # noqa: E402
    bootstrap_ci,
    per_user_ndcg,
    sampled_metrics,
    scale_split,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    abort,
    check_overlaps,
    git_commit,
    sha256_file,
    verify_fingerprints,
    write_csv,
    write_json,
)
from scripts.run_race_clean_3 import shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = PROTOCOL_CONFIG
RACE = materialized_root()
V1 = RACE / "A11_DISTRIBUTIONAL_REPRESENTATION_SERIES_V1"
B0_H = RACE / "race" / "a11_top25" / "features"
OUT = RACE / "A11_FUNCTIONAL_DISTRIBUTION_BENCHMARK_V2"
SEEDS = (101, 202, 303)
K_HIST = 25
N_BOOT = 20_000
N_PERM = 20_000
CONFIRM = 0.001
CENTERS = np.array([-1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)
# Predeclared: median adjacent spacing of the fixed centers.
SIGMA = float(np.median(np.diff(CENTERS)))  # 0.25


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    for sub in (
        "audit", "representations", "results", "bootstrap", "randomization",
        "correlations", "plots", "report", "runs", "cache",
    ):
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def masked_mean(val: np.ndarray, mask: np.ndarray, n: np.ndarray) -> np.ndarray:
    s = (val * mask).sum(axis=1)
    out = s / np.maximum(n.astype(np.float64), 1.0)
    out[n <= 0] = 0.0
    return out


def legendre_stack(x: np.ndarray, k_max: int) -> np.ndarray:
    """P1..Pk_max, shape [N, 25, k_max]. P1=x."""
    p1 = x
    p0 = np.ones_like(x)
    cols = [p1]
    if k_max >= 2:
        prev, cur = p0, p1
        for n in range(1, k_max):
            nxt = ((2 * n + 1) * x * cur - n * prev) / (n + 1)
            cols.append(nxt)
            prev, cur = cur, nxt
    return np.stack(cols, axis=-1)


def chebyshev_stack(x: np.ndarray, k_max: int) -> np.ndarray:
    xc = np.clip(x, -1.0, 1.0)
    cols = []
    for k in range(1, k_max + 1):
        cols.append(np.cos(k * np.arccos(xc)))
    return np.stack(cols, axis=-1)


def feats_orthogonal(a11: np.ndarray, n: np.ndarray, k_max: int, kind: str) -> np.ndarray:
    mask = mask_from_n(n).astype(np.float64)
    x = a11.astype(np.float64)
    stack = legendre_stack(x, k_max) if kind == "leg" else chebyshev_stack(x, k_max)
    # drop k=1 (mean-equivalent); keep L2..LK / C2..CK → dim k_max-1
    out = np.stack(
        [masked_mean(stack[:, :, j], mask, n) for j in range(1, k_max)],
        axis=1,
    ).astype(np.float32)
    return out


def feats_quantile(a11: np.ndarray, n: np.ndarray) -> np.ndarray:
    m = mask_from_n(n)
    x = np.where(m, a11.astype(np.float64), np.nan)
    with np.errstate(all="ignore"):
        qs = np.nanpercentile(x, [5, 10, 25, 50, 75, 90, 95], axis=1).T
        std = np.nanstd(x, axis=1, ddof=0)
    iqr = qs[:, 4] - qs[:, 2]
    out = np.column_stack([qs, std, iqr]).astype(np.float32)
    out[n <= 0] = 0
    n1 = n == 1
    if n1.any():
        v = a11[n1, 0]
        out[n1, :7] = v[:, None]
        out[n1, 7:] = 0
    return np.nan_to_num(out, nan=0.0)


def feats_kernel(a11: np.ndarray, n: np.ndarray, sigma: float) -> np.ndarray:
    mask = mask_from_n(n).astype(np.float64)
    x = a11.astype(np.float64)[..., None]
    phi = np.exp(-((x - CENTERS) ** 2) / (2.0 * sigma * sigma))
    out = np.stack([masked_mean(phi[:, :, j], mask, n) for j in range(len(CENTERS))], axis=1)
    return out.astype(np.float32)


class Temp4Branch(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.theta_plus = nn.Parameter(torch.tensor([THETA0], dtype=torch.float32))
        self.theta_minus = nn.Parameter(torch.tensor([THETA0], dtype=torch.float32))
        self.mlp = ResMLP(4)
        self.register_buffer("mu", torch.tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(np.maximum(np.asarray(std), 1e-6), dtype=torch.float32))

    def pooled(self, x: Tensor, mask: Tensor) -> Tensor:
        tau_p = torch.nn.functional.softplus(self.theta_plus)
        tau_m = torch.nn.functional.softplus(self.theta_minus)
        mf = mask.float()
        wp = torch.softmax((tau_p * x).masked_fill(~mask, -1e9), dim=1) * mf
        wm = torch.softmax((-tau_m * x).masked_fill(~mask, -1e9), dim=1) * mf
        empty = ~mask.any(dim=1)
        sp = (wp * x).sum(1)
        sm = (wm * x).sum(1)
        hp = -(wp.clamp_min(1e-12).log() * wp).sum(1)
        hm = -(wm.clamp_min(1e-12).log() * wm).sum(1)
        r = torch.stack([sp, sm, hp, hm], dim=1)
        r = torch.where(empty.unsqueeze(1), torch.zeros_like(r), r)
        return (r - self.mu) / self.sd

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.mlp(self.pooled(x, mask))

    def taus(self) -> tuple[float, float]:
        with torch.no_grad():
            return (
                float(torch.nn.functional.softplus(self.theta_plus)),
                float(torch.nn.functional.softplus(self.theta_minus)),
            )


def tau1_temp4(a11: np.ndarray, mask: np.ndarray, device, bs: int = 8192) -> np.ndarray:
    dummy = Temp4Branch(np.zeros(4), np.ones(4)).to(device)
    with torch.no_grad():
        dummy.mu.zero_()
        dummy.sd.fill_(1.0)
    outs = []
    with torch.no_grad():
        for start in range(0, len(a11), bs):
            sl = slice(start, start + bs)
            outs.append(
                dummy.pooled(
                    torch.from_numpy(a11[sl]).to(device),
                    torch.from_numpy(mask[sl]).to(device),
                )
                .cpu()
                .numpy()
            )
    del dummy
    return np.concatenate(outs)


def signflip_p(delta: np.ndarray, n: int = N_PERM, seed: int = 20260816) -> float:
    """Two-sided paired sign-flip p-value for mean(delta)."""
    rng = np.random.default_rng(seed)
    obs = float(np.abs(delta.mean()))
    hits = 0
    bs = 256
    for start in range(0, n, bs):
        b = min(bs, n - start)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(b, delta.size))
        null = np.abs((signs * delta).mean(axis=1))
        hits += int((null >= obs - 1e-18).sum())
    return (hits + 1) / (n + 1)


def holm(pvals: list[tuple[str, float]]) -> list[dict[str, Any]]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        raw = (m - rank) * pvals[i][1]
        running = max(running, raw)
        adj[i] = min(1.0, running)
    return [
        {"variant": name, "raw_p": p, "holm_p": adj[i], "rank": order.index(i) + 1}
        for i, (name, p) in enumerate(pvals)
    ]


def contrast_users(cells0, cells1) -> np.ndarray:
    arrs = []
    for i in range(len(SEEDS)):
        common = sorted(set(cells0[i]["pu"]) & set(cells1[i]["pu"]))
        arrs.append(np.asarray([cells1[i]["pu"][u] - cells0[i]["pu"][u] for u in common], dtype=np.float64))
    return np.concatenate(arrs)


VARIANTS: list[dict[str, Any]] = (
    [{"code": f"LEG_K{k}", "family": "LEGENDRE", "k": k, "dim": k - 1, "learnable": False} for k in (2, 4, 6, 8)]
    + [{"code": f"CHEB_K{k}", "family": "CHEBYSHEV", "k": k, "dim": k - 1, "learnable": False} for k in (2, 4, 6, 8)]
    + [{"code": "QUANTILE_FUNC", "family": "QUANTILE", "k": None, "dim": 9, "learnable": False}]
    + [{"code": "KERNEL9", "family": "KERNEL", "k": None, "dim": 9, "learnable": False}]
    + [{"code": "TEMP4", "family": "TEMP", "k": None, "dim": 4, "learnable": True}]
)


def write_overlap() -> None:
    text = """# Overlap with A11_DISTRIBUTIONAL_REPRESENTATION_SERIES_V1

V2 is a **frozen-B0 functional benchmark**. It does **not** rerun:

- joint HGT retraining / continuation training
- alternate routing (abs / dual-tail)
- A11-only learned attention (R4)
- combination of families

Reuse (identical objects, not identical protocol):

| V1 | V2 | action |
|---|---|---|
| signed Top25 A11 caches | same empirical object | **reuse files** |
| frozen B0 train/val logits | F0 | **reuse files** |
| R1 LEGENDRE3 = [L2,L3,L4] | LEG_K4 | same features; **retrain** under V2 protocol (20k boot, Holm, identical decoder) |
| R2 QUANTILE7 | QUANTILE_FUNC | V2 adds q05,q95; **retrain** |
| R3 TEMP2 = [S+,S-] | TEMP4 = [S+,S-,H+,H-] | extra entropies; **retrain** |

New in V2: LEG_K2/K6/K8 order curve, Chebyshev family, RBF kernel mean, TEMP entropies,
20k bootstrap, sign-flip randomization, Holm correction, Spearman correlations.

V1 joint +0.007 vs original B0 is **out of scope** (continuation training of HGT).
V2 only measures incremental information of functions of A11 on frozen B0.
"""
    (OUT / "audit" / "OVERLAP_WITH_V1.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_dirs()
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n")
    (OUT / "FULLRANK_LOCKED").write_text("LOCKED_NOT_RUN\n")
    write_overlap()
    print("[func-v2] load", flush=True)
    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    check_overlaps()
    va_u, va_y = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"]
    tr_y = bundle["train_pairs"]["label"].astype(np.float32)
    mt = bundle["model_train"]
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))

    tr_path = V1 / "01_FIXED_ROUTING" / "cache" / "top25_train_signed.npz"
    va_path = V1 / "00_AUDIT" / "cache" / "top25_val_signed.npz"
    if not tr_path.exists() or not va_path.exists():
        abort("missing V1 signed Top25 caches — V1 must finish materialize first")
    top_tr = np.load(tr_path)
    top_va = np.load(va_path)
    a11_tr, n_tr = top_tr["a11"], top_tr["n"]
    a11_va, n_va = top_va["a11"], top_va["n"]
    mask_tr, mask_va = mask_from_n(n_tr), mask_from_n(n_va)
    write_json(
        OUT / "audit" / "A11_OBJECT.json",
        {
            "routing": "SIGNED_TOP25_UNCHANGED",
            "sigma_kernel": SIGMA,
            "centers": CENTERS.tolist(),
            "sigma_rule": "median adjacent spacing of fixed centers",
            "n_train": int(len(n_tr)),
            "n_val": int(len(n_va)),
            "top25_train": str(tr_path),
            "top25_val": str(va_path),
        },
    )

    print("[func-v2] features", flush=True)
    feat_tr: dict[str, np.ndarray] = {}
    feat_va: dict[str, np.ndarray] = {}
    scalers: dict[str, StandardScaler] = {}
    for spec in VARIANTS:
        code = spec["code"]
        if spec["family"] == "LEGENDRE":
            raw_tr, raw_va = feats_orthogonal(a11_tr, n_tr, spec["k"], "leg"), feats_orthogonal(a11_va, n_va, spec["k"], "leg")
        elif spec["family"] == "CHEBYSHEV":
            raw_tr, raw_va = feats_orthogonal(a11_tr, n_tr, spec["k"], "cheb"), feats_orthogonal(a11_va, n_va, spec["k"], "cheb")
        elif spec["family"] == "QUANTILE":
            raw_tr, raw_va = feats_quantile(a11_tr, n_tr), feats_quantile(a11_va, n_va)
        elif spec["family"] == "KERNEL":
            raw_tr, raw_va = feats_kernel(a11_tr, n_tr, SIGMA), feats_kernel(a11_va, n_va, SIGMA)
        else:
            continue
        sc = StandardScaler().fit(raw_tr)
        scalers[code] = sc
        feat_tr[code], feat_va[code] = scale_split(sc, raw_tr), scale_split(sc, raw_va)
        np.save(OUT / "representations" / f"{code}_train.npy", feat_tr[code])
        np.save(OUT / "representations" / f"{code}_val.npy", feat_va[code])
        with (OUT / "representations" / f"{code}_scaler.pkl").open("wb") as f:
            pickle.dump(sc, f)

    init_temp = tau1_temp4(a11_tr, mask_tr, device)
    sc_t = StandardScaler().fit(init_temp)
    scalers["TEMP4"] = sc_t
    write_json(
        OUT / "audit" / "TEMP4_SCALER.json",
        {
            "strategy": "StandardScaler on [S+,S-,H+,H-] at tau=1 (init), MODEL_TRAIN only. Internal attention weights not pre-standardized.",
            "mean": sc_t.mean_.tolist(),
            "scale": sc_t.scale_.tolist(),
            "init_theta": THETA0,
            "init_tau": 1.0,
        },
    )
    with (OUT / "representations" / "TEMP4_scaler.pkl").open("wb") as f:
        pickle.dump(sc_t, f)

    # correlations vs H3 on val (train-only scaler already fit; corr on raw unscaled val is fine / diagnostic)
    H_va = np.load(B0_H / "X_val.npy").astype(np.float32)
    raw_blocks = {
        "H_mean": H_va[:, 0],
        "H_max": H_va[:, 1],
        "H_top3": H_va[:, 2],
        "L2": feats_orthogonal(a11_va, n_va, 4, "leg")[:, 0],
        "L3": feats_orthogonal(a11_va, n_va, 4, "leg")[:, 1],
        "L4": feats_orthogonal(a11_va, n_va, 4, "leg")[:, 2],
        "C2": feats_orthogonal(a11_va, n_va, 4, "cheb")[:, 0],
        "C3": feats_orthogonal(a11_va, n_va, 4, "cheb")[:, 1],
        "q10": feats_quantile(a11_va, n_va)[:, 1],
        "q50": feats_quantile(a11_va, n_va)[:, 3],
        "q90": feats_quantile(a11_va, n_va)[:, 5],
        "K_c0": feats_kernel(a11_va, n_va, SIGMA)[:, 4],
        "K_cneg1": feats_kernel(a11_va, n_va, SIGMA)[:, 0],
        "K_cpos1": feats_kernel(a11_va, n_va, SIGMA)[:, 8],
        "S_plus_init": tau1_temp4(a11_va, mask_va, device)[:, 0],
        "S_minus_init": tau1_temp4(a11_va, mask_va, device)[:, 1],
    }
    names_c = list(raw_blocks)
    mat = np.column_stack([raw_blocks[k] for k in names_c])
    with np.errstate(invalid="ignore"):
        P = np.corrcoef(mat.T)
        S, _ = spearmanr(mat)
    corr_rows = []
    for i, a in enumerate(names_c):
        for j, b in enumerate(names_c):
            if j <= i:
                continue
            corr_rows.append(
                {"i": a, "j": b, "pearson": float(P[i, j]), "spearman": float(S[i, j])}
            )
    write_csv(OUT / "correlations" / "FUNCTIONAL_CORRELATIONS.csv", corr_rows, ["i", "j", "pearson", "spearman"])
    write_csv(OUT / "FUNCTIONAL_CORRELATIONS.csv", corr_rows, ["i", "j", "pearson", "spearman"])

    b0_cells = []
    b0_tr = {}
    b0_va = {}
    for seed in SEEDS:
        p_tr = V1 / "01_FIXED_ROUTING" / "cache" / f"b0_train_logits_seed{seed}.npy"
        p_va = V1 / "01_FIXED_ROUTING" / "cache" / f"b0_val_logits_seed{seed}.npy"
        if not p_tr.exists() or not p_va.exists():
            abort(f"missing frozen B0 logits for seed {seed}")
        b0_tr[seed] = np.load(p_tr).astype(np.float64)
        b0_va[seed] = np.load(p_va).astype(np.float64)
        b0_cells.append(cell_from_scores(va_u, va_y, b0_va[seed]))
        print(f"[func-v2] F0 seed={seed} NDCG={b0_cells[-1]['metrics']['NDCG@20']:.6f}", flush=True)

    cells: dict[str, list] = {"F0": b0_cells}
    tau_rows = []

    for spec in VARIANTS:
        code = spec["code"]
        cells[code] = []
        for seed in SEEDS:
            ckpt = OUT / "runs" / f"{code}_seed{seed}"
            if spec["learnable"]:
                branch = Temp4Branch(sc_t.mean_, sc_t.scale_).to(device)
                ftr, fva = a11_tr, a11_va
                learnable = True
            else:
                branch = ResMLP(spec["dim"]).to(device)
                ftr, fva = feat_tr[code], feat_va[code]
                learnable = False
            if not (ckpt / "branch.pt").exists():
                assert_zero(branch, fva, mask_va, learnable, device, f"{code} s{seed}")
            scores = train_frozen_branch(
                branch=branch,
                b0_tr=b0_tr[seed],
                b0_va=b0_va[seed],
                y_tr=tr_y,
                va_u=va_u,
                va_y=va_y,
                feat_tr=ftr,
                feat_va=fva,
                mask_tr=mask_tr,
                mask_va=mask_va,
                learnable=learnable,
                seed=seed,
                ckpt=ckpt,
                acfg=acfg,
                device=device,
            )
            cell = cell_from_scores(va_u, va_y, scores)
            cells[code].append(cell)
            if spec["learnable"]:
                tp, tm = branch.taus()
                tau_rows.append({"variant": code, "seed": seed, "tau_plus": tp, "tau_minus": tm})
                write_json(ckpt / "learned_tau.json", {"tau_plus": tp, "tau_minus": tm})
            print(
                f"[func-v2] {code} seed={seed} NDCG={cell['metrics']['NDCG@20']:.6f} "
                f"R20={cell['metrics'].get('Recall@20', float('nan')):.6f} "
                f"HR={cell['metrics'].get('HitRate@20', float('nan')):.6f} "
                f"MRR={cell['metrics'].get('MRR', float('nan')):.6f}",
                flush=True,
            )
            del branch
            empty_cache()

    if tau_rows:
        write_csv(OUT / "results" / "TEMP4_LEARNED_TAU.csv", tau_rows, ["variant", "seed", "tau_plus", "tau_minus"])

    # by-seed table
    codes = ["F0"] + [s["code"] for s in VARIANTS]
    seed_rows = []
    for i, seed in enumerate(SEEDS):
        row: dict[str, Any] = {"seed": seed}
        for c in codes:
            m = cells[c][i]["metrics"]
            row[f"{c}_NDCG20"] = m["NDCG@20"]
            row[f"{c}_Recall20"] = m.get("Recall@20", float("nan"))
            row[f"{c}_HitRate20"] = m.get("HitRate@20", float("nan"))
            row[f"{c}_MRR"] = m.get("MRR", float("nan"))
        seed_rows.append(row)
    mean_ndcg = {c: float(np.mean([cells[c][i]["metrics"]["NDCG@20"] for i in range(3)])) for c in codes}
    fields = ["seed"] + [f"{c}_{k}" for c in codes for k in ("NDCG20", "Recall20", "HitRate20", "MRR")]
    write_csv(OUT / "FUNCTIONAL_RESULTS_BY_SEED.csv", seed_rows, fields)
    write_csv(OUT / "results" / "FUNCTIONAL_RESULTS_BY_SEED.csv", seed_rows, fields)

    boot_rows = []
    rand_rows = []
    holm_in = []
    hist_rows = []
    for spec in VARIANTS:
        code = spec["code"]
        seed_d = [cells[code][i]["metrics"]["NDCG@20"] - cells["F0"][i]["metrics"]["NDCG@20"] for i in range(3)]
        cat = contrast_users(cells["F0"], cells[code])
        mu, lo, hi = bootstrap_ci(cat, n=N_BOOT)
        p_sf = signflip_p(cat, n=N_PERM)
        holm_in.append((code, p_sf))
        boot_rows.append(
            {
                "variant": code,
                "family": spec["family"],
                "dim": spec["dim"],
                "mean_seed_delta": float(np.mean(seed_d)),
                "n_seeds_pos": int(sum(1 for x in seed_d if x > 0)),
                "seed101": seed_d[0],
                "seed202": seed_d[1],
                "seed303": seed_d[2],
                "mean": mu,
                "median": float(np.median(cat)),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "frac_gt0": float((cat > 0).mean()),
                "frac_eq0": float((cat == 0).mean()),
                "frac_lt0": float((cat < 0).mean()),
                "ci_entirely_gt0": bool(lo > 0),
                "practical_class": mag(float(np.mean(seed_d))),
                "mean_ndcg20": mean_ndcg[code],
            }
        )
        rand_rows.append({"variant": code, "raw_p": p_sf, "n_perm": N_PERM, "n_users_concat": int(cat.size)})
        for hb in ("1-5", "6-10", "11-25", "26-50", ">50"):
            a0, a1 = [], []
            for i in range(3):
                for u, v0 in cells["F0"][i]["pu"].items():
                    if hist_bucket(len(mt.get(int(u), ()))) != hb:
                        continue
                    if u in cells[code][i]["pu"]:
                        a0.append(v0)
                        a1.append(cells[code][i]["pu"][u])
            dlt = float(np.mean(np.asarray(a1) - np.asarray(a0))) if a0 else float("nan")
            hist_rows.append({"variant": code, "hist": hb, "delta": dlt, "n": len(a0)})

    holm_rows = holm(holm_in)
    holm_map = {r["variant"]: r for r in holm_rows}
    for r in boot_rows:
        r["raw_p"] = holm_map[r["variant"]]["raw_p"]
        r["holm_p"] = holm_map[r["variant"]]["holm_p"]
    for r in rand_rows:
        r["holm_p"] = holm_map[r["variant"]]["holm_p"]
        r["rank"] = holm_map[r["variant"]]["rank"]

    bfields = [
        "variant", "family", "dim", "mean_seed_delta", "n_seeds_pos",
        "seed101", "seed202", "seed303", "mean", "median", "ci95_lo", "ci95_hi",
        "frac_gt0", "frac_eq0", "frac_lt0", "ci_entirely_gt0", "practical_class",
        "mean_ndcg20", "raw_p", "holm_p",
    ]
    write_csv(OUT / "FUNCTIONAL_BOOTSTRAP.csv", boot_rows, bfields)
    write_csv(OUT / "bootstrap" / "FUNCTIONAL_BOOTSTRAP.csv", boot_rows, bfields)
    write_csv(OUT / "FUNCTIONAL_RANDOMIZATION.csv", rand_rows, ["variant", "raw_p", "holm_p", "rank", "n_perm", "n_users_concat"])
    write_csv(OUT / "randomization" / "FUNCTIONAL_RANDOMIZATION.csv", rand_rows, ["variant", "raw_p", "holm_p", "rank", "n_perm", "n_users_concat"])
    write_csv(OUT / "FUNCTIONAL_HOLM_CORRECTION.csv", holm_rows, ["variant", "raw_p", "holm_p", "rank"])
    write_csv(OUT / "FUNCTIONAL_HISTORY_LENGTH.csv", hist_rows, ["variant", "hist", "delta", "n"])
    write_csv(OUT / "results" / "FUNCTIONAL_HISTORY_LENGTH.csv", hist_rows, ["variant", "hist", "delta", "n"])

    order_rows = []
    for fam, prefix in (("LEGENDRE", "LEG_K"), ("CHEBYSHEV", "CHEB_K")):
        prev = None
        for k in (2, 4, 6, 8):
            code = f"{prefix}{k}"
            br = next(x for x in boot_rows if x["variant"] == code)
            incr = float("nan") if prev is None else br["mean_seed_delta"] - prev
            order_rows.append(
                {
                    "family": fam,
                    "K": k,
                    "dim": k - 1,
                    "mean_ndcg20": br["mean_ndcg20"],
                    "mean_delta": br["mean_seed_delta"],
                    "incremental_vs_prevK": incr,
                    "ci95_lo": br["ci95_lo"],
                    "ci95_hi": br["ci95_hi"],
                }
            )
            prev = br["mean_seed_delta"]
    write_csv(OUT / "FUNCTIONAL_ORDER_CURVES.csv", order_rows, list(order_rows[0]))
    write_csv(OUT / "results" / "FUNCTIONAL_ORDER_CURVES.csv", order_rows, list(order_rows[0]))

    # order plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam, marker in (("LEGENDRE", "o"), ("CHEBYSHEV", "s")):
        sub = [r for r in order_rows if r["family"] == fam]
        ax.plot([r["K"] for r in sub], [r["mean_delta"] for r in sub], marker=marker, label=fam)
    ax.axhline(0.001, color="gray", ls="--", lw=1, label="MATERIAL bar")
    ax.set_xlabel("polynomial order K")
    ax.set_ylabel("mean Δ NDCG@20 vs frozen B0")
    ax.set_title("Orthogonal order vs incremental ranking gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "order_curves.png", dpi=120)
    plt.close(fig)

    # reconstructions (diagnostic)
    rng_i = []
    for i, k in enumerate(n_va.tolist()):
        if int(k) >= 8:
            rng_i.append(i)
            if len(rng_i) >= 4000:
                break
    xs = np.linspace(-1, 1, 201)

    def pick(crit):
        best, bv = None, None
        for i in rng_i:
            x = a11_va[i, : int(n_va[i])].astype(np.float64)
            v = crit(x)
            if best is None or v > bv:
                best, bv = i, v
        return best

    picks = {
        "high_mean": pick(lambda x: float(x.mean())),
        "high_max_low_mean": pick(lambda x: float(x.max() - x.mean())),
        "near_zero": pick(lambda x: -abs(float(x.mean())) - float(x.std())),
    }
    pstack = legendre_stack(xs[None, :], 4)[0]  # [201, 4] P1..P4
    for name, idx in picks.items():
        xv = a11_va[idx, : int(n_va[idx])].astype(np.float64)
        L = [float(xv.mean())] + [float(legendre_stack(xv[None, :], 4)[0, :, j].mean()) for j in range(1, 4)]
        fhat = 0.5 * (1.0 + sum((2 * (kk + 1) + 1) * L[kk] * pstack[:, kk] for kk in range(4)))
        # kernel mean profile: sum_j K_j * phi_j(x) with K_j = mean_i phi_j(x_i)
        Kj = np.exp(-((xv[:, None] - CENTERS) ** 2) / (2 * SIGMA * SIGMA)).mean(0)
        kprof = (Kj * np.exp(-((xs[:, None] - CENTERS) ** 2) / (2 * SIGMA * SIGMA))).sum(1)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].hist(xv, bins=min(12, max(int(n_va[idx]), 3)), range=(-1, 1), density=True, color="#4c78a8")
        ax[0].plot(xs, fhat, "r--", lw=1.5, label="Legendre series M=4")
        ax[0].set_title(name)
        ax[0].legend(fontsize=8)
        ax[1].plot(xs, kprof, color="#f58518", label="kernel mean profile")
        ax[1].hist(xv, bins=min(12, max(int(n_va[idx]), 3)), range=(-1, 1), density=True, color="#4c78a8", alpha=0.35)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "plots" / f"recon_{name}.png", dpi=120)
        plt.close(fig)

    confirmed = []
    for r in boot_rows:
        if (
            r["mean_seed_delta"] >= CONFIRM
            and r["n_seeds_pos"] >= 2
            and r["ci_entirely_gt0"]
            and r["holm_p"] < 0.05
        ):
            confirmed.append(r)
    winner = "F0_H3"
    winner_row = None
    if confirmed:
        confirmed.sort(key=lambda r: (-r["mean_ndcg20"], r["dim"]))
        best = confirmed[0]
        for alt in confirmed[1:]:
            if abs(alt["mean_ndcg20"] - best["mean_ndcg20"]) < 0.0002 and alt["dim"] < best["dim"]:
                best = alt
        winner = best["variant"]
        winner_row = best
        verdict = "DISTRIBUTIONAL_SIGNAL_CONFIRMED"
    else:
        pos = [r for r in boot_rows if r["mean_seed_delta"] > 0 and r["n_seeds_pos"] >= 2]
        verdict = "DISTRIBUTIONAL_SIGNAL_WEAK" if pos else "H3_SUFFICIENT"

    def q_ans():
        by = {r["variant"]: r for r in boot_rows}
        best_any = max(boot_rows, key=lambda r: r["mean_seed_delta"])
        sat = ""
        leg = [r for r in order_rows if r["family"] == "LEGENDRE"]
        incs = [r["incremental_vs_prevK"] for r in leg if r["K"] > 2]
        if all(abs(x) < 0.0002 or (isinstance(x, float) and x < 0.0002) for x in incs if x == x):
            sat = "saturates after K2/K4"
        else:
            sat = "order still moves NDCG"
        cheb_best = max((r for r in boot_rows if r["family"] == "CHEBYSHEV"), key=lambda r: r["mean_seed_delta"])
        leg_best = max((r for r in boot_rows if r["family"] == "LEGENDRE"), key=lambda r: r["mean_seed_delta"])
        return best_any, sat, cheb_best, leg_best, by

    best_any, sat, cheb_best, leg_best, by = q_ans()
    wr = winner_row or best_any
    report = f"""# A11_FUNCTIONAL_DISTRIBUTION_REPORT

Frozen B0. Same signed Top25 A11 values for every family. No joint HGT, no routing change.

Kernel sigma (predeclared) = {SIGMA}

## Mean sampled NDCG@20

| variant | dim | seed101 | seed202 | seed303 | mean | Δ vs F0 | class | holm p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
"""
    for c in codes:
        if c == "F0":
            report += (
                f"| F0 | 3 | {cells['F0'][0]['metrics']['NDCG@20']:.6f} | "
                f"{cells['F0'][1]['metrics']['NDCG@20']:.6f} | {cells['F0'][2]['metrics']['NDCG@20']:.6f} | "
                f"{mean_ndcg['F0']:.6f} | 0 | ANCHOR | — |\n"
            )
            continue
        r = by[c]
        report += (
            f"| {c} | {r['dim']} | {cells[c][0]['metrics']['NDCG@20']:.6f} | "
            f"{cells[c][1]['metrics']['NDCG@20']:.6f} | {cells[c][2]['metrics']['NDCG@20']:.6f} | "
            f"{r['mean_ndcg20']:.6f} | {r['mean_seed_delta']:+.6f} | {r['practical_class']} | {r['holm_p']:.4g} |\n"
        )

    report += f"""
Confirmed (Δ≥0.001, ≥2/3 seeds, CI>0, Holm p<0.05): {[r['variant'] for r in confirmed]}

Q1. Distribution beyond H3? **{"yes" if verdict != "H3_SUFFICIENT" else "no measurable"}**.
Q2. Strongest family/variant: **{winner}**.
Q3. Low-order moments enough? Legendre best={leg_best['variant']} Δ={leg_best['mean_seed_delta']:+.6f}.
Q4. Saturate with K? {sat}.
Q5. Chebyshev vs Legendre near extremes: CHEB best {cheb_best['variant']} Δ={cheb_best['mean_seed_delta']:+.6f} vs LEG {leg_best['variant']} Δ={leg_best['mean_seed_delta']:+.6f}.
Q6. Kernel vs moments: KERNEL9 Δ={by['KERNEL9']['mean_seed_delta']:+.6f}.
Q7. TEMP4 Δ={by['TEMP4']['mean_seed_delta']:+.6f}. Learned tau in TEMP4_LEARNED_TAU.csv.
Q8. Seed robustness: see n_seeds_pos.
Q9. Holm: winner holm_p={wr['holm_p']:.4g}.
Q10. Smallest confirmed: {min(confirmed, key=lambda r: r['dim'])['variant'] if confirmed else "none"}.

A11_FUNCTIONAL_WINNER = {winner}
A11_FUNCTIONAL_VERDICT = {verdict}
BEST_MEAN_DELTA = {wr['mean_seed_delta']:+.6f}
BEST_BOOTSTRAP_CI = [{wr['ci95_lo']:.6f}, {wr['ci95_hi']:.6f}]
BEST_HOLM_P = {wr['holm_p']:.6g}
BEST_REPRESENTATION_DIM = {wr['dim']}
TEST_STATUS = LOCKED_NOT_RUN
FULLRANK_STATUS = LOCKED_NOT_RUN
"""
    (OUT / "A11_FUNCTIONAL_DISTRIBUTION_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "report" / "A11_FUNCTIONAL_DISTRIBUTION_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    write_json(
        OUT / "MANIFEST.json",
        {
            "timestamp": utc_now(),
            "git_commit": git_commit(),
            "A11_FUNCTIONAL_WINNER": winner,
            "A11_FUNCTIONAL_VERDICT": verdict,
            "BEST_MEAN_DELTA": wr["mean_seed_delta"],
            "BEST_BOOTSTRAP_CI": [wr["ci95_lo"], wr["ci95_hi"]],
            "BEST_HOLM_P": wr["holm_p"],
            "BEST_REPRESENTATION_DIM": wr["dim"],
            "sigma_kernel": SIGMA,
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "FULLRANK_STATUS": "LOCKED_NOT_RUN",
            "reused_v1_top25": True,
            "reused_v1_b0_logits": True,
            "joint_not_run": True,
            "routing_not_changed": True,
            **{f"{k}_hash": v for k, v in hashes.items()},
        },
    )


if __name__ == "__main__":
    main()
