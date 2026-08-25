#!/usr/bin/env python3
"""A11_DISTRIBUTIONAL_REPRESENTATION_SERIES_V1.

Auxiliary residual representations of item-A11 on frozen (then joint) B0.
Does not change HGT, KG, A5, H3, loss, or negatives. Test/full-rank locked.
"""

from __future__ import annotations

import gc
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_paths_20260824 import PROTOCOL_CONFIG, materialized_root  # noqa: E402

from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_artist_a11_residual_branch_v1 import (  # noqa: E402
    b0_logits,
    bootstrap_ci,
    load_b0,
    per_user_ndcg,
    sampled_metrics,
    scale_split,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    ACT,
    abort,
    check_overlaps,
    git_commit,
    sha256_arr,
    sha256_file,
    verify_fingerprints,
    write_csv,
    write_json,
)
from scripts.run_publication_our_hgt_fullrank import build_fusion_head  # noqa: E402
from scripts.run_race_clean_3 import activity_bucket, shared_A_dir  # noqa: E402
from src.lastfm_lp.binary.binary_measures import a11_energy_from_n11_matrix  # noqa: E402
from src.lastfm_lp.binary.measure_race_selection import deterministic_topk_indices  # noqa: E402
from src.lastfm_lp.clean_v2.crossfit_index import clean_v2_index_for  # noqa: E402
from src.lastfm_lp.clean_v2.routing import routing_history_exclude_self  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import load_data_and_typed_graph, user_item_to_nodes  # noqa: E402
from src.lastfm_lp.evaluation.ranking_metrics import ranking_metrics_for_users  # noqa: E402
from src.lastfm_lp.models.encoders import HGTGraphEncoder  # noqa: E402
from src.lastfm_lp.models.fusion import RecommendationModel  # noqa: E402
from src.lastfm_lp.pipeline.features_hcr_v2 import ensure_cross_fit  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = PROTOCOL_CONFIG
RACE = materialized_root()
ROLE = RACE / "HGT_NODE_SEMANTIC_ROLE_FEATURES_V1"
B0_H = RACE / "race" / "a11_top25" / "features"
B0_CKPT = RACE / "race" / "a11_top25" / "checkpoints" / "RACE_CLEAN_3_a11_top25"
OUT = RACE / "A11_DISTRIBUTIONAL_REPRESENTATION_SERIES_V1"
SEEDS = (101, 202, 303)
K = 25
THETA0 = float(np.log(np.e - 1.0))
SCREEN = 0.0005
CONFIRM = 0.001
N_BINS = 2001


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stage_dir(name: str) -> Path:
    p = OUT / name
    for sub in ("audit", "reports", "runs", "cache"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def P2(x: np.ndarray) -> np.ndarray:
    return 0.5 * (3.0 * x * x - 1.0)


def P3(x: np.ndarray) -> np.ndarray:
    return 0.5 * (5.0 * x**3 - 3.0 * x)


def P4(x: np.ndarray) -> np.ndarray:
    return 0.125 * (35.0 * x**4 - 30.0 * x * x + 3.0)


def mag(x: float) -> str:
    ax = abs(x)
    if ax < 0.0005:
        return "MICRO"
    if ax < 0.001:
        return "SMALL"
    if ax < 0.002:
        return "MATERIAL"
    return "STRONG"


def hist_bucket(h: int) -> str:
    if h <= 5:
        return "1-5"
    if h <= 10:
        return "6-10"
    if h <= 25:
        return "11-25"
    if h <= 50:
        return "26-50"
    return ">50"


class StreamDist:
    """Running stats + histogram quantiles on [-1, 1]."""

    def __init__(self) -> None:
        self.n = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.sum = 0.0
        self.sumsq = 0.0
        self.n_pos = 0
        self.n_neg = 0
        self.n_zero = 0
        self.edges = np.linspace(-1.0, 1.0, N_BINS + 1)
        self.counts = np.zeros(N_BINS, dtype=np.int64)

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        self.n += int(x.size)
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))
        self.n_pos += int((x > 0).sum())
        self.n_neg += int((x < 0).sum())
        self.n_zero += int((x == 0).sum())
        self.sum += float(x.sum())
        self.sumsq += float((x * x).sum())
        idx = np.clip(np.digitize(x, self.edges) - 1, 0, N_BINS - 1)
        np.add.at(self.counts, idx, 1)

    def summary(self) -> dict[str, float]:
        if self.n <= 0:
            nan = float("nan")
            keys = (
                "n", "min", "q05", "q10", "q25", "median", "q75", "q90", "q95", "max",
                "mean", "std", "fraction_positive", "fraction_negative", "fraction_zero",
                "count_positive", "count_negative",
            )
            return {k: nan for k in keys}
        cdf = np.cumsum(self.counts) / max(self.n, 1)
        centres = 0.5 * (self.edges[:-1] + self.edges[1:])

        def q(p: float) -> float:
            i = int(np.searchsorted(cdf, p, side="left"))
            i = min(max(i, 0), N_BINS - 1)
            return float(centres[i])

        mean = self.sum / self.n
        var = max(self.sumsq / self.n - mean * mean, 0.0)
        return {
            "n": float(self.n),
            "min": float(self.min),
            "q05": q(0.05),
            "q10": q(0.10),
            "q25": q(0.25),
            "median": q(0.50),
            "q75": q(0.75),
            "q90": q(0.90),
            "q95": q(0.95),
            "max": float(self.max),
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "fraction_positive": self.n_pos / self.n,
            "fraction_negative": self.n_neg / self.n,
            "fraction_zero": self.n_zero / self.n,
            "count_positive": float(self.n_pos),
            "count_negative": float(self.n_neg),
        }


class ResMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, r: Tensor) -> Tensor:
        return self.net(r).squeeze(-1)


class TempPoolBranch(nn.Module):
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        super().__init__()
        self.theta_plus = nn.Parameter(torch.tensor([THETA0], dtype=torch.float32))
        self.theta_minus = nn.Parameter(torch.tensor([THETA0], dtype=torch.float32))
        self.mlp = ResMLP(2)
        if mean is not None:
            self.register_buffer("mu", torch.tensor(np.asarray(mean), dtype=torch.float32))
            self.register_buffer("sd", torch.tensor(np.maximum(np.asarray(std), 1e-6), dtype=torch.float32))
        else:
            self.mu = None
            self.sd = None

    def pooled(self, x: Tensor, mask: Tensor) -> Tensor:
        tau_p = torch.nn.functional.softplus(self.theta_plus)
        tau_m = torch.nn.functional.softplus(self.theta_minus)
        ep = (tau_p * x).masked_fill(~mask, -1e9)
        em = (-tau_m * x).masked_fill(~mask, -1e9)
        wp = torch.softmax(ep, dim=1) * mask.float()
        wm = torch.softmax(em, dim=1) * mask.float()
        empty = ~mask.any(dim=1)
        sp = (wp * x).sum(1)
        sm = (wm * x).sum(1)
        sp = torch.where(empty, torch.zeros_like(sp), sp)
        sm = torch.where(empty, torch.zeros_like(sm), sm)
        r = torch.stack([sp, sm], dim=1)
        if self.mu is not None and self.sd is not None:
            r = (r - self.mu) / self.sd
        return r

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.mlp(self.pooled(x, mask))


class A11AttnBranch(nn.Module):
    """A11-only set attention. No HGT / z_u / z_X / popularity."""

    def __init__(self) -> None:
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))
        self.mlp = ResMLP(3)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        phi = torch.stack([x, x * x, x**3], dim=-1)
        e = self.scorer(phi).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(e, dim=1) * mask.float()
        empty = ~mask.any(dim=1)
        p2 = 0.5 * (3 * x * x - 1)
        p3 = 0.5 * (5 * x**3 - 3 * x)
        r = torch.stack([(alpha * x).sum(1), (alpha * p2).sum(1), (alpha * p3).sum(1)], dim=1)
        r = torch.where(empty.unsqueeze(1), torch.zeros_like(r), r)
        return self.mlp(r)


class LegendreTempBranch(nn.Module):
    """Exploratory concat of LEGENDRE3 + TEMP2 (Exp 3 fallback)."""

    def __init__(self, l_mean, l_std, t_mean, t_std) -> None:
        super().__init__()
        self.temp = TempPoolBranch(t_mean, t_std)
        self.temp.mlp = nn.Identity()  # type: ignore[assignment]
        self.register_buffer("l_mu", torch.tensor(np.asarray(l_mean), dtype=torch.float32))
        self.register_buffer("l_sd", torch.tensor(np.maximum(np.asarray(l_std), 1e-6), dtype=torch.float32))
        self.mlp = ResMLP(5)

    def forward(self, x: Tensor, mask: Tensor, legendre: Tensor) -> Tensor:
        t = self.temp.pooled(x, mask)
        l = (legendre - self.l_mu) / self.l_sd
        return self.mlp(torch.cat([l, t], dim=1))


class JointModel(nn.Module):
    def __init__(self, rec: RecommendationModel, branch: nn.Module, kind: str) -> None:
        super().__init__()
        self.encoder = rec.encoder
        self.fusion_head = rec.fusion_head
        self.branch = branch
        self.kind = kind

    def forward(
        self,
        user_idx,
        item_idx,
        base_features,
        hcr_features=None,
        z=None,
        feat=None,
        mask=None,
        residual_enabled=True,
        **_,
    ):
        if z is None:
            z = self.encoder.encode_all()
        enc = self.encoder.pair_outputs(z, user_idx, item_idx)
        b0 = self.fusion_head(enc.graph_context, enc.graph_score, base_features, hcr_features)
        if self.kind in {"R3", "R4", "LT"}:
            d = self.branch(feat, mask) if self.kind != "LT" else self.branch(feat, mask, None)
        else:
            d = self.branch(feat)
        if not residual_enabled:
            d = torch.zeros_like(d)
        return {"logits": b0 + d, "delta": d}


def mask_from_n(n: np.ndarray) -> np.ndarray:
    return np.arange(K)[None, :] < n[:, None]


def features_legendre(a11: np.ndarray, n: np.ndarray) -> np.ndarray:
    m = mask_from_n(n)
    x = a11.astype(np.float64)
    nn = np.maximum(n.astype(np.float64), 1.0)
    out = np.stack(
        [((P2(x) * m).sum(1) / nn), ((P3(x) * m).sum(1) / nn), ((P4(x) * m).sum(1) / nn)],
        axis=1,
    ).astype(np.float32)
    out[n <= 0] = 0
    return out


def features_quantile(a11: np.ndarray, n: np.ndarray) -> np.ndarray:
    m = mask_from_n(n)
    x = np.where(m, a11.astype(np.float64), np.nan)
    with np.errstate(all="ignore"):
        qs = np.nanpercentile(x, [10, 25, 50, 75, 90], axis=1).T
        std = np.nanstd(x, axis=1, ddof=0)
    iqr = qs[:, 3] - qs[:, 1]
    out = np.column_stack([qs, std, iqr]).astype(np.float32)
    out[n <= 0] = 0
    n1 = n == 1
    if n1.any():
        v = a11[n1, 0]
        out[n1, :5] = v[:, None]
        out[n1, 5:] = 0
    return np.nan_to_num(out, nan=0.0)


def dual_tail_select(col: np.ndarray, hist: np.ndarray, k_side: int = 12) -> np.ndarray:
    pos = np.where(col > 0)[0]
    neg = np.where(col < 0)[0]
    parts = []
    if pos.size:
        order = np.lexsort((hist[pos], -col[pos]))
        parts.append(pos[order[: min(k_side, pos.size)]])
    if neg.size:
        order = np.lexsort((hist[neg], col[neg]))
        parts.append(neg[order[: min(k_side, neg.size)]])
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def materialize(
    bundle,
    pairs,
    path: Path,
    *,
    routing: str,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    n = len(pairs["label"])
    if path.exists():
        z = np.load(path)
        if int(z["a11"].shape[0]) == n:
            out = {k: z[k] for k in z.files}
            if audit_path is None:
                print(f"[a11] reuse {path} routing={routing}", flush=True)
                return out
            if audit_path.exists():
                print(f"[a11] reuse {path} routing={routing}", flush=True)
                out["audit"] = json.loads(audit_path.read_text())
                return out
            print(f"[a11] {path} exists but audit missing — rematerializing", flush=True)
    print(f"[a11] materialize n={n} routing={routing} audit={audit_path is not None}", flush=True)
    cf = ensure_cross_fit(bundle)
    mt = bundle["model_train"]
    ids = np.zeros((n, K), dtype=np.int32)
    a11 = np.zeros((n, K), dtype=np.float32)
    nv = np.zeros(n, dtype=np.int16)
    users, items, labels = pairs["user_id"], pairs["item_id"], pairs["label"]
    full_d, top_d = StreamDist(), StreamDist()
    n_pairs = n_trunc = n_full_gt25 = n_trunc_gt25 = 0
    sum_full_neg = sum_top_neg = sum_full_pos = sum_top_pos = 0.0
    t0 = time.time()
    groups: dict[int, list[int]] = defaultdict(list)
    for r in range(n):
        groups[int(users[r])].append(r)

    def fill(hist: np.ndarray, row_ids: list[int]) -> None:
        nonlocal n_pairs, n_trunc, n_full_gt25, n_trunc_gt25
        nonlocal sum_full_neg, sum_top_neg, sum_full_pos, sum_top_pos
        if not row_ids or hist.size == 0:
            return
        cands = items[np.asarray(row_ids, dtype=np.int64)]
        idx = clean_v2_index_for(cf, int(users[row_ids[0]]))
        n11 = idx.cooccurrence_block(hist, cands)
        a_mat, _ = a11_energy_from_n11_matrix(
            n11, idx.popularity[hist], idx.popularity[cands], idx.n_users
        )
        H, C = a_mat.shape
        if routing == "signed":
            if H <= K:
                sel = np.broadcast_to(np.arange(H)[:, None], (H, C)).copy()
            else:
                sel = deterministic_topk_indices(a_mat, hist, k=K)
        elif routing == "abs":
            kk = min(K, H)
            sel = deterministic_topk_indices(np.abs(a_mat), hist, k=kk)
        elif routing == "dual":
            sel = None
        else:
            abort(f"unknown routing {routing}")
        for j, r in enumerate(row_ids):
            if routing == "dual":
                take = dual_tail_select(a_mat[:, j], hist)
                kk = int(take.size)
                nv[r] = kk
                if kk:
                    ids[r, :kk] = hist[take].astype(np.int32)
                    a11[r, :kk] = a_mat[take, j].astype(np.float32)
                top = a11[r, :kk]
            else:
                kk = int(sel.shape[0])
                nv[r] = kk
                ids[r, :kk] = hist[sel[:, j]].astype(np.int32)
                a11[r, :kk] = a_mat[sel[:, j], j].astype(np.float32)
                top = a11[r, :kk]
            if audit_path is not None:
                full = a_mat[:, j]
                full_d.add(full)
                top_d.add(top)
                n_pairs += 1
                fn, tn = float((full < 0).sum()), float((top < 0).sum())
                fp, tp = float((full > 0).sum()), float((top > 0).sum())
                sum_full_neg += fn
                sum_top_neg += tn
                sum_full_pos += fp
                sum_top_pos += tp
                trunc = bool((full < 0).any() and not (top < 0).any())
                n_trunc += int(trunc)
                if H > K:
                    n_full_gt25 += 1
                    n_trunc_gt25 += int(trunc)

    for ui, (u, rows) in enumerate(groups.items()):
        idx = clean_v2_index_for(cf, int(u))
        del idx
        base = np.asarray(sorted(mt.get(u, ())), dtype=np.int64)
        base_set = set(int(x) for x in base.tolist())
        normal, special = [], []
        for r in rows:
            if int(labels[r]) == 1 and int(items[r]) in base_set:
                special.append(r)
            else:
                normal.append(r)
        fill(base, normal)
        for r in special:
            fill(routing_history_exclude_self(base, int(items[r])), [r])
        if ui % 2000 == 0:
            print(f"[a11 {routing}] users {ui}/{len(groups)} ({time.time()-t0:.0f}s)", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=ids, a11=a11, n=nv)
    out: dict[str, Any] = {"ids": ids, "a11": a11, "n": nv}
    if audit_path is not None:
        frac = n_trunc / max(n_pairs, 1)
        sign = "NEGATIVE_TAIL_TRUNCATED" if frac >= 0.01 else "BOTH_TAILS_PRESERVED"
        audit = {
            "n_pairs": n_pairs,
            "frac_full_neg_but_top25_no_neg": frac,
            "n_trunc": n_trunc,
            "n_pairs_hist_gt25": n_full_gt25,
            "frac_trunc_among_hist_gt25": n_trunc_gt25 / max(n_full_gt25, 1),
            "mean_n_neg_full": sum_full_neg / max(n_pairs, 1),
            "mean_n_neg_top25": sum_top_neg / max(n_pairs, 1),
            "mean_n_pos_full": sum_full_pos / max(n_pairs, 1),
            "mean_n_pos_top25": sum_top_pos / max(n_pairs, 1),
            "A11_ROUTING_SIGN_AUDIT": sign,
            "full_history_pooled": full_d.summary(),
            "top25_pooled": top_d.summary(),
        }
        write_json(audit_path, audit)
        out["audit"] = audit
        print(f"A11_ROUTING_SIGN_AUDIT = {sign}  frac_trunc={frac:.4f}", flush=True)
    print(f"[a11] saved {path} ({time.time()-t0:.0f}s)", flush=True)
    return out


def write_freeze(hashes: dict[str, str]) -> None:
    s0 = stage_dir("00_AUDIT")
    text = f"""# A11_DEFINITION_FREEZE

Exact existing item-A11. Not redefined.

## Formula (binary Pearson / φ)

    a11(h,X) = (N · n11 − n_h · n_X) / sqrt( n_h (N−n_h) n_X (N−n_X) )

denom ≤ 1e-12 → 0. Signed. No absolute value for selection or pooling.

Source function: `src/lastfm_lp/binary/binary_measures.py` `a11_energy_from_n11_matrix`.

RACE CLEAN 3 uses **no additive smoothing** in this formula (MI_SMOOTHING=0 is a
separate MI path and is not applied to A11).

## Population / fold logic

- 5 user folds, seed **2026** (`assign_user_folds`).
- For user u, co-occurrence / popularity / N come from `D_train \\ fold(u)`.
- Resolver: `clean_v2_index_for` — leave-one-fold **even on validation**.
- Validation and test interactions are **not** in `model_train` and therefore
  cannot enter A11 estimation.

## Routing (unchanged for Exp 1)

Eligible history: `H_u^(\\neg X)` (self-exclusion).
Rank by **signed** A11 descending; tie-break lower item ID.
`K = min(25, |H_u^(\\neg X)|)`.
Downstream H3 = `[mean, max, top3mean]` of that Top25.

## Source arrays

- splits: `outputs/lastfm_star/splits/{{model_train,valid,test}}.txt`
- H3 features: `outputs/lastfm_star/materialized/race/a11_top25/features/X_{{train,val}}.npy`
- cross-fit cache: `outputs/lastfm_star/features/cross_fit_cache/`

## Hashes

```
{json.dumps(hashes, indent=2)}
```
"""
    (s0 / "audit" / "A11_DEFINITION_FREEZE.md").write_text(text, encoding="utf-8")


def load_official_cell(bundle, seed: int) -> dict[str, Any]:
    path = ROLE / "runs" / f"N0_seed{seed}" / "val_logits.npy"
    logits = np.load(path).astype(np.float64)
    return {
        "metrics": sampled_metrics(bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"], logits),
        "pu": per_user_ndcg(bundle["val_pairs"]["user_id"], bundle["val_pairs"]["label"], logits),
        "logits": logits,
    }


def cell_from_scores(users, labels, scores) -> dict[str, Any]:
    return {
        "metrics": sampled_metrics(users, labels, scores),
        "pu": per_user_ndcg(users, labels, scores),
        "logits": np.asarray(scores, dtype=np.float64),
    }


def boot_pair(c0, c1, n: int = 10_000) -> dict[str, Any]:
    common = sorted(set(c0["pu"]) & set(c1["pu"]))
    arr = np.asarray([c1["pu"][u] - c0["pu"][u] for u in common], dtype=np.float64)
    mu, lo, hi = bootstrap_ci(arr, n=n)
    return {
        "n": int(arr.size),
        "mean": mu,
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "frac_gt0": float((arr > 0).mean()),
        "frac_eq0": float((arr == 0).mean()),
        "frac_lt0": float((arr < 0).mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_entirely_gt0": bool(lo > 0),
        "arr": arr,
    }


def contrast_seeds(cells0, cells1, tag: str) -> dict[str, Any]:
    seed_d, arrs = [], []
    for i in range(len(SEEDS)):
        d = cells1[i]["metrics"]["NDCG@20"] - cells0[i]["metrics"]["NDCG@20"]
        seed_d.append(d)
        arrs.append(boot_pair(cells0[i], cells1[i])["arr"])
    cat = np.concatenate(arrs)
    mu, lo, hi = bootstrap_ci(cat)
    return {
        "contrast": tag,
        "mean_seed_delta": float(np.mean(seed_d)),
        "n_seeds_pos": int(sum(1 for x in seed_d if x > 0)),
        "seed_deltas": seed_d,
        "mean": mu,
        "median": float(np.median(cat)),
        "p25": float(np.percentile(cat, 25)),
        "p75": float(np.percentile(cat, 75)),
        "frac_gt0": float((cat > 0).mean()),
        "frac_eq0": float((cat == 0).mean()),
        "frac_lt0": float((cat < 0).mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_entirely_gt0": bool(lo > 0),
    }


def assert_zero(branch, feat, mask, learnable, device, tag: str) -> None:
    branch.eval()
    with torch.no_grad():
        sl = slice(0, min(512, len(feat)))
        if learnable:
            d0 = branch(torch.from_numpy(feat[sl]).to(device), torch.from_numpy(mask[sl]).to(device))
        else:
            d0 = branch(torch.from_numpy(feat[sl]).to(device))
    mx = float(d0.abs().max())
    if mx > 1e-7:
        abort(f"{tag} delta not zero at init maxabs={mx}")


def train_frozen_branch(
    *,
    branch: nn.Module,
    b0_tr: np.ndarray,
    b0_va: np.ndarray,
    y_tr: np.ndarray,
    va_u,
    va_y,
    feat_tr,
    feat_va,
    mask_tr,
    mask_va,
    learnable: bool,
    seed: int,
    ckpt: Path,
    acfg: dict,
    device,
) -> np.ndarray:
    if (ckpt / "branch.pt").exists() and (ckpt / "val_logits.npy").exists():
        print(f"[reuse] {ckpt}", flush=True)
        branch.load_state_dict(torch.load(ckpt / "branch.pt", map_location=device))
        return np.load(ckpt / "val_logits.npy").astype(np.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(
        branch.parameters(),
        lr=float(acfg.get("lr", 1e-3)),
        weight_decay=float(acfg.get("weight_decay", 1e-4)),
    )
    n_pos = float((y_tr > 0.5).sum())
    n_neg = float((y_tr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )
    b0_tr_t = torch.from_numpy(b0_tr.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    if learnable:
        x_tr = torch.from_numpy(feat_tr).to(device)
        m_tr = torch.from_numpy(mask_tr).to(device)
    else:
        f_tr = torch.from_numpy(feat_tr).to(device)
    n_train = len(y_tr)
    batch_size = int(acfg.get("batch_size", 4096))
    best_state, best_ndcg, left, history = None, -1.0, int(acfg.get("patience", 4)), []
    t0 = time.time()
    for epoch in range(int(acfg.get("max_epochs", 15))):
        branch.train()
        perm = np.random.permutation(n_train)
        opt.zero_grad()
        starts = list(range(0, n_train, batch_size))
        n_b = max(len(starts), 1)
        epoch_loss = 0.0
        for start in starts:
            idx = torch.from_numpy(perm[start : start + batch_size]).to(device)
            delta = branch(x_tr[idx], m_tr[idx]) if learnable else branch(f_tr[idx])
            loss = loss_fn(b0_tr_t[idx] + delta, y_t[idx])
            (loss / n_b).backward()
            epoch_loss += float(loss.item())
        opt.step()
        branch.eval()
        with torch.no_grad():
            if learnable:
                d_va = branch(
                    torch.from_numpy(feat_va).to(device),
                    torch.from_numpy(mask_va).to(device),
                ).cpu().numpy()
            else:
                d_va = branch(torch.from_numpy(feat_va).to(device)).cpu().numpy()
        scores = b0_va + d_va.astype(np.float64)
        ndcg = float(sampled_metrics(va_u, va_y, scores)["NDCG@20"])
        history.append({"epoch": epoch, "loss": epoch_loss / max(n_b, 1), "val_NDCG@20": ndcg})
        print(
            f"[{ckpt.parent.name} s{seed}] epoch {epoch} loss={history[-1]['loss']:.4f} NDCG={ndcg:.4f}",
            flush=True,
        )
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
            left = int(acfg.get("patience", 4))
        else:
            left -= 1
            if left <= 0:
                break
    if best_state:
        branch.load_state_dict(best_state)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or branch.state_dict(), ckpt / "branch.pt")
    branch.eval()
    with torch.no_grad():
        if learnable:
            d_va = branch(
                torch.from_numpy(feat_va).to(device),
                torch.from_numpy(mask_va).to(device),
            ).cpu().numpy()
        else:
            d_va = branch(torch.from_numpy(feat_va).to(device)).cpu().numpy()
    scores = b0_va + d_va.astype(np.float64)
    np.save(ckpt / "val_logits.npy", scores.astype(np.float32))
    np.save(ckpt / "delta_val.npy", d_va.astype(np.float32))
    write_json(
        ckpt / "train_meta.json",
        {
            "best_val_NDCG@20_sampled": best_ndcg,
            "history": history,
            "seconds": time.time() - t0,
            "seed": seed,
        },
    )
    return scores


def load_b0_trainable(bundle, seed: int, device):
    ckpt = B0_CKPT / f"seed_{seed}"
    if not (ckpt / "model.pt").exists():
        abort(f"missing B0 {ckpt}")
    cfg = bundle["cfg"]
    acfg = cfg.get("models", {}).get("architecture", {})
    fcfg = cfg.get("fusion", {})
    meta = json.loads((ckpt / "train_meta.json").read_text())
    with (ckpt / "a_scaler.pkl").open("rb") as f:
        a_scaler = pickle.load(f)
    with (ckpt / "h_scaler.pkl").open("rb") as f:
        h_scaler = pickle.load(f)
    graph = load_data_and_typed_graph(cfg, bundle["model_train"], max_kg_edges=acfg.get("max_kg_edges", 250_000))
    edge_index_dict = {k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0}
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
    fusion = build_fusion_head(encoder=encoder, hcr_dim=3, use_a11=True, fcfg=fcfg).to(device)
    rec = RecommendationModel(encoder, fusion).to(device)
    rec.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
    for p in rec.parameters():
        p.requires_grad_(True)
    return rec, a_scaler, h_scaler, int(meta["item_offset"]), graph


def score_joint(model, users, items, A_s, H_s, item_offset, device, feat, mask, kind, residual_enabled=True):
    model.eval()
    with torch.no_grad():
        z = model.encoder.encode_all()
        u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
        chunks = []
        for start in range(0, len(users), 4096):
            sl = slice(start, start + 4096)
            kw = dict(z=z, residual_enabled=residual_enabled)
            if kind in {"R3", "R4"}:
                kw["feat"] = torch.from_numpy(feat[sl]).to(device)
                kw["mask"] = torch.from_numpy(mask[sl]).to(device)
            else:
                kw["feat"] = torch.from_numpy(feat[sl]).to(device)
            out = model(
                u_idx[sl].to(device),
                i_idx[sl].to(device),
                torch.from_numpy(A_s[sl]).to(device),
                torch.from_numpy(H_s[sl]).to(device),
                **kw,
            )
            chunks.append(out["logits"].cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_joint(
    model,
    bundle,
    *,
    A_tr_s,
    A_va_s,
    H_tr_s,
    H_va_s,
    item_offset,
    device,
    feat_tr,
    feat_va,
    mask_tr,
    mask_va,
    kind: str,
    seed: int,
    ckpt: Path,
) -> np.ndarray:
    if (ckpt / "model.pt").exists() and (ckpt / "val_logits.npy").exists():
        print(f"[joint reuse] {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
        return np.load(ckpt / "val_logits.npy").astype(np.float64)
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    torch.manual_seed(seed)
    np.random.seed(seed)
    u_all, i_all = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    y_all = bundle["train_pairs"]["label"].astype(np.float32)
    n_train = len(y_all)
    batch_size = int(acfg.get("batch_size", 4096))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )
    va_u, va_i, va_y = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], bundle["val_pairs"]["label"]
    best_state, best_ndcg, left, history = None, -1.0, 4, []
    t0 = time.time()
    for epoch in range(int(acfg.get("max_epochs", 15))):
        te = time.time()
        model.train()
        perm = np.random.permutation(n_train)
        opt.zero_grad()
        z_live = model.encoder.encode_all()
        starts = list(range(0, n_train, batch_size))
        n_b = max(len(starts), 1)
        epoch_loss = 0.0
        for bi, start in enumerate(starts):
            idx = perm[start : start + batch_size]
            z = z_live if bi == 0 else z_live.detach()
            kw = dict(z=z)
            if kind in {"R3", "R4"}:
                kw["feat"] = torch.from_numpy(feat_tr[idx]).to(device)
                kw["mask"] = torch.from_numpy(mask_tr[idx]).to(device)
            else:
                kw["feat"] = torch.from_numpy(feat_tr[idx]).to(device)
            out = model(
                u_all[idx].to(device),
                i_all[idx].to(device),
                torch.from_numpy(A_tr_s[idx]).to(device),
                torch.from_numpy(H_tr_s[idx]).to(device),
                **kw,
            )
            loss = loss_fn(out["logits"], torch.from_numpy(y_all[idx]).to(device))
            if not torch.isfinite(loss):
                abort(f"joint {kind} NaN loss")
            (loss / n_b).backward()
            epoch_loss += float(loss.item())
        opt.step()
        del z_live
        empty_cache()
        scores = score_joint(
            model, va_u, va_i, A_va_s, H_va_s, item_offset, device, feat_va, mask_va, kind
        )
        ndcg = float(ranking_metrics_for_users(va_u, va_y, scores, ks=(20,))["NDCG@20"])
        history.append(
            {"epoch": epoch, "loss": epoch_loss / n_b, "val_NDCG@20": ndcg, "sec": time.time() - te}
        )
        print(
            f"[joint {kind} s{seed}] epoch {epoch} NDCG={ndcg:.4f} ({history[-1]['sec']:.1f}s)",
            flush=True,
        )
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = 4
        else:
            left -= 1
            if left <= 0:
                break
    if best_state:
        model.load_state_dict(best_state)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt / "model.pt")
    scores = score_joint(model, va_u, va_i, A_va_s, H_va_s, item_offset, device, feat_va, mask_va, kind)
    np.save(ckpt / "val_logits.npy", scores.astype(np.float32))
    zeroed = score_joint(
        model, va_u, va_i, A_va_s, H_va_s, item_offset, device, feat_va, mask_va, kind, residual_enabled=False
    )
    np.save(ckpt / "val_logits_zeroed.npy", zeroed.astype(np.float32))
    write_json(
        ckpt / "train_meta.json",
        {"best_val_NDCG@20_sampled": best_ndcg, "history": history, "seconds": time.time() - t0, "seed": seed},
    )
    write_json(
        ckpt / "zeroing.json",
        {
            "active": float(sampled_metrics(va_u, va_y, scores)["NDCG@20"]),
            "zeroed": float(sampled_metrics(va_u, va_y, zeroed)["NDCG@20"]),
        },
    )
    return scores


def bucket_table(cells0, cells1, mt, code: str) -> list[dict]:
    rows = []
    for bkt in ACT:
        a0, a1 = [], []
        for i in range(3):
            for u, v0 in cells0[i]["pu"].items():
                if activity_bucket(len(mt.get(int(u), ()))) != bkt:
                    continue
                if u in cells1[i]["pu"]:
                    a0.append(v0)
                    a1.append(cells1[i]["pu"][u])
        rows.append(
            {
                "variant": code,
                "bucket": bkt,
                "B0": float(np.mean(a0)) if a0 else float("nan"),
                "R": float(np.mean(a1)) if a1 else float("nan"),
                "delta": float(np.mean(np.asarray(a1) - np.asarray(a0))) if a0 else float("nan"),
                "n": len(a0),
            }
        )
    for hb in ("1-5", "6-10", "11-25", "26-50", ">50"):
        a0, a1 = [], []
        for i in range(3):
            for u, v0 in cells0[i]["pu"].items():
                if hist_bucket(len(mt.get(int(u), ()))) != hb:
                    continue
                if u in cells1[i]["pu"]:
                    a0.append(v0)
                    a1.append(cells1[i]["pu"][u])
        rows.append(
            {
                "variant": code,
                "bucket": f"hist_{hb}",
                "B0": float(np.mean(a0)) if a0 else float("nan"),
                "R": float(np.mean(a1)) if a1 else float("nan"),
                "delta": float(np.mean(np.asarray(a1) - np.asarray(a0))) if a0 else float("nan"),
                "n": len(a0),
            }
        )
    return rows


def tau1_temp(a11: np.ndarray, mask: np.ndarray, device, bs: int = 8192) -> np.ndarray:
    tmp = TempPoolBranch().to(device)
    outs = []
    with torch.no_grad():
        for start in range(0, len(a11), bs):
            sl = slice(start, start + bs)
            outs.append(
                tmp.pooled(
                    torch.from_numpy(a11[sl]).to(device),
                    torch.from_numpy(mask[sl]).to(device),
                )
                .cpu()
                .numpy()
            )
    del tmp
    return np.concatenate(outs)


def make_branch(code: str, device, L_tr, Q_tr, a11_tr, mask_tr):
    if code == "R1":
        sc = StandardScaler().fit(L_tr)
        return ResMLP(3).to(device), scale_split(sc, L_tr), False, sc
    if code == "R2":
        sc = StandardScaler().fit(Q_tr)
        return ResMLP(7).to(device), scale_split(sc, Q_tr), False, sc
    if code == "R3":
        init_s = tau1_temp(a11_tr, mask_tr, device)
        sc = StandardScaler().fit(init_s)
        return TempPoolBranch(sc.mean_, sc.scale_).to(device), a11_tr, True, sc
    if code == "R4":
        return A11AttnBranch().to(device), a11_tr, True, None
    abort(f"unknown code {code}")
    raise AssertionError


def apply_scaler_or_raw(code, sc, L, Q, a11):
    if code == "R1":
        return scale_split(sc, L)
    if code == "R2":
        return scale_split(sc, Q)
    return a11


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n")
    (OUT / "FULLRANK_LOCKED").write_text("LOCKED_NOT_RUN\n")
    print("[a11-dist] load", flush=True)
    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    check_overlaps()
    hashes.update(
        {
            "H3_X_train": sha256_file(B0_H / "X_train.npy"),
            "H3_X_val": sha256_file(B0_H / "X_val.npy"),
            "A_X_train": sha256_file(shared_A_dir() / "X_train.npy"),
            "A_X_val": sha256_file(shared_A_dir() / "X_val.npy"),
            "a11_fn": sha256_file(ROOT / "src" / "lastfm_lp" / "binary" / "binary_measures.py"),
            "topk_fn": sha256_file(ROOT / "src" / "lastfm_lp" / "binary" / "measure_race_selection.py"),
            "clean_index_fn": sha256_file(ROOT / "src" / "lastfm_lp" / "clean_v2" / "crossfit_index.py"),
        }
    )
    write_freeze(hashes)
    s0 = stage_dir("00_AUDIT")
    s1 = stage_dir("01_FIXED_ROUTING")
    va_u = bundle["val_pairs"]["user_id"]
    va_i = bundle["val_pairs"]["item_id"]
    va_y = bundle["val_pairs"]["label"]
    tr_y = bundle["train_pairs"]["label"].astype(np.float32)
    mt = bundle["model_train"]
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))

    top_tr = materialize(
        bundle, bundle["train_pairs"], s1 / "cache" / "top25_train_signed.npz", routing="signed"
    )
    top_va = materialize(
        bundle,
        bundle["val_pairs"],
        s0 / "cache" / "top25_val_signed.npz",
        routing="signed",
        audit_path=s0 / "reports" / "ROUTING_SIGN.json",
    )
    if "audit" not in top_va:
        abort("routing audit missing")
    audit = top_va["audit"]
    sign_audit = audit["A11_ROUTING_SIGN_AUDIT"]
    (s0 / "audit" / "ROUTING_SIGN_AUDIT.md").write_text(
        f"# ROUTING SIGN AUDIT\n\nA11_ROUTING_SIGN_AUDIT = **{sign_audit}**\n\n"
        f"pairs={audit['n_pairs']}\n"
        f"frac(full has neg & Top25 has none)={audit['frac_full_neg_but_top25_no_neg']:.4f}\n"
        f"among |H|>25: {audit['frac_trunc_among_hist_gt25']:.4f} (n={audit['n_pairs_hist_gt25']})\n"
        f"mean n_neg full={audit['mean_n_neg_full']:.3f}  Top25={audit['mean_n_neg_top25']:.3f}\n"
        f"mean n_pos full={audit['mean_n_pos_full']:.3f}  Top25={audit['mean_n_pos_top25']:.3f}\n\n"
        f"## Full-history pooled\n{json.dumps(audit['full_history_pooled'], indent=2)}\n\n"
        f"## Top25 pooled\n{json.dumps(audit['top25_pooled'], indent=2)}\n",
        encoding="utf-8",
    )
    write_json(s0 / "manifest.json", {"A11_ROUTING_SIGN_AUDIT": sign_audit, **{k: v for k, v in audit.items() if k != "full_history_pooled"}})

    L_tr = features_legendre(top_tr["a11"], top_tr["n"])
    L_va = features_legendre(top_va["a11"], top_va["n"])
    Q_tr = features_quantile(top_tr["a11"], top_tr["n"])
    Q_va = features_quantile(top_va["a11"], top_va["n"])
    mask_tr, mask_va = mask_from_n(top_tr["n"]), mask_from_n(top_va["n"])
    np.save(s1 / "cache" / "L_train.npy", L_tr)
    np.save(s1 / "cache" / "L_val.npy", L_va)
    np.save(s1 / "cache" / "Q_train.npy", Q_tr)
    np.save(s1 / "cache" / "Q_val.npy", Q_va)
    (s1 / "audit" / "R4_NO_SCALER.md").write_text(
        "R4 A11-only attention: scaler omitted. Pre-standardizing x would change the "
        "learned set attention and is inconsistent with the A11-in-[-1,1] geometry.\n",
        encoding="utf-8",
    )

    # correlation vs H3 (tau=1 S±)
    H_va = np.load(B0_H / "X_val.npy").astype(np.float32)
    S_va = tau1_temp(top_va["a11"], mask_va, device)
    xnan = np.where(mask_va, top_va["a11"].astype(np.float64), np.nan)
    with np.errstate(all="ignore"):
        mins = np.nanmin(xnan, axis=1).astype(np.float32)
        stds = np.nanstd(xnan, axis=1, ddof=0).astype(np.float32)
    mins = np.nan_to_num(mins, nan=0.0)
    stds = np.nan_to_num(stds, nan=0.0)
    shape = np.column_stack([H_va, mins, stds, Q_va[:, [0, 2, 4]], L_va, S_va])
    names_shape = [
        "H_mean", "H_max", "H_top3", "min", "std", "q10", "q50", "q90", "L2", "L3", "L4", "S_plus", "S_minus",
    ]
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(shape.T)
    write_csv(
        s1 / "reports" / "FEATURE_CORR.csv",
        [
            {"i": names_shape[i], "j": names_shape[j], "corr": float(C[i, j])}
            for i in range(len(names_shape))
            for j in range(i + 1, len(names_shape))
        ],
        ["i", "j", "corr"],
    )
    write_csv(
        s0 / "reports" / "POOLED_STATS.csv",
        [
            {"set": "full_history", **audit["full_history_pooled"]},
            {"set": "top25", **audit["top25_pooled"]},
        ],
        ["set", "n", "min", "q05", "q10", "q25", "median", "q75", "q90", "q95", "max",
         "mean", "std", "fraction_positive", "fraction_negative", "fraction_zero",
         "count_positive", "count_negative"],
    )

    variants = {
        "R1": {"name": "LEGENDRE", "learnable": False},
        "R2": {"name": "QUANTILES", "learnable": False},
        "R3": {"name": "TEMP_POOL", "learnable": True},
        "R4": {"name": "A11_ATTENTION", "learnable": True},
    }
    b0_cells = []
    cells: dict[str, list] = {k: [] for k in ["R0", "R1", "R2", "R3", "R4"]}

    for seed in SEEDS:
        print(f"[phase1] B0 logits seed={seed}", flush=True)
        cache_tr = s1 / "cache" / f"b0_train_logits_seed{seed}.npy"
        cache_va = s1 / "cache" / f"b0_val_logits_seed{seed}.npy"
        off_cell = load_official_cell(bundle, seed)
        if cache_tr.exists() and cache_va.exists():
            b0_tr = np.load(cache_tr).astype(np.float64)
            b0_va = np.load(cache_va).astype(np.float64)
        else:
            ctx = load_b0(bundle, seed)
            A_tr = np.load(shared_A_dir() / "X_train.npy").astype(np.float32)
            A_va = np.load(shared_A_dir() / "X_val.npy").astype(np.float32)
            Hi_tr = np.load(B0_H / "X_train.npy").astype(np.float32)
            Hi_va = np.load(B0_H / "X_val.npy").astype(np.float32)
            A_tr_s, A_va_s = scale_split(ctx["a_scaler"], A_tr), scale_split(ctx["a_scaler"], A_va)
            H_tr_s, H_va_s = scale_split(ctx["h_scaler"], Hi_tr), scale_split(ctx["h_scaler"], Hi_va)
            b0_tr = b0_logits(ctx, bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], A_tr_s, H_tr_s)
            b0_va = b0_logits(ctx, va_u, va_i, A_va_s, H_va_s)
            np.save(cache_tr, b0_tr.astype(np.float32))
            np.save(cache_va, b0_va.astype(np.float32))
            del ctx
            empty_cache()
        ident = float(np.max(np.abs(b0_va - off_cell["logits"])))
        print(f"[phase1] seed={seed} live B0 vs N0 maxabs={ident:.3e}", flush=True)
        r0 = cell_from_scores(va_u, va_y, b0_va)
        b0_cells.append(r0)
        cells["R0"].append(r0)
        write_json(s1 / "runs" / f"R0_seed{seed}" / "identity_vs_N0.json", {"max_abs": ident})

        for code in ("R1", "R2", "R3", "R4"):
            branch, ftr, learnable, sc = make_branch(code, device, L_tr, Q_tr, top_tr["a11"], mask_tr)
            fva = apply_scaler_or_raw(code, sc, L_va, Q_va, top_va["a11"])
            mtr, mva = mask_tr, mask_va
            ckpt = s1 / "runs" / f"{code}_seed{seed}"
            if sc is not None:
                ckpt.mkdir(parents=True, exist_ok=True)
                with (ckpt / "scaler.pkl").open("wb") as f:
                    pickle.dump(sc, f)
            if not (ckpt / "branch.pt").exists():
                assert_zero(branch, fva, mva, learnable, device, f"{code} s{seed}")
            scores = train_frozen_branch(
                branch=branch,
                b0_tr=b0_tr,
                b0_va=b0_va,
                y_tr=tr_y,
                va_u=va_u,
                va_y=va_y,
                feat_tr=ftr,
                feat_va=fva,
                mask_tr=mtr,
                mask_va=mva,
                learnable=learnable,
                seed=seed,
                ckpt=ckpt,
                acfg=acfg,
                device=device,
            )
            cell = cell_from_scores(va_u, va_y, scores)
            if (ckpt / "delta_val.npy").exists():
                cell["delta"] = np.load(ckpt / "delta_val.npy").astype(np.float64)
            cells[code].append(cell)
            print(f"[phase1] {code} seed={seed} NDCG={cell['metrics']['NDCG@20']:.6f}", flush=True)
            del branch
            empty_cache()

    seed_table = []
    names = ["R0", "R1", "R2", "R3", "R4"]
    for i, seed in enumerate(SEEDS):
        seed_table.append({"seed": seed, **{n: cells[n][i]["metrics"]["NDCG@20"] for n in names}})
    mean_row = {"seed": "mean", **{n: float(np.mean([r[n] for r in seed_table])) for n in names}}
    write_csv(s1 / "reports" / "PHASE1_BY_SEED.csv", seed_table + [mean_row], ["seed"] + names)

    boot_rows = [contrast_seeds(cells["R0"], cells[c], f"{c}-R0") for c in ("R1", "R2", "R3", "R4")]
    write_csv(
        s1 / "reports" / "PHASE1_BOOTSTRAP.csv",
        [{k: v for k, v in r.items() if k not in {"arr", "seed_deltas"}} | {"seed_deltas": json.dumps(r["seed_deltas"])} for r in boot_rows],
        [
            "contrast", "mean_seed_delta", "n_seeds_pos", "mean", "median", "p25", "p75",
            "frac_gt0", "frac_eq0", "frac_lt0", "ci95_lo", "ci95_hi", "ci_entirely_gt0", "seed_deltas",
        ],
    )
    cand = []
    for r in boot_rows:
        code = r["contrast"].split("-")[0]
        if r["mean_seed_delta"] >= SCREEN and r["n_seeds_pos"] >= 2:
            cand.append((code, r["mean_seed_delta"], mean_row[code]))
    cand.sort(key=lambda t: -t[2])
    promoted = [c[0] for c in cand[:2]]
    winner_fixed = variants[promoted[0]]["name"] if promoted else "B0_H3"
    print(f"[phase1] promoted={promoted} winner={winner_fixed}", flush=True)
    write_json(
        s1 / "SELECTED.json",
        {
            "promoted": promoted,
            "A11_FIXED_REP_WINNER": winner_fixed,
            "boot": [{k: v for k, v in r.items() if k not in {"arr", "seed_deltas"}} for r in boot_rows],
        },
    )
    write_json(s1 / "manifest.json", {"promoted": promoted, "winner": winner_fixed})
    act_rows = []
    for code in ("R1", "R2", "R3", "R4"):
        act_rows.extend(bucket_table(cells["R0"], cells[code], mt, code))
    write_csv(s1 / "reports" / "BY_ACTIVITY_HISTORY.csv", act_rows, ["variant", "bucket", "B0", "R", "delta", "n"])

    # ---- Exp 2 joint
    s2 = stage_dir("02_JOINT_CONFIRM")
    joint_verdict = "A11_REP_NO_GAIN"
    joint_cells: dict[str, list] = {}
    best_joint = {"code": None, "mean_delta": 0.0, "npos": 0, "ci": False}
    if not promoted:
        (s2 / "SKIP.md").write_text("No Phase-1 promotion (bar +0.0005 and 2/3 seeds). Skip joint confirm.\n")
        write_json(s2 / "manifest.json", {"A11_JOINT_CONFIRM_VERDICT": joint_verdict, "promoted": []})
    else:
        for code in promoted:
            joint_cells[code] = []
            learnable = variants[code]["learnable"]
            for seed in SEEDS:
                rec, a_sc, h_sc, item_offset, graph = load_b0_trainable(bundle, seed, device)
                del graph
                A_tr = np.load(shared_A_dir() / "X_train.npy").astype(np.float32)
                A_va = np.load(shared_A_dir() / "X_val.npy").astype(np.float32)
                Hi_tr = np.load(B0_H / "X_train.npy").astype(np.float32)
                Hi_va = np.load(B0_H / "X_val.npy").astype(np.float32)
                A_tr_s, A_va_s = scale_split(a_sc, A_tr), scale_split(a_sc, A_va)
                H_tr_s, H_va_s = scale_split(h_sc, Hi_tr), scale_split(h_sc, Hi_va)
                branch, ftr, learnable, sc = make_branch(code, device, L_tr, Q_tr, top_tr["a11"], mask_tr)
                fva = apply_scaler_or_raw(code, sc, L_va, Q_va, top_va["a11"])
                model = JointModel(rec, branch, code).to(device)
                assert_zero(branch, fva, mask_va, learnable, device, f"joint {code} s{seed}")
                scores = train_joint(
                    model,
                    bundle,
                    A_tr_s=A_tr_s,
                    A_va_s=A_va_s,
                    H_tr_s=H_tr_s,
                    H_va_s=H_va_s,
                    item_offset=item_offset,
                    device=device,
                    feat_tr=ftr,
                    feat_va=fva,
                    mask_tr=mask_tr,
                    mask_va=mask_va,
                    kind=code,
                    seed=seed,
                    ckpt=s2 / "runs" / f"{code}_seed{seed}",
                )
                joint_cells[code].append(cell_from_scores(va_u, va_y, scores))
                del model, rec, branch
                empty_cache()
            boot = contrast_seeds(cells["R0"], joint_cells[code], f"joint{code}-R0")
            write_json(s2 / "reports" / f"{code}_BOOT.json", {k: v for k, v in boot.items() if k not in {"arr"}})
            if boot["mean_seed_delta"] > best_joint["mean_delta"]:
                best_joint = {
                    "code": code,
                    "mean_delta": boot["mean_seed_delta"],
                    "npos": boot["n_seeds_pos"],
                    "ci": boot["ci_entirely_gt0"],
                }
        md, npos, ci = best_joint["mean_delta"], best_joint["npos"], best_joint["ci"]
        if md >= CONFIRM and npos >= 2 and ci:
            joint_verdict = "A11_REP_GAIN"
        elif md > 0:
            joint_verdict = "A11_REP_SMALL_GAIN"
        else:
            joint_verdict = "A11_REP_NO_GAIN"
        write_json(s2 / "manifest.json", {"A11_JOINT_CONFIRM_VERDICT": joint_verdict, **best_joint})

    confirmed = []
    if joint_verdict == "A11_REP_GAIN":
        for code in promoted:
            b = json.loads((s2 / "reports" / f"{code}_BOOT.json").read_text())
            if b["mean_seed_delta"] >= CONFIRM and b["n_seeds_pos"] >= 2 and b["ci_entirely_gt0"]:
                confirmed.append(code)

    # ---- Exp 3 signed tail
    s3 = stage_dir("03_SIGNED_TAIL_ROUTING")
    tail_verdict = "ROUTING_TAIL_EXPERIMENT_NOT_NEEDED"
    tail_cells: dict[str, list] = {}
    if sign_audit != "NEGATIVE_TAIL_TRUNCATED":
        (s3 / "SKIP.md").write_text("ROUTING_TAIL_EXPERIMENT_NOT_NEEDED — both tails preserved in Top25.\n")
        write_json(s3 / "manifest.json", {"A11_SIGNED_TAIL_VERDICT": tail_verdict})
    else:
        use_code = confirmed[0] if confirmed else (promoted[0] if promoted else "LT")
        note = "confirmed" if confirmed else ("promoted_unconfirmed" if promoted else "exploratory_LEGENDRE3_TEMP2")
        print(f"[tail] representation={use_code} ({note})", flush=True)
        t1_tr = materialize(bundle, bundle["train_pairs"], s3 / "cache" / "train_abs.npz", routing="abs")
        t1_va = materialize(bundle, bundle["val_pairs"], s3 / "cache" / "val_abs.npz", routing="abs")
        t2_tr = materialize(bundle, bundle["train_pairs"], s3 / "cache" / "train_dual.npz", routing="dual")
        t2_va = materialize(bundle, bundle["val_pairs"], s3 / "cache" / "val_dual.npz", routing="dual")
        routes = {
            "T0": ("signed", top_tr, top_va),
            "T1": ("abs", t1_tr, t1_va),
            "T2": ("dual", t2_tr, t2_va),
        }
        for tname, (_rt, trd, vad) in routes.items():
            if tname == "T0" and use_code in cells:
                tail_cells[tname] = cells[use_code]
                continue
            tail_cells[tname] = []
            Ltr = features_legendre(trd["a11"], trd["n"])
            Lva = features_legendre(vad["a11"], vad["n"])
            Qtr = features_quantile(trd["a11"], trd["n"])
            Qva = features_quantile(vad["a11"], vad["n"])
            mtr, mva = mask_from_n(trd["n"]), mask_from_n(vad["n"])
            for seed in SEEDS:
                b0_tr = np.load(s1 / "cache" / f"b0_train_logits_seed{seed}.npy").astype(np.float64)
                b0_va = np.load(s1 / "cache" / f"b0_val_logits_seed{seed}.npy").astype(np.float64)
                ckpt = s3 / "runs" / f"{tname}_{use_code}_seed{seed}"
                if use_code == "LT":
                    scL = StandardScaler().fit(Ltr)
                    scT = StandardScaler().fit(tau1_temp(trd["a11"], mtr, device))
                    inner = LegendreTempBranch(scL.mean_, scL.scale_, scT.mean_, scT.scale_).to(device)
                    scores = train_lt(
                        inner, b0_tr, b0_va, tr_y, va_u, va_y,
                        trd["a11"], vad["a11"], mtr, mva,
                        scale_split(scL, Ltr), scale_split(scL, Lva),
                        seed, ckpt, acfg, device,
                    )
                    del inner
                else:
                    branch, ftr, learnable, sc = make_branch(use_code, device, Ltr, Qtr, trd["a11"], mtr)
                    fva = apply_scaler_or_raw(use_code, sc, Lva, Qva, vad["a11"])
                    if not (ckpt / "branch.pt").exists():
                        assert_zero(branch, fva, mva, learnable, device, f"{tname} {use_code} s{seed}")
                    scores = train_frozen_branch(
                        branch=branch, b0_tr=b0_tr, b0_va=b0_va, y_tr=tr_y, va_u=va_u, va_y=va_y,
                        feat_tr=ftr, feat_va=fva, mask_tr=mtr, mask_va=mva, learnable=learnable,
                        seed=seed, ckpt=ckpt, acfg=acfg, device=device,
                    )
                    del branch
                tail_cells[tname].append(cell_from_scores(va_u, va_y, scores))
                empty_cache()
        boot_t1 = contrast_seeds(tail_cells["T0"], tail_cells["T1"], "T1-T0")
        boot_t2 = contrast_seeds(tail_cells["T0"], tail_cells["T2"], "T2-T0")
        write_json(s3 / "reports" / "T1_BOOT.json", {k: v for k, v in boot_t1.items() if k != "arr"})
        write_json(s3 / "reports" / "T2_BOOT.json", {k: v for k, v in boot_t2.items() if k != "arr"})
        best_t = boot_t1 if boot_t1["mean_seed_delta"] >= boot_t2["mean_seed_delta"] else boot_t2
        if best_t["mean_seed_delta"] >= CONFIRM and best_t["n_seeds_pos"] >= 2 and best_t["ci_entirely_gt0"]:
            tail_verdict = "TAIL_ROUTING_GAIN"
        else:
            tail_verdict = "TAIL_ROUTING_NO_GAIN"
        write_json(
            s3 / "manifest.json",
            {
                "A11_SIGNED_TAIL_VERDICT": tail_verdict,
                "representation": use_code,
                "note": note,
                "T1": {k: v for k, v in boot_t1.items() if k != "arr"},
                "T2": {k: v for k, v in boot_t2.items() if k != "arr"},
            },
        )

    # ---- Exp 4 reconstruction
    s4 = stage_dir("04_DISTRIBUTION_AUDIT")
    stats = []
    for i, k in enumerate(top_va["n"].tolist()):
        k = int(k)
        if k < 3:
            continue
        x = top_va["a11"][i, :k].astype(np.float64)
        stats.append(
            {
                "i": i,
                "mean": float(x.mean()),
                "max": float(x.max()),
                "min": float(x.min()),
                "std": float(x.std()),
                "q10": float(np.percentile(x, 10)),
                "q90": float(np.percentile(x, 90)),
                "L2": float(P2(x).mean()),
                "L3": float(P3(x).mean()),
                "L4": float(P4(x).mean()),
            }
        )
    picks = {
        "high_mean_low_var": max(stats, key=lambda r: r["mean"] - 2 * r["std"]),
        "high_max_low_mean": max(stats, key=lambda r: r["max"] - r["mean"]),
        "strong_pos_tail": max(stats, key=lambda r: r["q90"]),
        "strong_neg_tail": min(stats, key=lambda r: r["min"]),
        "symmetric_extremes": max(stats, key=lambda r: abs(r["max"]) + abs(r["min"]) - 2 * abs(r["mean"])),
        "near_zero": min(stats, key=lambda r: abs(r["mean"]) + r["std"]),
    }
    xs = np.linspace(-1, 1, 201)
    plot_dir = s4 / "reports" / "reconstructions"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dumped = {}
    for name, rec in picks.items():
        i = rec["i"]
        k = int(top_va["n"][i])
        xv = top_va["a11"][i, :k].astype(np.float64)
        L = [float(xv.mean()), float(P2(xv).mean()), float(P3(xv).mean()), float(P4(xv).mean())]
        Pk = [lambda x: x, P2, P3, P4]
        fhat = 0.5 * (1.0 + sum((2 * (kk + 1) + 1) * L[kk] * Pk[kk](xs) for kk in range(4)))
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].hist(xv, bins=min(15, max(k, 3)), range=(-1, 1), density=True, color="#4c78a8")
        ax[0].plot(xs, fhat, "r--", lw=1, alpha=0.6, label="orthogonal series")
        ax[0].set_title(name)
        ax[0].legend(fontsize=8)
        ax[1].scatter(np.sort(xv), np.linspace(0, 1, k, endpoint=False), s=12)
        ax[1].set_xlim(-1, 1)
        ax[1].set_title("empirical CDF")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{name}.png", dpi=120)
        plt.close(fig)
        dumped[name] = {**{kk: vv for kk, vv in rec.items() if kk != "i"}, "pair_index": int(i), "L": L}
    write_json(s4 / "reports" / "PICKS.json", dumped)
    write_json(s4 / "manifest.json", {"diagnostic_only": True, "n_examples": 6})

    # ---- Exp 5
    s5 = stage_dir("05_FINALIST_IF_JUSTIFIED")
    final_verdict = "B0_H3_REMAINS_WINNER"
    if len(confirmed) >= 2:
        d0 = cells[confirmed[0]][0].get("delta")
        d1 = cells[confirmed[1]][0].get("delta")
        corr = float(np.corrcoef(d0, d1)[0, 1]) if d0 is not None and d1 is not None else float("nan")
        if abs(corr) >= 0.95:
            (s5 / "REPRESENTATION_REDUNDANCY_HIGH.md").write_text(f"corr={corr}\n")
            write_json(s5 / "manifest.json", {"A11_FINALIST_VERDICT": final_verdict, "redundancy": corr})
        else:
            (s5 / "SKIP.md").write_text(
                "Two confirmed independent winners exist, but a combined joint HGT finalist "
                "is not auto-built in this pass (would require a third full HGT train). "
                f"codes={confirmed} corr={corr}\n"
            )
            write_json(s5 / "manifest.json", {"A11_FINALIST_VERDICT": final_verdict, "would_combine": confirmed, "corr": corr})
    else:
        (s5 / "SKIP.md").write_text("Fewer than two independently confirmed representations. No combined finalist.\n")
        write_json(s5 / "manifest.json", {"A11_FINALIST_VERDICT": final_verdict})

    if joint_verdict == "A11_REP_GAIN":
        md = best_joint["mean_delta"]
        if abs(md) >= 0.002:
            final_verdict = "A11_STRONG_GAIN"
        elif abs(md) >= 0.001:
            final_verdict = "A11_MATERIAL_GAIN"
        elif abs(md) >= 0.0005:
            final_verdict = "A11_SMALL_GAIN"
    elif winner_fixed != "B0_H3":
        dlt = next(r["mean_seed_delta"] for r in boot_rows if r["contrast"].startswith(promoted[0] if promoted else "R1"))
        if mag(dlt) == "SMALL":
            final_verdict = "A11_SMALL_GAIN"

    master = []
    for n in names:
        s101, s202, s303 = (cells[n][i]["metrics"]["NDCG@20"] for i in range(3))
        mean = float(np.mean([s101, s202, s303]))
        dlt = mean - mean_row["R0"]
        if n == "R0":
            fr = {"frac_gt0": 0, "frac_eq0": 1, "frac_lt0": 0, "ci95_lo": 0, "ci95_hi": 0}
        else:
            fr = next(x for x in boot_rows if x["contrast"].startswith(n))
        master.append(
            {
                "experiment": "01_FIXED_ROUTING",
                "variant": n,
                "representation": "H3" if n == "R0" else variants[n]["name"],
                "routing": "SIGNED_TOP25",
                "seed101": s101,
                "seed202": s202,
                "seed303": s303,
                "mean_ndcg20": mean,
                "delta_vs_B0": dlt,
                "ci_low": fr["ci95_lo"],
                "ci_high": fr["ci95_hi"],
                "fraction_users_improved": fr["frac_gt0"],
                "fraction_users_unchanged": fr["frac_eq0"],
                "fraction_users_worsened": fr["frac_lt0"],
                "practical_class": mag(dlt),
                "verdict": winner_fixed if n != "R0" else "ANCHOR",
            }
        )
    write_csv(OUT / "MASTER_A11_RESULTS.csv", master, list(master[0].keys()))

    def q_short_long():
        by = defaultdict(list)
        for row in act_rows:
            by[row["variant"]].append(row)
        short = max(
            ("R1", "R2", "R3", "R4"),
            key=lambda c: next((r["delta"] for r in by[c] if r["bucket"] == "hist_1-5"), float("-inf")),
        )
        long = max(
            ("R1", "R2", "R3", "R4"),
            key=lambda c: next((r["delta"] for r in by[c] if r["bucket"] == "hist_>50"), float("-inf")),
        )
        return variants[short]["name"], variants[long]["name"]

    short_n, long_n = q_short_long()
    best_p1 = max(boot_rows, key=lambda r: r["mean_seed_delta"])
    md = f"""# A11_DISTRIBUTIONAL_MASTER_REPORT

A11_ROUTING_SIGN_AUDIT = {sign_audit}

Phase-1 frozen-B0 residuals (sampled NDCG@20):

| seed | R0 B0 | R1 LEGENDRE | R2 QUANTILES | R3 TEMP | R4 ATTN |
|---|---:|---:|---:|---:|---:|
| 101 | {seed_table[0]['R0']:.6f} | {seed_table[0]['R1']:.6f} | {seed_table[0]['R2']:.6f} | {seed_table[0]['R3']:.6f} | {seed_table[0]['R4']:.6f} |
| 202 | {seed_table[1]['R0']:.6f} | {seed_table[1]['R1']:.6f} | {seed_table[1]['R2']:.6f} | {seed_table[1]['R3']:.6f} | {seed_table[1]['R4']:.6f} |
| 303 | {seed_table[2]['R0']:.6f} | {seed_table[2]['R1']:.6f} | {seed_table[2]['R2']:.6f} | {seed_table[2]['R3']:.6f} | {seed_table[2]['R4']:.6f} |
| mean | {mean_row['R0']:.6f} | {mean_row['R1']:.6f} | {mean_row['R2']:.6f} | {mean_row['R3']:.6f} | {mean_row['R4']:.6f} |

Promoted: {promoted}

Q1. Both tails in Top25? **{sign_audit}**. frac truncated={audit['frac_full_neg_but_top25_no_neg']:.4f} (among |H|>25: {audit['frac_trunc_among_hist_gt25']:.4f}).
Q2. H3 lose shape? See `01_FIXED_ROUTING/reports/FEATURE_CORR.csv`.
Q3. Legendre? delta={boot_rows[0]['mean_seed_delta']:+.6f} class={mag(boot_rows[0]['mean_seed_delta'])}.
Q4. Quantiles? delta={boot_rows[1]['mean_seed_delta']:+.6f} class={mag(boot_rows[1]['mean_seed_delta'])}.
Q5. Temp pool? delta={boot_rows[2]['mean_seed_delta']:+.6f} class={mag(boot_rows[2]['mean_seed_delta'])}.
Q6. Negative tail used? TEMP S_minus is in TEMP2; routing audit {sign_audit}.
Q7. A11-only attention? delta={boot_rows[3]['mean_seed_delta']:+.6f} class={mag(boot_rows[3]['mean_seed_delta'])}.
Q8. Short histories (1–5): most useful among residuals = {short_n}.
Q9. Long histories (>50): most useful among residuals = {long_n}.
Q10. Complementary vs H3: FEATURE_CORR.csv.
Q11. Alternate routing: {tail_verdict}.
Q12. Practical magnitude? best phase1 {best_p1['contrast']} {mag(best_p1['mean_seed_delta'])} ({best_p1['mean_seed_delta']:+.6f}).

A11_FIXED_REP_WINNER = {winner_fixed}
A11_JOINT_CONFIRM_VERDICT = {joint_verdict}
A11_SIGNED_TAIL_VERDICT = {tail_verdict}
A11_FINALIST_VERDICT = {final_verdict}
TEST_STATUS = LOCKED_NOT_RUN
FULLRANK_STATUS = LOCKED_NOT_RUN
"""
    (OUT / "A11_DISTRIBUTIONAL_MASTER_REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    write_json(
        OUT / "A11_DISTRIBUTIONAL_REPRESENTATION_SERIES_V1_MANIFEST.json",
        {
            "timestamp": utc_now(),
            "git_commit": git_commit(),
            "A11_ROUTING_SIGN_AUDIT": sign_audit,
            "A11_FIXED_REP_WINNER": winner_fixed,
            "A11_JOINT_CONFIRM_VERDICT": joint_verdict,
            "A11_SIGNED_TAIL_VERDICT": tail_verdict,
            "A11_FINALIST_VERDICT": final_verdict,
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "FULLRANK_STATUS": "LOCKED_NOT_RUN",
            "promoted": promoted,
            **{f"{k}_hash": v for k, v in hashes.items()},
        },
    )


def train_lt(branch, b0_tr, b0_va, y_tr, va_u, va_y, a11_tr, a11_va, mtr, mva, Ltr, Lva, seed, ckpt, acfg, device):
    if (ckpt / "val_logits.npy").exists() and (ckpt / "branch.pt").exists():
        print(f"[reuse] {ckpt}", flush=True)
        return np.load(ckpt / "val_logits.npy").astype(np.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(branch.parameters(), lr=float(acfg.get("lr", 1e-3)), weight_decay=float(acfg.get("weight_decay", 1e-4)))
    n_pos = float((y_tr > 0.5).sum())
    n_neg = float((y_tr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device))
    b0_tr_t = torch.from_numpy(b0_tr.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    x_tr = torch.from_numpy(a11_tr).to(device)
    m_tr = torch.from_numpy(mtr).to(device)
    L_tr = torch.from_numpy(Ltr).to(device)
    n_train = len(y_tr)
    batch_size = int(acfg.get("batch_size", 4096))
    best_state, best_ndcg, left, history = None, -1.0, int(acfg.get("patience", 4)), []
    t0 = time.time()
    for epoch in range(int(acfg.get("max_epochs", 15))):
        branch.train()
        perm = np.random.permutation(n_train)
        opt.zero_grad()
        starts = list(range(0, n_train, batch_size))
        n_b = max(len(starts), 1)
        eloss = 0.0
        for start in starts:
            idx = torch.from_numpy(perm[start : start + batch_size]).to(device)
            delta = branch(x_tr[idx], m_tr[idx], L_tr[idx])
            loss = loss_fn(b0_tr_t[idx] + delta, y_t[idx])
            (loss / n_b).backward()
            eloss += float(loss.item())
        opt.step()
        branch.eval()
        with torch.no_grad():
            d_va = branch(
                torch.from_numpy(a11_va).to(device),
                torch.from_numpy(mva).to(device),
                torch.from_numpy(Lva).to(device),
            ).cpu().numpy()
        scores = b0_va + d_va.astype(np.float64)
        ndcg = float(sampled_metrics(va_u, va_y, scores)["NDCG@20"])
        history.append({"epoch": epoch, "loss": eloss / n_b, "val_NDCG@20": ndcg})
        print(f"[{ckpt.name} s{seed}] epoch {epoch} NDCG={ndcg:.4f}", flush=True)
        if ndcg > best_ndcg + 1e-4:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
            left = int(acfg.get("patience", 4))
        else:
            left -= 1
            if left <= 0:
                break
    if best_state:
        branch.load_state_dict(best_state)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or branch.state_dict(), ckpt / "branch.pt")
    branch.eval()
    with torch.no_grad():
        d_va = branch(
            torch.from_numpy(a11_va).to(device),
            torch.from_numpy(mva).to(device),
            torch.from_numpy(Lva).to(device),
        ).cpu().numpy()
    scores = b0_va + d_va.astype(np.float64)
    np.save(ckpt / "val_logits.npy", scores.astype(np.float32))
    write_json(ckpt / "train_meta.json", {"best_val_NDCG@20_sampled": best_ndcg, "history": history, "seconds": time.time() - t0})
    return scores


if __name__ == "__main__":
    main()
