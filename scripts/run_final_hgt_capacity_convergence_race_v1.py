#!/usr/bin/env python3
"""FINAL_HGT_CAPACITY_CONVERGENCE_RACE_V1.

A11 closed at LEG_K2. This race is HGT convergence + encoder capacity only.
Test and full-rank stay locked. One train at a time.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import platform
import subprocess
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
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hgt_aggregation_common import empty_cache  # noqa: E402
from scripts.run_a11_distributional_representation_series_v1 import (  # noqa: E402
    ResMLP,
    cell_from_scores,
    mag,
)
from scripts.run_a11_functional_distribution_benchmark_v2 import (  # noqa: E402
    holm,
    signflip_p,
)
from scripts.run_artist_a11_residual_branch_v1 import (  # noqa: E402
    bootstrap_ci,
    sampled_metrics,
    scale_split,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    abort,
    check_overlaps,
    git_commit,
    verify_fingerprints,
    write_csv,
    write_json,
)
from src.lastfm_lp.models.fusion import LateFusionHead  # noqa: E402
from scripts.run_race_clean_3 import shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import (  # noqa: E402
    load_data_and_typed_graph,
    user_item_to_nodes,
)
from src.lastfm_lp.models.encoders import HGTGraphEncoder  # noqa: E402
from src.lastfm_lp.models.hgt_encoder import HGTEncoder  # noqa: E402
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = ROOT / "configs" / "lastfm_star_race_clean_3.yaml"
RACE = ROOT / "KRAM_FINAL_WORK" / "RACE_CLEAN_3"
V2 = RACE / "A11_FUNCTIONAL_DISTRIBUTION_BENCHMARK_V2"
B0_H = RACE / "race" / "a11_top25" / "features"
OUT = RACE / "FINAL_HGT_CAPACITY_CONVERGENCE_RACE_V1"
SEEDS = (101, 202, 303)
N_BOOT = 20_000
N_PERM = 20_000
READOUT_DIM = 64
GRAPH_CTX = 256
CONFIRM = 0.001
STRONG = 0.002
INDIFF_CAP = 0.0003
MIN_IMPROVE = 1e-4
MEM_FRAC = 0.92
REP_SEED = 20260817

ARCHS = {
    "H0": {"code": "H0_64_2L_2H", "d": 64, "layers": 2, "heads": 2},
    "H1": {"code": "H1_64_2L_4H", "d": 64, "layers": 2, "heads": 4},
    "H2": {"code": "H2_128_2L_2H", "d": 128, "layers": 2, "heads": 2},
    "H3": {"code": "H3_64_3L_2H", "d": 64, "layers": 3, "heads": 2},
    "H4": {"code": "H4_128_2L_4H", "d": 128, "layers": 2, "heads": 4},
    "H5": {"code": "H5_128_3L_4H", "d": 128, "layers": 3, "heads": 4},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dirs() -> dict[str, Path]:
    mapping = {
        "audit": OUT / "00_AUDIT",
        "conv": OUT / "01_CONVERGENCE",
        "cap": OUT / "02_CAPACITY",
        "repro": OUT / "03_REPRODUCTION",
        "stats": OUT / "04_STATS",
        "plots": OUT / "05_PLOTS",
        "report": OUT / "06_REPORT",
    }
    for p in mapping.values():
        p.mkdir(parents=True, exist_ok=True)
        (p / "runs").mkdir(parents=True, exist_ok=True)
    return mapping


class FinalHGTModel(nn.Module):
    """HGT → optional d→64 readout → stock 265D B0 decoder + LEG_K2 residual."""

    def __init__(self, encoder: HGTGraphEncoder, readout: nn.Module, fusion: nn.Module, leg: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.readout = readout
        self.fusion_head = fusion
        self.leg = leg

    def node_z(self, z_raw: Tensor | None = None) -> Tensor:
        if z_raw is None:
            z_raw = self.encoder.encode_all()
        return self.readout(z_raw)

    def forward(
        self,
        user_idx: Tensor,
        item_idx: Tensor,
        base_features: Tensor,
        hcr_features: Tensor,
        r_l2: Tensor,
        z: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if z is None:
            z = self.node_z()
        zu, zi = z[user_idx], z[item_idx]
        gctx = torch.cat([zu, zi, zu * zi, (zu - zi).abs()], dim=-1)
        gscore = (zu * zi).sum(dim=-1)
        b0 = self.fusion_head(gctx, gscore, base_features, hcr_features)
        return {"logits": b0 + self.leg(r_l2), "b0": b0}


def build_race_model(bundle, graph, device, *, d: int, layers: int, heads: int) -> FinalHGTModel:
    cfg = bundle["cfg"]
    fcfg = cfg.get("fusion", {})
    acfg = cfg.get("models", {}).get("architecture", {})
    edge_index_dict = {k: v.to(device) for k, v in graph["edge_index_dict"].items() if v.numel() > 0}
    metadata = (["user", "entity"], list(edge_index_dict.keys()))
    encoder = HGTGraphEncoder(
        graph["meta"]["n_users"],
        graph["meta"]["n_entities"],
        metadata,
        edge_index_dict,
        embed_dim=d,
        n_layers=layers,
        heads=heads,
        dropout=float(acfg.get("dropout", 0.1)),
    ).to(device)
    readout: nn.Module
    if d == READOUT_DIM:
        readout = nn.Identity()
    else:
        readout = nn.Linear(d, READOUT_DIM, bias=False).to(device)
    fusion = LateFusionHead(
        graph_dim=GRAPH_CTX,
        base_feature_dim=5,
        hcr_feature_dim=3,
        hidden_dim=int(fcfg.get("hidden_dim", 128)),
        dropout=float(fcfg.get("dropout", 0.2)),
        use_hcr=True,
        activation=str(fcfg.get("activation", "gelu")),
    ).to(device)
    leg = ResMLP(1).to(device)
    return FinalHGTModel(encoder, readout, fusion, leg).to(device)


def param_blocks(model: FinalHGTModel) -> dict[str, int]:
    def n(m):
        return int(sum(p.numel() for p in m.parameters()))

    return {
        "hgt": n(model.encoder),
        "readout": 0 if isinstance(model.readout, nn.Identity) else n(model.readout),
        "decoder": n(model.fusion_head),
        "leg_k2": n(model.leg),
        "total": n(model),
    }


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch_geometric

        info["pyg"] = torch_geometric.__version__
    except Exception as e:
        info["pyg"] = f"unavailable:{e}"
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    else:
        info["gpu_name"] = "Apple MPS" if torch.backends.mps.is_available() else "CPU"
        info["vram_bytes"] = None
    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["cpu_brand"] = brand
        info["unified_memory_bytes"] = mem
    except Exception:
        pass
    return info


def write_audit(d: dict[str, Path], bundle, host: dict[str, Any]) -> dict[str, Any]:
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    fcfg = bundle["cfg"].get("fusion", {})
    rec = {
        "hidden_dimension": int(acfg.get("embedding_dim", 64)),
        "n_hgt_layers": int(acfg.get("num_layers", 2)),
        "heads": int(acfg.get("heads", 2)),
        "dropout_hgt": float(acfg.get("dropout", 0.1)),
        "inter_layer_activation": "ReLU + Dropout after every HGT layer except last (hgt_encoder.py)",
        "skip_residual": "PyG HGTConv stock skip (learnable sigmoid mix of conv out and node input). Unchanged.",
        "optimizer": "Adam",
        "lr0": float(acfg.get("lr", 1e-3)),
        "weight_decay": float(acfg.get("weight_decay", 1e-4)),
        "scheduler_current": "NONE",
        "max_epochs_current": int(acfg.get("max_epochs", 15)),
        "patience_current": int(acfg.get("patience", 4)),
        "early_stopping_metric": "sampled validation NDCG@20",
        "improvement_threshold": MIN_IMPROVE,
        "batch_size": int(acfg.get("batch_size", 4096)),
        "batching_strategy": "one Adam step per epoch; loss averaged over all train batches (gradient accumulation)",
        "gradient_clipping": "NONE",
        "amp_precision": "fp32 (no AMP)",
        "seed_handling": "torch.manual_seed(seed); numpy.random.seed(seed) at train start",
        "decoder": "LateFusionHead 265 → 128 LayerNorm GELU Dropout(0.2) → 64 GELU Dropout → 1",
        "decoder_input": "256D pair context + 1D graph_dot + 5D A5 + 3D H3 = 265D",
        "leg_k2": "Linear(1,16) GELU Linear(16,1); last layer zero-init; trains jointly",
        "loss": "BCEWithLogitsLoss pos_weight=n_neg/n_pos",
        "readout_d_gt_64": "Linear(d,64,bias=False) after final HGT layer; identity if d=64",
        "source_files": [
            "src/lastfm_lp/models/hgt_encoder.py",
            "src/lastfm_lp/models/encoders/hgt_wrap.py",
            "src/lastfm_lp/models/fusion/late_fusion.py",
            "scripts/run_publication_our_hgt_fullrank.py:train_and_save",
            "configs/lastfm_full_sota_v1.yaml models.architecture / fusion",
            "configs/lastfm_star_race_clean_3.yaml (does not override architecture)",
        ],
        "host": host,
    }
    md = f"""# CURRENT_HGT_TRAINING_CONFIG

Recorded from live config/code at {utc_now()}. Nothing below is a silent change.

## HGT encoder (B0 / C0 architecture)

- hidden / embed_dim: **{rec['hidden_dimension']}**
- layers: **{rec['n_hgt_layers']}**
- heads: **{rec['heads']}**
- dropout: **{rec['dropout_hgt']}** (applied with ReLU after every layer except the last)
- skip: {rec['skip_residual']}
- conv class: `torch_geometric.nn.HGTConv` (stock ADD). No aux aggregation in this program.

## Optimization (historical / C0 recipe)

- optimizer: **{rec['optimizer']}**
- lr0: **{rec['lr0']}**
- weight decay: **{rec['weight_decay']}**
- scheduler: **{rec['scheduler_current']}**
- max_epochs: **{rec['max_epochs_current']}**
- early-stopping patience: **{rec['patience_current']}**
- metric: {rec['early_stopping_metric']}
- improvement bar: {rec['improvement_threshold']}
- batch size: **{rec['batch_size']}**
- batching: {rec['batching_strategy']}
- grad clip: {rec['gradient_clipping']}
- precision: {rec['amp_precision']}
- seeds: {rec['seed_handling']}

## Decoder / A11

- {rec['decoder']}
- input: {rec['decoder_input']}
- LEG_K2: {rec['leg_k2']}
- width>64: {rec['readout_d_gt_64']}

## Loss / negatives

- {rec['loss']}
- negatives frozen in prepared pair tables (train_per_positive=4, val 20)

## Host

```
{json.dumps(host, indent=2)}
```

This machine is **not CUDA**. Capacity preflight uses MPS/CPU allocated memory vs unified RAM.
"""
    (d["audit"] / "CURRENT_HGT_TRAINING_CONFIG.md").write_text(md, encoding="utf-8")
    (OUT / "CURRENT_HGT_TRAINING_CONFIG.md").write_text(md, encoding="utf-8")
    write_json(d["audit"] / "CURRENT_HGT_TRAINING_CONFIG.json", rec)
    return rec


def mps_bytes() -> dict[str, int]:
    out = {"allocated": 0, "driver": 0}
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            out["allocated"] = int(torch.mps.current_allocated_memory())
        except Exception:
            pass
        try:
            out["driver"] = int(torch.mps.driver_allocated_memory())
        except Exception:
            pass
    if torch.cuda.is_available():
        out["allocated"] = int(torch.cuda.memory_allocated())
        out["driver"] = int(torch.cuda.max_memory_allocated())
    return out


@torch.no_grad()
def score_model(model, users, items, A, H, r, item_offset, device, bs: int = 8192) -> np.ndarray:
    model.eval()
    z = model.node_z()
    u_idx, i_idx = user_item_to_nodes(users, items, item_offset)
    chunks = []
    for start in range(0, len(users), bs):
        sl = slice(start, start + bs)
        out = model(
            u_idx[sl].to(device),
            i_idx[sl].to(device),
            torch.from_numpy(A[sl]).to(device),
            torch.from_numpy(H[sl]).to(device),
            torch.from_numpy(r[sl]).to(device),
            z=z,
        )
        chunks.append(out["logits"].cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def preflight(model, bundle, A_tr_s, H_tr_s, r_tr, item_offset, device, batch_size: int) -> dict[str, Any]:
    empty_cache()
    u_all, i_all = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    y_all = bundle["train_pairs"]["label"].astype(np.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )
    t0 = time.time()
    status = "OK"
    err = None
    try:
        model.train()
        opt.zero_grad()
        z = model.node_z()
        idx = np.arange(min(batch_size, len(y_all)))
        out = model(
            u_all[idx].to(device),
            i_all[idx].to(device),
            torch.from_numpy(A_tr_s[idx]).to(device),
            torch.from_numpy(H_tr_s[idx]).to(device),
            torch.from_numpy(r_tr[idx]).to(device),
            z=z,
        )
        loss = loss_fn(out["logits"], torch.from_numpy(y_all[idx]).to(device))
        loss.backward()
        opt.step()
    except RuntimeError as e:
        status = "RESOURCE_INFEASIBLE" if "out of memory" in str(e).lower() or "oom" in str(e).lower() else "ERROR"
        err = str(e)[:500]
    mem = mps_bytes()
    unified = host_info().get("unified_memory_bytes") or 0
    peak = max(mem["allocated"], mem["driver"])
    risk = bool(unified and peak > MEM_FRAC * unified)
    return {
        "status": "MEMORY_RISK" if status == "OK" and risk else status,
        "error": err,
        "peak_allocated": mem["allocated"],
        "peak_reserved": mem["driver"],
        "wall_sec": time.time() - t0,
        "unified_memory_bytes": unified,
    }


def recipe_kwargs(kind: str, rec: dict[str, Any], max_epochs_override: int | None = None) -> dict[str, Any]:
    lr0 = float(rec["lr0"])
    if kind == "C0":
        return {
            "max_epochs": int(rec["max_epochs_current"]),
            "patience": int(rec["patience_current"]),
            "min_epochs": 0,
            "use_plateau": False,
            "lr0": lr0,
            "min_lr": lr0 / 16.0,
        }
    return {
        "max_epochs": int(max_epochs_override or max(60, 4 * int(rec["max_epochs_current"]))),
        "patience": 8,
        "min_epochs": 15,
        "use_plateau": True,
        "lr0": lr0,
        "min_lr": lr0 / 16.0,
    }


def train_one(
    *,
    model: FinalHGTModel,
    bundle,
    A_tr_s,
    A_va_s,
    H_tr_s,
    H_va_s,
    r_tr,
    r_va,
    item_offset: int,
    device,
    seed: int,
    ckpt: Path,
    rec: dict[str, Any],
    recipe: str,
    max_epochs_override: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if (ckpt / "model.pt").exists() and (ckpt / "val_logits.npy").exists() and not resume:
        print(f"[reuse] {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
        meta = json.loads((ckpt / "train_meta.json").read_text())
        logits = np.load(ckpt / "val_logits.npy").astype(np.float64)
        return {"meta": meta, "logits": logits, "reused": True}
    kw = recipe_kwargs(recipe, rec, max_epochs_override)
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    batch_size = int(acfg.get("batch_size", 4096))
    u_all, i_all = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    y_all = bundle["train_pairs"]["label"].astype(np.float32)
    n_train = len(y_all)
    va_u, va_i, va_y = bundle["val_pairs"]["user_id"], bundle["val_pairs"]["item_id"], bundle["val_pairs"]["label"]
    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_state = None
    best_ndcg = -1.0
    best_epoch = -1
    left = kw["patience"]
    if resume and (ckpt / "last.pt").exists():
        model.load_state_dict(torch.load(ckpt / "last.pt", map_location=device))
        prev = json.loads((ckpt / "train_meta.json").read_text())
        history = list(prev.get("history", []))
        start_epoch = int(prev.get("final_epoch", 0)) + 1
        best_ndcg = float(prev.get("best_val_NDCG@20_sampled", -1.0))
        best_epoch = int(prev.get("best_epoch", -1))
        print(f"[resume] {ckpt} from epoch {start_epoch}", flush=True)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=kw["lr0"], weight_decay=float(rec["weight_decay"]))
    sched = None
    if kw["use_plateau"]:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=3, threshold=1e-4, min_lr=kw["min_lr"]
        )
    if resume and (ckpt / "opt.pt").exists():
        opt.load_state_dict(torch.load(ckpt / "opt.pt", map_location=device))
        if sched is not None and (ckpt / "sched.pt").exists():
            sched.load_state_dict(torch.load(ckpt / "sched.pt", map_location=device))
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )
    t0 = time.time()
    last_state = None
    for epoch in range(start_epoch, kw["max_epochs"]):
        te = time.time()
        model.train()
        perm = np.random.permutation(n_train)
        opt.zero_grad()
        z_live = model.node_z()
        starts = list(range(0, n_train, batch_size))
        n_b = max(len(starts), 1)
        epoch_loss = 0.0
        for bi, start in enumerate(starts):
            idx = perm[start : start + batch_size]
            z = z_live if bi == 0 else z_live.detach()
            out = model(
                u_all[idx].to(device),
                i_all[idx].to(device),
                torch.from_numpy(A_tr_s[idx]).to(device),
                torch.from_numpy(H_tr_s[idx]).to(device),
                torch.from_numpy(r_tr[idx]).to(device),
                z=z,
            )
            loss = loss_fn(out["logits"], torch.from_numpy(y_all[idx]).to(device))
            if not torch.isfinite(loss):
                abort(f"NaN loss {ckpt} epoch {epoch}")
            (loss / n_b).backward()
            epoch_loss += float(loss.item())
        opt.step()
        del z_live
        empty_cache()
        scores = score_model(model, va_u, va_i, A_va_s, H_va_s, r_va, item_offset, device)
        ndcg = float(sampled_metrics(va_u, va_y, scores)["NDCG@20"])
        lr_now = float(opt.param_groups[0]["lr"])
        if sched is not None:
            sched.step(ndcg)
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / n_b,
                "val_NDCG@20": ndcg,
                "lr": lr_now,
                "sec": time.time() - te,
            }
        )
        print(
            f"[{ckpt.parent.name}/{ckpt.name} s{seed}] epoch {epoch} "
            f"loss={history[-1]['loss']:.4f} NDCG={ndcg:.4f} lr={lr_now:.2e} ({history[-1]['sec']:.1f}s)",
            flush=True,
        )
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ndcg > best_ndcg + MIN_IMPROVE:
            best_ndcg = ndcg
            best_epoch = epoch
            best_state = last_state
            left = kw["patience"]
        elif epoch + 1 >= kw["min_epochs"]:
            left -= 1
            if left <= 0:
                break
    if best_state:
        model.load_state_dict(best_state)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt / "model.pt")
    if last_state:
        torch.save(last_state, ckpt / "last.pt")
    torch.save(opt.state_dict(), ckpt / "opt.pt")
    if sched is not None:
        torch.save(sched.state_dict(), ckpt / "sched.pt")
    scores = score_model(model, va_u, va_i, A_va_s, H_va_s, r_va, item_offset, device)
    np.save(ckpt / "val_logits.npy", scores.astype(np.float32))
    m = sampled_metrics(va_u, va_y, scores)
    meta = {
        "seed": seed,
        "recipe": recipe,
        "best_epoch": best_epoch,
        "final_epoch": history[-1]["epoch"] if history else -1,
        "best_val_NDCG@20_sampled": best_ndcg,
        "final_lr": float(opt.param_groups[0]["lr"]),
        "min_lr": kw["min_lr"],
        "max_epochs": kw["max_epochs"],
        "history": history,
        "seconds": time.time() - t0,
        "metrics": m,
        "params": param_blocks(model),
    }
    write_json(ckpt / "train_meta.json", meta)
    return {"meta": meta, "logits": scores, "reused": False}


def contrast_users(cells0, cells1) -> np.ndarray:
    arrs = []
    for i in range(len(SEEDS)):
        common = sorted(set(cells0[i]["pu"]) & set(cells1[i]["pu"]))
        arrs.append(np.asarray([cells1[i]["pu"][u] - cells0[i]["pu"][u] for u in common], dtype=np.float64))
    return np.concatenate(arrs)


def boot_vs(cells0, cells1) -> dict[str, Any]:
    seed_d = [cells1[i]["metrics"]["NDCG@20"] - cells0[i]["metrics"]["NDCG@20"] for i in range(3)]
    cat = contrast_users(cells0, cells1)
    mu, lo, hi = bootstrap_ci(cat, n=N_BOOT)
    p = signflip_p(cat, n=N_PERM)
    return {
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
        "raw_p": p,
        "practical_class": mag(float(np.mean(seed_d))),
        "mean_ndcg20": float(np.mean([c["metrics"]["NDCG@20"] for c in cells1])),
    }


@torch.no_grad()
def representation_audit(model: FinalHGTModel, graph, device, n_users: int) -> list[dict[str, Any]]:
    """Per-layer node-type norms / cosine on a deterministic sample. Native HGT dim, before readout."""
    core: HGTEncoder = model.encoder.core
    x_dict = {"user": core.user_embed.weight, "entity": core.entity_input(True)}
    rows = []
    rng = np.random.default_rng(REP_SEED)
    for li, conv in enumerate(core.convs):
        x_dict = conv(x_dict, model.encoder._edge_index_dict)
        if li < len(core.convs) - 1:
            x_dict = {k: torch.nn.functional.relu(v) for k, v in x_dict.items()}
        for nt, x in x_dict.items():
            nrm = x.norm(dim=1)
            n = int(x.size(0))
            take = min(512, n)
            idx = rng.choice(n, size=take, replace=False)
            xs = torch.nn.functional.normalize(x[torch.from_numpy(idx).to(device)], dim=1)
            # mean pairwise cosine of a 128-subset to keep it cheap
            m = min(128, take)
            sim = xs[:m] @ xs[:m].T
            iu = torch.triu_indices(m, m, offset=1)
            rows.append(
                {
                    "layer": li,
                    "node_type": nt,
                    "mean_norm": float(nrm.mean()),
                    "std_norm": float(nrm.std(unbiased=False)),
                    "mean_pairwise_cosine": float(sim[iu[0], iu[1]].mean()) if iu.numel() else 0.0,
                    "repr_variance": float(x.var(unbiased=False)),
                    "n_sampled": take,
                }
            )
    return rows


def plot_curves(histories: dict[str, list[list[dict]]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, per_seed in histories.items():
        # mean NDCG across seeds where history exists
        max_e = max((h[-1]["epoch"] for hs in per_seed for h in [hs] if hs), default=0)
        xs = list(range(max_e + 1))
        ys, lrs = [], []
        for e in xs:
            vals = [hs[e]["val_NDCG@20"] for hs in per_seed if e < len(hs)]
            lr = [hs[e]["lr"] for hs in per_seed if e < len(hs)]
            ys.append(float(np.mean(vals)) if vals else np.nan)
            lrs.append(float(np.mean(lr)) if lr else np.nan)
        axes[0].plot(xs, ys, label=name)
        axes[1].plot(xs, lrs, label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("sampled val NDCG@20")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("LR vs epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = dirs()
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n")
    (OUT / "FULLRANK_LOCKED").write_text("LOCKED_NOT_RUN\n")
    print("[race] load", flush=True)
    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    check_overlaps()
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    host = host_info()
    rec = write_audit(d, bundle, host)
    print(f"[race] device={device} host={host.get('gpu_name')}", flush=True)

    p_tr = V2 / "representations" / "LEG_K2_train.npy"
    p_va = V2 / "representations" / "LEG_K2_val.npy"
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

    # scalers fitted on MODEL_TRAIN (from-scratch protocol)
    a_scaler = StandardScaler().fit(A_tr)
    h_scaler = StandardScaler().fit(H_tr)
    A_tr_s, A_va_s = scale_split(a_scaler, A_tr), scale_split(a_scaler, A_va)
    H_tr_s, H_va_s = scale_split(h_scaler, H_tr), scale_split(h_scaler, H_va)
    with (d["audit"] / "a_scaler.pkl").open("wb") as f:
        pickle.dump(a_scaler, f)
    with (d["audit"] / "h_scaler.pkl").open("wb") as f:
        pickle.dump(h_scaler, f)

    def run_family(tag: str, spec: dict, recipe: str, folder: Path, max_ep=None, resume=False):
        cells = []
        metas = []
        for seed in SEEDS:
            print(f"[race] {tag} seed={seed} d={spec['d']} L={spec['layers']} H={spec['heads']} recipe={recipe}", flush=True)
            model = build_race_model(bundle, graph, device, d=spec["d"], layers=spec["layers"], heads=spec["heads"])
            ckpt = folder / "runs" / f"{tag}_seed{seed}"
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
                ckpt=ckpt,
                rec=rec,
                recipe=recipe,
                max_epochs_override=max_ep,
                resume=resume,
            )
            cell = cell_from_scores(va_u, va_y, out["logits"])
            cells.append(cell)
            metas.append(out["meta"])
            print(f"[race] {tag} seed={seed} NDCG={cell['metrics']['NDCG@20']:.6f}", flush=True)
            del model
            empty_cache()
            gc.collect()
        return cells, metas

    # ---- Stage A
    print("[race] STAGE A C0", flush=True)
    c0_cells, c0_meta = run_family("C0_CURRENT_RECIPE", ARCHS["H0"], "C0", d["conv"])
    print("[race] STAGE A C1", flush=True)
    c1_cells, c1_meta = run_family("C1_EXTENDED_64_2L_2H", ARCHS["H0"], "C1", d["conv"])

    def not_converged(metas):
        hits = 0
        for m in metas:
            fe, be = int(m["final_epoch"]), int(m["best_epoch"])
            if be >= fe - 2 and float(m["final_lr"]) > float(m["min_lr"]) + 1e-12:
                hits += 1
        return hits >= 2

    conv_flag = "OK"
    if not_converged(c1_meta):
        conv_flag = "TRAINING_NOT_CONVERGED"
        print("[race] C1 TRAINING_NOT_CONVERGED → extend max_epochs=100", flush=True)
        c1_cells, c1_meta = run_family(
            "C1_EXTENDED_64_2L_2H", ARCHS["H0"], "C1", d["conv"], max_ep=100, resume=True
        )

    conv_boot = boot_vs(c0_cells, c1_cells)
    if conv_boot["mean_seed_delta"] >= STRONG and conv_boot["n_seeds_pos"] >= 2 and conv_boot["ci_entirely_gt0"]:
        conv_verdict = "CONVERGENCE_STRONG_GAIN"
    elif conv_boot["mean_seed_delta"] >= CONFIRM and conv_boot["n_seeds_pos"] >= 2 and conv_boot["ci_entirely_gt0"]:
        conv_verdict = "CONVERGENCE_GAIN"
    else:
        conv_verdict = "NO_CONVERGENCE_GAIN"

    conv_rows = []
    for i, seed in enumerate(SEEDS):
        conv_rows.append(
            {
                "seed": seed,
                "C0_NDCG20": c0_cells[i]["metrics"]["NDCG@20"],
                "C1_NDCG20": c1_cells[i]["metrics"]["NDCG@20"],
                "C1_minus_C0": c1_cells[i]["metrics"]["NDCG@20"] - c0_cells[i]["metrics"]["NDCG@20"],
                "C0_best_epoch": c0_meta[i]["best_epoch"],
                "C1_best_epoch": c1_meta[i]["best_epoch"],
                "C0_final_epoch": c0_meta[i]["final_epoch"],
                "C1_final_epoch": c1_meta[i]["final_epoch"],
                "C1_final_lr": c1_meta[i]["final_lr"],
            }
        )
    write_csv(
        OUT / "CONVERGENCE_RESULTS_BY_SEED.csv",
        conv_rows,
        list(conv_rows[0]),
    )
    write_csv(d["conv"] / "CONVERGENCE_RESULTS_BY_SEED.csv", conv_rows, list(conv_rows[0]))
    write_json(d["conv"] / "CONVERGENCE_BOOTSTRAP.json", {k: v for k, v in conv_boot.items()})
    print(f"[race] CONVERGENCE_VERDICT={conv_verdict} Δ={conv_boot['mean_seed_delta']:+.6f} flag={conv_flag}", flush=True)

    plot_curves(
        {
            "C0": [m["history"] for m in c0_meta],
            "C1": [m["history"] for m in c1_meta],
        },
        d["plots"] / "convergence_curves.png",
        "Stage A sampled NDCG@20 vs epoch",
    )

    # C1 is H0 for the capacity race (matched extended protocol)
    cap_cells: dict[str, list] = {"H0": c1_cells}
    cap_meta: dict[str, list] = {"H0": c1_meta}
    resource_rows = []
    param_rows = []
    repr_rows = []

    # preflight H0..H5
    for hid, spec in ARCHS.items():
        model = build_race_model(bundle, graph, device, d=spec["d"], layers=spec["layers"], heads=spec["heads"])
        pf = preflight(model, bundle, A_tr_s, H_tr_s, r_tr, item_offset, device, int(acfg.get("batch_size", 4096)))
        pb = param_blocks(model)
        resource_rows.append({"arch": hid, "code": spec["code"], **pf, **{f"p_{k}": v for k, v in pb.items()}})
        param_rows.append({"arch": hid, "code": spec["code"], **pb})
        write_json(d["cap"] / f"PREFLIGHT_{hid}.json", {"preflight": pf, "params": pb})
        print(f"[preflight] {hid} status={pf['status']} alloc={pf['peak_allocated']}", flush=True)
        del model
        empty_cache()

    write_csv(OUT / "CAPACITY_RESOURCE_USAGE.csv", resource_rows, list(resource_rows[0]))
    write_csv(d["stats"] / "CAPACITY_RESOURCE_USAGE.csv", resource_rows, list(resource_rows[0]))
    write_csv(OUT / "CAPACITY_PARAMETER_COUNTS.csv", param_rows, list(param_rows[0]))
    write_csv(d["stats"] / "CAPACITY_PARAMETER_COUNTS.csv", param_rows, list(param_rows[0]))

    infeasible = {r["arch"] for r in resource_rows if r["status"] == "RESOURCE_INFEASIBLE"}

    print("[race] STAGE B capacity", flush=True)
    for hid, spec in ARCHS.items():
        if hid == "H0":
            continue
        if hid in infeasible:
            print(f"[race] {hid} RESOURCE_INFEASIBLE — skip", flush=True)
            cap_cells[hid] = None
            cap_meta[hid] = None
            continue
        cells, metas = run_family(spec["code"], spec, "C1", d["cap"])
        cap_cells[hid] = cells
        cap_meta[hid] = metas
        # representation audit on seed 101 checkpoint
        model = build_race_model(bundle, graph, device, d=spec["d"], layers=spec["layers"], heads=spec["heads"])
        model.load_state_dict(torch.load(d["cap"] / "runs" / f"{spec['code']}_seed101" / "model.pt", map_location=device))
        for row in representation_audit(model, graph, device, graph["meta"]["n_users"]):
            repr_rows.append({"arch": hid, **row})
        del model
        empty_cache()

    # H0 representation
    model = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
    model.load_state_dict(
        torch.load(d["conv"] / "runs" / "C1_EXTENDED_64_2L_2H_seed101" / "model.pt", map_location=device)
    )
    for row in representation_audit(model, graph, device, graph["meta"]["n_users"]):
        repr_rows.append({"arch": "H0", **row})
    del model
    empty_cache()
    if repr_rows:
        write_csv(OUT / "CAPACITY_REPRESENTATION_AUDIT.csv", repr_rows, list(repr_rows[0]))
        write_csv(d["stats"] / "CAPACITY_REPRESENTATION_AUDIT.csv", repr_rows, list(repr_rows[0]))

    seed_rows = []
    for i, seed in enumerate(SEEDS):
        row: dict[str, Any] = {"seed": seed, "H0_NDCG20": cap_cells["H0"][i]["metrics"]["NDCG@20"]}
        for hid in ("H1", "H2", "H3", "H4", "H5"):
            if not cap_cells.get(hid):
                row[f"{hid}_NDCG20"] = float("nan")
                row[f"{hid}_minus_H0"] = float("nan")
                continue
            row[f"{hid}_NDCG20"] = cap_cells[hid][i]["metrics"]["NDCG@20"]
            row[f"{hid}_minus_H0"] = cap_cells[hid][i]["metrics"]["NDCG@20"] - cap_cells["H0"][i]["metrics"]["NDCG@20"]
            for mk in ("MRR", "Recall@20", "HitRate@20"):
                row[f"{hid}_{mk}"] = cap_cells[hid][i]["metrics"].get(mk, float("nan"))
        seed_rows.append(row)
    write_csv(OUT / "CAPACITY_RESULTS_BY_SEED.csv", seed_rows, list(seed_rows[0]))
    write_csv(d["cap"] / "CAPACITY_RESULTS_BY_SEED.csv", seed_rows, list(seed_rows[0]))

    boot_rows = []
    rand_rows = []
    holm_in = []
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        if not cap_cells.get(hid):
            continue
        br = boot_vs(cap_cells["H0"], cap_cells[hid])
        br["variant"] = hid
        br["code"] = ARCHS[hid]["code"]
        nds = [cap_cells[hid][i]["metrics"]["NDCG@20"] for i in range(3)]
        br["std_seeds"] = float(np.std(nds, ddof=1))
        br["min_seed"] = float(np.min(nds))
        br["max_seed"] = float(np.max(nds))
        h0s = [cap_cells["H0"][i]["metrics"]["NDCG@20"] for i in range(3)]
        br["HIGH_VARIANCE"] = bool(br["std_seeds"] >= 0.0015 and br["std_seeds"] > 2 * float(np.std(h0s, ddof=1) + 1e-12))
        boot_rows.append(br)
        holm_in.append((hid, br["raw_p"]))
        rand_rows.append({"variant": hid, "raw_p": br["raw_p"], "n_perm": N_PERM})
    holm_rows = holm(holm_in) if holm_in else []
    holm_map = {r["variant"]: r for r in holm_rows}
    for r in boot_rows:
        r["holm_p"] = holm_map[r["variant"]]["holm_p"]
        r["gate"] = bool(
            r["mean_seed_delta"] >= CONFIRM
            and r["n_seeds_pos"] >= 2
            and r["ci_entirely_gt0"]
            and r["holm_p"] < 0.05
        )
        r["strong"] = bool(r["gate"] and r["mean_seed_delta"] >= STRONG)
    bfields = [
        "variant", "code", "mean_seed_delta", "n_seeds_pos", "seed101", "seed202", "seed303",
        "mean", "median", "ci95_lo", "ci95_hi", "frac_gt0", "frac_eq0", "frac_lt0",
        "ci_entirely_gt0", "raw_p", "holm_p", "practical_class", "mean_ndcg20",
        "std_seeds", "min_seed", "max_seed", "HIGH_VARIANCE", "gate", "strong",
    ]
    if boot_rows:
        write_csv(OUT / "CAPACITY_BOOTSTRAP.csv", boot_rows, bfields)
        write_csv(d["stats"] / "CAPACITY_BOOTSTRAP.csv", boot_rows, bfields)
        write_csv(OUT / "CAPACITY_RANDOMIZATION.csv", rand_rows, ["variant", "raw_p", "n_perm"])
        write_csv(OUT / "CAPACITY_HOLM.csv", holm_rows, ["variant", "raw_p", "holm_p", "rank"])
        write_csv(d["stats"] / "CAPACITY_HOLM.csv", holm_rows, ["variant", "raw_p", "holm_p", "rank"])

    qualified = [r for r in boot_rows if r["gate"]]
    winner = "H0"
    winner_row = None
    if qualified:
        qualified.sort(key=lambda r: -r["mean_ndcg20"])
        best = qualified[0]
        for alt in qualified[1:]:
            if abs(alt["mean_ndcg20"] - best["mean_ndcg20"]) < INDIFF_CAP:
                pa = next(p for p in param_rows if p["arch"] == alt["variant"])
                pb = next(p for p in param_rows if p["arch"] == best["variant"])
                if pa["total"] < pb["total"]:
                    best = alt
        winner = best["variant"]
        winner_row = best
        cap_verdict = "HGT_STRONG_CAPACITY_GAIN" if best["strong"] else "HGT_CAPACITY_GAIN"
    else:
        cap_verdict = "NO_HGT_CAPACITY_GAIN"

    cap_hist = {"H0": [m["history"] for m in cap_meta["H0"]]}
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        if cap_meta.get(hid):
            cap_hist[hid] = [m["history"] for m in cap_meta[hid]]
    plot_curves(cap_hist, d["plots"] / "capacity_curves.png", "Capacity race sampled NDCG@20 vs epoch")

    # NDCG vs params / VRAM
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs, ys, names = [], [], []
    for hid, spec in ARCHS.items():
        if not cap_cells.get(hid):
            continue
        p = next(r for r in param_rows if r["arch"] == hid)
        xs.append(p["total"])
        ys.append(float(np.mean([c["metrics"]["NDCG@20"] for c in cap_cells[hid]])))
        names.append(hid)
    axes[0].scatter(xs, ys)
    for x, y, n in zip(xs, ys, names):
        axes[0].annotate(n, (x, y))
    axes[0].set_xlabel("trainable parameters")
    axes[0].set_ylabel("mean sampled NDCG@20")
    vr = [next(r for r in resource_rows if r["arch"] == n)["peak_allocated"] for n in names]
    axes[1].scatter(vr, ys)
    for x, y, n in zip(vr, ys, names):
        axes[1].annotate(n, (x, y))
    axes[1].set_xlabel("preflight peak allocated bytes")
    axes[1].set_ylabel("mean sampled NDCG@20")
    fig.tight_layout()
    fig.savefig(d["plots"] / "ndcg_vs_cost.png", dpi=120)
    plt.close(fig)

    # ---- Stage C
    print("[race] STAGE C reproduction", flush=True)
    final_id = winner
    final_spec = ARCHS[final_id]
    repro_targets = ["H0"] if winner == "H0" else ["H0", winner]
    repro_cells: dict[str, list] = {}
    repro_rows = []
    for hid in repro_targets:
        spec = ARCHS[hid]
        cells, metas = run_family(f"REPRO_{spec['code']}", spec, "C1", d["repro"])
        repro_cells[hid] = cells
        for i, seed in enumerate(SEEDS):
            repro_rows.append(
                {
                    "arch": hid,
                    "seed": seed,
                    "NDCG20": cells[i]["metrics"]["NDCG@20"],
                    "best_epoch": metas[i]["best_epoch"],
                }
            )
    write_csv(OUT / "FINAL_REPRODUCTION.csv", repro_rows, ["arch", "seed", "NDCG20", "best_epoch"])
    write_csv(d["repro"] / "FINAL_REPRODUCTION.csv", repro_rows, ["arch", "seed", "NDCG20", "best_epoch"])

    h0_mean = float(np.mean([c["metrics"]["NDCG@20"] for c in cap_cells["H0"]]))
    if winner != "H0" and winner_row is not None:
        dlt = winner_row["mean_seed_delta"]
        # reproduction direction
        repro_d = float(np.mean([c["metrics"]["NDCG@20"] for c in repro_cells[winner]])) - float(
            np.mean([c["metrics"]["NDCG@20"] for c in repro_cells["H0"]])
        )
        direction_ok = (dlt > 0 and repro_d > 0) or (dlt < 0 and repro_d < 0)
        if dlt >= STRONG and winner_row["n_seeds_pos"] >= 2 and winner_row["ci_entirely_gt0"] and direction_ok:
            sampled_verdict = "FINAL_SAMPLED_GAIN"
        elif dlt >= CONFIRM and winner_row["n_seeds_pos"] >= 2 and winner_row["ci_entirely_gt0"] and direction_ok:
            sampled_verdict = "FINAL_SAMPLED_MATERIAL"
        else:
            sampled_verdict = "MATCHED_H0_REMAINS_WINNER"
            final_id = "H0"
    else:
        sampled_verdict = "MATCHED_H0_REMAINS_WINNER"
        dlt = 0.0
        repro_d = 0.0

    if sampled_verdict == "MATCHED_H0_REMAINS_WINNER":
        final_id = "H0"
        cap_verdict = cap_verdict if winner == "H0" else cap_verdict

    final_code = ARCHS[final_id]["code"]
    final_repro = float(np.mean([c["metrics"]["NDCG@20"] for c in repro_cells[final_id]]))
    by = {r["variant"]: r for r in boot_rows}

    def q(hid):
        if hid not in by:
            return "not run / infeasible"
        r = by[hid]
        return f"{r['mean_seed_delta']:+.6f} class={r['practical_class']} gate={r['gate']}"

    report = f"""# FINAL_HGT_CAPACITY_CONVERGENCE_REPORT

A11 closed at LEG_K2. Primary capacity baseline = C1/H0 under extended convergence.
C1 completeness flag = {conv_flag}.

## Stage A

| seed | C0 | C1 | Δ | C0 best ep | C1 best ep |
|---:|---:|---:|---:|---:|---:|
"""
    for r in conv_rows:
        report += (
            f"| {r['seed']} | {r['C0_NDCG20']:.6f} | {r['C1_NDCG20']:.6f} | "
            f"{r['C1_minus_C0']:+.6f} | {r['C0_best_epoch']} | {r['C1_best_epoch']} |\n"
        )
    report += f"""
Q1. Historical recipe under-converged? **{"yes" if conv_verdict != "NO_CONVERGENCE_GAIN" or conv_flag == "TRAINING_NOT_CONVERGED" else "no material extra from C1"}**.
Q2. Convergence-only gain C1−C0 = {conv_boot['mean_seed_delta']:+.6f} ({conv_verdict}).

## Stage B vs matched H0

Q3. Heads 2→4 (H1): {q("H1")}
Q4. Width 64→128 (H2): {q("H2")}
Q5. Depth 2→3 (H3): {q("H3")}
Q6. Width+heads (H4): {q("H4")}
Q7. Scaled 128/3/4 (H5): {q("H5")}
Q8. Robust? see Holm / n_seeds_pos in CAPACITY_BOOTSTRAP.csv
Q9. Oversmoothing: CAPACITY_REPRESENTATION_AUDIT.csv (2L vs 3L cosine / variance).
Q10. Cost: CAPACITY_PARAMETER_COUNTS.csv and CAPACITY_RESOURCE_USAGE.csv
Q11. Final sampled architecture: {final_code} + A5 + H3 + LEG_K2
Q12. Gain source: convergence {conv_verdict}; capacity {cap_verdict}.

A11_STATUS = CLOSED_LEG_K2
CONVERGENCE_VERDICT = {conv_verdict}
CONVERGENCE_DELTA = {conv_boot['mean_seed_delta']:+.6f}
HGT_CAPACITY_WINNER = {ARCHS[winner]['code']}
HGT_CAPACITY_DELTA_VS_MATCHED_H0 = {0.0 if winner == "H0" else winner_row['mean_seed_delta']:+.6f}
HGT_CAPACITY_VERDICT = {cap_verdict}
FINAL_SAMPLED_ARCHITECTURE = {final_code}+A5+H3+LEG_K2
FINAL_REPRODUCTION_NDCG20 = {final_repro:.6f}
FINAL_SAMPLED_VERDICT = {sampled_verdict}
TEST_STATUS = LOCKED_NOT_RUN
FULLRANK_STATUS = LOCKED_NOT_RUN
"""
    (OUT / "FINAL_HGT_CAPACITY_CONVERGENCE_REPORT.md").write_text(report, encoding="utf-8")
    (d["report"] / "FINAL_HGT_CAPACITY_CONVERGENCE_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    write_json(
        OUT / "MANIFEST.json",
        {
            "timestamp": utc_now(),
            "git_commit": git_commit(),
            "A11_STATUS": "CLOSED_LEG_K2",
            "CONVERGENCE_VERDICT": conv_verdict,
            "CONVERGENCE_DELTA": conv_boot["mean_seed_delta"],
            "HGT_CAPACITY_WINNER": ARCHS[winner]["code"],
            "HGT_CAPACITY_VERDICT": cap_verdict,
            "FINAL_SAMPLED_ARCHITECTURE": f"{final_code}+A5+H3+LEG_K2",
            "FINAL_REPRODUCTION_NDCG20": final_repro,
            "FINAL_SAMPLED_VERDICT": sampled_verdict,
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "FULLRANK_STATUS": "LOCKED_NOT_RUN",
            "infeasible": sorted(infeasible),
            "convergence_flag": conv_flag,
            **{f"{k}_hash": v for k, v in hashes.items()},
        },
    )


if __name__ == "__main__":
    main()
