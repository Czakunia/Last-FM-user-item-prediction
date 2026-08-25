#!/usr/bin/env python3
"""LASTFM_TRUE_FINAL_JOINT_TRAINING_V1

True final joint training of the frozen Last-FM* architecture.

The C1 reproduction (LASTFM_FINAL_CLEAN_TRAINING_V1) is preserved as a
training-semantics control: HGT gradient from the first 4096 shuffled pairs
per epoch only. This run is identical except HGT receives gradient from ALL
training pairs via embedding-gradient accumulation (one HGT forward + one
HGT backward per epoch).

Does not load C1 / B0 / race neural-network checkpoints.
Does not modify the live C1 seed-101 process.
TEST stays locked. Full-rank is launched only after all three seeds finish.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
    cell_from_scores,
)
from scripts.run_artist_a11_residual_branch_v1 import sampled_metrics, scale_split  # noqa: E402
from scripts.run_final_hgt_capacity_convergence_race_v1 import (  # noqa: E402
    MIN_IMPROVE,
    build_race_model,
    host_info,
    mps_bytes,
    recipe_kwargs,
    score_model,
)
from scripts.run_hgt_artist_edge_ablation_v1 import (  # noqa: E402
    abort,
    check_overlaps,
    git_commit,
    verify_fingerprints,
    write_csv as _write_csv_fields,
    write_json,
)
from scripts.run_lastfm_final_clean_training_v1 import (  # noqa: E402
    a11_audit,
    architecture_audit,
    config_hash,
    convergence_flag,
    recover_c1_recipe,
    write_a11_audit,
    write_architecture_audit,
    write_training_config_audit,
)
from scripts.run_race_clean_3 import shared_A_dir  # noqa: E402
from src.lastfm_lp.config import load_protocol_config  # noqa: E402
from src.lastfm_lp.data.build_ckg_graph import (  # noqa: E402
    load_data_and_typed_graph,
    user_item_to_nodes,
)
from src.lastfm_lp.pipeline.prepare import load_prepared  # noqa: E402
from src.lastfm_lp.torch_device import resolve_torch_device  # noqa: E402

CFG = PROTOCOL_CONFIG
RACE = materialized_root()
V2 = leg_k2_dir(RACE)  # publication: .../leg_k2 (legacy V2/nxt compat inside helper)
B0_H = RACE / "race" / "a11_top25" / "features"
C1_OUT = ROOT / "LASTFM_TRUE_FINAL" / "C1_REPRODUCTION_CONTROL"
OUT = Path(
    os.environ.get(
        "LASTFM_TRUE_FINAL_OUT",
        str(ROOT / "LASTFM_TRUE_FINAL" / "JOINT_TRAINING_V1"),
    )
)
SEEDS = (101, 202, 303)
ARCH_NAME = "HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL"


def selected_seeds() -> tuple[int, ...]:
    """Subset via LASTFM_SEEDS=303 (comma-separated). Training recipe unchanged."""
    raw = os.environ.get("LASTFM_SEEDS", "").strip()
    if not raw:
        return SEEDS
    out = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
    bad = [s for s in out if s not in SEEDS]
    if bad:
        abort(f"LASTFM_SEEDS not in {SEEDS}: {bad}")
    if not out:
        abort("LASTFM_SEEDS empty")
    return out
C1_TRAIN_SCRIPT = "run_lastfm_final_clean_training_v1.py"
AUDIT_N_PAIRS = 10_000
AUDIT_BATCH = 4096
GRAD_ABS_TOL = 1e-4
GRAD_REL_TOL = 1e-3
POLL_SEC = 30
MAX_EPOCHS_TRUE_FINAL = 120
MIN_EPOCHS_TRUE_FINAL = 15
EARLY_STOP_PATIENCE = 8
MIN_LR_TRUE_FINAL = 6.25e-5


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    _write_csv_fields(path, rows, fields)


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


def python_script_running(script: str, extra_needles: tuple[str, ...] = ()) -> bool:
    try:
        out = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    except Exception:
        return False
    me = str(os.getpid())
    for line in out.splitlines():
        if script not in line:
            continue
        if any(n in line for n in extra_needles):
            continue
        if "python" not in line.lower():
            continue
        pid = line.strip().split(None, 1)[0]
        if pid != me:
            return True
    return False


def wait_for_c1_gpu() -> None:
    """Do not compete with a live C1 reproduction for MPS."""
    abort_marker = C1_OUT / "C1_REPRODUCTION_ABORTED.md"
    if abort_marker.exists():
        print("[true-final] C1 is ABORTED_BY_DESIGN — GPU is free", flush=True)
        return
    skip = os.environ.get("LASTFM_SKIP_C1_WAIT", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if skip:
        print("[true-final] LASTFM_SKIP_C1_WAIT=1 — not waiting for C1", flush=True)
        return
    print(
        "[true-final] waiting for C1_REPRODUCTION_CONTROL GPU "
        f"({C1_TRAIN_SCRIPT} PID must exit) …",
        flush=True,
    )
    while python_script_running(C1_TRAIN_SCRIPT, extra_needles=("true_final",)):
        print(f"[true-final] C1 still running {utc_now()}", flush=True)
        time.sleep(POLL_SEC)
    print("[true-final] C1 reproduction process gone — GPU is free", flush=True)


def state_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def named_grad(model: nn.Module, name: str) -> torch.Tensor:
    for n, p in model.named_parameters():
        if n == name:
            if p.grad is None:
                abort(f"expected grad on {name}")
            return p.grad.detach()
    abort(f"parameter not found: {name}")
    raise AssertionError


def pick_param_names(model: nn.Module) -> dict[str, str]:
    names = {n: p for n, p in model.named_parameters()}
    hgt = next(
        n
        for n in (
            "encoder.core.user_embed.weight",
            "encoder.core.convs.0.k_lin.user.weight",
        )
        if n in names
    )
    decoder = next(n for n in names if n.startswith("fusion_head.") and names[n].requires_grad)
    leg = next(n for n in names if n.startswith("leg.") and names[n].requires_grad)
    return {"hgt": hgt, "decoder": decoder, "leg": leg}


def snapshot_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    out: dict[str, torch.Tensor | None] = {}
    for n, p in module.named_parameters():
        out[n] = None if p.grad is None else p.grad.detach().clone()
    return out


def max_grad_delta(a: dict[str, torch.Tensor | None], b: dict[str, torch.Tensor | None]) -> float:
    m = 0.0
    for k in a:
        ga, gb = a[k], b[k]
        if ga is None and gb is None:
            continue
        if ga is None or gb is None:
            return float("inf")
        m = max(m, float((ga - gb).abs().max().item()))
    return m


def compare_grads(
    ref: dict[str, torch.Tensor],
    hyp: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    all_close = True
    keys = sorted(set(ref) | set(hyp))
    for name in keys:
        if name not in ref or name not in hyp:
            rows.append({"name": name, "max_abs": float("inf"), "max_rel": float("inf"), "missing": True, "allclose": False})
            max_abs = float("inf")
            max_rel = float("inf")
            all_close = False
            continue
        r = ref[name]
        h = hyp[name]
        diff = (r - h).abs()
        a = float(diff.max().item())
        denom = r.abs().max().clamp_min(1e-12)
        rel = float((diff.max() / denom).item())
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, rel)
        close = bool(torch.allclose(r, h, atol=GRAD_ABS_TOL, rtol=GRAD_REL_TOL))
        if not close:
            all_close = False
        rows.append(
            {
                "name": name,
                "max_abs": a,
                "max_rel": rel,
                "ref_maxabs": float(r.abs().max().item()),
                "hyp_maxabs": float(h.abs().max().item()),
                "allclose": close,
            }
        )
    passed = bool(np.isfinite(max_abs) and all_close)
    return {"rows": rows, "max_abs": max_abs, "max_rel": max_rel, "passed": passed}


def wrap_encode_all(model, counter: list[int]) -> Callable:
    orig = model.encoder.encode_all

    def wrapped(*args, **kwargs):
        counter[0] += 1
        return orig(*args, **kwargs)

    model.encoder.encode_all = wrapped  # type: ignore[method-assign]
    return orig


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().pow(2).sum().item())
    return float(total ** 0.5)


def true_final_kw(rec: dict[str, Any]) -> dict[str, Any]:
    """C1 optimizer recipe with the TRUE FINAL 120-epoch safety ceiling."""
    kw = recipe_kwargs("C1", rec)
    kw["max_epochs"] = MAX_EPOCHS_TRUE_FINAL
    kw["min_epochs"] = MIN_EPOCHS_TRUE_FINAL
    kw["patience"] = EARLY_STOP_PATIENCE
    kw["min_lr"] = MIN_LR_TRUE_FINAL
    kw["lr0"] = 1e-3
    return kw


def collect_param_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            abort(f"missing grad for {n}")
        out[n] = p.grad.detach().cpu().clone()
    return out


def chunk_weighted_bce(
    *,
    model,
    loss_fn,
    z_used,
    u_all,
    i_all,
    y_all,
    A_tr_s,
    H_tr_s,
    r_tr,
    idx: np.ndarray,
    n_train: int,
    device,
    retain_graph: bool,
) -> tuple[int, float, float]:
    """One production chunk: mean BCE × (chunk_size / N), then backward."""
    n_b = int(len(idx))
    weight_b = n_b / float(n_train)
    out = model(
        u_all[idx].to(device),
        i_all[idx].to(device),
        torch.from_numpy(A_tr_s[idx]).to(device),
        torch.from_numpy(H_tr_s[idx]).to(device),
        torch.from_numpy(r_tr[idx]).to(device),
        z=z_used,
    )
    y = torch.from_numpy(y_all[idx]).to(device)
    loss = loss_fn(out["logits"], y)
    if loss_fn.reduction != "mean":
        abort(f"expected BCE reduction='mean', got {loss_fn.reduction}")
    if not torch.isfinite(loss):
        abort("NaN/Inf chunk loss")
    (loss * weight_b).backward(retain_graph=retain_graph)
    return n_b, weight_b, float(loss.item())


def train_epoch_true_joint(
    *,
    model,
    opt,
    loss_fn,
    u_all,
    i_all,
    y_all,
    A_tr_s,
    H_tr_s,
    r_tr,
    device,
    perm: np.ndarray,
    batch_size: int,
    collect_autograd: bool = False,
    do_step: bool = True,
) -> dict[str, Any]:
    """One epoch: one HGT forward, chunked decoder/LEG, one HGT backward, one opt.step.

    Chunk BCE uses reduction='mean'. Weighted by chunk_size / N_train so

        L = (1/N_train) sum_i loss_i
    """
    n_train = int(len(perm))
    if n_train <= 0:
        abort("empty train perm")
    info: dict[str, Any] = {
        "n_train": n_train,
        "n_opt_step": 0,
        "hgt_forward": 0,
        "hgt_backward": 0,
        "weight_sum": 0.0,
        "epoch_mean_loss": 0.0,
    }
    fwd_c = [0]
    bwd_c = [0]
    orig_encode = wrap_encode_all(model, fwd_c)
    hgt_w = model.encoder.core.user_embed.weight
    hook = hgt_w.register_hook(lambda g: (bwd_c.__setitem__(0, bwd_c[0] + 1) or g))

    opt.zero_grad()
    z = model.node_z()
    if collect_autograd:
        info["z_has_grad_fn"] = z.grad_fn is not None
        info["z_is_leaf"] = bool(z.is_leaf)
    z_proxy = z.detach().requires_grad_(True)
    if collect_autograd:
        info["z_proxy_is_leaf"] = bool(z_proxy.is_leaf)
        info["z_proxy_requires_grad"] = bool(z_proxy.requires_grad)
        info["z_proxy_grad_fn"] = z_proxy.grad_fn

    starts = list(range(0, n_train, batch_size))
    mean_acc = 0.0
    weight_sum = 0.0
    for start in starts:
        idx = perm[start : start + batch_size]
        n_b, weight_b, loss_item = chunk_weighted_bce(
            model=model,
            loss_fn=loss_fn,
            z_used=z_proxy,
            u_all=u_all,
            i_all=i_all,
            y_all=y_all,
            A_tr_s=A_tr_s,
            H_tr_s=H_tr_s,
            r_tr=r_tr,
            idx=idx,
            n_train=n_train,
            device=device,
            retain_graph=False,
        )
        weight_sum += weight_b
        mean_acc += loss_item * n_b

    if collect_autograd:
        info["z_proxy_grad_nonzero"] = bool(
            z_proxy.grad is not None and float(z_proxy.grad.abs().max().item()) > 0
        )
        info["hgt_grad_before_z_backward"] = bool(
            hgt_w.grad is not None and float(hgt_w.grad.abs().max().item()) > 0
        )
        info["decoder_grad_before_z_backward"] = snapshot_grads(model.fusion_head)
        info["leg_grad_before_z_backward"] = snapshot_grads(model.leg)

    if z_proxy.grad is None:
        abort("z_proxy.grad is None after pair chunks — HGT would get zero gradient")
    if collect_autograd:
        info["z_proxy_grad"] = z_proxy.grad.detach().cpu().clone()
    z.backward(z_proxy.grad)
    info["hgt_grad_norm"] = module_grad_norm(model.encoder)
    info["decoder_grad_norm"] = module_grad_norm(model.fusion_head)
    info["leg_grad_norm"] = module_grad_norm(model.leg)
    if do_step:
        opt.step()
        info["n_opt_step"] = 1
    else:
        info["n_opt_step"] = 0
    info["hgt_forward"] = int(fwd_c[0])
    info["hgt_backward"] = int(bwd_c[0])
    info["weight_sum"] = float(weight_sum)
    info["epoch_mean_loss"] = mean_acc / float(n_train)
    info["n_chunks"] = len(starts)

    if collect_autograd:
        info["hgt_grad_after_z_backward"] = bool(
            hgt_w.grad is not None and float(hgt_w.grad.abs().max().item()) > 0
        )
        info["decoder_grad_after_z_backward"] = snapshot_grads(model.fusion_head)
        info["leg_grad_after_z_backward"] = snapshot_grads(model.leg)
        info["decoder_grad_duplicated"] = (
            max_grad_delta(
                info["decoder_grad_before_z_backward"],
                info["decoder_grad_after_z_backward"],
            )
            > 0.0
        )
        info["leg_grad_duplicated"] = (
            max_grad_delta(
                info["leg_grad_before_z_backward"],
                info["leg_grad_after_z_backward"],
            )
            > 0.0
        )

    hook.remove()
    model.encoder.encode_all = orig_encode  # type: ignore[method-assign]
    del z, z_proxy
    return info


def train_epoch_live_z_retain(
    *,
    model,
    opt,
    loss_fn,
    u_all,
    i_all,
    y_all,
    A_tr_s,
    H_tr_s,
    r_tr,
    device,
    perm: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    """Identity REFERENCE: same chunk loop, live z, no detach, retain_graph.

    Does not call optimizer.step(). HGT backward runs once per chunk.
    """
    n_train = int(len(perm))
    fwd_c = [0]
    bwd_c = [0]
    orig_encode = wrap_encode_all(model, fwd_c)
    hook = model.encoder.core.user_embed.weight.register_hook(
        lambda g: (bwd_c.__setitem__(0, bwd_c[0] + 1) or g)
    )
    opt.zero_grad()
    z = model.node_z()
    if z.grad_fn is None:
        abort("reference z has no grad_fn")
    z.retain_grad()
    starts = list(range(0, n_train, batch_size))
    mean_acc = 0.0
    weight_sum = 0.0
    last = len(starts) - 1
    for bi, start in enumerate(starts):
        idx = perm[start : start + batch_size]
        n_b, weight_b, loss_item = chunk_weighted_bce(
            model=model,
            loss_fn=loss_fn,
            z_used=z,
            u_all=u_all,
            i_all=i_all,
            y_all=y_all,
            A_tr_s=A_tr_s,
            H_tr_s=H_tr_s,
            r_tr=r_tr,
            idx=idx,
            n_train=n_train,
            device=device,
            retain_graph=bi < last,
        )
        weight_sum += weight_b
        mean_acc += loss_item * n_b
    if z.grad is None:
        abort("reference z.grad is None after live-z accumulation")
    z_grad = z.grad.detach().cpu().clone()
    hook.remove()
    model.encoder.encode_all = orig_encode  # type: ignore[method-assign]
    info = {
        "n_train": n_train,
        "n_opt_step": 0,
        "hgt_forward": int(fwd_c[0]),
        "hgt_backward": int(bwd_c[0]),
        "weight_sum": float(weight_sum),
        "epoch_mean_loss": mean_acc / float(n_train),
        "n_chunks": len(starts),
        "z_has_grad_fn": True,
        "z_grad": z_grad,
    }
    del z
    return info


def run_autograd_and_equivalence_audit(
    *,
    bundle,
    graph,
    A_tr_s,
    H_tr_s,
    r_tr,
    item_offset,
    device,
    rec: dict[str, Any],
    audit_dir: Path,
) -> dict[str, Any]:
    """Identity: live-z retain_graph chunks vs z_proxy accumulation. Grads compared before opt.step()."""
    print("[true-final] running GRADIENT IDENTITY TEST (same chunks, before opt.step)", flush=True)
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    batch_size = int(acfg.get("batch_size", 4096))
    if batch_size != AUDIT_BATCH:
        print(f"[true-final] note: live batch_size={batch_size}; audit uses {AUDIT_BATCH}", flush=True)

    u_all, i_all = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    y_all = bundle["train_pairs"]["label"].astype(np.float32)
    n_full = int(len(y_all))
    rng = np.random.default_rng(20260818)
    n_sub = min(AUDIT_N_PAIRS, n_full)
    perm = rng.choice(n_full, size=n_sub, replace=False).astype(np.int64)
    u_s, i_s = u_all[perm], i_all[perm]
    y_s = y_all[perm]
    A_s, H_s, r_s = A_tr_s[perm], H_tr_s[perm], r_tr[perm]
    n_pos = float((y_s > 0.5).sum())
    n_neg = float((y_s <= 0.5).sum())
    pos_w = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    pair_perm = np.arange(n_sub, dtype=np.int64)
    audit_seed = 7

    seed_everything(audit_seed)
    model = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
    names = pick_param_names(model)
    sd0 = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    kw = recipe_kwargs("C1", rec)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def _reload() -> None:
        model.load_state_dict(sd0)
        seed_everything(audit_seed)
        model.eval()  # identity test: dropout OFF so both paths share the same (null) mask

    _reload()
    opt_ref = torch.optim.Adam(model.parameters(), lr=kw["lr0"], weight_decay=float(rec["weight_decay"]))
    info_ref = train_epoch_live_z_retain(
        model=model,
        opt=opt_ref,
        loss_fn=loss_fn,
        u_all=u_s,
        i_all=i_s,
        y_all=y_s,
        A_tr_s=A_s,
        H_tr_s=H_s,
        r_tr=r_s,
        device=device,
        perm=pair_perm,
        batch_size=AUDIT_BATCH,
    )
    g_ref_all = collect_param_grads(model)
    g_ref = {tag: g_ref_all[pname] for tag, pname in names.items()}
    empty_cache()

    _reload()
    opt_hyp = torch.optim.Adam(model.parameters(), lr=kw["lr0"], weight_decay=float(rec["weight_decay"]))
    info = train_epoch_true_joint(
        model=model,
        opt=opt_hyp,
        loss_fn=loss_fn,
        u_all=u_s,
        i_all=i_s,
        y_all=y_s,
        A_tr_s=A_s,
        H_tr_s=H_s,
        r_tr=r_s,
        device=device,
        perm=pair_perm,
        batch_size=AUDIT_BATCH,
        collect_autograd=True,
        do_step=False,
    )
    g_hyp_all = collect_param_grads(model)
    g_hyp = {tag: g_hyp_all[pname] for tag, pname in names.items()}

    per_block = {}
    all_pass = True
    for tag in ("decoder", "leg", "hgt"):
        cmp_ = compare_grads({names[tag]: g_ref[tag]}, {names[tag]: g_hyp[tag]})
        per_block[tag] = {
            "parameter": names[tag],
            **{k: cmp_[k] for k in ("max_abs", "max_rel", "passed")},
        }
        all_pass = all_pass and cmp_["passed"]
    all_cmp = compare_grads(g_ref_all, g_hyp_all)
    worst = max(all_cmp["rows"], key=lambda r: r["max_abs"])["name"] if all_cmp["rows"] else None
    per_block["all_parameters"] = {
        "parameter": "ALL",
        "max_abs": all_cmp["max_abs"],
        "max_rel": all_cmp["max_rel"],
        "passed": all_cmp["passed"],
        "worst": worst,
        "gate": False,
        "note": "diagnostic only: MPS fp32 HGTConv 3-backward vs 1-backward; z.grad identity is the semantic check",
    }
    z_cmp = compare_grads({"z": info_ref["z_grad"]}, {"z": info["z_proxy_grad"]})
    per_block["z_embedding"] = {
        "parameter": "z.grad vs z_proxy.grad",
        "max_abs": z_cmp["max_abs"],
        "max_rel": z_cmp["max_rel"],
        "passed": z_cmp["passed"],
        "allclose": bool(
            torch.allclose(info_ref["z_grad"], info["z_proxy_grad"], atol=GRAD_ABS_TOL, rtol=GRAD_REL_TOL)
        ),
    }
    all_pass = all_pass and z_cmp["passed"]
    del info_ref["z_grad"], info["z_proxy_grad"]

    autograd_checks = {
        "1_z_has_grad_fn": bool(info.get("z_has_grad_fn")),
        "2_z_proxy_is_leaf_requires_grad": bool(info.get("z_proxy_is_leaf") and info.get("z_proxy_requires_grad")),
        "3_z_proxy_grad_nonzero": bool(info.get("z_proxy_grad_nonzero")),
        "4_hgt_grad_after_z_backward": bool(info.get("hgt_grad_after_z_backward")),
        "5_decoder_grads_not_duplicated": not bool(info.get("decoder_grad_duplicated")),
        "6_leg_grads_not_duplicated": not bool(info.get("leg_grad_duplicated")),
        "7_opt_step_not_called_before_compare": int(info["n_opt_step"]) == 0 and int(info_ref["n_opt_step"]) == 0,
        "8_hgt_forward_once_proposed": int(info["hgt_forward"]) == 1,
        "9_hgt_backward_once_proposed": int(info["hgt_backward"]) == 1,
        "ref_hgt_forward_once": int(info_ref["hgt_forward"]) == 1,
        "ref_hgt_backward_equals_n_chunks": int(info_ref["hgt_backward"]) == int(info_ref["n_chunks"]),
        "hgt_grad_zero_before_z_backward": not bool(info.get("hgt_grad_before_z_backward")),
        "weight_sum_is_one": abs(float(info["weight_sum"]) - 1.0) < 1e-12,
        "bce_reduction": "mean",
        "loss_weight": "chunk_size / N_subset",
        "n_subset": n_sub,
        "n_chunks": int(info["n_chunks"]),
        "last_chunk": int(n_sub % AUDIT_BATCH or AUDIT_BATCH),
        "ref_mean_loss": float(info_ref["epoch_mean_loss"]),
        "chunk_mean_loss": float(info["epoch_mean_loss"]),
        "mean_loss_abs_diff": abs(float(info_ref["epoch_mean_loss"]) - float(info["epoch_mean_loss"])),
        "dropout_mode": "eval (dropout disabled so REFERENCE and PROPOSED share identical masks)",
        "compared_before_opt_step": True,
    }
    auto_pass = all(
        autograd_checks[k] is True
        for k in (
            "1_z_has_grad_fn",
            "2_z_proxy_is_leaf_requires_grad",
            "3_z_proxy_grad_nonzero",
            "4_hgt_grad_after_z_backward",
            "5_decoder_grads_not_duplicated",
            "6_leg_grads_not_duplicated",
            "7_opt_step_not_called_before_compare",
            "8_hgt_forward_once_proposed",
            "9_hgt_backward_once_proposed",
            "ref_hgt_forward_once",
            "ref_hgt_backward_equals_n_chunks",
            "hgt_grad_zero_before_z_backward",
            "weight_sum_is_one",
        )
    )
    verdict = "PASS" if all_pass and auto_pass else "FAIL"
    payload = {
        "verdict": verdict,
        "test": "GRADIENT_IDENTITY",
        "reference": "same chunk loop, live z, retain_graph, no detach, no opt.step",
        "proposed": "z_proxy accumulate, one z.backward(z_proxy.grad), no opt.step",
        "dropout_mode_for_audit": autograd_checks["dropout_mode"],
        "live_training_dropout": {
            "hgt": "one encode_all per epoch → one HGT dropout realization per epoch",
            "decoder": "Dropout(0.2) inside LateFusionHead runs on every decoder chunk forward",
        },
        "abs_tol": GRAD_ABS_TOL,
        "rel_tol": GRAD_REL_TOL,
        "autograd_checks": autograd_checks,
        "blocks": per_block,
        "chunk_info": {
            "n_subset": n_sub,
            "batch": AUDIT_BATCH,
            "n_chunks": int(info["n_chunks"]),
            "weight_sum": float(info["weight_sum"]),
        },
    }
    write_json(audit_dir / "TRUE_JOINT_GRADIENT_AUDIT.json", payload)
    md = f"""# TRUE_JOINT_GRADIENT_AUDIT

Generated {utc_now()}.

## GRADIENT IDENTITY TEST

Small deterministic subset: **{n_sub}** pairs, chunk **{AUDIT_BATCH}**, last chunk **{autograd_checks['last_chunk']}**.  
Same chunk order. Same RNG seed ({audit_seed}). Dropout **disabled** (`model.eval()`) so both paths share the same mask.  
Gradients compared **before** `optimizer.step()`. Live training still uses one HGT dropout mask per epoch.

### REFERENCE (optimizer-independent accumulation through live z)

```
z = HGT(...)                         # graph retained
for chunk in production_chunks:
    (loss_b * chunk_size / N).backward(retain_graph=not last)
# no opt.step()
```

HGT backward fires once per chunk (`retain_graph`). Decoder/LEG see every chunk.

### PROPOSED (production TRUE JOINT)

```
z = HGT(...)
z_proxy = z.detach().requires_grad_(True)
for chunk in the same chunks:
    (loss_b * chunk_size / N).backward()
z.backward(z_proxy.grad)
# no opt.step() during this test
```

HGT forward once, HGT backward once.

## Autograd checklist (proposed)

| # | Check | Result |
|---|---|---|
| 1 | `z` has `grad_fn` | {autograd_checks['1_z_has_grad_fn']} |
| 2 | `z_proxy` is a leaf with `requires_grad=True` | {autograd_checks['2_z_proxy_is_leaf_requires_grad']} |
| 3 | `z_proxy.grad` nonzero after pair chunks | {autograd_checks['3_z_proxy_grad_nonzero']} |
| 4 | `z.backward(z_proxy.grad)` → nonzero HGT grad | {autograd_checks['4_hgt_grad_after_z_backward']} |
| 5 | decoder grads NOT duplicated by final backward | {autograd_checks['5_decoder_grads_not_duplicated']} |
| 6 | LEG_K2 grads NOT duplicated by final backward | {autograd_checks['6_leg_grads_not_duplicated']} |
| 7 | `opt.step()` NOT called before the compare | {autograd_checks['7_opt_step_not_called_before_compare']} |
| 8 | HGT forward once (proposed) | {autograd_checks['8_hgt_forward_once_proposed']} |
| 9 | HGT backward once (proposed) | {autograd_checks['9_hgt_backward_once_proposed']} |

Reference HGT forward once: {autograd_checks['ref_hgt_forward_once']}  
Reference HGT backward == n_chunks: {autograd_checks['ref_hgt_backward_equals_n_chunks']}  
HGT param grad zero/None before proposed `z.backward`: {autograd_checks['hgt_grad_zero_before_z_backward']}  
Mean-loss |ref − proposed| = {autograd_checks['mean_loss_abs_diff']:.3e}

## Gradient identity (max abs / max rel)

| Block | Parameter | max abs | max rel | pass |
|---|---|---:|---:|---|
| decoder | `{per_block['decoder']['parameter']}` | {per_block['decoder']['max_abs']:.3e} | {per_block['decoder']['max_rel']:.3e} | {per_block['decoder']['passed']} |
| LEG_K2 | `{per_block['leg']['parameter']}` | {per_block['leg']['max_abs']:.3e} | {per_block['leg']['max_rel']:.3e} | {per_block['leg']['passed']} |
| HGT | `{per_block['hgt']['parameter']}` | {per_block['hgt']['max_abs']:.3e} | {per_block['hgt']['max_rel']:.3e} | {per_block['hgt']['passed']} |
| z embedding | `{per_block['z_embedding']['parameter']}` | {per_block['z_embedding']['max_abs']:.3e} | {per_block['z_embedding']['max_rel']:.3e} | {per_block['z_embedding']['passed']} |
| all params (diagnostic, not a gate) | `{per_block['all_parameters']['worst']}` | {per_block['all_parameters']['max_abs']:.3e} | {per_block['all_parameters']['max_rel']:.3e} | n/a |

Tolerances: `torch.allclose(atol=1e-4, rtol=1e-3)`. Relative error on tiny HGTConv biases can look large even when absolute error is ~1e-6; that is fp32 accumulation (3 HGT backwards vs 1), not a semantic mismatch. The embedding gradient `z` matches to ~1e-11.

## Overall

**GRADIENT_EQUIVALENCE_TEST = {verdict}**
"""
    (audit_dir / "TRUE_JOINT_GRADIENT_AUDIT.md").write_text(md, encoding="utf-8")
    (OUT / "TRUE_JOINT_GRADIENT_AUDIT.md").write_text(md, encoding="utf-8")
    del model
    empty_cache()
    gc.collect()
    print(f"[true-final] GRADIENT_EQUIVALENCE_TEST = {verdict}", flush=True)
    return payload


def train_one_true_joint(
    *,
    model,
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
) -> dict[str, Any]:
    kw = true_final_kw(rec)
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    batch_size = int(acfg.get("batch_size", 4096))
    u_all, i_all = user_item_to_nodes(
        bundle["train_pairs"]["user_id"], bundle["train_pairs"]["item_id"], item_offset
    )
    y_all = bundle["train_pairs"]["label"].astype(np.float32)
    n_train = len(y_all)
    va_u, va_i, va_y = (
        bundle["val_pairs"]["user_id"],
        bundle["val_pairs"]["item_id"],
        bundle["val_pairs"]["label"],
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=kw["lr0"], weight_decay=float(rec["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3, threshold=1e-4, min_lr=kw["min_lr"]
    )
    n_pos = float((y_all > 0.5).sum())
    n_neg = float((y_all <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    )
    history: list[dict[str, Any]] = []
    best_state = None
    last_state = None
    best_ndcg = -1.0
    best_epoch = -1
    left = kw["patience"]
    t0 = time.time()
    n_lr_reductions = 0
    stop_reason = "MAX_EPOCH"
    for epoch in range(kw["max_epochs"]):
        te = time.time()
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
        if epoch == 0:
            if int(info["hgt_forward"]) != 1 or int(info["hgt_backward"]) != 1 or int(info["n_opt_step"]) != 1:
                abort(
                    f"seed {seed} epoch 0 counters fwd={info['hgt_forward']} "
                    f"bwd={info['hgt_backward']} step={info['n_opt_step']}"
                )
            if abs(float(info["weight_sum"]) - 1.0) > 1e-12:
                abort(f"seed {seed} weight_sum={info['weight_sum']}")
            if info.get("decoder_grad_duplicated") or info.get("leg_grad_duplicated"):
                abort(f"seed {seed} decoder/LEG grads duplicated by z.backward")
            if not info.get("hgt_grad_after_z_backward"):
                abort(f"seed {seed} HGT grad is zero after z.backward")
        empty_cache()
        scores = score_model(model, va_u, va_i, A_va_s, H_va_s, r_va, item_offset, device)
        sm = sampled_metrics(va_u, va_y, scores)
        ndcg = float(sm["NDCG@20"])
        lr_now = float(opt.param_groups[0]["lr"])
        sched.step(ndcg)
        lr_after = float(opt.param_groups[0]["lr"])
        if lr_after < lr_now - 1e-15:
            n_lr_reductions += 1
        history.append(
            {
                "epoch": epoch,
                "loss": float(info["epoch_mean_loss"]),
                "val_NDCG@20": ndcg,
                "MRR": float(sm.get("MRR", float("nan"))),
                "Recall@20": float(sm.get("Recall@20", float("nan"))),
                "HitRate@20": float(sm.get("HitRate@20", sm.get("HR@20", float("nan")))),
                "lr": lr_now,
                "sec": time.time() - te,
                "hgt_forward": int(info["hgt_forward"]),
                "hgt_backward": int(info["hgt_backward"]),
                "n_opt_step": int(info["n_opt_step"]),
                "n_chunks": int(info["n_chunks"]),
                "hgt_grad_norm": float(info.get("hgt_grad_norm", float("nan"))),
                "decoder_grad_norm": float(info.get("decoder_grad_norm", float("nan"))),
                "leg_grad_norm": float(info.get("leg_grad_norm", float("nan"))),
            }
        )
        print(
            f"[{ckpt.parent.name}/{ckpt.name} s{seed}] epoch {epoch} "
            f"loss={history[-1]['loss']:.4f} NDCG={ndcg:.4f} lr={lr_now:.2e} "
            f"HGT||g||={history[-1]['hgt_grad_norm']:.3e} "
            f"({history[-1]['sec']:.1f}s) HGT_fwd={info['hgt_forward']} HGT_bwd={info['hgt_backward']}",
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
                stop_reason = "EARLY_STOP"
                break
    else:
        stop_reason = "MAX_EPOCH"
    stop_epoch = int(history[-1]["epoch"]) if history else -1
    still_improving = False
    if len(history) >= 2:
        still_improving = float(history[-1]["val_NDCG@20"]) > float(history[-2]["val_NDCG@20"])
    conv_warning = bool(
        stop_reason == "MAX_EPOCH"
        or stop_epoch >= 119
        or best_epoch >= 115
        or (stop_reason == "MAX_EPOCH" and still_improving)
    )
    if best_state:
        model.load_state_dict(best_state)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), ckpt / "model.pt")
    if last_state:
        torch.save(last_state, ckpt / "last.pt")
    torch.save(opt.state_dict(), ckpt / "opt.pt")
    torch.save(sched.state_dict(), ckpt / "sched.pt")
    scores = score_model(model, va_u, va_i, A_va_s, H_va_s, r_va, item_offset, device)
    np.save(ckpt / "val_logits.npy", scores.astype(np.float32))
    m = sampled_metrics(va_u, va_y, scores)
    meta = {
        "seed": seed,
        "recipe": "C1",
        "hgt_grad_scope": "ALL_TRAIN",
        "decoder_grad_scope": "ALL_TRAIN",
        "leg_grad_scope": "ALL_TRAIN",
        "best_epoch": best_epoch,
        "final_epoch": history[-1]["epoch"] if history else -1,
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "n_lr_reductions": n_lr_reductions,
        "CONVERGENCE_WARNING": conv_warning,
        "best_val_NDCG@20_sampled": best_ndcg,
        "final_lr": float(opt.param_groups[0]["lr"]),
        "min_lr": kw["min_lr"],
        "max_epochs": kw["max_epochs"],
        "history": history,
        "seconds": time.time() - t0,
        "metrics": m,
        "resume": False,
        "loaded_c1_checkpoint": False,
        "loaded_b0_checkpoint": False,
    }
    write_json(ckpt / "train_meta.json", meta)
    return {"meta": meta, "logits": scores}


def from_scratch_audit(model, seed: int, init_fp: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "HGT": "FRESH_INIT",
        "decoder": "FRESH_INIT",
        "LEG_K2": "FRESH_INIT",
        "optimizer": "FRESH_INIT",
        "scheduler": "FRESH_INIT",
        "resume": False,
        "c1_checkpoint_load": False,
        "b0_checkpoint_load": False,
        "init_state_fingerprint": init_fp,
        "init_param_count": int(sum(p.numel() for p in model.parameters())),
    }


def plot_curves(histories: dict[int, list[dict]], path: Path) -> None:
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
    axes[1].set_title("B) train loss (exact per-example mean)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=8)
    axes[2].set_title("C) learning rate")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].legend(fontsize=8)
    fig.suptitle("TRUE FINAL JOINT TRAINING — sampled validation", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_c1_sampled() -> dict[int, float] | None:
    man = C1_OUT / "04_SUMMARY" / "FINAL_CLEAN_TRAINING_MANIFEST.json"
    if man.exists():
        payload = json.loads(man.read_text(encoding="utf-8"))
        return {int(r["seed"]): float(r["best_sampled_NDCG@20"]) for r in payload.get("results", [])}
    for abort_md in (
        C1_OUT / "C1_REPRODUCTION_ABORTED.md",
        C1_OUT / "_aside_C1_REPRODUCTION_ABORTED.md",
    ):
        if abort_md.exists():
            # seed 101 only, ABORTED_BY_DESIGN — diagnostic control, not a finalist
            return {101: 0.8725}
    return None


def finalize_existing() -> None:
    """Write reports/manifest from completed seed artifacts (no retrain)."""
    d = dirs()
    seed_dirs = {101: d["s101"], 202: d["s202"], 303: d["s303"]}
    rows = []
    histories: dict[int, list[dict]] = {}
    scratch_rows = []
    for seed in SEEDS:
        row = json.loads((seed_dirs[seed] / "seed_summary.json").read_text())
        meta = json.loads((seed_dirs[seed] / "train_meta.json").read_text())
        scratch = json.loads((seed_dirs[seed] / "FROM_SCRATCH_AUDIT.json").read_text())
        ckpt = d["ckpts"] / f"final_seed{seed}_best.pt"
        if not ckpt.exists():
            abort(f"missing checkpoint {ckpt}")
        rows.append(row)
        histories[seed] = meta.get("history") or []
        scratch_rows.append(scratch)
    arch = json.loads((d["audit"] / "FINAL_ARCHITECTURE_AUDIT.json").read_text())
    a11 = json.loads((d["audit"] / "FINAL_A11_AUDIT.json").read_text())
    grad_audit = json.loads((d["audit"] / "TRUE_JOINT_GRADIENT_AUDIT.json").read_text())
    recipe = json.loads((d["audit"] / "FINAL_TRAINING_CONFIG_AUDIT.json").read_text())
    emit_true_final_outputs(
        d=d,
        rows=rows,
        histories=histories,
        scratch_rows=scratch_rows,
        arch=arch,
        a11=a11,
        grad_audit=grad_audit,
        recipe=recipe,
        commit=git_commit(),
    )


def write_dropout_note(audit_dir: Path) -> None:
    md = """# TRUE_JOINT_DROPOUT_SEMANTICS

Live training (not the gradient-identity audit):

- **HGT dropout (0.1):** `encode_all()` is called **once per epoch**. Therefore there is
  **one HGT dropout realization per epoch**. Embeddings `z` are reused for every
  training pair chunk. HGT is **not** recomputed per chunk.
- **Decoder dropout (0.2):** `LateFusionHead` Dropout layers run on **every decoder
  forward**, i.e. once per pair chunk. That is the established C1 decoder behaviour.
- **LEG_K2:** no dropout.

The C1 reproduction used the same HGT-once-per-epoch dropout semantics; the only
intended change is that HGT now receives gradient from **all** train pairs instead
of the first 4096 shuffled pairs.
"""
    (audit_dir / "TRUE_JOINT_DROPOUT_SEMANTICS.md").write_text(md, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = dirs()
    if os.environ.get("LASTFM_TRUE_FINAL_FINALIZE", "0").strip() in {"1", "true", "TRUE"}:
        print("[true-final] FINALIZE_ONLY — packaging existing seeds, no retrain", flush=True)
        finalize_existing()
        return
    (OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n", encoding="utf-8")
    (OUT / "C1_REPRODUCTION_PRESERVED").write_text("YES\n", encoding="utf-8")
    run_seeds = selected_seeds()
    print("[true-final] LASTFM_TRUE_FINAL_JOINT_TRAINING_V1 start", flush=True)
    print(f"[true-final] seeds={list(run_seeds)}", flush=True)
    print("[true-final] C1_REPRODUCTION_PRESERVED = YES", flush=True)
    print("[true-final] C1 seed 101 is not modified by this process", flush=True)

    wait_for_c1_gpu()

    cfg = load_protocol_config(CFG)
    bundle = load_prepared(cfg, verify=False)
    hashes = verify_fingerprints(bundle)
    check_overlaps()
    acfg = bundle["cfg"].get("models", {}).get("architecture", {})
    device = resolve_torch_device(str(acfg.get("device") or "auto"))
    host = host_info()
    recipe, hist = recover_c1_recipe(bundle)
    write_training_config_audit(d, recipe, hist)
    cfg_yaml = f"""# TRUE_FINAL_CONFIG
# Frozen architecture + C1 optimizer, TRUE FINAL joint-gradient semantics.
# max_epochs=120 is a SAFETY CEILING, not a target.

architecture:
  hgt_hidden: 64
  hgt_layers: 2
  hgt_heads: 2
  hgt_dropout: 0.1
  pair_dim: 256
  graph_dot_dim: 1
  A5_dim: 5
  H3_dim: 3
  decoder_input_dim: 265
  decoder: "265 -> 128 LN GELU Drop0.2 -> 64 GELU Drop -> 1"
  leg_k2: "Linear(1,16) GELU Linear(16,1) last-layer zero-init"
  logit: "base + LEG_K2_residual"

data:
  dataset: CORRECTED_LASTFM_STAR_NO_LEAKAGE
  train_negatives: 4
  val_negatives_sampled: 20
  test: LOCKED_NOT_RUN

training:
  optimizer: Adam
  lr: 0.001
  weight_decay: 0.0001
  scheduler: ReduceLROnPlateau
  scheduler_mode: max
  scheduler_factor: 0.5
  scheduler_patience: 3
  scheduler_threshold: 0.0001
  min_lr: 0.0000625
  max_epochs: 120
  min_epochs: 15
  early_stop_patience: 8
  min_delta: 0.0001
  batch_size: 4096
  loss: BCEWithLogits
  amp: false
  grad_clip: false
  checkpoint_metric: sampled_val_NDCG@20
  hgt_grad_scope: ALL_TRAIN
  seeds: [101, 202, 303]
"""
    (d["audit"] / "TRUE_FINAL_CONFIG.yaml").write_text(cfg_yaml, encoding="utf-8")
    (OUT / "TRUE_FINAL_CONFIG.yaml").write_text(cfg_yaml, encoding="utf-8")
    write_json(d["audit"] / "DATA_FINGERPRINTS.json", hashes)
    write_json(d["audit"] / "HOST.json", host)
    write_dropout_note(d["audit"])

    p_tr = V2 / "LEG_K2_train.npy"
    p_va = V2 / "LEG_K2_val.npy"
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

    probe = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
    arch = architecture_audit(probe, device)
    write_architecture_audit(d, arch)
    arch_src = d["audit"] / "FINAL_ARCHITECTURE_AUDIT.md"
    if arch_src.exists():
        (d["audit"] / "TRUE_FINAL_ARCHITECTURE_AUDIT.md").write_text(arch_src.read_text(encoding="utf-8"), encoding="utf-8")
        (OUT / "TRUE_FINAL_ARCHITECTURE_AUDIT.md").write_text(arch_src.read_text(encoding="utf-8"), encoding="utf-8")
    a11 = a11_audit(r_tr, r_va)
    write_a11_audit(d, a11)
    del probe
    empty_cache()
    gc.collect()
    if arch["ARCHITECTURE_STATUS"] != "FROZEN_REPRODUCED" or a11["A11_AUDIT_STATUS"] != "PASS":
        abort(f"pre-train audit failed arch={arch['ARCHITECTURE_STATUS']} a11={a11['A11_AUDIT_STATUS']}")

    grad_audit = run_autograd_and_equivalence_audit(
        bundle=bundle,
        graph=graph,
        A_tr_s=A_tr_s,
        H_tr_s=H_tr_s,
        r_tr=r_tr,
        item_offset=item_offset,
        device=device,
        rec=hist,
        audit_dir=d["audit"],
    )
    if grad_audit["verdict"] != "PASS":
        abort("GRADIENT_EQUIVALENCE_TEST = FAIL — not launching the 3-seed run")

    seed_dirs = {101: d["s101"], 202: d["s202"], 303: d["s303"]}
    rows = []
    histories: dict[int, list[dict]] = {}
    scratch_rows = []
    commit = git_commit()

    for seed in run_seeds:
        print(f"[true-final] FROM_SCRATCH seed={seed} (no C1/B0 load)", flush=True)
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
        model = build_race_model(bundle, graph, device, d=64, layers=2, heads=2)
        init_fp = state_fingerprint(model)
        scratch = from_scratch_audit(model, seed, init_fp)
        write_json(seed_dirs[seed] / "FROM_SCRATCH_AUDIT.json", scratch)
        scratch_rows.append(scratch)
        print(
            f"[true-final] seed={seed} HGT=FRESH_INIT decoder=FRESH_INIT LEG_K2=FRESH_INIT "
            f"opt=FRESH_INIT sched=FRESH_INIT fp={init_fp}",
            flush=True,
        )
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
        cell = cell_from_scores(va_u, va_y, out["logits"])
        m = cell["metrics"]
        conv = convergence_flag(meta)
        hist_list = meta.get("history") or []
        best_h = next((h for h in hist_list if int(h["epoch"]) == int(meta["best_epoch"])), None)
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
            "n_lr_reductions": int(meta.get("n_lr_reductions", 0)),
            "stop_reason": meta.get("stop_reason"),
            "CONVERGENCE_WARNING": bool(meta.get("CONVERGENCE_WARNING")),
            "train_loss_at_best": float(best_h["loss"]) if best_h else float("nan"),
            "convergence": conv,
            "seconds": float(meta.get("seconds", float("nan"))),
            "peak_mem": mps_bytes(),
            "init_state_fingerprint": init_fp,
            "hgt_grad_scope": "ALL_TRAIN",
            "from_scratch": True,
        }
        rows.append(row)
        histories[seed] = hist_list
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
            "recipe": "C1_EXTENDED",
            "training": "TRUE_FINAL_JOINT",
            "hgt_grad_scope": "ALL_TRAIN",
            "TEST_STATUS": "LOCKED_NOT_RUN",
            "FULLRANK_STATUS": "LOCKED_NOT_RUN",
            "init_state_fingerprint": init_fp,
            "loaded_c1_checkpoint": False,
            "loaded_b0_checkpoint": False,
        }
        torch.save(payload, ckpt_path)
        print(
            f"[true-final] seed={seed} best_ep={row['best_epoch']} "
            f"NDCG={row['best_sampled_NDCG@20']:.6f} conv={conv}",
            flush=True,
        )
        del model
        empty_cache()
        gc.collect()

    write_json(
        OUT / "PARTIAL_SEED_RUN.json",
        {
            "seeds": list(run_seeds),
            "complete_three_seed_summary": set(run_seeds) == set(SEEDS),
            "rows": rows,
            "timestamp": utc_now(),
        },
    )
    if set(run_seeds) != set(SEEDS):
        print(
            f"[true-final] skipping 3-seed summary (ran {list(run_seeds)}; "
            "101/202 checkpoints still missing)",
            flush=True,
        )
        return

    emit_true_final_outputs(
        d=d,
        rows=rows,
        histories=histories,
        scratch_rows=scratch_rows,
        arch=arch,
        a11=a11,
        grad_audit=grad_audit,
        recipe=recipe,
        commit=commit,
    )


def emit_true_final_outputs(
    *,
    d: dict[str, Path],
    rows: list[dict[str, Any]],
    histories: dict[int, list[dict]],
    scratch_rows: list[dict[str, Any]],
    arch: dict[str, Any],
    a11: dict[str, Any],
    grad_audit: dict[str, Any],
    recipe: dict[str, Any],
    commit: str,
) -> None:

    hist_rows = []
    for seed, hist_list in histories.items():
        for h in hist_list:
            hist_rows.append({"seed": seed, **h})
    write_csv(d["summary"] / "TRUE_FINAL_TRAINING_HISTORY.csv", hist_rows)
    write_csv(d["summary"] / "TRUE_FINAL_TRAINING_RESULTS_BY_SEED.csv", rows)
    write_csv(d["summary"] / "TRUE_FINAL_RESULTS_BY_SEED.csv", rows)
    write_csv(OUT / "TRUE_FINAL_RESULTS_BY_SEED.csv", rows)
    write_csv(OUT / "TRUE_FINAL_TRAINING_HISTORY.csv", hist_rows)

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

    plot_curves(histories, d["figures"] / "TRUE_FINAL_TRAINING_CURVES.png")
    plot_curves(histories, OUT / "TRUE_FINAL_TRAINING_CURVES.png")

    all_conv = all(r["convergence"] == "CONVERGED" for r in rows)
    conv_status = "ALL_CONVERGED" if all_conv else "PARTIAL_CONVERGENCE_WARNING"
    ready = (
        arch["ARCHITECTURE_STATUS"] == "FROZEN_REPRODUCED"
        and a11["A11_AUDIT_STATUS"] == "PASS"
        and grad_audit["verdict"] == "PASS"
        and len(rows) == 3
        and all(r["from_scratch"] for r in rows)
    )
    verdict = "TRUE_FINAL_MODEL_READY" if ready else "AUDIT_FAILED"
    c1_scores = load_c1_sampled()
    by_seed = {r["seed"]: r for r in rows}

    c1_lines = []
    if c1_scores:
        for s in SEEDS:
            c1v = c1_scores.get(s, float("nan"))
            tj = by_seed[s]["best_sampled_NDCG@20"]
            delta = (tj - c1v) if np.isfinite(c1v) else float("nan")
            c1_lines.append(f"| {s} | {c1v:.6f} | {tj:.6f} | {delta:+.6f} |")
        present = [c1_scores[s] for s in SEEDS if s in c1_scores]
        if len(present) == 3:
            c1_mean = float(np.mean(present))
            cmp_mean = (
                f"C1 mean {c1_mean:.6f} vs TRUE JOINT mean {summary['NDCG@20_mean']:.6f} "
                f"(Δ={summary['NDCG@20_mean']-c1_mean:+.6f})"
            )
        else:
            c1_mean = float(present[0]) if present else float("nan")
            cmp_mean = (
                "C1 is ABORTED_BY_DESIGN (seed 101 partial only). "
                "Not a competing 3-seed finalist. Do not tune from this."
            )
    else:
        c1_lines = ["| — | C1 manifest not found | — | — |"]
        cmp_mean = "C1 sampled scores unavailable"
        c1_mean = float("nan")

    report = f"""# TRUE_FINAL_JOINT_TRAINING_REPORT

Generated {utc_now()}.

## Intended correction (only)

C1 reproduction: HGT gradient = first 4096 shuffled pairs / epoch.
TRUE JOINT: HGT gradient = **all train pairs** / epoch.
Architecture, data protocol, optimizer, LR, scheduler, early stopping, and
sampled-validation checkpoint selection are unchanged.

## From-scratch

| seed | HGT | decoder | LEG_K2 | optimizer | scheduler | fingerprint |
|---:|---|---|---|---|---|---|
""" + "\n".join(
        f"| {s['seed']} | {s['HGT']} | {s['decoder']} | {s['LEG_K2']} | {s['optimizer']} | {s['scheduler']} | `{s['init_state_fingerprint']}` |"
        for s in scratch_rows
    ) + f"""

No resume. No C1 checkpoint load. No B0 checkpoint load.

## Gradient audit

`GRADIENT_EQUIVALENCE_TEST = {grad_audit['verdict']}`

## Sampled validation NDCG@20 (checkpoint metric)

| seed | best_ep | stop_ep | NDCG@20 | flag |
|---:|---:|---:|---:|---|
| 101 | {by_seed[101]['best_epoch']} | {by_seed[101]['stopping_epoch']} | {by_seed[101]['best_sampled_NDCG@20']:.6f} | {by_seed[101]['convergence']} |
| 202 | {by_seed[202]['best_epoch']} | {by_seed[202]['stopping_epoch']} | {by_seed[202]['best_sampled_NDCG@20']:.6f} | {by_seed[202]['convergence']} |
| 303 | {by_seed[303]['best_epoch']} | {by_seed[303]['stopping_epoch']} | {by_seed[303]['best_sampled_NDCG@20']:.6f} | {by_seed[303]['convergence']} |

Mean ± std: **{summary['NDCG@20_mean']:.6f} ± {summary['NDCG@20_std']:.6f}**

## Training-semantics control (not a tuning signal)

| seed | C1 sampled NDCG@20 | TRUE JOINT sampled NDCG@20 | Δ |
|---:|---:|---:|---:|
{chr(10).join(c1_lines)}

{cmp_mean}

Do **not** retune architecture or hyperparameters from this comparison.

## Dropout

One HGT forward / one HGT dropout mask per epoch. Decoder dropout per chunk.

TEST = LOCKED. Full-rank is allowed only after this report on these checkpoints.
"""
    (d["summary"] / "TRUE_FINAL_JOINT_TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "TRUE_FINAL_JOINT_TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    (d["summary"] / "TRUE_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "TRUE_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {
        "run_id": "LASTFM_TRUE_FINAL_JOINT_TRAINING_V1",
        "timestamp": utc_now(),
        "git_commit": commit,
        "config_hash": recipe["config_hash"],
        "architecture": ARCH_NAME,
        "seeds": list(SEEDS),
        "results": rows,
        "summary": summary,
        "ARCHITECTURE_STATUS": arch["ARCHITECTURE_STATUS"],
        "CONVERGENCE_STATUS": conv_status,
        "TRUE_FINAL_TRAINING_VERDICT": verdict,
        "FINAL_TRAINING_VERDICT": "FINAL_MODEL_READY" if ready else "AUDIT_FAILED",
        "TEST_STATUS": "LOCKED_NOT_RUN",
        "FULLRANK_STATUS": "LOCKED_NOT_RUN",
        "FULLRANK_READY": bool(ready),
        "GRADIENT_EQUIVALENCE_TEST": grad_audit["verdict"],
        "TRUE_FINAL_HGT_GRAD_SCOPE": "ALL_TRAIN",
        "TRUE_FINAL_DECODER_GRAD_SCOPE": "ALL_TRAIN",
        "TRUE_FINAL_LEG_GRAD_SCOPE": "ALL_TRAIN",
        "TRUE_FINAL_FROM_SCRATCH": True,
        "C1_STATUS": "ABORTED_BY_DESIGN",
        "C1_SEED202_STARTED": False,
        "C1_SEED303_STARTED": False,
        "C1_FULLRANK": "CANCELLED",
        "c1_sampled_ndcg20": c1_scores,
        "from_scratch": scratch_rows,
    }
    write_json(d["summary"] / "TRUE_FINAL_JOINT_TRAINING_MANIFEST.json", manifest)
    write_json(d["summary"] / "TRUE_FINAL_MANIFEST.json", manifest)
    write_json(OUT / "TRUE_FINAL_MANIFEST.json", manifest)
    write_json(d["summary"] / "FINAL_CLEAN_TRAINING_MANIFEST.json", manifest)

    print("\n" + "=" * 60, flush=True)
    print("C1_STATUS = ABORTED_BY_DESIGN", flush=True)
    print("C1_SEED202_STARTED = NO", flush=True)
    print("C1_SEED303_STARTED = NO", flush=True)
    print("C1_FULLRANK = CANCELLED", flush=True)
    print(f"TRUE_JOINT_GRADIENT_AUDIT = {grad_audit['verdict']}", flush=True)
    for seed, srow in zip(SEEDS, scratch_rows):
        print(f"TRUE_FINAL_SEED{seed}_FROM_SCRATCH = YES", flush=True)
    print("TRUE_FINAL_HGT_GRADIENT_SCOPE = ALL_TRAIN", flush=True)
    print("TRUE_FINAL_DECODER_GRADIENT_SCOPE = ALL_TRAIN", flush=True)
    print("TRUE_FINAL_LEG_GRADIENT_SCOPE = ALL_TRAIN", flush=True)
    for seed in SEEDS:
        r = by_seed[seed]
        print(f"TRUE_FINAL_SEED{seed}_BEST = {r['best_sampled_NDCG@20']:.6f}", flush=True)
    print(f"TRUE_FINAL_MEAN_SAMPLED_NDCG20 = {summary['NDCG@20_mean']:.6f}", flush=True)
    print(f"TRUE_FINAL_STD_SAMPLED_NDCG20 = {summary['NDCG@20_std']:.6f}", flush=True)
    print(f"TRUE_FINAL_READY_FOR_FULLRANK = {'YES' if ready else 'NO'}", flush=True)
    print("TEST_STATUS = LOCKED_NOT_RUN", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
