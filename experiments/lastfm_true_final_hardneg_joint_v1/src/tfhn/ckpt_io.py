"""Write TRUE-FINAL-compatible checkpoint payloads + JOINT dirs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tfhn.paths import ARCH, ART, JOINT_OUT


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_joint_dirs() -> dict[str, Path]:
    mapping = {
        "audit": JOINT_OUT / "00_AUDIT",
        "s101": JOINT_OUT / "01_SEED101",
        "s202": JOINT_OUT / "02_SEED202",
        "s303": JOINT_OUT / "03_SEED303",
        "summary": JOINT_OUT / "04_SUMMARY",
        "figures": JOINT_OUT / "05_FIGURES",
        "ckpts": JOINT_OUT / "06_CHECKPOINTS",
    }
    for p in mapping.values():
        p.mkdir(parents=True, exist_ok=True)
    (JOINT_OUT / "TEST_LOCKED").write_text("LOCKED_NOT_RUN\n", encoding="utf-8")
    return mapping


def build_payload(
    *,
    state: dict,
    seed: int,
    best_epoch: int | str,
    sampled_ndcg: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Payload contract accepted by fullrank / external loaders."""
    ep = best_epoch if isinstance(best_epoch, int) else (
        -1 if best_epoch == "init" else int(best_epoch)
    )
    payload: dict[str, Any] = {
        "model_state_dict": state,
        "seed": int(seed),
        "best_epoch": ep,
        "best_epoch_raw": best_epoch,
        "sampled_NDCG@20": float(sampled_ndcg),
        "architecture_name": ARCH,
        "timestamp": utc_now(),
        "recipe": "TRUE_FINAL_HARDNEG_R3",
        "training": "TRUE_FINAL_HARDNEG_FROM_SCRATCH",
        "hgt_grad_scope": "ALL_TRAIN",
        "from_scratch": True,
        "lambda_bpr": 0.5,
        "TEST_STATUS": "LOCKED_NOT_RUN",
        "FULLRANK_STATUS": "LOCKED_NOT_RUN",
        "EXTERNAL_STATUS": "LOCKED_NOT_RUN",
        "loaded_c1_checkpoint": False,
        "loaded_b0_checkpoint": False,
        "joint_out": str(JOINT_OUT),
        "experiment": str(ART.parent),
    }
    if extra:
        payload.update(extra)
    return payload


def save_final_seed_ckpt(seed: int, payload: dict[str, Any]) -> Path:
    d = ensure_joint_dirs()
    path = d["ckpts"] / f"final_seed{seed}_best.pt"
    torch.save(payload, path)
    # mirror under experiment artifacts
    mirror = ART / "06_CHECKPOINTS"
    mirror.mkdir(parents=True, exist_ok=True)
    torch.save(payload, mirror / f"final_seed{seed}_best.pt")
    # Primary seeds mirror into numbered JOINT dirs; extras get EXTRA_SEED{N}/
    primary = {101: "s101", 202: "s202", 303: "s303"}
    seed_i = int(seed)
    if seed_i in primary:
        torch.save(payload, d[primary[seed_i]] / "model_best.pt")
    else:
        extra_dir = JOINT_OUT / f"EXTRA_SEED{seed_i}"
        extra_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, extra_dir / "model_best.pt")
    return path


def copy_scalers_to_audit(feat_dir: Path) -> None:
    d = ensure_joint_dirs()
    for name in ("a_scaler.pkl", "h_scaler.pkl", "l_scaler.pkl"):
        src = feat_dir / name
        if src.exists():
            shutil.copy2(src, d["audit"] / name)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
