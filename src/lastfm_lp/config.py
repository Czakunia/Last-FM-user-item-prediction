"""Load / freeze protocol config for LASTFM_USER_CONDITIONED_LINK_PREDICTION_V1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "lastfm_lp_v1.yaml"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_protocol_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.get("extends")
    if extends:
        parent = load_protocol_config(extends)
        cfg = _deep_merge(parent, cfg)
    # resolve relative output paths
    for key, val in list(cfg.get("paths", {}).items()):
        p = Path(val)
        if not p.is_absolute():
            cfg["paths"][key] = str((ROOT / p).resolve())
    data_path = Path(cfg["data"]["path"])
    if not data_path.is_absolute():
        cfg["data"]["path"] = str((ROOT / data_path).resolve())
    return cfg


def protocol_document(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_protocol_config()
    return {
        "protocol": cfg["protocol"],
        "data": {
            "source": cfg["data"]["source"],
            "original_test_preserved": cfg["data"]["original_test_preserved"],
            "timestamps_available": cfg["data"]["timestamps_available"],
        },
        "split": {
            "validation_from_train": cfg["split"]["validation_from_train"],
            "split_unit": cfg["split"]["split_unit"],
            "validation_ratio": cfg["split"]["validation_ratio"],
            "min_train_items": cfg["split"]["min_train_items"],
            "seed": cfg["split"]["seed"],
        },
        "negatives": {
            "validation_per_positive": cfg["negatives"]["validation_per_positive"],
            "test_per_positive": cfg["negatives"]["test_per_positive"],
            "train_per_positive": cfg["negatives"]["train_per_positive"],
            "types": cfg["negatives"]["types"],
            "active_type": cfg["negatives"]["active_type"],
        },
        "evaluation": cfg["evaluation"],
        "hcr": cfg["hcr"],
        "features": cfg["features"],
        "models": {
            "seed": cfg["models"]["seed"],
            "mlp": cfg["models"]["mlp"],
            "kan": cfg["models"]["kan"],
            "logistic": cfg["models"]["logistic"],
        },
        "stages_first_full_run": cfg["stages"]["first_full_run"],
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "secondary_metrics": cfg["evaluation"]["secondary_metrics"],
    }


def freeze_manifest(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_protocol_config()
    doc = protocol_document(cfg)
    out = Path(cfg["paths"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != doc:
            raise FileExistsError(
                f"Manifest already frozen at {out} and differs from expected protocol. "
                "Delete it only if you intentionally change the protocol."
            )
        return out
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return out


def verify_manifest(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_protocol_config()
    out = Path(cfg["paths"]["manifest"])
    if not out.exists():
        raise FileNotFoundError(f"Missing manifest: {out}. Run freeze_lastfm_manifest.py")
    existing = json.loads(out.read_text(encoding="utf-8"))
    expected = protocol_document(cfg)
    if existing != expected:
        raise ValueError("Frozen manifest does not match current protocol config.")
    return existing
