"""Train/eval one protocol stage (A0–A2, B1–B4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.lastfm_lp.evaluation.calibration import PlattCalibrator, brier_score
from src.lastfm_lp.evaluation.paired_runs import evaluate_split
from src.lastfm_lp.models.kan_predictor import KANPredictor
from src.lastfm_lp.models.logistic_probe import LogisticProbe
from src.lastfm_lp.models.mlp_predictor import MLPPredictor, TrainConfig
from src.lastfm_lp.pipeline.features_build import materialize_stage_features


def _stage_out_dir(cfg: dict[str, Any], stage: str) -> Path:
    root = Path(cfg["paths"]["stage_a" if stage.startswith("A") else "stage_b"])
    out = root / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_stage(bundle: dict[str, Any], stage: str) -> dict[str, Any]:
    cfg = bundle["cfg"]
    ks = cfg["evaluation"]["ranking_k"]
    feats = materialize_stage_features(bundle, stage)
    Xtr, ytr = feats["X_train"], feats["y_train"].astype(np.int32)
    Xva, yva = feats["X_val"], feats["y_val"].astype(np.int32)
    Xte, yte = feats["X_test"], feats["y_test"].astype(np.int32)

    seed = cfg["models"]["seed"]
    train_info: dict[str, Any] = {}

    if stage in {"A0", "B0"}:
        model = LogisticProbe(
            C=cfg["models"]["logistic"]["C"],
            max_iter=cfg["models"]["logistic"]["max_iter"],
            seed=seed,
        )
        model.fit(Xtr, ytr)
        train_info = {"n_parameters": model.n_parameters()}
        predict = model.predict_proba
    elif stage in {"A1", "B1", "B3"}:
        mcfg = cfg["models"]["mlp"]
        tcfg = TrainConfig(
            lr=mcfg["lr"],
            weight_decay=mcfg["weight_decay"],
            max_epochs=mcfg["max_epochs"],
            batch_size=mcfg["batch_size"],
            patience=mcfg["patience"],
            seed=seed,
        )
        model = MLPPredictor(Xtr.shape[1], mcfg["hidden_dims"], mcfg["dropout"], tcfg)
        train_info = model.fit(Xtr, ytr, Xva, yva)
        train_info["n_parameters"] = model.n_parameters()
        predict = model.predict_proba
    elif stage in {"A2", "B2", "B4"}:
        kcfg = cfg["models"]["kan"]
        tcfg = TrainConfig(
            lr=kcfg["lr"],
            weight_decay=kcfg["weight_decay"],
            max_epochs=kcfg["max_epochs"],
            batch_size=kcfg["batch_size"],
            patience=kcfg["patience"],
            seed=seed,
        )
        model = KANPredictor(
            Xtr.shape[1],
            kcfg["hidden_dim"],
            kcfg["dropout"],
            kcfg["grid_size"],
            kcfg["spline_order"],
            tcfg,
        )
        train_info = model.fit(Xtr, ytr, Xva, yva)
        train_info["n_parameters"] = model.n_parameters()
        predict = model.predict_proba
    else:
        raise ValueError(f"Unsupported stage {stage}")

    val_scores = predict(Xva)
    test_scores = predict(Xte)
    cal = PlattCalibrator().fit(val_scores, yva)
    val_probs = cal.transform(val_scores)
    test_probs = cal.transform(test_scores)

    val_metrics = evaluate_split(bundle["val_pairs"], val_scores, ks)
    test_metrics = evaluate_split(bundle["test_pairs"], test_scores, ks)
    val_metrics["Brier_calibrated"] = brier_score(yva, val_probs)
    test_metrics["Brier_calibrated"] = brier_score(yte, test_probs)

    result = {
        "stage": stage,
        "feature_names": feats["feature_names"],
        "n_features": len(feats["feature_names"]),
        "train_info": train_info,
        "validation": val_metrics,
        "test": test_metrics,
        "primary_metric": cfg["evaluation"]["primary_metric"],
        "primary_test": test_metrics.get(cfg["evaluation"]["primary_metric"]),
    }
    out = _stage_out_dir(cfg, stage)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "test_scores.npy", test_scores)
    print(
        f"[{stage}] test {cfg['evaluation']['primary_metric']}="
        f"{result['primary_test']:.4f}  AUPRC={test_metrics['AUPRC']:.4f}  "
        f"MRR={test_metrics['MRR']:.4f}"
    )
    return result
