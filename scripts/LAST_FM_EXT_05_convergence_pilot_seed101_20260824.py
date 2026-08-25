#!/usr/bin/env python3
"""POST_HOC_CONVERGENCE_CONTROLLED_REFIT — seed 101 pilot only.

Builds TRAIN_INNER / VAL_INNER inside TRAIN_EXTERNAL, retrains TRUE FINAL from
scratch with max_epochs=300 and inner-val NDCG@20 early stopping (patience=30).
Does not touch sealed test or original external benchmark artefacts.

Usage:
  .venv/bin/python scripts/LAST_FM_EXT_05_convergence_pilot_seed101_20260824.py

Optional:
  LASTFM_TORCH_DEVICE=cpu   # force CPU (recommended if MPS OOM during encode_all)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_convergence_refit_20260824 import (  # noqa: E402
    EXPERIMENT_TAG,
    MODELS_DIR,
    build_convergence_bundle,
    generate_pilot_report,
    load_or_build_bundle,
    train_convergence_pilot,
)


def main() -> None:
    print(f"[convergence] {EXPERIMENT_TAG} — seed 101 pilot", flush=True)
    bundle = load_or_build_bundle()
    run_dir = MODELS_DIR / "PILOT_SEED101"
    resume = os.environ.get("LASTFM_CONVERGENCE_RESUME", "1") != "0"
    meta = train_convergence_pilot(seed=101, bundle=bundle, run_dir=run_dir, resume=resume)
    report = generate_pilot_report(seed=101, run_dir=run_dir)
    print(f"[convergence] finished seed=101 best_epoch={meta.get('best_epoch_1based')} "
          f"best_NDCG@20={meta.get('best_val_inner_NDCG@20'):.6f} stop={meta.get('stop_reason')}", flush=True)
    print(f"[convergence] report: {report}", flush=True)


if __name__ == "__main__":
    main()
