#!/usr/bin/env python3
"""Refit frozen TRUE FINAL on TRAIN_EXTERNAL with frozen epoch counts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_external_benchmark_20260824 import build_external_bundle, fit_true_final_refit


if __name__ == "__main__":
    bundle = build_external_bundle()
    rows = fit_true_final_refit(bundle)
    print("LASTFM_EXT_01 = OK", flush=True)
    for row in rows:
        print(f"seed={row['seed']} epoch_count={row['epoch_count']} sha256={row['sha256']}", flush=True)
