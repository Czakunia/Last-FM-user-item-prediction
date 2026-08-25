#!/usr/bin/env python3
"""Fit frozen classical baselines on TRAIN_EXTERNAL only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_external_benchmark_20260824 import build_external_bundle, fit_baselines


if __name__ == "__main__":
    import os

    bundle = build_external_bundle()
    # EASE-R on ~48k items is a dense Gram inversion; optional and may OOM.
    include_easer = os.environ.get("LASTFM_FIT_EASER", "0") in {"1", "true", "TRUE"}
    rows = fit_baselines(bundle, include_userknn=True, include_easer=include_easer)
    print("LASTFM_EXT_02 = OK", flush=True)
    print(f"EASE-R included={include_easer}", flush=True)
    for row in rows:
        print(f"{row['model']} sha256={row['artifact_sha256']}", flush=True)
