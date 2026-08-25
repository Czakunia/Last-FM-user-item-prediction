#!/usr/bin/env python3
"""Validate upstream evaluator reproduction and protocol metadata."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_external_benchmark_20260824 import build_external_bundle, verify_upstream_evaluator_reproduction


if __name__ == "__main__":
    bundle = build_external_bundle()
    payload = verify_upstream_evaluator_reproduction(bundle)
    print("LASTFM_EXT_03 = OK", flush=True)
    print(f"SHEHZAD_EVALUATOR_REPRODUCTION={payload['SHEHZAD_EVALUATOR_REPRODUCTION']}", flush=True)
