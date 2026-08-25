#!/usr/bin/env python3
"""Freeze-gate audit + protocol manifest. Does not touch the sealed test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_external_benchmark_20260824 import (  # noqa: E402
    build_external_bundle,
    run_leakage_and_freeze_audit,
    verify_upstream_evaluator_reproduction,
    write_protocol_manifest,
)


if __name__ == "__main__":
    bundle = build_external_bundle()
    repro = verify_upstream_evaluator_reproduction(bundle)
    print(f"SHEHZAD_EVALUATOR_REPRODUCTION={repro['SHEHZAD_EVALUATOR_REPRODUCTION']}", flush=True)
    freeze = run_leakage_and_freeze_audit()
    path = write_protocol_manifest(freeze)
    print(f"FINAL_MODELS_FROZEN={freeze['FINAL_MODELS_FROZEN']}", flush=True)
    print(f"READY_TO_UNSEAL={freeze['READY_TO_UNSEAL']}", flush=True)
    print(f"manifest={path}", flush=True)
    print("LASTFM_EXT_05 = OK", flush=True)
