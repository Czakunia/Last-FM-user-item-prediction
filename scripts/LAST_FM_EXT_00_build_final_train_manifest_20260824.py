#!/usr/bin/env python3
"""Build the final external-train manifest without unsealing test metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_external_benchmark_20260824 import build_external_bundle, write_manifest_inputs


if __name__ == "__main__":
    bundle = build_external_bundle()
    payload = write_manifest_inputs(bundle)
    print("LASTFM_EXT_00 = OK", flush=True)
    print(f"upstream_train_equals_model_train_plus_validation={payload['checks']['upstream_train_equals_model_train_plus_validation']}", flush=True)
    print(f"upstream_test_equals_sealed_test={payload['checks']['upstream_test_equals_sealed_test']}", flush=True)
