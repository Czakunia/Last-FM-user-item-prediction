#!/usr/bin/env python3
"""Single guarded entrypoint for sealed Last-FM* external test evaluation.

Exactly-once protocol: requires CONFIRM_UNSEAL_LASTFM_TEST=YES.
Does not retrain, retune, or reselect models after observing results.
No commit / no push.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Guard MUST run before heavy imports so protocol tests and dry launches stay safe.
if os.environ.get("CONFIRM_UNSEAL_LASTFM_TEST") != "YES":
    raise RuntimeError(
        "Sealed Last-FM* test evaluation is disabled. "
        "Set CONFIRM_UNSEAL_LASTFM_TEST=YES only after protocol approval."
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._lastfm_sealed_test_eval_20260824 import run_sealed_test_once  # noqa: E402


if __name__ == "__main__":
    payload = run_sealed_test_once()
    print("LASTFM_EXT_04 = OK", flush=True)
    print(f"SEALED_TEST_EXECUTED={payload['SEALED_TEST_EXECUTED']}", flush=True)
    print(f"results_sha256={payload['results_sha256']}", flush=True)
    a = payload["block_a"]["true_final_mean"]
    b = payload["block_b"]["true_final_mean"]
    print(
        f"BlockA TRUE_FINAL mean NDCG@20={a['NDCG@20']:.6f} Recall@20={a['RECALL@20']:.6f}",
        flush=True,
    )
    print(
        f"BlockB TRUE_FINAL mean NDCG@20={b['NDCG@20']:.6f} Recall@20={b['Recall@20']:.6f} "
        f"MRR={b['MRR']:.6f} HitRate@20={b['HitRate@20']:.6f}",
        flush=True,
    )
    print("STOP after sealed-test report (no commit, no push).", flush=True)
