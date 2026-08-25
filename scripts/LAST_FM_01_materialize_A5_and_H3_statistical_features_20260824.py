#!/usr/bin/env python3
"""LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824

Step 1 — Precompute explicit candidate context used by the ranker.

Materializes (once, ~2 min CPU):
  • A5 (5-D): set overlap + popularity/degree under cross-fold protocol
  • H3 (3-D): signed A11 pool on Top25 routed neighbors (policy=a11 only)

Output root (local, gitignored .npy):
  outputs/lastfm_star/materialized/

LEG_K2 arrays for TRUE FINAL are expected under:
  outputs/lastfm_star/materialized/leg_k2/
If missing, copy from a prior LEG_K2 cache or regenerate with the closed A11
functional series (not part of the numbered 00–04 publication path).

Prerequisite: step 0 complete.
"""

from __future__ import annotations

import subprocess
import sys

from _lastfm_pipeline_common_20260824 import ROOT, base_env, venv_python

if __name__ == "__main__":
    py = venv_python()
    # Frozen routing = A11 Top25 only (do not rematerialize exploratory policies).
    rc = subprocess.run(
        [
            str(py),
            "-u",
            str(ROOT / "scripts" / "run_race_clean_3.py"),
            "materialize-all",
            "--policies",
            "a11",
        ],
        cwd=str(ROOT),
        env=base_env(),
    ).returncode
    sys.exit(rc)
