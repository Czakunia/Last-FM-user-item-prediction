#!/usr/bin/env python3
"""LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824

Step 1 — Precompute explicit candidate context used by the ranker.

Materializes (once, ~2 min CPU):
  • A5 (5-D): set overlap + popularity/degree under cross-fold protocol
  • H3 (3-D): signed A11 pool on Top25 routed neighbors

Output paths (local, gitignored .npy):
  KRAM_FINAL_WORK/RACE_CLEAN_3/features/A_true/X_{train,val}.npy
  KRAM_FINAL_WORK/RACE_CLEAN_3/race/a11_top25/features/X_{train,val}.npy
  LEG scaler for TRUE FINAL residual branch (pkl, gitignored)

Prerequisite: step 0 complete.
"""

from __future__ import annotations

import subprocess
import sys

from _lastfm_pipeline_common_20260824 import ROOT, base_env, venv_python

if __name__ == "__main__":
    py = venv_python()
    rc = subprocess.run(
        [str(py), "-u", str(ROOT / "scripts" / "run_race_clean_3.py"), "materialize-all"],
        cwd=str(ROOT),
        env=base_env(),
    ).returncode
    sys.exit(rc)
