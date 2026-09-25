#!/usr/bin/env python3
"""02 — Materialise A5, H3 and LEG statistical features on model-train.

What this script does
---------------------
For every training pair (user, item, label) it computes the frozen empirical
vector used by the joint ranker:

* **A5** (5-D): log history length, log item popularity, log CKG degree,
  Jaccard and cosine of the user history ``H_u`` with the co-occurrence
  neighbourhood ``N_X`` of the candidate. Counts are *cross-fitted*:
  user ``u`` is held out of the co-occurrence tables used to score ``u``.
* **H3** (3-D): after Pearson φ / HCR ``a11`` is computed for each history
  item vs the candidate, at most 25 associations are kept and reduced to
  mean, signed max, and mean of the three strongest signed values.
* **LEG / L2**: mean of the second-order Legendre polynomial ``P2`` on the
  same retained φ list. This is *not* concatenated into the 8-D vector; the
  residual MLP reads this scalar later.

All quantities come from ``model_train`` only. The sealed test file is not
read for feature fitting.

Inputs
------
- splits from step 01
- collaborative knowledge graph built from Last-FM* train + ``kg_final.txt``

Outputs
-------
- ``outputs/lastfm_star/materialized/`` (A5 / H3 / L2 arrays + scalers)

Prerequisite: step 01. Large ``.npy`` files are gitignored; regenerate here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from _lastfm_pipeline_common_20260824 import run_script  # noqa: E402

if __name__ == "__main__":
    run_script("LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824.py")
