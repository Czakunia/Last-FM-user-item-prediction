#!/usr/bin/env python3
"""01 — Prepare Last-FM* anti-leak splits and pair tables.

What this script does
---------------------
Reads the leakage-corrected Last-FM* dump (IntentAwareRS,
``--resolveDataLeakage yes``) from ``data/LastFM_star_IntentAwareRS/``.

It then carves a *model-train* / *validation* split from the official train
file only (seed 2026, 10% per user, at least 5 training items kept). The
official test file is copied into ``outputs/lastfm_star/splits/test.txt``
and is **not scored** at this step.

It also writes pair tables used later for A5/H3 materialisation.

Inputs
------
- ``data/LastFM_star_IntentAwareRS/{train,test,kg_final,user_list,item_list,entity_list,relation_list}.txt``
- ``configs/lastfm_star_proxy_20260824.yaml``

Outputs
-------
- ``outputs/lastfm_star/splits/model_train.txt``
- ``outputs/lastfm_star/splits/valid.txt``
- ``outputs/lastfm_star/splits/test.txt``
- ``outputs/lastfm_star/splits/eval_users.json``
- ``outputs/lastfm_star/splits/meta.json``

This package already ships the frozen splits (see ``SHA256_DATA.json``).
Re-running this step must reproduce those hashes. TEST remains sealed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from _lastfm_pipeline_common_20260824 import run_module_prepare  # noqa: E402

if __name__ == "__main__":
    run_module_prepare()
