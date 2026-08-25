#!/usr/bin/env python3
"""LAST_FM_00_prepare_antileak_splits_and_pair_tables_20260824

Step 0 — Last-FM* anti-leakage setup.

Reads official Last-FM* under data/LastFM_star_IntentAwareRS/ (IntentAwareRS
--resolveDataLeakage yes). Builds model_train, valid (10%% from train, seed 2026),
and pair tables under outputs/lastfm_star/.

Prerequisite: raw Last-FM* files in data/LastFM_star_IntentAwareRS/
Config: configs/lastfm_star_proxy_20260824.yaml
TEST is preserved but never scored during development.
"""

from __future__ import annotations

from _lastfm_pipeline_common_20260824 import run_module_prepare

if __name__ == "__main__":
    run_module_prepare()
