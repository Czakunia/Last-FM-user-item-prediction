"""Publication paths for the frozen Last-FM* proxy (2026-08-24).

Materialized statistical features live under outputs/ — not under the
legacy exploratory KRAM_FINAL_WORK tree.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Preferred publication location (regenerable via LAST_FM_01).
_DEFAULT_MATERIALIZED = ROOT / "outputs" / "lastfm_star" / "materialized"

# Legacy cache from the RACE_CLEAN_3 exploration phase (local only).
_LEGACY_KRAM = ROOT / "KRAM_FINAL_WORK" / "RACE_CLEAN_3"

PROTOCOL_CONFIG = ROOT / "configs" / "lastfm_star_proxy_20260824.yaml"


def materialized_root() -> Path:
    """Root for A5 / H3 / LEG_K2 tables.

    Order:
      1. LASTFM_MATERIALIZED_ROOT env override
      2. outputs/lastfm_star/materialized if present
      3. legacy KRAM_FINAL_WORK/RACE_CLEAN_3 if still on disk (compat)
      4. otherwise create under outputs/lastfm_star/materialized
    """
    env = os.environ.get("LASTFM_MATERIALIZED_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if _DEFAULT_MATERIALIZED.exists():
        return _DEFAULT_MATERIALIZED
    if _LEGACY_KRAM.exists():
        return _LEGACY_KRAM
    return _DEFAULT_MATERIALIZED


def leg_k2_dir(root: Path | None = None) -> Path:
    """Directory holding LEG_K2_{train,val}.npy and LEG_K2_scaler.pkl."""
    r = root if root is not None else materialized_root()
    pub = r / "leg_k2"
    if pub.exists():
        return pub
    legacy = r / "A11_FUNCTIONAL_DISTRIBUTION_BENCHMARK_V2" / "nxt"
    if legacy.exists():
        return legacy
    return pub
