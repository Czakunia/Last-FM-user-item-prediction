#!/usr/bin/env python3
"""FAST full-rank evaluator — copy of the frozen reference with a faster scorer.

Does not overwrite ``scripts/run_lastfm_noleak_fullrank_validation_v1.py``.
Default output: LASTFM_TRUE_FINAL/NOLEAK_FULLRANK_VALIDATION_V1_FAST.

Do not launch this on the full catalog while the reference seed-202 job
(PID 76313) is still running. Equivalence must PASS first.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "LASTFM_FULLRANK_OUT" not in os.environ:
    os.environ["LASTFM_FULLRANK_OUT"] = str(
        ROOT / "LASTFM_TRUE_FINAL" / "NOLEAK_FULLRANK_VALIDATION_V1_FAST"
    )

import scripts.run_lastfm_noleak_fullrank_validation_v1 as ref  # noqa: E402
from scripts.run_lastfm_noleak_fullrank_validation_v1 import main as reference_main  # noqa: E402

from src.lastfm_lp.evaluation.fullrank_fast_features import (  # noqa: E402
    score_user_catalog_fast,
)

ref.score_user_catalog = score_user_catalog_fast


if __name__ == "__main__":
    print(
        "[fast-fullrank] using optimized scorer; OUT="
        f"{os.environ.get('LASTFM_FULLRANK_OUT')}",
        flush=True,
    )
    print("[fast-fullrank] reference scorer module preserved at", ref.__file__, flush=True)
    reference_main()
