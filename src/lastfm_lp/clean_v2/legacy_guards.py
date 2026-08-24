"""Guards: CLEAN V2 must not call legacy Tabular-A proxies."""

from __future__ import annotations

import src.lastfm_lp.features.tabular_pair_features as legacy_tab


def assert_legacy_proxies_not_used_in_clean_call_stack(called_names: set[str]) -> None:
    """Fail if CLEAN V2 forward traced into legacy proxy helpers."""

    forbidden = {
        "_cosine_pop",
        "build_tabular_pair_features",  # old A builder with proxies
    }
    hit = called_names & forbidden
    if hit:
        raise AssertionError(f"CLEAN V2 called legacy proxy path(s): {sorted(hit)}")


def legacy_cosine_pop_is_not_clean_v2() -> bool:
    """Document: legacy _cosine_pop exists but is not CLEAN V2."""

    return hasattr(legacy_tab, "_cosine_pop")
