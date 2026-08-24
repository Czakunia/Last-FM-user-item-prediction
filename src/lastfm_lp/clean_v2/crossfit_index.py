"""CLEAN V2 always uses leave-one-fold population stats for user u.

V1 DIFFERENCE (intentional, documented):
  V1 ``CrossFitHCRBundle.index_for`` returns ``full_index`` on val/test.
  CLEAN V2 ALWAYS uses fold-excluded index for population item-item quantities
  (n11, A11, Jaccard/Cosine route, N_X), including validation and test.
"""

from __future__ import annotations

from src.lastfm_lp.binary.cross_fit import CrossFitHCRBundle
from src.lastfm_lp.binary.hcr_pairwise import PairwiseStatsIndex


def clean_v2_index_for(cf: CrossFitHCRBundle, user_id: int) -> PairwiseStatsIndex:
    """Return PairwiseStatsIndex built on R_f = users with fold != fold(u)."""

    fold = cf.user_to_fold.get(int(user_id))
    if fold is None:
        # Ambiguity documented in audit: rare unseen user → full_index fallback.
        return cf.full_index
    return cf.fold_indices[int(fold)]
