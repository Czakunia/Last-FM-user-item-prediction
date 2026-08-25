# Last-FM* TRUE FINAL outputs (2026-08-24)

Frozen publication outputs for the Last-FM* proxy on branch `lastfm-proxy`.

- Architecture + reproduction: [`docs/LASTFM_PROXY_REPRODUCTION.md`](../docs/LASTFM_PROXY_REPRODUCTION.md)
- Entry points: `scripts/LAST_FM_00` … `LAST_FM_04` (`*_20260824.py`)

## Contents

| Directory | Role |
|---|---|
| `JOINT_TRAINING_V1/` | TRUE FINAL training (seeds 101/202/303); checkpoints gitignored |
| `JOINT_HGT_ONLY_RANKER_V1/` | Ablation: HGT only |
| `JOINT_HGT_A5_RANKER_V1/` | Ablation: HGT + A5 |
| `JOINT_HGT_A5_H3_NOLEG_V1/` | Ablation: HGT + A5 + H3 (no LEG) |
| `NOLEAK_FULLRANK_*` | Full-catalog validation reports |

**TEST sealed.** Do not mix sampled validation NDCG@20 with full-catalog NDCG@20.
