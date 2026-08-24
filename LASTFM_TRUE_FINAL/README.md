# LASTFM_TRUE_FINAL

Project home for the **Last-FM* proxy** experiment (branch `lastfm-proxy`).

## Frozen outputs on this branch

| Directory | Status | Role |
|---|---|---|
| `JOINT_TRAINING_V1/` | TRUE FINAL training reports | Reference model checkpoints (local, gitignored) |
| `NOLEAK_FULLRANK_VALIDATION_V1/` | **Complete** (3 seeds) | TRUE FINAL full-catalog metrics |
| `JOINT_HGT_A5_RANKER_V1/` | Sampled training complete | Ablation: HGT + A5 |
| `JOINT_HGT_ONLY_RANKER_V1/` | Sampled training complete | Ablation: HGT only |
| `JOINT_HGT_A5_H3_NOLEG_V1/` | Sampled training complete | Ablation: HGT + A5 + H3, no LEG |
| `NOLEAK_FULLRANK_HGT_*_V1/` | Full-rank complete | Ablation full-catalog scores |
| `ABLATION_TRAIN_THEN_FULLRANK/logs/` | Local only | Pipeline logs (gitignored) |

## Metrics (do not mix)

- **Sampled** NDCG@20 (~0.87 TRUE FINAL): checkpoint selection during training  
- **Full-catalog** NDCG@20 (~0.28 TRUE FINAL): publication evaluation  

TEST locked.

## Reproduce

See repository root [README.md](../README.md) and [docs/LASTFM_PROXY_REPRODUCTION.md](../docs/LASTFM_PROXY_REPRODUCTION.md).

Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
