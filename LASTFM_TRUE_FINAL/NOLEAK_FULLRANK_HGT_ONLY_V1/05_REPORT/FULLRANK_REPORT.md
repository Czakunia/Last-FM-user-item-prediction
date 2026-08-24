# FULLRANK_REPORT

Generated 2026-08-24T00:44:53Z.

**EVALUATION_PROTOCOL:** `FULL_CATALOG` = all Last-FM\* items minus `model_train` history.  
**SAMPLED_20_NEG** is the frozen training selection metric. The two columns are **not comparable**.

Dataset = corrected NO-LEAKAGE Last-FM\*. Wording: *full-ranking evaluation on corrected Last-FM\**.  
Not a claim of "directly reproduced KGAT benchmark".

## Main table

| seed | EVALUATION_PROTOCOL (sampled) | sampled NDCG@20 | EVALUATION_PROTOCOL (full-rank) | full-rank NDCG@20 | full-rank Recall@20 | MRR |
|---:|---|---:|---|---:|---:|---:|
| 101 | SAMPLED_20_NEG | 0.475024 | FULL_CATALOG | 0.008548 | 0.016167 | 0.013704 |
| 202 | SAMPLED_20_NEG | 0.475407 | FULL_CATALOG | 0.009520 | 0.016752 | 0.015520 |
| 303 | SAMPLED_20_NEG | 0.475953 | FULL_CATALOG | 0.009935 | 0.017691 | 0.015889 |
| mean | SAMPLED_20_NEG | 0.475461 | FULL_CATALOG | 0.009334 | 0.016870 | 0.015038 |

sampled NDCG@20 − full-rank NDCG@20 (means) = 0.466127

## Secondary (full-rank)

| seed | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | P@20 | MAP@20 | HitRate@20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.005371 | 0.006894 | 0.006799 | 0.011077 | 0.002469 | 0.003918 | 0.045590 |
| 202 | 0.005985 | 0.007417 | 0.006547 | 0.010450 | 0.002618 | 0.004807 | 0.048137 |
| 303 | 0.006190 | 0.007763 | 0.006753 | 0.011114 | 0.002789 | 0.004960 | 0.051332 |

## Beyond-accuracy (Top-20, pop from model_train)

| seed | CatalogCoverage@20 | ARP@20 | LongTailShare@20 |
|---:|---:|---:|---:|
| 101 | 0.000935 | 472.6225 | 0.000000 |
| 202 | 0.000914 | 503.3518 | 0.000000 |
| 303 | 0.000914 | 536.5144 | 0.000000 |

## Frozen checkpoints (selected BEFORE this evaluation)

| seed | best_epoch (sampled NDCG@20) |
|---:|---:|
| 101 | 97 |
| 202 | 58 |
| 303 | 77 |

No retrain. No epoch reselection. No new negatives.

## Q&A

1. Corrected NO-LEAKAGE Last-FM\*? **YES**
2. Checkpoint selected before full-rank? **YES**
3. Model retrained? **NO**
4. Sampled negatives used in ranking? **NO**
5. Whole catalog after model_train mask? **YES**
6. All validation positives retained? **YES**
7. Train history masked? **YES**
8. Full-rank NDCG@20 mean = **0.009334** (std 0.000712)
9. Full-rank Recall@20 mean = **0.016870**
10. Seed range NDCG@20: 0.008548 … 0.009935
11. Gap sampled − full-rank NDCG@20 (means) = 0.466127
12. TEST accessed for scoring? **NO** (`LOCKED_NOT_RUN`)
