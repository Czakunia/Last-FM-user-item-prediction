# FULLRANK_REPORT

Generated 2026-08-23T22:50:53Z.

**EVALUATION_PROTOCOL:** `FULL_CATALOG` = all Last-FM\* items minus `model_train` history.  
**SAMPLED_20_NEG** is the frozen training selection metric. The two columns are **not comparable**.

Dataset = corrected NO-LEAKAGE Last-FM\*. Wording: *full-ranking evaluation on corrected Last-FM\**.  
Not a claim of "directly reproduced KGAT benchmark".

## Main table

| seed | EVALUATION_PROTOCOL (sampled) | sampled NDCG@20 | EVALUATION_PROTOCOL (full-rank) | full-rank NDCG@20 | full-rank Recall@20 | MRR |
|---:|---|---:|---|---:|---:|---:|
| 101 | SAMPLED_20_NEG | 0.791067 | FULL_CATALOG | 0.064571 | 0.113031 | 0.091840 |
| 202 | SAMPLED_20_NEG | 0.790560 | FULL_CATALOG | 0.060866 | 0.109796 | 0.084353 |
| 303 | SAMPLED_20_NEG | 0.790768 | FULL_CATALOG | 0.066630 | 0.115579 | 0.093273 |
| mean | SAMPLED_20_NEG | 0.790798 | FULL_CATALOG | 0.064022 | 0.112802 | 0.089822 |

sampled NDCG@20 − full-rank NDCG@20 (means) = 0.726776

## Secondary (full-rank)

| seed | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | P@20 | MAP@20 | HitRate@20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.042028 | 0.051081 | 0.040625 | 0.069825 | 0.021638 | 0.029913 | 0.296205 |
| 202 | 0.038119 | 0.047266 | 0.038040 | 0.066399 | 0.020000 | 0.028155 | 0.283772 |
| 303 | 0.044062 | 0.053365 | 0.043174 | 0.072738 | 0.021796 | 0.031814 | 0.297975 |

## Beyond-accuracy (Top-20, pop from model_train)

| seed | CatalogCoverage@20 | ARP@20 | LongTailShare@20 |
|---:|---:|---:|---:|
| 101 | 0.775596 | 157.2699 | 0.086914 |
| 202 | 0.777861 | 197.7973 | 0.091912 |
| 303 | 0.842944 | 153.0043 | 0.148530 |

## Frozen checkpoints (selected BEFORE this evaluation)

| seed | best_epoch (sampled NDCG@20) |
|---:|---:|
| 101 | 54 |
| 202 | 70 |
| 303 | 67 |

No retrain. No epoch reselection. No new negatives.

## Q&A

1. Corrected NO-LEAKAGE Last-FM\*? **YES**
2. Checkpoint selected before full-rank? **YES**
3. Model retrained? **NO**
4. Sampled negatives used in ranking? **NO**
5. Whole catalog after model_train mask? **YES**
6. All validation positives retained? **YES**
7. Train history masked? **YES**
8. Full-rank NDCG@20 mean = **0.064022** (std 0.002921)
9. Full-rank Recall@20 mean = **0.112802**
10. Seed range NDCG@20: 0.060866 … 0.066630
11. Gap sampled − full-rank NDCG@20 (means) = 0.726776
12. TEST accessed for scoring? **NO** (`LOCKED_NOT_RUN`)
