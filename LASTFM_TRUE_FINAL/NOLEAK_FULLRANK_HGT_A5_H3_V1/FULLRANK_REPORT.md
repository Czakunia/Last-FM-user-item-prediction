# FULLRANK_REPORT

Generated 2026-08-23T15:28:45Z.

**EVALUATION_PROTOCOL:** `FULL_CATALOG` = all Last-FM\* items minus `model_train` history.  
**SAMPLED_20_NEG** is the frozen training selection metric. The two columns are **not comparable**.

Dataset = corrected NO-LEAKAGE Last-FM\*. Wording: *full-ranking evaluation on corrected Last-FM\**.  
Not a claim of "directly reproduced KGAT benchmark".

## Main table

| seed | EVALUATION_PROTOCOL (sampled) | sampled NDCG@20 | EVALUATION_PROTOCOL (full-rank) | full-rank NDCG@20 | full-rank Recall@20 | MRR |
|---:|---|---:|---|---:|---:|---:|
| 101 | SAMPLED_20_NEG | 0.872794 | FULL_CATALOG | 0.227969 | 0.361670 | 0.265290 |
| 202 | SAMPLED_20_NEG | 0.872930 | FULL_CATALOG | 0.241332 | 0.369962 | 0.286141 |
| 303 | SAMPLED_20_NEG | 0.873007 | FULL_CATALOG | 0.227054 | 0.362345 | 0.262417 |
| mean | SAMPLED_20_NEG | 0.872910 | FULL_CATALOG | 0.232119 | 0.364659 | 0.271283 |

sampled NDCG@20 − full-rank NDCG@20 (means) = 0.640792

## Secondary (full-rank)

| seed | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 | P@20 | MAP@20 | HitRate@20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.165875 | 0.193469 | 0.166827 | 0.255956 | 0.069475 | 0.131106 | 0.650002 |
| 202 | 0.182672 | 0.209157 | 0.181531 | 0.270592 | 0.071992 | 0.143010 | 0.656348 |
| 303 | 0.164123 | 0.192393 | 0.165125 | 0.256118 | 0.070205 | 0.129627 | 0.651211 |

## Beyond-accuracy (Top-20, pop from model_train)

| seed | CatalogCoverage@20 | ARP@20 | LongTailShare@20 |
|---:|---:|---:|---:|
| 101 | 0.755481 | 92.7553 | 0.092153 |
| 202 | 0.798516 | 87.1467 | 0.117353 |
| 303 | 0.761507 | 90.8384 | 0.089589 |

## Frozen checkpoints (selected BEFORE this evaluation)

| seed | best_epoch (sampled NDCG@20) |
|---:|---:|
| 101 | 36 |
| 202 | 34 |
| 303 | 39 |

No retrain. No epoch reselection. No new negatives.

## Q&A

1. Corrected NO-LEAKAGE Last-FM\*? **YES**
2. Checkpoint selected before full-rank? **YES**
3. Model retrained? **NO**
4. Sampled negatives used in ranking? **NO**
5. Whole catalog after model_train mask? **YES**
6. All validation positives retained? **YES**
7. Train history masked? **YES**
8. Full-rank NDCG@20 mean = **0.232119** (std 0.007993)
9. Full-rank Recall@20 mean = **0.364659**
10. Seed range NDCG@20: 0.227054 … 0.241332
11. Gap sampled − full-rank NDCG@20 (means) = 0.640792
12. TEST accessed for scoring? **NO** (`LOCKED_NOT_RUN`)
