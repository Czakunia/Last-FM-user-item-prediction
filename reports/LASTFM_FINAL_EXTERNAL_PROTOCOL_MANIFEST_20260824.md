# Last-FM* final external protocol manifest (2026-08-24)

Status after **one-shot sealed test** evaluation. Models were not retrained or reselected after observing results.

- `FINAL_MODELS_FROZEN` = **YES**
- `READY_TO_UNSEAL` = **YES** (pre-run)
- `sealed_test_accessed` = **YES**
- `SEALED_TEST_EXECUTED` = **YES**
- evaluation timestamp = `2026-08-25T12:11:04Z`
- our git commit = `df96f695b3cbaadaec23253c486b483bf7c83861`
- IntentAwareRS commit = `63ece4444659b9505c36058be219c2db951ea087`
- `TRAIN_EXTERNAL` hash = `ed398530a92126c261b8312e6183ab204316ae0b9754f10bef4df57a6c55f779`
- sealed test hash = `05189cddff68db850c79df3b318868e3aed22c6ef66896bef439a73c953f1cd9`
- results artifact SHA256 = `ff25f0013b028ff077ab8a0a028d7107a3a1c0828edcb017b7ec1a6fcddfbf63`

## A. IntentAwareRS-compatible evaluation (literature comparison)

- evaluator = `EvaluatorHoldout` (`external_repos/IntentAwareRS/topn_baselines_neurals/Evaluation/Evaluator.py`)
- cutoffs = `[1, 5, 10, 20, 40, 50, 100]`
- train masking = `exclude_seen=True passed to recommender.recommend/remove_seen_flag`

| Model | NDCG@20 | Recall@20 |
|---|---:|---:|
| TopPop | 0.009127 | 0.015320 |
| ItemKNN | 0.205134 | 0.236903 |
| P3alpha | 0.203513 | 0.238527 |
| RP3beta | 0.231756 | 0.258837 |
| UserKNN | 0.162233 | 0.181929 |
| TRUE_FINAL_mean | 0.171778 | 0.213878 |

### TRUE FINAL per-seed (Block A)

| Seed | epochs | NDCG@20 | Recall@20 |
|---|---:|---:|---:|
| 101 | 36 | 0.168467 | 0.212269 |
| 202 | 34 | 0.176218 | 0.216693 |
| 303 | 43 | 0.170649 | 0.212673 |
| **mean** | — | **0.171778** | **0.213878** |

## B. Our frozen full-catalog evaluator (internal robustness)

| Model | NDCG@5 | NDCG@10 | NDCG@20 | Recall@20 | MRR | HitRate@20 |
|---|---:|---:|---:|---:|---:|---:|
| TopPop | 0.005405 | 0.007060 | 0.009130 | 0.015320 | 0.016563 | 0.070934 |
| ItemKNN | 0.184321 | 0.189567 | 0.205134 | 0.236903 | 0.294405 | 0.597433 |
| P3alpha | 0.177826 | 0.185820 | 0.203487 | 0.238474 | 0.293030 | 0.626971 |
| RP3beta | 0.212878 | 0.217056 | 0.231759 | 0.258837 | 0.332108 | 0.645374 |
| UserKNN | 0.151010 | 0.152650 | 0.162223 | 0.182176 | 0.255631 | 0.522632 |
| TRUE_FINAL_mean | 0.138613 | 0.151538 | 0.171778 | 0.213878 | 0.240868 | 0.582954 |

### TRUE FINAL per-seed (Block B)

| Seed | NDCG@20 | Recall@20 | MRR | HitRate@20 |
|---|---:|---:|---:|---:|
| 101 | 0.168467 | 0.212269 | 0.232442 | 0.578095 |
| 202 | 0.176218 | 0.216693 | 0.247530 | 0.584980 |
| 303 | 0.170649 | 0.212673 | 0.242631 | 0.585788 |
| **mean** | **0.171778** | **0.213878** | **0.240868** | **0.582954** |

**Never mix Block A and Block B values in the same comparison column.**

## Development ablation (reference only; not external test)

| Variant | NDCG@20 | Recall@20 | MRR |
|---|---:|---:|---:|
| HGT only | ~0.0093 | ~0.0169 | — |
| +A5 | ~0.0640 | ~0.1128 | — |
| +H3 | ~0.2321 | ~0.3647 | — |
| TRUE FINAL | ~0.2764 | ~0.3768 | ~0.3534 |

