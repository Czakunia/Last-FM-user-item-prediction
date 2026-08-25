# Last-FM* results discussion (development ~0.276 vs sealed ~0.172)

**Date:** 2026-08-24 / 2026-08-25  
**Model:** full Last-FM* stack (HGT + A5 + H3 + LEG$_{K2}$), seeds `{101, 202, 303}`  
**Status:** sealed external evaluation executed once; no post-hoc reselection

This note explains why the same frozen model reports **two different NDCG@20 numbers** and how to cite them.

---

## Headline numbers (do not mix)

| Protocol | What is scored | NDCG@20 | Recall@20 | Role |
|---|---|---:|---:|---|
| **Full-catalog development** | Official validation catalog, checkpoints frozen on sampled NDCG | **0.2764 ± 0.0041** | 0.3768 | Ablation / architecture discussion |
| **Sealed external test** | Official corrected test, IntentAwareRS holdout evaluator | **0.1718 ± 0.0040** | 0.2139 ± 0.0024 | Literature-comparable claim |

Classical strongest sealed baseline: **RP3β** NDCG@20 = **0.2318**, Recall@20 = **0.2588**.

---

## Why ~0.27 and ~0.17 are both correct

They answer different questions on different candidate universes.

1. **Sampled validation (~0.87)**  
   Cheap proxy: rank one positive against 20 sampled negatives. Used only for LR schedule / early stopping / epoch freeze. Not a publication ranking score.

2. **Full-catalog development (~0.276)**  
   Same frozen checkpoints, but rank every eligible item in the validation catalog. This is where A5 / H3 / LEG show large gains over HGT-only (~0.009 → ~0.276).  
   It is **not** an external test claim.

3. **Sealed external (~0.172)**  
   Train on `TRAIN_EXTERNAL = model_train ∪ validation`, freeze epochs from development, score the official test **once**.  
   Comparable to published Last-FM* recommender tables under the same IntentAwareRS evaluator.

A large gap between (2) and (3) is expected: different splits, different catalogs, and sealed evaluation is harder. It does **not** by itself indicate leakage.

---

## What the experiments support

**Supported**
- Explicit candidate-specific context (A5, H3, LEG) substantially improves the neural HGT model under the development full-catalog protocol.
- The modelling principle transfers from Task A–style dual evidence to ranking after a domain change.
- Sealed evaluation places the proposed model below strong classical collaborative-filtering baselines (notably RP3β).

**Not supported**
- Sealed superiority over RP3β / ItemKNN.
- Treating sampled NDCG (~0.87) as full-catalog or sealed performance.
- Mixing development full-catalog and sealed numbers in one comparison column.

---

## Sealed per-seed (IntentAwareRS Block A)

| Seed | Frozen epoch | NDCG@20 | Recall@20 |
|---:|---:|---:|---:|
| 101 | 36 | 0.1685 | 0.2123 |
| 202 | 34 | 0.1762 | 0.2167 |
| 303 | 43 | 0.1706 | 0.2127 |
| mean ± SD | — | **0.1718 ± 0.0040** | **0.2139 ± 0.0024** |

Authoritative machine-readable artefacts:
- `reports/LASTFM_FINAL_EXTERNAL_PROTOCOL_MANIFEST_20260824.md`
- `LASTFM_EXTERNAL_BENCHMARK_20260824/sealed_test_results/` (JSON; weights not committed)

---

## Development ablation (full-catalog reference)

| Variant | Sampled NDCG@20 | Full-catalog NDCG@20 | Full-catalog Recall@20 |
|---|---:|---:|---:|
| HGT only | 0.4755 | 0.0093 | 0.0169 |
| HGT + A5 | 0.7908 | 0.0640 | 0.1128 |
| HGT + A5 + H3 | 0.8729 | 0.2321 | 0.3647 |
| Full model (+ LEG) | 0.8733 | **0.2764** | 0.3768 |

---

## Citation cheat-sheet

- **Internal architecture claim:** use full-catalog development **0.276**.
- **External ranking claim:** use sealed **0.172** and report RP3β **0.232** on the same protocol.
- Always name the protocol in the sentence.
