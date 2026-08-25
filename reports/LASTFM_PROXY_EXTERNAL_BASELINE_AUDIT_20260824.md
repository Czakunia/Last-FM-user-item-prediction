# LAST-FM* External Baseline Audit — RP3β vs Frozen TRUE FINAL

**Frozen comparison protocol — 2026-08-24**

Auditor: automated pipeline (`scripts/_audit_rp3beta_fullrank_20260824.py` + artefact trace)  
**No commits. No pushes. TRUE FINAL not retrained. Test not scored.**

---

## 1. Frozen model identity

### Architecture
`HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL` (LateFusion 265-D + LEG residual)

### Checkpoint selection protocol
Checkpoints were chosen by **sampled validation NDCG@20 only** (`SAMPLED_20_NEG`, 20 negatives). Full-catalog metrics were computed **after** epoch freeze. No evidence of re-selection from full-rank results.

| Seed | Checkpoint path | SHA256 | On disk | Sampled best epoch | Sampled NDCG@20 | Full NDCG@5 | Full NDCG@10 | Full NDCG@20 | Recall@20 | MRR | HitRate@20 | n_users |
|------|-----------------|--------|---------|-------------------|-----------------|-------------|--------------|--------------|-----------|-----|------------|---------|
| 101 | `JOINT_TRAINING_V1/06_CHECKPOINTS/final_seed101_best.pt` | — | **NO** | 36 | 0.873224 | 0.235750 | 0.251517 | **0.277432** | 0.375599 | 0.356988 | 0.663386 | 23163 |
| 202 | `JOINT_TRAINING_V1/06_CHECKPOINTS/final_seed202_best.pt` | — | **NO** | 34 | 0.873253 | 0.237608 | 0.254203 | **0.279879** | 0.378722 | 0.359681 | 0.665328 | 23163 |
| 303 | `JOINT_TRAINING_V1/06_CHECKPOINTS/final_seed303_best.pt` | `d2f68a87e80f3648bc41a508528c60afbe79355cd7d8c1a230e774da0a2daa47` | **YES** | 43 | 0.873515 | 0.227321 | 0.245037 | **0.271886** | 0.376197 | 0.343391 | 0.664335 | 23163 |

**Source:** `LASTFM_TRUE_FINAL/NOLEAK_FULLRANK_VALIDATION_V1/{01,02,03}_SEED*/SEED_METRICS.json`  
Seed 101 metrics note: `"recovered_from": "in-memory SEED_METRICS.json read 2026-08-19"` — checkpoint file absent.

### Mean full-rank (seeds 101/202/303)

| Metric | Mean | Matches README? |
|--------|-----:|-----------------|
| NDCG@20 | **0.2764** | yes (~0.276) |
| Recall@20 | **0.3768** | yes (~0.377) |
| MRR | 0.3534 | — |

### Ablation ladder (full-catalog means, same evaluator)

| Model | Sampled NDCG@20 | Full NDCG@20 | Recall@20 | MRR |
|-------|----------------:|-------------:|----------:|----:|
| HGT only | 0.4755 | 0.0093 | 0.0169 | 0.0150 |
| HGT + A5 | 0.7908 | 0.0640 | 0.1128 | 0.0898 |
| HGT + A5 + H3 (no LEG) | 0.8729 | 0.2321 | 0.3647 | 0.2713 |
| **TRUE FINAL (+ LEG)** | **0.8732** | **0.2764** | **0.3768** | **0.3534** |
| **RP3β REFERENCE** | n/a | **0.3227** | **0.4122** | **0.4183** |

Sources: ablation CSVs under `NOLEAK_FULLRANK_HGT_*_V1/`; RP3β from `reports/rp3beta_fullrank_20260824.json`.

**Checkpoint protocol verdict:** PASS — epoch IDs in full-rank JSON match training `seed_summary.json` where present; no post-hoc full-rank selection detected.

---

## 2. Dataset provenance

### Upstream source
- Paper: Shehzad, Ferrari Dacrema, Jannach — *A Worrying Reproducibility Study of Intent-Aware Recommendation Models* (SIGIR 2025)
- Official repo: [IntentAwareRS](https://github.com/Faisalse/IntentAwareRS)
- Local corrected bundle: `data/LastFM_star_IntentAwareRS/` (`MANIFEST.json` documents `--resolveDataLeakage yes` path)
- Step 0: `LAST_FM_00_prepare_antileak_splits_and_pair_tables_20260824.py` — 10% per-user holdout from corrected **train** → `model_train` + `valid`; **test sealed**

### Interaction counts (verified 2026-08-24)

| Split | Pairs | Users | Items |
|-------|------:|------:|------:|
| Corrected upstream train | 1,235,905 | 23,529 | 47,973 |
| `model_train` | 1,111,803 | 23,529 | 47,914 |
| `validation` | 124,102 | 23,529 | 35,154 |
| Sealed test | 306,914 | 23,529 | 38,478 |

### Split integrity

| Check | Result |
|-------|--------|
| `model_train ∩ validation` | **0** |
| `model_train ∩ test` | **0** |
| `validation ∩ test` | **0** |
| `upstream_train == model_train ∪ validation` | **exact** |
| `local_test == upstream_test` | **exact** (symdiff 0) |

### File hashes (SHA256)

| File | Hash |
|------|------|
| `model_train.txt` | `479d40d9e81cb18f6e47b4f2934b4832ac66764665d482c1e76baa8275917278` |
| `valid.txt` | `4178acf9108ce0be0f12501ae0bce6f857bf0901c28ce0111a18095c0a02d77f` |
| `test.txt` | `f684145429010deba000da57b4c6192d3787382d64ee5bdf5c7554e64d8bc03f` |

Matches `NOLEAK_FULLRANK_VALIDATION_V1/00_AUDIT/FULLRANK_DATA_AUDIT.json`.

### Dataset identity vs IntentAwareRS bundled corrected files

**DATASET_IDENTITY = EXACT** for the materialized Last-FM* used in this experiment (`data/LastFM_star_IntentAwareRS/` train+test fingerprints match `MANIFEST.json`; local `model_train`/`valid` partition verified as exact 90/10 decomposition of upstream train).

*Note:* Re-running IntentAwareRS leakage repair from raw **leaking** KGIN dumps offline can yield ~326 pair differences (non-deterministic `list(set)` ordering in upstream re-split). That affects **reproduction from raw** only; it does **not** affect identity of our already-materialized corrected bundle.

---

## 3. Leakage checks

### A5 (`A5_LEAKAGE_AUDIT = PASS`)
- Materialized in `run_race_clean_3.py` → `vectorized_clean_v2_A_for_user`
- Popularity / KG degree from `model_train` only
- Association overlap via `clean_v2_index_for(cf, u)` — **leave-one-fold** on `model_train` users (5-fold, seed 2026)
- Validation labels never enter A5 inputs; val rows scored with fold-excluded population index

### H3 (`H3_LEAKAGE_AUDIT = PASS`)
- Same cross-fit index and Top25 signed-A11 routing as A5
- Pool stats (mean/max/top3mean) from fold-excluded co-occurrence only
- Full-rank recomputation path in `run_lastfm_noleak_fullrank_validation_v1.py` uses identical `h3_and_l2` + `clean_v2_index_for`

### LEG (`LEG_LEAKAGE_AUDIT = PASS`)
Evidence chain:
1. **Formula:** `L2(u,X) = mean_{h ∈ signed Top25} P2(A11(h,X))`, `P2(x)=(3x²−1)/2` — audited in `FINAL_A11_AUDIT.json` (`A11_AUDIT_STATUS = PASS`)
2. **Cross-fit:** same `clean_v2_index_for` as H3; user `u`'s fold excluded from co-occurrence population
3. **Scaler:** `LEG_K2_scaler.pkl` — StandardScaler; documentation + training code: fit on **train pair rows only** (`FINAL_A11_AUDIT.json`: `"scaler_note": "... fit on MODEL_TRAIN only"`)
4. **Full-rank equivalence:** `LEG_K2_PAIR_MATCH.json` — recomputed L2 vs cached `LEG_K2_val.npy`, n=64, max_abs=9.5e-7, **PASS**
5. **Candidate universe:** LEG adds logit residual only; does not filter/prune candidates
6. **Sampled vs full-rank gap (0.232→0.276):** sampled NDCG@20 differs by ~0.0004 (H3 vs TRUE FINAL) while full-rank differs by ~0.044 — consistent with LEG helping most when ranking ~48k items, not when ranking 20 negatives

**Caveat:** `LEG_K2_{train,val}.npy` were precomputed in a prior closed A11 series (not regenerated in publication step 01). Provenance is documented; live full-rank formula matches cache on held-out val pairs.

---

## 4. External repository

| Field | Value |
|-------|-------|
| Path | `external_repos/IntentAwareRS/` |
| Commit | `63ece4444659b9505c36058be219c2db951ea087` |
| `git status --short` | clean (no local modifications) |
| Added to `.gitignore` | `external_repos/` (uncommitted) |

Upstream not vendored into publication tree.

---

## 5. RP3β implementation and hyperparameters

| Field | Value |
|-------|-------|
| Implementation | `external_repos/IntentAwareRS/topn_baselines_neurals/Recommenders/GraphBased/RP3betaRecommender.py` |
| Class | `RP3betaRecommender` |
| Reference config source | `run_experiments_for_KGIN_original_baselines.py`, `dataset == "lastFm"` |

```python
REFERENCE_RP3BETA = {
    "topK": 350,
    "alpha": 0.7681732734954694,
    "beta": 0.4181395996963926,
    "normalize_similarity": True,
    "implicit": False,   # default; matches upstream fit() call
    "min_rating": 0,
}
```

- Binary implicit feedback (URM entries = 1.0)
- Upstream evaluates on **sealed test** with **full corrected train** — different from our `model_train → validation` headline comparison

### Shehzad et al. paper table (Last-FM*, corrected, **their test split**)

| Metric | Reported |
|--------|----------:|
| NDCG@20 | 0.233 |
| Recall@20 | 0.253 |

(`data/LastFM_star_IntentAwareRS/MANIFEST.json`) — **not comparable** to our validation numbers without re-running their exact split/evaluator.

---

## 6. Candidate-universe equivalence

From `NOLEAK_FULLRANK_VALIDATION_V1/00_AUDIT/CATALOG.json` (TRUE FINAL) and RP3β audit (same splits):

| Quantity | TRUE FINAL | RP3β |
|----------|----------:|-----:|
| `n_catalog_items` | 48,123 | 48,123 |
| `n_validation_users` (with ≥1 positive) | 23,163 | 23,163 |
| mean candidates / user | 48,075.05 | 48,075.05 |
| min candidates | 46,355 | 46,355 |
| max candidates | 48,119 | 48,119 |
| `candidate_universe_hash` | `315fd2f124edfe20…` | same construction |

Protocol: `C_u = {0,…,48122} \ H_u^{model_train}`; validation positives remain rankable; no ANN/popularity/KG pruning.

---

## 7. Metric-equivalence checks

| Check | Result |
|-------|--------|
| Evaluator | `src/lastfm_lp/evaluation/publication_full_rank_evaluator.py` |
| TRUE FINAL full-rank mode | `evaluate_user_dense` (exact catalog) |
| RP3β full-rank mode | `evaluate_publication_full_rank(..., mode="dense")` — same dense path |
| Binary relevance | yes |
| Train-item exclusion at rank time | yes (`dense_topk` masks train) |
| Tie-break | higher score; lower item_id |
| Multi-positive users | Recall/NDCG denom = \|P_u\| |
| Users with 0 val positives | skipped |
| Unit tests | `pytest -q` → **18 passed** |

### Sanity controls (RP3β, users 0–4)
See `reports/rp3beta_fullrank_20260824.json` → `sanity_users` (per-user top-20, hits, NDCG@20, Recall@20 manually consistent with evaluator).

---

## 8. Full-rank results — headline comparison

### RP3β REFERENCE (our protocol)

Fit: `model_train` only (1,111,803 pairs, binary URM)  
Eval: validation users, frozen dense full-catalog evaluator  
Runtime: fit 33.5s, eval 361.7s

| Metric | RP3β | TRUE FINAL (mean) |
|--------|-----:|------------------:|
| NDCG@5 | 0.2879 | 0.2336 |
| NDCG@10 | 0.3009 | 0.2504 |
| **NDCG@20** | **0.3227** | **0.2764** |
| Recall@20 | 0.4122 | 0.3768 |
| MRR | 0.4183 | 0.3534 |
| HitRate@20 | 0.7019 | 0.6644 |

**TUNED_RP3BETA:** not run (reference config only, per protocol).

---

## 9. Absolute and relative differences (TRUE FINAL − RP3β)

Only valid after equivalence gates above.

| Metric | Δ_abs (TRUE FINAL − RP3β) | Δ_rel vs TRUE FINAL |
|--------|--------------------------:|--------------------:|
| NDCG@20 | **−0.0463** | RP3β higher by **16.8%** relative to TRUE FINAL |
| Recall@20 | −0.0354 | RP3β higher by 9.4% |
| MRR | −0.0649 | RP3β higher by 18.4% |

Under this strictly comparable protocol, **RP3β REFERENCE outperforms frozen TRUE FINAL** on validation full-catalog ranking.

---

## 10. Limitations

1. **Split difference vs Shehzad table:** paper RP3β (0.233 NDCG@20) uses full train → test; we use `model_train` → validation (10% holdout). Direct numeric comparison to their table is invalid.
2. **Checkpoints 101/202 absent on disk** — full-rank metrics trusted from frozen JSON; only seed 303 checkpoint verifiable by SHA256.
3. **LEG_K2 cache** not regenerated in 00–04 pipeline; live formula audit PASS on sample.
4. **RP3β dense-only** in audit script; chunked path not used (neural full-rank also uses dense scoring).
5. **No TUNED_RP3BETA** — reference hyperparameters only; asymmetric vs neural early-stop tuning documented if tuning were added later.
6. **Test remains sealed** — no claim about test-set superiority either way.

---

## 11. Final verdict

```
LAST-FM* EXTERNAL BASELINE AUDIT — 2026-08-24

Frozen TRUE FINAL unchanged: YES

DATA
IntentAwareRS provenance verified: YES
Leakage-corrected source verified: YES
Dataset identity: EXACT
model_train-validation overlap: 0
model_train-test overlap: 0
validation-test overlap: 0

FEATURE LEAKAGE
A5: PASS
H3: PASS
LEG: PASS

EVALUATION
Same validation users: YES
Same item universe: YES
Same candidate exclusion: YES
Same evaluator: YES
Same metric implementation: YES
Full catalog: YES

EXTERNAL BASELINE
RP3β implementation source: external_repos/IntentAwareRS/topn_baselines_neurals/Recommenders/GraphBased/RP3betaRecommender.py
IntentAwareRS commit: 63ece4444659b9505c36058be219c2db951ea087
RP3β reference parameters: topK=350, alpha=0.7681732734954694, beta=0.4181395996963926, normalize_similarity=True, implicit=False
RP3β NDCG@20: 0.3227
RP3β Recall@20: 0.4122
RP3β MRR: 0.4183

TRUE FINAL
NDCG@20: 0.2764
Recall@20: 0.3768
MRR: 0.3534

COMPARISON STATUS:
STRICTLY COMPARABLE

PUBLICATION CLAIM SAFE:
NO

READY FOR HUMAN AUDIT:
YES
```

### Interpretation for publication

- The comparison itself is **methodologically sound**: same corrected data, same `model_train`, same validation users, same full-catalog candidate sets, same metric code, RP3β through **our** evaluator (not cross-repo metric tables).
- The frozen TRUE FINAL full-rank numbers (**≈0.276 / 0.377**) are **confirmed** from artefact JSON.
- **LEG train-only provenance passes** — the +0.044 full-rank lift over H3+no-LEG (0.232→0.276) is not explained by validation leakage under audited cross-fit rules.
- **However:** under this protocol, reproduced **REFERENCE_RP3β beats TRUE FINAL** on NDCG@20 (0.323 vs 0.276). Do **not** claim TRUE FINAL outperforms RP3β on this validation benchmark without additional experiments (e.g., sealed-test protocol, TUNED_RP3β ablation, or model improvements).

---

*Artefacts:* `reports/rp3beta_fullrank_20260824.json`, `scripts/_audit_rp3beta_fullrank_20260824.py`  
*No git commit. No push. Test not scored.*
