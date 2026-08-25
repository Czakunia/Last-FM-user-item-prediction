# Last-FM* proxy — reproduction protocol

**Frozen date:** 2026-08-24  
**Branch:** `lastfm-proxy`  
**Purpose:** Cross-domain proxy — does combining HGT structure with explicit candidate-specific context (A5, H3, LEG_K2) help user–item ranking on a large music CKG?

**Data:** Last-FM* (`data/LastFM_star_IntentAwareRS/`) — IntentAwareRS `--resolveDataLeakage yes` (leakage-corrected split).  
**Config:** `configs/lastfm_star_proxy_20260824.yaml`  
**Splits:** `outputs/lastfm_star/splits/` (val carved from train only, seed 2026).  
**TEST:** sealed — never scored during development.

Do **not** redistribute raw KGAT dumps. Keep `data/external/` ignored.

---

## 1. Numbered execution order (00 → 04)

| Step | Entry script | Role |
|---:|---|---|
| 0 | `LAST_FM_00_prepare_antileak_splits_and_pair_tables_20260824.py` | `model_train`, valid, pair tables |
| 1 | `LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824.py` | A5 + H3 tables under `outputs/lastfm_star/materialized/` |
| 2 | `LAST_FM_02_train_true_final_joint_model_HGT_A5_H3_LEG_20260824.py` | TRUE FINAL (HGT+A5+H3+LEG_K2), seeds **101, 202, 303** |
| 3 | `LAST_FM_03_train_ablation_layers_sampled_val_NDCG_20260824.py` | Ablations HGT / +A5 / +H3 (sampled validation NDCG@20) |
| 4 | `LAST_FM_04_fullrank_validation_frozen_checkpoints_20260824.py` | Full-catalog validation on frozen checkpoints |

Shared helper: `scripts/_lastfm_pipeline_common_20260824.py`  
Index: `scripts/PIPELINE_LAST_FM_20260824.txt`  
Internal implementations (do not run directly): `scripts/INTERNAL_LASTFM_SCRIPTS_20260824.txt`

### Environment

- Python 3.10+ recommended; project uses `.venv`
- PyTorch 2.x + PyTorch Geometric (MPS on Apple Silicon OK)
- **One GPU / MPS job at a time** on 16 GB machines
- ~20 GB free disk (data + materialized features + checkpoints)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Outputs / checkpoints

| Artifact | Location |
|---|---|
| Splits | `outputs/lastfm_star/splits/` |
| A5 / H3 / LEG_K2 tables | `outputs/lastfm_star/materialized/` (gitignored `.npy`) |
| TRUE FINAL checkpoints | `LASTFM_TRUE_FINAL/JOINT_TRAINING_V1/06_CHECKPOINTS/` (gitignored) |
| Ablation checkpoints | `LASTFM_TRUE_FINAL/JOINT_HGT_*` (gitignored `.pt`) |
| Full-catalog reports | `LASTFM_TRUE_FINAL/NOLEAK_FULLRANK_*` (JSON/CSV committed when present) |

---

## 2. Metrics — do not mix protocols

| Metric | Candidate set | Use |
|---|---|---|
| **Sampled validation NDCG@20** | 20 random negatives per val positive | Training, early stopping |
| **Full-catalog development** | All items minus `H_u` in `model_train` | Architecture ablation (~**0.276**) |
| **Sealed external test** | Official corrected test (IntentAwareRS) | Literature comparison (~**0.172**) |

Example full model (3 seeds): sampled ≈ **0.873**, full-catalog NDCG@20 ≈ **0.276**, sealed NDCG@20 ≈ **0.172**.  
Do **not** mix these protocols. See `reports/LASTFM_RESULTS_DISCUSSION_20260824.md`.

---

## 3. Frozen architecture (TRUE FINAL)

Authoritative description (single source — do not duplicate elsewhere).

```
s(u,X) = s_B0(u,X) + δ_LEG(u,X)
```

`s_B0` is the 265-D late-fusion MLP. `δ_LEG` is a 1-D residual (zero at init).

### Graph + HGT

Heterogeneous CKG (`build_ckg_graph.py`): user / entity nodes; interact edges + KG relations (KG edges capped at 250_000).

HGT (frozen capacity): **64-D**, **2 layers**, **2 heads**, dropout **0.1**.

```
g_ctx(u,X) = [ z_u ‖ z_X ‖ z_u⊙z_X ‖ |z_u−z_X| ] ∈ R^{256}
graph_dot  = ⟨z_u, z_X⟩ ∈ R^1
```

### A5 (5-D tabular)

Cross-fit association counts (fold seed 2026). Set-overlap of history `H_u` with co-occurrence neighbourhood of candidate `X`, plus log user activity / item popularity / KG degree.

### H3 (3-D)

Routing winner: **A11 / Pearson φ**, Top25 (`K = min(25, |H_u\{X}|)`), ties → lower item id.  
Pool signed A11 values: `[mean, max, top3mean]`.

### LEG_K2 residual

Same Top25 / signed A11. Legendre P2 mean → Linear(1→16)→GELU→Linear(16→1), **zero-init** last layer.

### Decoder

```
[ g_ctx (256) ‖ graph_dot (1) ‖ A5 (5) ‖ H3 (3) ] = 265
265 → 128 → LN → GELU → Dropout(0.2) → 64 → GELU → Dropout(0.2) → 1
```

Optimizer: Adam `1e-3`, wd `1e-4`, ReduceLROnPlateau, max 120 epochs, patience 8.  
Checkpoint metric: **sampled validation NDCG@20**.

### Ablation decoder dims

| Layer | Decoder input |
|---|---:|
| HGT only | 257 |
| HGT + A5 | 262 |
| HGT + A5 + H3 (no LEG) | 265 |
| TRUE FINAL (+ LEG_K2) | 265 + residual |

---

## 4. Top25 / A11 routing (method note)

LEVEL B contingency on leave-one-fold population:

```
A11 / Pearson φ = (N n11 − n_h n_X) / sqrt( n_h(N−n_h) n_X(N−n_X) )
```

After Top25 is chosen, routing scores are discarded; downstream always evaluates **signed A11** on the selected history items. This is the frozen routing for H3 and LEG_K2.

Implementation: `src/lastfm_lp/binary/measure_race_selection.py` (`deterministic_topk_indices`).

---

## 5. Verification

```bash
.venv/bin/python -m pytest -q
```

Publication tests:

- `tests/test_clean_v2.py`
- `tests/test_publication_full_rank_evaluator.py`

Expected reference: **18 passed** (unless obsolete exploratory tests were intentionally removed — document any change).

---

## 6. What is intentionally excluded from git

- Neural checkpoints (`*.pt`)
- Large materialized arrays (`*.npy`, most `*.pkl`) — regenerate via step 01 (+ LEG_K2 cache)
- Training logs
- Legacy exploratory trees (`KRAM_FINAL_WORK/` exploration races)
- Official **test** labels for scoring (TEST sealed)
- Raw external KGAT dumps under `data/external/`
