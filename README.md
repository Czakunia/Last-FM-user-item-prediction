# Last-FM* proxy — frozen pipeline (2026-08-24)

**Branch:** `lastfm-proxy`  
Cross-domain proxy on **Last-FM\*** (IntentAwareRS leakage-corrected split): heterogeneous graph encoder **+** explicit candidate-specific statistical context.

This repository is **not** Task A (synthetic pharmacotherapy).

---

## Headline results (read before citing numbers)

| Protocol | NDCG@20 | Recall@20 | Use |
|---|---:|---:|---|
| Full-catalog **development** (frozen checkpoints) | **0.2764 ± 0.0041** | 0.3768 | Ablation / architecture |
| **Sealed external** test (IntentAwareRS holdout) | **0.1718 ± 0.0040** | 0.2139 ± 0.0024 | Literature comparison |
| Strongest classical sealed baseline (RP3β) | **0.2318** | 0.2588 | External reference |

Full discussion of why **~0.27** and **~0.17** differ (and why both are correct):  
→ [`reports/LASTFM_RESULTS_DISCUSSION_20260824.md`](reports/LASTFM_RESULTS_DISCUSSION_20260824.md)

Protocol freeze + hashes:  
→ [`reports/LASTFM_FINAL_EXTERNAL_PROTOCOL_MANIFEST_20260824.md`](reports/LASTFM_FINAL_EXTERNAL_PROTOCOL_MANIFEST_20260824.md)

---

## Run in order

### Development (00 → 04)

| Step | Script | What it does |
|---:|---|---|
| **0** | `scripts/LAST_FM_00_prepare_antileak_splits_and_pair_tables_20260824.py` | Build `model_train` / valid / pair tables |
| **1** | `scripts/LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824.py` | Precompute A5 + H3 under `outputs/lastfm_star/materialized/` |
| **2** | `scripts/LAST_FM_02_train_true_final_joint_model_HGT_A5_H3_LEG_20260824.py` | Train full model (HGT+A5+H3+LEG), seeds 101/202/303 |
| **3** | `scripts/LAST_FM_03_train_ablation_layers_sampled_val_NDCG_20260824.py` | Ablations: HGT / +A5 / +H3 |
| **4** | `scripts/LAST_FM_04_fullrank_validation_frozen_checkpoints_20260824.py` | Full-catalog development evaluation |

### Sealed external (EXT_00 → EXT_05)

| Step | Script | Role |
|---:|---|---|
| EXT_00 | `scripts/LAST_FM_EXT_00_build_final_train_manifest_20260824.py` | Build `TRAIN_EXTERNAL` |
| EXT_01 | `scripts/LAST_FM_EXT_01_refit_true_final_on_full_train_20260824.py` | Refit frozen epochs on full train |
| EXT_02 | `scripts/LAST_FM_EXT_02_fit_external_baselines_20260824.py` | Classical baselines |
| EXT_03 | `scripts/LAST_FM_EXT_03_validate_candidate_and_metric_equivalence_20260824.py` | Equivalence checks |
| EXT_04 | `scripts/LAST_FM_EXT_04_run_sealed_test_once_20260824.py` | **One-shot** sealed test |
| EXT_05 | `scripts/LAST_FM_EXT_05_freeze_and_protocol_manifest_20260824.py` | Manifest + freeze |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Development
.venv/bin/python scripts/LAST_FM_00_prepare_antileak_splits_and_pair_tables_20260824.py
.venv/bin/python scripts/LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824.py
.venv/bin/python scripts/LAST_FM_02_train_true_final_joint_model_HGT_A5_H3_LEG_20260824.py
.venv/bin/python scripts/LAST_FM_03_train_ablation_layers_sampled_val_NDCG_20260824.py
.venv/bin/python scripts/LAST_FM_04_fullrank_validation_frozen_checkpoints_20260824.py
```

**One GPU/MPS job at a time** on 16 GB machines. Use `.venv/bin/python`.

---

## Frozen model

`HGT 64/2L/2H + pair256 + graph_dot + A5 + H3 + LEG_K2 residual` → LateFusion **265-D**.

Authoritative detail: [`docs/LASTFM_PROXY_REPRODUCTION.md`](docs/LASTFM_PROXY_REPRODUCTION.md).

---

## Metrics — do not mix

| Protocol | Typical NDCG@20 | Use |
|---|---:|---|
| Sampled validation (20 negatives) | ~0.87 | training / early stopping only |
| Full-catalog development | **~0.276** | architecture ablation |
| Sealed external test | **~0.172** | literature comparison |

Never put development full-catalog and sealed numbers in the same comparison column.

---

## Data & config

- Raw Last-FM*: `data/LastFM_star_IntentAwareRS/` (not the leaking KGAT dump)
- Protocol: `configs/lastfm_star_proxy_20260824.yaml`
- Splits: `outputs/lastfm_star/splits/` (val 10% from train, seed 2026)
- Materialized features: `outputs/lastfm_star/materialized/` (local `.npy`, gitignored)
- Sealed JSON results: `LASTFM_EXTERNAL_BENCHMARK_20260824/sealed_test_results/` (committed)
- Model weights / large caches: **gitignored** (regenerate via scripts)

Do **not** redistribute raw KGAT dumps. `data/external/` stays ignored.

---

## Layout

| Path | Role |
|---|---|
| `scripts/LAST_FM_*_20260824.py` | Development entry points |
| `scripts/LAST_FM_EXT_*_20260824.py` | Sealed external entry points |
| `scripts/_lastfm_pipeline_common_20260824.py` | Shared helpers |
| `src/lastfm_lp/` | Reusable library |
| `LASTFM_TRUE_FINAL/` | Training + full-catalog reports (weights ignored) |
| `reports/` | Manifest + results discussion |
| `docs/LASTFM_PROXY_REPRODUCTION.md` | Reproduction document |
| `tests/test_lastfm_external_protocol_20260824.py` | Protocol tests |
