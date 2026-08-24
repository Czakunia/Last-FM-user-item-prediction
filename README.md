# Last-FM* proxy — frozen pipeline (2026-08-24)

Branch **`lastfm-proxy`**. Cross-domain proxy on **Last-FM\*** (IntentAwareRS anti-leakage).  
Task A (synthetic) lives in `GraphNeuralNetwork_Thesis` — not here.

## Run in order (one command per step)

| Step | Script | What it does |
|---:|---|---|
| **0** | `scripts/LAST_FM_00_prepare_antileak_splits_and_pair_tables_20260824.py` | Build model_train / valid / pair tables from Last-FM* |
| **1** | `scripts/LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824.py` | Precompute A5 (5-D) + H3 (3-D) + LEG scaler (local `.npy`) |
| **2** | `scripts/LAST_FM_02_train_true_final_joint_model_HGT_A5_H3_LEG_20260824.py` | Train TRUE FINAL: HGT+A5+H3+LEG, 3 seeds |
| **3** | `scripts/LAST_FM_03_train_ablation_layers_sampled_val_NDCG_20260824.py` | Train HGT / +A5 / +H3 ablations (sampled val NDCG) |
| **4** | `scripts/LAST_FM_04_fullrank_validation_frozen_checkpoints_20260824.py` | Full-catalog NDCG on frozen checkpoints |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

.venv/bin/python scripts/LAST_FM_00_prepare_antileak_splits_and_pair_tables_20260824.py
.venv/bin/python scripts/LAST_FM_01_materialize_A5_and_H3_statistical_features_20260824.py
.venv/bin/python scripts/LAST_FM_02_train_true_final_joint_model_HGT_A5_H3_LEG_20260824.py
.venv/bin/python scripts/LAST_FM_03_train_ablation_layers_sampled_val_NDCG_20260824.py
.venv/bin/python scripts/LAST_FM_04_fullrank_validation_frozen_checkpoints_20260824.py
```

**One GPU job at a time** on 16 GB M1 Air. Use `.venv/bin/python`.

## Frozen model

`HGT 64/2L/2H + pair256 + graph_dot + A5 + H3 + LEG_K2` → LateFusion 265-D.  
Details: [LASTFM_TRUE_FINAL/ARCHITECTURE.md](LASTFM_TRUE_FINAL/ARCHITECTURE.md).

## Data & config

- Raw data: `data/LastFM_star_IntentAwareRS/` (not the leaking KGAT dump)
- Protocol: `configs/lastfm_star_race_clean_3.yaml`
- Splits: `outputs/lastfm_star/splits/` (val 10% from train, seed 2026)
- **TEST locked** — never scored

## Metrics — do not mix

| Protocol | NDCG@20 | Use |
|---|---:|---|
| Sampled (20 negs) | ~0.87 | training, early stopping, ablation on val |
| Full catalog | ~0.28 | publication ranking on frozen ckpts |

## What is in git vs local only

**Committed:** code, config, Last-FM* data, splits, seed summaries, full-rank CSV/JSON.  
**Local only (gitignored):** checkpoints `.pt`, materialized `.npy`/`.pkl`, logs, old exploration dirs.

Reproduction notes: [docs/LASTFM_PROXY_REPRODUCTION.md](docs/LASTFM_PROXY_REPRODUCTION.md)

## Local cleanup

Remove legacy exploration files from disk:

```bash
bash scripts/cleanup_legacy_local.sh --yes
```
