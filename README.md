# Last-FM* — HGT + A5 + H3 + LEG (final joint model)

Reproduction package for the **official** Last-FM* ranker of the thesis:
Heterogeneous Graph Transformer plus candidate-specific statistics
(A5, H3, second-order Legendre residual \(L_2\)), trained with hard
negatives (R3).

This is **not** the parameter-free \(\phi\) ranker (second GitHub package).
It is **not** the unofficial SVD-fed / stats-only diagnostics.

## Official numbers (do not mix protocols)

| Protocol | NDCG@20 |
|---|---:|
| Development full-catalogue, random negatives, 3 seeds | 0.2764 |
| Development full-catalogue, hard negatives, 5 seeds | **0.2819** |
| Sealed extra, random negatives, 3 seeds | 0.1718 |
| Sealed extra, hard negatives, 5 seeds (101/202/303/404/505) | **0.2015** |

Sampled validation NDCG@20 (~0.87) is only a checkpoint proxy. Never compare
it to full-catalogue or extra.

Per-seed extra: see `expected_results.json`.

## Data shipped in this repo

| Path | What it is |
|---|---|
| `data/LastFM_star_IntentAwareRS/` | Leakage-corrected Last-FM* from IntentAwareRS (`--resolveDataLeakage yes`). Train/test/KG lists. |
| `outputs/lastfm_star/splits/` | Frozen model-train / valid / test used in the thesis (valid carved from train, seed **2026**). |

SHA-256 fingerprints: `SHA256_DATA.json`.

Source of Last-FM*:
[IntentAwareRS](https://github.com/Faisalse/IntentAwareRS)
(Shehzad, Ferrari Dacrema, Jannach, SIGIR 2025, DOI:10.1145/3726302.3730307).
Do **not** use the original leaking KGAT Last-FM dump.

## Numbered pipeline

Run from the package root after `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

| Step | Script | Does |
|---:|---|---|
| 1 | `pipeline/01_prepare_lastfm_star_antileak_splits_and_pair_tables.py` | Rebuild splits (must match shipped hashes). TEST not scored. |
| 2 | `pipeline/02_materialize_A5_H3_and_LEG_statistical_features_on_model_train.py` | Cross-fitted A5 / H3 / \(L_2\) on model-train pairs. |
| 3 | `pipeline/03_build_item_A11_neighbourhoods_and_hard_negative_R3_pairs.py` | R3 hard-negatives (2× φ-band + pop + random) and SCREEN_3K. |
| 4 | `pipeline/04_train_HGT_A5_H3_LEG_from_scratch_official_seeds.py` | Train 20 epochs, seeds 101–505. HGT never sees A5/H3. |
| 5 | `pipeline/05_select_checkpoint_on_development_full_catalogue.py` | Freeze `selected_epoch` on DEV full-catalogue NDCG@20. |
| 6 | `pipeline/06_refit_on_train_external_and_evaluate_sealed_holdout.py` | `TRAIN_EXTERNAL = model_train ∪ valid`, then sealed extra. |

Step 6 requires:

```bash
export GO_TRUE_FINAL_EXTERNAL=YES
export CONFIRM_UNSEAL_LASTFM_TEST=YES
```

Each pipeline file has a long module docstring: inputs, outputs, and the
exact scientific role of that step.

## Architecture (frozen)

```
s(u,X) = MLP_265→128→64→1( [ q_struct (257) ‖ A5 (5) ‖ H3 (3) ] ) + δ_LEG(L2)
```

HGT: 64-D, 2 layers, 2 heads, dropout 0.1. CKG: user–item interactions plus
at most 250_000 KG edges. Loss: BCE + 0.5 BPR. Adam 1e-3, wd 1e-4.

A5 / H3 / \(L_2\) definitions match the thesis appendix (Last-FM* statistical
branch). Cross-fitting: user \(u\) is excluded from the co-occurrence tables
used to score \(u\).

## Hardware

Full-catalogue extra scores ~23.5k users × ~48k items. Plan many hours per
seed on CPU. One training job at a time on 16 GB machines.

## What is intentionally not in git

Checkpoints (`*.pt`), materialised `.npy` features, and training logs.
Regenerate with steps 2–4. The sealed **labels** *are* included (Last-FM*
`test.txt`) so extra is reproducible; do not use them before step 6.


## Changelog (this repository)

| Date | What |
|---|---|
| 2026-08-24 | First publish: random-negative sealed extra **NDCG@20 = 0.1718** (`scripts/LAST_FM_*_20260824.py`). |
| 2026-09 | Official thesis update: hard-negative R3 training, 5 seeds, sealed extra **NDCG@20 = 0.2015**. Use numbered `pipeline/01`–`06`. |

Archived random-negative sealed discussion:
[`reports/LASTFM_RESULTS_DISCUSSION_20260824.md`](reports/LASTFM_RESULTS_DISCUSSION_20260824.md).
Sealed JSON from that protocol:
`LASTFM_EXTERNAL_BENCHMARK_20260824/sealed_test_results/`.

The parameter-free φ ranker lives in a **separate** repo:
[`Czakunia/lastfm-star-softmax-phi-ranker`](https://github.com/Czakunia/lastfm-star-softmax-phi-ranker).

