# Data used by the Last-FM* HGT + stats model

## 1. Last-FM* (IntentAwareRS)

Directory: `LastFM_star_IntentAwareRS/`

| File | Role |
|---|---|
| `train.txt` | Official leakage-corrected train interactions |
| `test.txt` | Official leakage-corrected holdout (sealed extra) |
| `kg_final.txt` | Knowledge-graph triples for the CKG |
| `user_list.txt` / `item_list.txt` / `entity_list.txt` / `relation_list.txt` | ID maps |
| `MANIFEST.json` | Fingerprints vs IntentAwareRS (`train.txt` hash `b3584137980a372a`, leak users 17290 → 0 after correction) |

Paper: Shehzad, Ferrari Dacrema, Jannach, *A Worrying Reproducibility Study of Intent-Aware Recommendation Models*, SIGIR 2025.

## 2. Thesis splits (derived from train only)

These live in `outputs/lastfm_star/splits/` in the package root.

| File | Role |
|---|---|
| `model_train.txt` | 1_111_803 interactions — graph, histories, statistics, negatives |
| `valid.txt` | 124_102 interactions — DEV full-catalogue / checkpoint selection |
| `test.txt` | 306_914 interactions — sealed extra only |
| `eval_users.json` | Users eligible for development ranking |
| `meta.json` | `seed=2026`, `validation_ratio=0.1`, `min_train_items=5` |

`TRAIN_EXTERNAL` for extra is the **union** of `model_train` and `valid`
(1_235_905 interactions). That union is not stored as a third file; step 06
builds it in memory.

SHA-256 of every shipped file: `../SHA256_DATA.json`.
