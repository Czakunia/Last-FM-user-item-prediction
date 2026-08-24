# Last-FM* proxy reproduction (frozen 2026-08-24)

Branch: `lastfm-proxy`  
Dataset: Last-FM* (`data/LastFM_star_IntentAwareRS/`)  
Config: `configs/lastfm_star_race_clean_3.yaml`

---

## Pipeline (run scripts 0 → 4 in order)

All entry points live in `scripts/LAST_FM_*_20260824.py`.  
Each script’s docstring states prerequisites, outputs, and what not to mix.

### Step 0 — splits

Requires raw Last-FM* under `data/LastFM_star_IntentAwareRS/`.  
Writes `outputs/lastfm_star/` (model_train, valid from train only, test preserved).

### Step 1 — statistical features

Writes precomputed tables under `KRAM_FINAL_WORK/RACE_CLEAN_3/` (paths are historical;  
large `.npy` files stay local — regenerate with step 1 after clone).

| Block | Dim | Role |
|---|---:|---|
| A5 | 5 | overlap + popularity / degree |
| H3 | 3 | signed A11 pool on Top25 |
| LEG scaler | 1 | P2-mean input for TRUE FINAL residual |

### Step 2 — TRUE FINAL

Frozen architecture: HGT 64/2L/2H + A5 + H3 + LEG.  
Output: `LASTFM_TRUE_FINAL/JOINT_TRAINING_V1/` (checkpoints gitignored).

### Step 3 — ablations

Progressive layers for thesis comparison: HGT → +A5 → +H3 (no LEG).  
Checkpoint metric: **sampled** val NDCG@20.

### Step 4 — full-rank

Full-catalog ranking for TRUE FINAL + all ablation checkpoints.  
Publication metric: **full-catalog** NDCG@20. TEST never scored.

---

## Verification

```bash
.venv/bin/python -m pytest tests/test_clean_v2.py tests/test_publication_full_rank_evaluator.py -q
```

---

## Internal scripts (do not run directly)

Listed in `scripts/INTERNAL_LASTFM_SCRIPTS_20260824.txt`.  
Use only the numbered `LAST_FM_*` entry points above.

---

## Intentionally excluded from git

- KRAM audit CSV/MD from old exploration (redo audit separately if needed)
- Neural checkpoints, `.npy`, logs
- Legacy `scripts/run_lastfm_stage_*` and measure-race ladders
