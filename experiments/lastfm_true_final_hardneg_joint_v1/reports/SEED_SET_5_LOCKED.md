# Official seed set — LOCKED for reporting (pre-reviewer)

**Decision (2026-09-06):** report hardneg TRUE FINAL on **5 seeds**, not 3.

| Seed | Role |
|------|------|
| 101, 202, 303 | original primary triad |
| 404, 505 | added before any reviewer submission; **kept even if weaker** |

**Official claim (sealed extras DONE):**
- mean NDCG@20 over **{101, 202, 303, 404, 505}** = **0.2015**
- Δ vs easy TRUE FINAL (0.1718) = **+0.0297**
- **no dropping** of 303 (or any seed) post-hoc
- easy baseline comparison stays against the published easy mean (~0.1718)

See `artifacts/OFFICIAL_5SEED_EXTERNAL.json` and `reports/FIVE_SEED_AND_RANKING_COMPARE.md`.

**HParam track (`hardneg_hparam_v1`):** after pick on SCREEN (seed 101), Phase B / sealed external = **all 5 seeds** from scratch with winning `(lr, λ_BPR)`. No 3-seed shortcut.

**Replace official easy TRUE FINAL:** still DEFERRED until explicit GO.
