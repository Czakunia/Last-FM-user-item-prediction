# Frozen Last-FM* architecture — formulas for the thesis

Status: **frozen**. Do not retune. This is the TRUE FINAL model:

`HGT_64D_2L_2H + pair256 + graph_dot1 + A5 + H3 + LEG_K2 residual`

Code:

- encoder `src/lastfm_lp/models/hgt_encoder.py`
- model `scripts/run_final_hgt_capacity_convergence_race_v1.py` (`FinalHGTModel`)
- fusion `src/lastfm_lp/models/fusion/late_fusion.py`
- A5 `src/lastfm_lp/clean_v2/tabular_true.py`
- H3 pool `src/lastfm_lp/clean_v2/pooling.py`
- A11 / routing measures `src/lastfm_lp/binary/binary_measures.py`
- Top25 `src/lastfm_lp/binary/measure_race_selection.py`
- spec `configs/lastfm_star_race_clean_3.yaml` + `LASTFM_TRUE_FINAL/ARCHITECTURE.md`

---

## 1. What the model answers

Each example is a pair `(u, X)` with unknown label at inference.

- `H_u` = items of `u` in **model_train** only.
- `X` = candidate item. The user need not have seen `X`. The system must know `X` (id, KG, popularity, co-occurrence with other items).
- Training: positive train interactions + **4 random negatives** per positive, excluding `H_u` and known positives.
- Loss: `BCEWithLogits`, `pos_weight = n_neg / n_pos`.
- Target: user–item link prediction / recommendation, **not** KG edge prediction (`artist → genre`).

Score:

```
s(u,X) = s_B0(u,X) + δ_LEG(u,X)
```

`s_B0` is the 265-D late-fusion MLP. `δ_LEG` is a 1-D residual (zero at init).

---

## 2. Graph and HGT (learned)

Heterogeneous CKG (`build_ckg_graph.py`):

- node types: `user`, `entity` (items + KG entities; items are a prefix of entities)
- edges: `user —interact→ entity`, reverse `interact_rev`, plus KG relations and their reverses
- KG edges capped at `max_kg_edges = 250_000` (deterministic subsample)

HGT (PyG `HGTConv`, frozen capacity):

- embedding dim **64**, **2** layers, **2** heads, dropout **0.1**
- ReLU + Dropout after every layer **except the last**
- stock HGT skip (learnable sigmoid mix)
- readout: Identity at d=64 (no extra Linear)

One forward produces node embeddings `z`. Then:

```
z_u = z[user]
z_X = z[item]
```

These are **not** tabular statistics. They are learned from train interactions + KG.

---

## 3. Pair context and graph_dot (from z)

```
g_ctx(u,X) = [ z_u  ‖  z_X  ‖  z_u ⊙ z_X  ‖  |z_u − z_X| ]   ∈ R^{256}
graph_dot  = ⟨z_u, z_X⟩                                         ∈ R^1
```

Together with A5 and H3 this is the 265-D decoder input.

---

## 4. Two different “Jaccard / Cosine / A11” worlds

**Do not mix these.** Same names, different operands.

| Level | Operands | Where used in the **final** model |
|---|---|---|
| **A — item sets** | `H_u` vs neighborhood `N_X` | A5 features 4 and 5 |
| **B — user sets** | incidence `U_h`, `U_X` via `n11, n_h, n_X, N` | Top25 **routing** (which history items enter H3/LEG) and the A11 **values** that are pooled |

Cross-fit: for user `u`, all of `n11, n_h, n_X, N, N_X` come from the other 4 folds (`D_train \ fold(u)`). Fold seed 2026.

`N_X = { j ≠ X : n11(j,X) > 0 }`.

---

## 5. A5 — tabular 5-D (CLEAN V2, frozen)

```
A(u,X) = [
  log(1 + |H_u|),          # user
  log(1 + pop(X)),         # candidate
  log(1 + deg_KG(X)),      # candidate
  J_A(u,X),                # pair, LEVEL A
  C_A(u,X)                 # pair, LEVEL A
]
```

```
|H ∩ N_X| = #{ h ∈ H_u : n11(h,X) > 0 }     # n11 is thresholded; count strength is lost

J_A = |H ∩ N_X| / |H ∪ N_X|                 # 0 if union empty
C_A = |H ∩ N_X| / sqrt(|H| · |N_X|)         # 0 if H or N_X empty
```

This **is** set Jaccard/cosine, but between the **history item set** and the **co-occurrence neighbourhood of X**. It is **not** user-incidence Jaccard `n11/(n_h+n_X-n11)`.

StandardScaler on A5 is fit on **model_train pair rows only**.

### Legacy A5 (NOT in the final model)

`src/lastfm_lp/features/tabular_pair_features.py` still exists. Its yaml names `history_candidate_jaccard/cosine` were:

- support-rate: mean `1[n11>0]` (optionally on 40 history items)
- pop-profile: `pop(X)·mean(pop(H)) / (‖pop(H)‖₂ · max(pop(X),1))`  
  which **cancels X** when `pop(X)>0`

Do **not** describe the frozen model with those two proxies. Do **not** claim they won a race against real Jaccard/cosine.

---

## 6. LEVEL B measures (contingency 2×2)

For history item `h` and candidate `X`, on the leave-one-fold population:

```
n11 = n11(h,X)
n10 = n_h − n11
n01 = n_X − n11
n00 = N − n_h − n_X + n11
```

### Used to **select Top25** (routing). Winner in the final model: **A11**.

After Top25 is chosen, routing scores are **discarded**. Downstream always evaluates **signed A11** on the selected items.

| Policy | Formula | In final model? |
|---|---|---|
| **A11 / Pearson φ** | `(N n11 − n_h n_X) / sqrt( n_h(N−n_h) n_X(N−n_X) )` | **Yes — routing + H3/LEG values** |
| Jaccard (user-incidence) | `n11 / (n_h + n_X − n11)` | Race only |
| Cosine (user-incidence) | `n11 / sqrt(n_h n_X)` | Race only |
| Co-occurrence | `n11` | Race only |
| MI (nats, unsmoothed) | `Σ p_ab ln(p_ab / (p_a· p_·b))` | Race only |
| NPMI_11 | normalized MI on the (1,1) cell; `n11=0 → −1` | Race only |
| G² unsigned | likelihood-ratio on the 2×2 | Race only |

Top25 rule (all policies): score descending, tie → **lower item id**.  
`K = min(25, |H_u \ {X}|)`. Candidate `X` never enters its own Top25.

The Jaccard/Cosine/A11/MI **race** answers: *does the routing measure change which history items are selected, when the downstream representation of that Top25 stays signed-A11 pooling?*  
It does **not** answer: *are MI features better than A11 features?*

---

## 7. H3 — 3-D pool of signed A11 on Top25

Let `v_k = a11(h_k, X)` for the K selected history items (signed, no abs).

```
H3(u,X) = [ mean(v),  max(v),  top3mean(v) ]
```

`top3mean` = mean of the `min(3,K)` **largest signed** values (not largest `|v|`).  
Empty history → `[0,0,0]`. No `wmean`. Dim = 3.

StandardScaler on H3: train pairs only.

---

## 8. LEG_K2 — 1-D residual (winner A11 branch)

Same Top25, same signed A11 values `v_k`. Legendre P2:

```
P2(x) = (3x² − 1) / 2
L2(u,X) = mean_k P2(v_k)
```

Scaled with the frozen `LEG_K2_scaler` (train-only). Residual MLP:

```
Linear(1 → 16) → GELU → Linear(16 → 1)
```

Last layer **zero-initialized**, so at init `δ_LEG = 0` and `s = s_B0`.

```
s(u,X) = s_B0(u,X) + MLP(L2(u,X))
```

---

## 9. Decoder (B0)

Concatenation:

```
[ g_ctx (256)  ‖  graph_dot (1)  ‖  A5 (5)  ‖  H3 (3) ]  =  265
```

MLP (`LateFusionHead`):

```
265 → 128 → LayerNorm → GELU → Dropout(0.2)
    → 64  → GELU → Dropout(0.2)
    → 1
```

Optimizer (TRUE FINAL): Adam lr `1e-3`, wd `1e-4`, ReduceLROnPlateau factor 0.5 patience 3, min_lr `6.25e-5`. Checkpoint = sampled val NDCG@20. Full-rank is evaluation only, not selection.

---

## 10. Branches that are **not** in the final model

Keep them in the “what we tried” chapter, not in the architecture diagram.

- **Legacy A5 proxies** (support-rate / pop-profile cosine) — superseded by CLEAN V2 set Jaccard/cosine.
- **Routing Jaccard / Cosine / MI / NPMI / G² / raw n11** — selection alternatives; frozen routing is **A11 Top25**.
- **KAN fusion head** — same 265-D input, spline readout; not the final decoder.
- **LightGCN / Wide-deep** capacity race — not the frozen encoder.
- **TEMP / extra HGT layers / 4 heads / 128-D** — capacity race; frozen is 64 / 2L / 2H.
- **C1 training semantics** — HGT gradient from the first 4096 shuffled pairs/epoch only. Control, not the final trainer. TRUE FINAL uses gradient from **all** train pairs.

---

## 11. One-sentence picture for the figure

HGT embeds users and items on the train CKG. For each candidate `X`, A5 describes cheap set-overlap of `H_u` with the co-occurrence neighbourhood of `X`. Independently, A11 (Pearson φ) ranks the user’s history, keeps Top25, pools those signed φ values into H3, and maps their P2-mean through a zero-init residual. The MLP reads `[pair(z), ⟨z_u,z_X⟩, A5, H3]`; LEG adds a small correction. Ranking is over the full catalog minus `H_u`.
