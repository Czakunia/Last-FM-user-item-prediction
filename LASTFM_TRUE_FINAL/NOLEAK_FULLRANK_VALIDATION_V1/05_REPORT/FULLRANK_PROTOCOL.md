# FULLRANK_PROTOCOL

Generated 2026-08-20T06:59:23Z.

## Dataset
Corrected **Last-FM\*** (IntentAwareRS `--resolveDataLeakage yes`).  
Config: `configs/lastfm_star_race_clean_3.yaml`.  
TEST is read **only** for pair-overlap audit. TEST is **not scored**.

## Frozen model
`HGT_64D_2L_2H+PAIR256+GRAPH_DOT1+A5+H3+LEG_K2_RESIDUAL`  
Checkpoints: `/Users/martajasiewicz/Desktop/KGAT_KnowledgeGraphs/LASTFM_TRUE_FINAL/JOINT_TRAINING_V1/06_CHECKPOINTS/final_seed{101,202,303}_best.pt`  
Epoch = sampled-val NDCG@20 best from TRUE FINAL JOINT training. **Not re-selected after full-rank.**
C1 reproduction checkpoints are **not** evaluated here.

## Candidate universe
For user `u`: `C_u = {0, …, n_items-1} \ H_u^{model_train}`.  
No sampled negatives, no popularity/ANN/artist prune.  
Same item IDs for every seed.

## Train mask
Only `model_train` history. Validation positives are **not** masked.

## Relevance
Binary. `P_u` = **all** `valid.txt` positives. No capping. No test labels.

## Ranking
Score `s_final = s_B0 + delta_L2` on every `X ∈ C_u` in candidate batches of 4096.  
Sort: higher score first; **tie → lower item_id** (`publication_full_rank_evaluator.dense_topk`).  
Chunking changes memory layout only; ranking ≡ scoring the full `C_u` at once.

## Metrics
NDCG@K / Recall@K / Precision@K for K∈{5,10,20}.  
DCG@K = Σ_r rel_r / log2(r+1). IDCG@K = same for min(K, |P_u|) ones at the top.  
Recall@K = |TopK ∩ P_u| / |P_u|. Precision@K = |TopK ∩ P_u| / K.  
MRR = 1/rank of first relevant among **all eligible** (train-masked) items; 0 if none.  
MAP@20 = (1/|P_u|) Σ_{k: rel} P@k  (**standard denom**, not n_hits).  
Macro-mean over users with |P_u| ≥ 1. Users with 0 val positives are skipped.

## Users with |C_u| < K
Not expected (catalog ~48k, history << that). If it occurred, Precision still uses /K.

## Beyond-accuracy (secondary)
CatalogCoverage@20, ARP@20, LongTailShare@20 from Top-20.  
`pop(i)` = # model_train users with i. TAIL = below median among items with pop>0.
