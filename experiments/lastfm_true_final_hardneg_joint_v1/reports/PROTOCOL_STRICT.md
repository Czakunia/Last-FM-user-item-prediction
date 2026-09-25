# TRUE_FINAL_HARDNEG_JOINT_V1 — protocol (strict)

- Architecture: TRUE FINAL HGT+A5+H3+LEG **from scratch**
- Negatives: R3 (2 hard A11 [5%,30%) + 1 pop + 1 random), ~500k positives
- Loss: BCE + 0.5·BPR · Seeds 101/202/303
- MAX_EPOCHS=**20**, save **init + every epoch**, no **train** early stop
- SCREEN_3K: fixed user list (same hash all seeds/epochs)
- SCREEN_3K **eval early stop**: after min_epoch=5, stop if last 3 consecutive epoch-to-epoch NDCG@20 gains are all `< 0.002` (skip remaining screen epochs)
- FULLCAT candidates (**rule B**): top-2 SCREEN_3K epochs (best + 2nd); seed101 kept earlier broader set if already frozen
- Freeze: argmax FULLCAT DEV NDCG@20 → `TRUE_FINAL_HARDNEG_seed{}_FROZEN`
- Manifest: `FINAL_TRUE_FINAL_HARDNEG_MANIFEST`
- External: **STOP** until `GO_TRUE_FINAL_EXTERNAL=YES`

Loop: 1× HGT encode/epoch (BCE+BPR on same z_proxy). See `LOOP_CONTRACT.md`.
