# Publication deletion audit — 2026-08-24
# Format: FILE | frozen | tests | docs | action | reason

configs/lastfm_star_proxy_20260824.yaml | yes | no | yes | KEEP | Self-contained publication protocol config
configs/lastfm_star_race_clean_3.yaml | no* | no | no | DELETE | Replaced by lastfm_star_proxy_20260824.yaml (*was loaded; refs updated)
configs/lastfm_full_sota_v1.yaml | no* | no | no | DELETE | Only existed as extends parent; inlined into proxy config
configs/lastfm_lp_v1.yaml | no* | no | no | DELETE | Parent of full_sota; inlined into proxy config
configs/lastfm_architecture_hcr_v1.yaml | no | no | no | DELETE | Exploratory HCR track
configs/lastfm_full_sota_v2_higher_order_hcr.yaml | no | no | no | DELETE | Exploratory
configs/lastfm_orthonormal_hcr_v2.yaml | no | no | no | DELETE | Exploratory
configs/lastfm_path_h2_v1.yaml | no | no | no | DELETE | Exploratory
configs/config.yaml | no | no | no | DELETE | Unrelated root config
configs/data_schema.yaml | no | no | no | DELETE | Unrelated
configs/contracts/ | no | no | no | DELETE | Unrelated

LASTFM_TRUE_FINAL/ARCHITECTURE.md | no | no | yes* | DELETE | Migrated into docs/LASTFM_PROXY_REPRODUCTION.md (single authoritative source)
LASTFM_TRUE_FINAL/README.md | no | no | yes | KEEP | Points at outputs; slim pointer OK
docs/LASTFM_PROXY_REPRODUCTION.md | no | no | yes | KEEP | Authoritative reproduction + architecture
docs/data_location.md | no | no | no | DELETE | Stale/exploratory
docs/kg_hcr.md | no | no | no | DELETE | Exploratory
docs/protocol_*.md | no | no | no | DELETE | Exploratory protocols

KRAM_FINAL_WORK/ (entire tree) | path-compat only | no | no | DELETE from publication tree | Features migrated to outputs/lastfm_star/materialized/; code uses that root
KRAM_FINAL_WORK/RACE_CLEAN_3/audit/RACE_CLEAN_3_FORMULAS.md | no | no | no | MOVE TO DOCS | Top25/A11 notes extracted into docs/LASTFM_PROXY_REPRODUCTION.md §4

scripts/run_lastfm_true_final_joint_training_v1.py | yes | no | no | KEEP | Subprocessed by LAST_FM_02 (exploratory name; NEEDS_RENAME later)
scripts/run_lastfm_joint_hgt_a5_ranker_v1.py | yes | no | no | KEEP | Subprocessed by LAST_FM_03
scripts/run_lastfm_noleak_fullrank_validation_v1.py | yes | no | no | KEEP | Subprocessed by LAST_FM_04
scripts/run_race_clean_3.py | yes | no | no | KEEP | Subprocessed by LAST_FM_01; shared_A_dir import
scripts/run_lastfm_noleak_fullrank_fast_v1.py | no | no | no | DELETE | Not on numbered path
scripts/run_lastfm_train_all_then_fullrank_v1.py | no | no | no | DELETE | Superseded by LAST_FM_03/04
scripts/run_lastfm_true_final_then_fullrank_v1.py | no | no | no | DELETE | Superseded
scripts/run_lastfm_stage_*.py | no | no | no | DELETE | Stage ladder exploration
scripts/run_kram_*.py | no | no | no | DELETE | KRAM exploration
scripts/detach_*.py | no | no | no | DELETE | Process helpers for old races
scripts/watchdog_*.py | no | no | no | DELETE | Old race watchdogs

src/lastfm_lp/binary/measure_race_selection.py | yes | yes | yes | KEEP | Top25 routing used by H3/LEG + tests via clean_v2.routing
src/lastfm_lp/torch_device.py | yes | no | no | KEEP | Device resolution for train/eval

tests/test_clean_v2.py | yes | yes | yes | KEEP | Publication feature/routing tests
tests/test_publication_full_rank_evaluator.py | yes | yes | yes | KEEP | Full-rank evaluator unit tests
tests/test_h2_*.py | no | no | no | DELETE | Exploratory H2
tests/test_h3_*.py | no | no | no | DELETE | Exploratory H3 backends
tests/test_orthonormal_hcr*.py | no | no | no | DELETE | Exploratory HCR
tests/test_native_measure_screen.py | no | no | no | DELETE | Exploratory screen
