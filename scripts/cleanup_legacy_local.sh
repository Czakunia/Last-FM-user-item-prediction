#!/usr/bin/env bash
# Remove legacy exploration files from LOCAL disk.
# Keeps: numbered LAST_FM_* scripts, internal deps, materialized .npy under RACE_CLEAN_3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

YES=0
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && YES=1
if [[ "$YES" != 1 ]]; then
  echo "This will DELETE legacy exploration dirs and non-whitelisted scripts."
  echo "Pass --yes to skip confirmation."
  read -r -p "Continue? [y/N] " ans
  [[ "${ans:-}" == [yY] ]] || { echo "Aborted."; exit 0; }
fi

echo "[cleanup] removing KRAM exploration trees …"
rm -rf KRAM_FINAL_WORK/BASE_HP_SCREEN KRAM_FINAL_WORK/NATIVE_MEASURE_SCREEN \
       KRAM_FINAL_WORK/L2H4B8192_MLP_KAN KRAM_FINAL_WORK/measure_race KRAM_FINAL_WORK/CLEAN_V2 2>/dev/null || true

RC3=KRAM_FINAL_WORK/RACE_CLEAN_3
for d in A11_DISTRIBUTIONAL_REPRESENTATION_SERIES_V1 A11_HGT_CONDITIONAL_FUNCTIONAL_FINAL_V1 \
         ARTIST_A11_LEVEL_V1 B0_MESSAGE_HISTORY_INTERACTION_SERIES_V1 B0_STRONG_RANKING_SERIES_V1 \
         HGT_ARTIST_EDGE_ABLATION_V1 HGT_GLOBAL_KG_CONTRIBUTION_ABLATION_V1 \
         HGT_NODE_RELATION_DEGREE_FEATURES_V1 HGT_NODE_SEMANTIC_ROLE_FEATURES_V1 \
         WIDE_A11_DEEP_LIGHTGCN_V1 LASTFM_FINAL_CLEAN_TRAINING_V1 \
         FINAL_HGT_CAPACITY_CONVERGENCE_RACE_V1 A11_FUNCTIONAL_DISTRIBUTION_BENCHMARK_V2; do
  rm -rf "$RC3/$d" 2>/dev/null || true
done
rm -rf "$RC3/audit" "$RC3/results" 2>/dev/null || true
find "$RC3/race" -mindepth 1 -maxdepth 1 -type d ! -name a11_top25 -exec rm -rf {} + 2>/dev/null || true

rm -rf outputs/lastfm outputs/lastfm_full outputs/lastfm_race_v1 2>/dev/null || true
rm -rf configs/contracts docs/kg_hcr.md docs/logic_task_a_card.md docs/protocol_lastfm_architecture_hcr_v1.md \
       docs/protocol_lastfm_lp_v1.md docs/protocol_lastfm_orthonormal_hcr_v2.md docs/task_a_logic.md \
       docs/transferability.md docs/README.md 2>/dev/null || true
rm -f configs/config.yaml configs/data_schema.yaml configs/lastfm_architecture_hcr_v1.yaml \
      configs/lastfm_full_sota_v2_higher_order_hcr.yaml configs/lastfm_lp_v1.yaml \
      configs/lastfm_orthonormal_hcr_v2.yaml configs/lastfm_path_h2_v1.yaml \
      configs/lastfm_full_sota_v1.yaml 2>/dev/null || true
rm -f "Soil_elements_regression_based_on_hyperspectral_data 2 (2).pdf" 2>/dev/null || true

declare -A KEEP
for f in scripts/PIPELINE_LAST_FM_20260824.txt scripts/INTERNAL_LASTFM_SCRIPTS_20260824.txt; do
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    KEEP["$line"]=1
  done < "$f"
done
for f in scripts/LAST_FM_*_20260824.py; do
  [[ -e "$f" ]] && KEEP[$(basename "$f")]=1
done
KEEP[cleanup_legacy_local.sh]=1

echo "[cleanup] removing non-whitelisted scripts …"
for f in scripts/*.py scripts/*.sh scripts/*.txt; do
  [[ -e "$f" ]] || continue
  base=$(basename "$f")
  [[ ${KEEP[$base]+x} ]] && continue
  [[ "$base" == FROZEN_* || "$base" == README_LASTFM* ]] && rm -f "$f" && continue
  rm -f "$f"
done
rm -f scripts/FROZEN_LASTFM_PROXY_SCRIPTS.txt scripts/README_LASTFM_PROXY.md 2>/dev/null || true

rm -f tests/test_native_measure_screen.py tests/test_h2_materialize_fast.py \
      tests/test_h2_fast_v2.py tests/test_h3_triple_backends.py tests/test_h3_user_batch.py \
      tests/test_orthonormal_hcr.py tests/test_orthonormal_hcr_triple.py 2>/dev/null || true

# Drop audit-only artifacts inside LASTFM_TRUE_FINAL (keep seed_summary + fullrank metrics)
find LASTFM_TRUE_FINAL -type d -name '00_AUDIT' -exec rm -rf {} + 2>/dev/null || true
find LASTFM_TRUE_FINAL -name 'FROM_SCRATCH_AUDIT.json' -delete 2>/dev/null || true
find LASTFM_TRUE_FINAL -name 'PROGRESS.txt' -delete 2>/dev/null || true

echo "[cleanup] done. Entry scripts:"
ls scripts/LAST_FM_*_20260824.py 2>/dev/null
