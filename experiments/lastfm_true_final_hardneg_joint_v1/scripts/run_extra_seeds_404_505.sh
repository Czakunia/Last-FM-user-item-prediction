#!/usr/bin/env bash
# EXTRA seeds 404/505 — exploratory only. Does NOT enter primary hardneg external mean.
# Does NOT overwrite PROTOCOL_COMPLETE / FINAL_TRUE_FINAL_HARDNEG_MANIFEST / 07_EXTERNAL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p "$EXP/logs" "$EXP/artifacts"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TFHN_SEEDS=404,505
export TFHN_DEVICE="${TFHN_DEVICE:-cpu}"
export TFHN_MAX_EPOCHS="${TFHN_MAX_EPOCHS:-20}"
export TFHN_LAMBDA_BPR="${TFHN_LAMBDA_BPR:-0.5}"
export TFHN_SCREEN_EARLY_STOP=1
export TFHN_SCREEN_MIN_EPOCH=5
export TFHN_SCREEN_PATIENCE=3
export TFHN_SCREEN_MIN_DELTA=0.002
export TFHN_FULLCAT_TOP_K=2
export TFHN_SKIP_CACHED=1
export TFHN_EXT_MAX_POS="${TFHN_EXT_MAX_POS:-500676}"

log "EXTRA 404/505 START — NOT part of primary 101/202/303 claim"

log "02 train from scratch seeds=404,505"
"$PY" -u "$EXP/scripts/02_train_from_scratch.py" 2>&1 | tee -a "$EXP/logs/02_train_extra_404_505.log"

for seed in 404 505; do
  if [[ -f "$EXP/artifacts/seed${seed}/FULLCAT.json" ]]; then
    log "03 skip seed=$seed (FULLCAT exists)"
    continue
  fi
  log "03 SCREEN_3K + FULLCAT seed=$seed"
  TFHN_SEED="$seed" "$PY" -u "$EXP/scripts/03_eval_fullcat.py" 2>&1 | tee "$EXP/logs/03_eval_s${seed}.log"
done

log "08 EXTRA sealed external (separate tree; not primary mean)"
export GO_TRUE_FINAL_EXTERNAL=YES
export CONFIRM_UNSEAL_LASTFM_TEST=YES
export TFHN_REPLACE_TRUE_FINAL=DEFERRED
"$PY" -u "$EXP/scripts/08_run_extra_seeds_external.py" 2>&1 | tee -a "$EXP/logs/08_extra_404_505.log"

echo "EXTRA_404_505_DONE" > "$EXP/artifacts/EXTRA_404_505_DONE.flag"
log "EXTRA DONE — see JOINT_HARDNEG…/08_EXTRA_SEEDS_404_505/ (NOT in primary mean)"
