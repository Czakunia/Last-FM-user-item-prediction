#!/usr/bin/env bash
# Full TRUE FINAL hardneg protocol — runs AFTER M1 HP (does not interrupt HP).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
HP_DONE="$ROOT/experiments/lastfm_m1_final_v1/artifacts/AWAITING_GO_EXTERNAL.flag"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p "$EXP/logs" "$EXP/artifacts" "$EXP/reports"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "WAIT for M1 HP protocol (AWAITING_GO_EXTERNAL.flag) — do not touch HP"
while [[ ! -f "$HP_DONE" ]]; do
  sleep 120
done
log "HP done flag seen — start TRUE FINAL hardneg track"

export PYTHONUNBUFFERED=1
export TFHN_SEEDS="${TFHN_SEEDS:-101,202,303}"
export TFHN_LAMBDA_BPR="${TFHN_LAMBDA_BPR:-0.5}"
export TFHN_DEVICE="${TFHN_DEVICE:-cpu}"
export TFHN_MAX_POS="${TFHN_MAX_POS:-0}"

log "01 build full hardneg pairs + features"
"$PY" -u "$EXP/scripts/01_build_hardneg_data.py" 2>&1 | tee "$EXP/logs/01_build_data.log"

log "02 train from scratch seeds=$TFHN_SEEDS"
"$PY" -u "$EXP/scripts/02_train_from_scratch.py" 2>&1 | tee "$EXP/logs/02_train.log"

for seed in ${TFHN_SEEDS//,/ }; do
  log "03 fullcat screen+eval seed=$seed"
  TFHN_SEED="$seed" "$PY" -u "$EXP/scripts/03_eval_fullcat.py" 2>&1 | tee "$EXP/logs/03_eval_s${seed}.log"
done

log "04 summarize"
"$PY" -u "$EXP/scripts/04_summarize.py" 2>&1 | tee "$EXP/logs/04_summarize.log"

echo "PROTOCOL_COMPLETE" > "$EXP/artifacts/PROTOCOL_COMPLETE.flag"
log "TRUE FINAL hardneg protocol COMPLETE — external still LOCKED"
