#!/usr/bin/env bash
# Run TRUE FINAL hardneg NOW (no HP wait). HP was paused by user.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p "$EXP/logs" "$EXP/artifacts" "$EXP/reports"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

export PYTHONUNBUFFERED=1
export TFHN_SEEDS="${TFHN_SEEDS:-101,202,303}"
export TFHN_LAMBDA_BPR="${TFHN_LAMBDA_BPR:-0.5}"
export TFHN_DEVICE="${TFHN_DEVICE:-cpu}"
export TFHN_MAX_POS="${TFHN_MAX_POS:-0}"

log "START TRUE FINAL hardneg NOW (HP paused; no wait)"
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

log "05 freeze official JOINT ckpts (external LOCKED)"
"$PY" -u "$EXP/scripts/05_freeze_external_ready.py" 2>&1 | tee "$EXP/logs/05_freeze.log"

echo "PROTOCOL_COMPLETE" > "$EXP/artifacts/PROTOCOL_COMPLETE.flag"
log "TRUE FINAL hardneg COMPLETE — ckpts frozen; external LOCKED until GO_EXTERNAL=YES"
log "JOINT: LASTFM_TRUE_FINAL/JOINT_HARDNEG_R3_FROM_SCRATCH_V1/06_CHECKPOINTS/"
log "Resume HP later via: bash experiments/lastfm_m1_hp_search_v1/scripts/run_hp_protocol.sh"
