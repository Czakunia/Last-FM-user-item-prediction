#!/usr/bin/env bash
# TRUE_FINAL_HARDNEG_JOINT_V1 — strict protocol (stops before external).
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
export TFHN_MAX_EPOCHS="${TFHN_MAX_EPOCHS:-20}"
export TFHN_MAX_POS="${TFHN_MAX_POS:-0}"

log "WAIT for DATA_READY (01 may still be running)"
while [[ ! -f "$EXP/artifacts/DATA_READY.flag" ]]; do
  sleep 30
done

log "00 freeze SCREEN_3K users"
"$PY" -u "$EXP/scripts/00_freeze_screen3k_users.py" 2>&1 | tee "$EXP/logs/00_screen3k.log"

log "01b R3 negative audit"
"$PY" -u "$EXP/scripts/01b_audit_negatives.py" 2>&1 | tee "$EXP/logs/01b_audit.log"

log "02 train from scratch MAX_EPOCHS=$TFHN_MAX_EPOCHS seeds=$TFHN_SEEDS"
"$PY" -u "$EXP/scripts/02_train_from_scratch.py" 2>&1 | tee "$EXP/logs/02_train.log"

for seed in ${TFHN_SEEDS//,/ }; do
  log "03 SCREEN_3K + FULLCAT candidates seed=$seed"
  TFHN_SEED="$seed" "$PY" -u "$EXP/scripts/03_eval_fullcat.py" 2>&1 | tee "$EXP/logs/03_eval_s${seed}.log"
done

log "04 summarize + FINAL_TRUE_FINAL_HARDNEG_MANIFEST"
"$PY" -u "$EXP/scripts/04_summarize.py" 2>&1 | tee "$EXP/logs/04_summarize.log"

echo "PROTOCOL_COMPLETE" > "$EXP/artifacts/PROTOCOL_COMPLETE.flag"
log "STOP. EXTERNAL_SEEN=NO. Require GO_TRUE_FINAL_EXTERNAL=YES (not generic GO_EXTERNAL)."
