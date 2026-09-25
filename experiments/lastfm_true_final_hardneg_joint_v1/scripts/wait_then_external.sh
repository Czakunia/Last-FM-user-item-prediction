#!/usr/bin/env bash
# Wait for DEV protocol complete, then run hardneg sealed external (GO already granted).
# REPLACE_TRUE_FINAL stays DEFERRED — this only unseals evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p "$EXP/logs" "$EXP/artifacts"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ ! -f "$EXP/artifacts/GO_TRUE_FINAL_EXTERNAL.flag" ]]; then
  log "FATAL: missing GO_TRUE_FINAL_EXTERNAL.flag"
  exit 1
fi

if [[ -f "$EXP/artifacts/EXTERNAL_DONE.flag" ]]; then
  log "EXTERNAL already done — stop"
  exit 0
fi

log "WAIT for PROTOCOL_COMPLETE (DEV freeze+manifest); also wait so we don't fight FULLCAT CPU"
while [[ ! -f "$EXP/artifacts/PROTOCOL_COMPLETE.flag" ]]; do
  sleep 60
done
log "PROTOCOL_COMPLETE seen"

if [[ ! -f "$EXP/artifacts/FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json" ]]; then
  log "FATAL: missing FINAL_TRUE_FINAL_HARDNEG_MANIFEST.json"
  exit 1
fi

# Ensure eval/summarize chain finished (resume writes PROTOCOL after 04)
if pgrep -f '03_eval_fullcat.py' >/dev/null 2>&1; then
  log "WAIT for leftover 03_eval_fullcat to exit"
  while pgrep -f '03_eval_fullcat.py' >/dev/null 2>&1; do
    sleep 60
  done
fi

log "07 sealed external (hardneg) — REPLACE_TRUE_FINAL=DEFERRED"
export PYTHONUNBUFFERED=1
export GO_TRUE_FINAL_EXTERNAL=YES
export CONFIRM_UNSEAL_LASTFM_TEST=YES
export TFHN_REPLACE_TRUE_FINAL=DEFERRED
export TFHN_DEVICE="${TFHN_DEVICE:-cpu}"
"$PY" -u "$EXP/scripts/07_run_sealed_external.py" 2>&1 | tee "$EXP/logs/07_external.log"

echo "EXTERNAL_DONE" > "$EXP/artifacts/EXTERNAL_DONE.flag"
log "EXTERNAL done. REPLACE_TRUE_FINAL still DEFERRED until user decides."
