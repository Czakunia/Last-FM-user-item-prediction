#!/usr/bin/env bash
# After run_now finishes summarize, freeze official JOINT ckpts (external LOCKED).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p "$EXP/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "wait for SUMMARY.json from hardneg protocol"
while [[ ! -f "$EXP/artifacts/SUMMARY.json" ]]; do
  sleep 60
done
# also need all three FULLCAT
for s in 101 202 303; do
  while [[ ! -f "$EXP/artifacts/seed${s}/FULLCAT.json" ]]; do
    sleep 60
  done
done

log "05 freeze official JOINT ckpts"
"$PY" -u "$EXP/scripts/05_freeze_external_ready.py" 2>&1 | tee "$EXP/logs/05_freeze.log"
echo "EXTERNAL_FREEZE_DONE" > "$EXP/artifacts/EXTERNAL_FREEZE_DONE.flag"
log "done — external still LOCKED"
