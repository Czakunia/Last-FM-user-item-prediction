#!/usr/bin/env bash
# Resume SCREEN_3K + FULLCAT + summarize (train already done). Uses screen eval early-stop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"
mkdir -p "$EXP/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

export PYTHONUNBUFFERED=1
export TFHN_SEEDS="${TFHN_SEEDS:-101,202,303}"
export TFHN_DEVICE="${TFHN_DEVICE:-cpu}"
export TFHN_SKIP_CACHED="${TFHN_SKIP_CACHED:-1}"
export TFHN_SCREEN_EARLY_STOP=1
export TFHN_SCREEN_MIN_EPOCH="${TFHN_SCREEN_MIN_EPOCH:-5}"
export TFHN_SCREEN_PATIENCE="${TFHN_SCREEN_PATIENCE:-3}"
export TFHN_SCREEN_MIN_DELTA="${TFHN_SCREEN_MIN_DELTA:-0.002}"
export TFHN_FULLCAT_TOP_K="${TFHN_FULLCAT_TOP_K:-2}"

log "RESUME eval with SCREEN_3K early_stop min_epoch=$TFHN_SCREEN_MIN_EPOCH patience=$TFHN_SCREEN_PATIENCE min_delta=$TFHN_SCREEN_MIN_DELTA FULLCAT_TOP_K=$TFHN_FULLCAT_TOP_K (rule B)"

for seed in ${TFHN_SEEDS//,/ }; do
  if [[ -f "$EXP/artifacts/seed${seed}/FULLCAT.json" ]]; then
    log "03 skip seed=$seed (FULLCAT.json exists)"
    continue
  fi
  log "03 SCREEN_3K + FULLCAT candidates seed=$seed"
  TFHN_SEED="$seed" "$PY" -u "$EXP/scripts/03_eval_fullcat.py" 2>&1 | tee "$EXP/logs/03_eval_s${seed}.log"
done

log "04 summarize + FINAL_TRUE_FINAL_HARDNEG_MANIFEST"
"$PY" -u "$EXP/scripts/04_summarize.py" 2>&1 | tee "$EXP/logs/04_summarize.log"

echo "PROTOCOL_COMPLETE" > "$EXP/artifacts/PROTOCOL_COMPLETE.flag"
log "STOP. EXTERNAL_SEEN=NO. Require GO_TRUE_FINAL_EXTERNAL=YES."
