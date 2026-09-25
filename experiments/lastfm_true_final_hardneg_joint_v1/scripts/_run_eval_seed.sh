#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
PY="$ROOT/.venv/bin/python"
SEED="$1"
export PYTHONUNBUFFERED=1
export TFHN_SEED="$SEED"
export TFHN_DEVICE=cpu
export TFHN_SKIP_CACHED=1
export TFHN_SCREEN_EARLY_STOP=1
export TFHN_SCREEN_MIN_EPOCH=5
export TFHN_SCREEN_PATIENCE=3
export TFHN_SCREEN_MIN_DELTA=0.002
LOG="$EXP/logs/03_eval_s${SEED}.log"
echo "[$(date '+%F %T')] START seed=$SEED pid=$$" | tee -a "$EXP/logs/resume_eval_es.log"
set +e
"$PY" -u "$EXP/scripts/03_eval_fullcat.py" >"$LOG" 2>&1
ec=$?
echo "[$(date '+%F %T')] END seed=$SEED exit=$ec" | tee -a "$EXP/logs/resume_eval_es.log"
exit $ec
