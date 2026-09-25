#!/usr/bin/env bash
# Watchdog: keep EXTRA 404/505 chain alive until EXTRA_404_505_DONE.flag.
# Does NOT touch primary 101/202/303 claim / 07_EXTERNAL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/lastfm_true_final_hardneg_joint_v1"
LOG="$EXP/logs/watch_extra_404_505.log"
CHAIN="$EXP/scripts/run_extra_seeds_404_505.sh"
DONE="$EXP/artifacts/EXTRA_404_505_DONE.flag"
mkdir -p "$EXP/logs" "$EXP/artifacts"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export TFHN_DEVICE="${TFHN_DEVICE:-cpu}"
export TFHN_EXT_MAX_POS="${TFHN_EXT_MAX_POS:-500676}"
export TFHN_SEEDS=404,505
export TFHN_SCREEN_EARLY_STOP=1
export TFHN_SCREEN_MIN_EPOCH=5
export TFHN_SCREEN_PATIENCE=3
export TFHN_SCREEN_MIN_DELTA=0.002
export TFHN_FULLCAT_TOP_K=2
export TFHN_SKIP_CACHED=1
export GO_TRUE_FINAL_EXTERNAL=YES
export CONFIRM_UNSEAL_LASTFM_TEST=YES
export TFHN_REPLACE_TRUE_FINAL=DEFERRED

chain_alive() {
  pgrep -f 'run_extra_seeds_404_505\.sh' >/dev/null 2>&1 \
    || pgrep -f '03_eval_fullcat\.py' >/dev/null 2>&1 \
    || pgrep -f '08_run_extra_seeds_external\.py' >/dev/null 2>&1 \
    || pgrep -f '02_train_from_scratch\.py' >/dev/null 2>&1
}

log "WATCHDOG START pid=$$"
while [[ ! -f "$DONE" ]]; do
  if chain_alive; then
    log "ok — chain alive; sleep 600s"
    sleep 600
    continue
  fi
  log "CHAIN DEAD — resume run_extra_seeds_404_505.sh"
  # shell script skips completed FULLCAT seeds; train skips DONE seeds
  nohup /usr/bin/caffeinate -i /bin/bash "$CHAIN" >>"$EXP/logs/extra_404_505_chain.log" 2>&1 &
  log "respawned pid=$!"
  sleep 120
done

log "EXTRA_404_505_DONE seen — writing brief summary"
OUT="$ROOT/LASTFM_TRUE_FINAL/JOINT_HARDNEG_R3_FROM_SCRATCH_V1/08_EXTRA_SEEDS_404_505"
{
  echo "EXTRA 404/505 COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PRIMARY claim unchanged (101/202/303 mean ~0.2009)"
  for s in 404 505; do
    fc="$EXP/artifacts/seed${s}/FULLCAT.json"
    if [[ -f "$fc" ]]; then
      "$ROOT/.venv/bin/python" -c "import json;d=json.load(open('$fc'));print('DEV seed$s selected_ep',d.get('selected_epoch'),'NDCG@20',d.get('NDCG@20',d.get('ndcg@20')))"
    fi
  done
  if [[ -f "$OUT/sealed_test_results.json" ]]; then
    "$ROOT/.venv/bin/python" -c "import json;d=json.load(open('$OUT/sealed_test_results.json'));a=d.get('block_a',{}).get('true_final_hardneg_mean',{});print('EXTRA external mean NDCG@20',a.get('NDCG@20'))"
  elif [[ -f "$OUT/EXTRA_SUMMARY.json" ]]; then
    cat "$OUT/EXTRA_SUMMARY.json"
  else
    ls -la "$OUT" 2>/dev/null || echo "no OUT yet"
  fi
} | tee -a "$LOG" | tee "$EXP/reports/EXTRA_404_505_FINAL.txt"

log "WATCHDOG EXIT"
