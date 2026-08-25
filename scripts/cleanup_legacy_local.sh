#!/usr/bin/env bash
# Optional local cleanup for leftovers outside the publication tree (2026-08-24).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
YES=0
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && YES=1
if [[ "$YES" != 1 ]]; then
  echo "Removes legacy KRAM_FINAL_WORK and non-whitelisted scripts if they reappear."
  read -r -p "Continue? [y/N] " ans
  [[ "${ans:-}" == [yY] ]] || { echo "Aborted."; exit 0; }
fi
rm -rf KRAM_FINAL_WORK 2>/dev/null || true
KEEPFILE=scripts/INTERNAL_LASTFM_SCRIPTS_20260824.txt
declare -A KEEP
while IFS= read -r f; do
  [[ -z "$f" || "$f" =~ ^# ]] && continue
  KEEP[$f]=1
done < "$KEEPFILE"
for f in scripts/LAST_FM_*_20260824.py; do [[ -e "$f" ]] && KEEP[$(basename "$f")]=1; done
KEEP[cleanup_legacy_local.sh]=1
KEEP[PIPELINE_LAST_FM_20260824.txt]=1
KEEP[INTERNAL_LASTFM_SCRIPTS_20260824.txt]=1
KEEP[_lastfm_pipeline_common_20260824.py]=1
KEEP[_lastfm_paths_20260824.py]=1
for f in scripts/*; do
  [[ -f "$f" ]] || continue
  base=$(basename "$f")
  [[ ${KEEP[$base]+x} ]] && continue
  rm -f "$f"
done
echo "Done."
