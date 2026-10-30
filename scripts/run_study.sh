#!/usr/bin/env bash
# Full certkkt study (CPU only, resumable). Every finished unit is one fsynced line in
# $RUN_DIR/units.jsonl; a rerun skips finished units. Retries up to 3 times on failure.
# Smoke test:  MAX_UNITS=3 RUN_DIR=tmp/longrun-smoke SUITES="portfolio" bash scripts/run_study.sh
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$(dirname "${LOG:-runs/longrun.log}")"
LOG=${LOG:-runs/longrun.log}
RUN_DIR=${RUN_DIR:-runs/full}
WORKERS=${WORKERS:-3}
SUITES=${SUITES:-"mm cblib portfolio mittelmann"}
EXTRA=()
[ -n "${MAX_UNITS:-}" ] && EXTRA+=(--max-units "$MAX_UNITS")
SUITE_ARGS=()
FETCH=()
for s in $SUITES; do
  SUITE_ARGS+=(--suite "$s")
  case $s in portfolio) FETCH+=(french) ;; *) FETCH+=("$s") ;; esac
done

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') longrun: $*" >> "$LOG"; }

log "start run_dir=$RUN_DIR suites=[$SUITES] workers=$WORKERS"
for attempt in 1 2 3; do
  uv run --locked certkkt fetch "${FETCH[@]}" >> "$LOG" 2>&1 \
    && uv run --locked certkkt study run --profile full "${SUITE_ARGS[@]}" \
         --run-dir "$RUN_DIR" --workers "$WORKERS" "${EXTRA[@]}" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    log "study finished on attempt $attempt"
    uv run --locked certkkt study report --profile full --run-dir "$RUN_DIR" \
      --out "$RUN_DIR/report.md" --json "$RUN_DIR/summary.json" >> "$LOG" 2>&1
    log "report written to $RUN_DIR/report.md"
    exit 0
  fi
  log "attempt $attempt failed (exit $rc); retrying in 60 s"
  sleep 60
done
log "giving up after 3 attempts"
exit 1
