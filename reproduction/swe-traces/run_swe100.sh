#!/usr/bin/env bash
set -uo pipefail
umask 077
ROOT=/export/home/ext.luohaowen1/continuum/reproduction/swe-traces
RUN="$ROOT/runs/deepseek-flash-swe100-v2-20260918"
mkdir -p "$RUN"
stty -echo
trap 'stty echo' EXIT
/export/home/ext.luohaowen1/continuum/envs/continuum-traces/bin/python -u "$ROOT/run_swe100.py" "$@" 2>&1 | tee -a "$RUN/controller-console.log"
exit "${PIPESTATUS[0]}"
