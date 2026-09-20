#!/usr/bin/env bash
set -uo pipefail
umask 077
ROOT=/export/home/ext.luohaowen1/continuum/reproduction/swe-traces
PYTHON=/export/home/ext.luohaowen1/continuum/envs/continuum-traces/bin/python
RUN="$ROOT/runs/deepseek-flash-100-20260918"
cd "$ROOT" || exit 1
stty -echo
trap 'stty echo' EXIT
"$PYTHON" -u collect.py --config "$ROOT/deepseek_config.json" --run-dir "$RUN" \
    --limit 3 --key-stdin "$@" 2>&1 | tee -a "$RUN/pilot-console.log"
status=${PIPESTATUS[0]}
"$PYTHON" summarize.py --run-dir "$RUN" > "$RUN/summary-console.json"
"$PYTHON" quality.py --run-dir "$RUN" > "$RUN/quality-console.json"
"$PYTHON" -c 'import json,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"collector_exit_code":int(sys.argv[2]),"finished_at_unix":time.time(),"stage":"three-task-pilot"}))' \
    "$RUN/controller-exit.json" "$status"
printf '\nDEEPSEEK_PILOT_EXIT=%s\n' "$status"
exit "$status"
