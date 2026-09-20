#!/usr/bin/env bash
set -uo pipefail
umask 077
ROOT=/export/home/ext.luohaowen1/continuum/reproduction/swe-traces
PYTHON=/export/home/ext.luohaowen1/continuum/envs/continuum-traces/bin/python
RUN="$ROOT/runs/nowcoding-mixed-100-20260918"
cd "$ROOT"
stty -echo
"$PYTHON" -u collect.py --limit 100 --resume-incomplete --prepare-on-demand \
    --group-by-dependencies --release-temporary-images --key-stdin 2>&1 | tee -a "$RUN/batch-console.log"
status=${PIPESTATUS[0]}
"$PYTHON" summarize.py > "$RUN/summary-console.json"
"$PYTHON" quality.py > "$RUN/quality-console.json"
"$PYTHON" -c 'import json,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"collector_exit_code":int(sys.argv[2]),"finished_at_unix":time.time()}))' \
    "$RUN/controller-exit.json" "$status"
printf '\nCONTINUUM_BATCH_EXIT=%s\n' "$status"
tmux wait-for -S continuum-swe100-20260918-completed
exit "$status"
