import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('/export/home/ext.luohaowen1/continuum/reproduction/swe-traces')
RUN = ROOT / 'runs/deepseek-flash-swe100-v2-20260918'
SESSION = 'continuum-deepseek-swe100-v2-20260918-r2'
REASON = ('User explicitly requested removal of all spending caps on 2026-09-18 and will '
          'top up the provider account as needed. Disable per-task and global monetary '
          'stops, including the previous 270/300 CNY limits. Preserve accounting, '
          'all previous attempts, identical API parameters and the fixed 100-task sample.')


def main():
    print('AUTH_CHECK_READY', flush=True)
    key = sys.stdin.readline().strip()
    if not key:
        raise RuntimeError('Missing credential')
    command = shlex.join(['bash', str(ROOT / 'run_swe100.sh'), '--resume-reviewed',
                          '--reviewed-budget-change', REASON])
    subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, command], check=True)
    ready = False
    for attempt in range(60):
        pane = subprocess.check_output(['tmux', 'capture-pane', '-p', '-t', SESSION], text=True)
        if 'AUTH_CHECK_READY' in pane:
            ready = True
            break
        time.sleep(1)
    if not ready:
        raise RuntimeError('Controller did not request its credential')
    buffer = SESSION + '-credential'
    try:
        subprocess.run(['tmux', 'load-buffer', '-b', buffer, '-'], input=key + '\n', text=True, check=True)
        subprocess.run(['tmux', 'paste-buffer', '-d', '-b', buffer, '-t', SESSION + ':0.0'], check=True)
    finally:
        subprocess.run(['tmux', 'delete-buffer', '-b', buffer], capture_output=True)
        key = ''
    for attempt in range(60):
        state = json.loads((RUN / 'controller-status.json').read_text())
        if state.get('state') == 'collecting':
            finalizer = shlex.join([sys.executable, str(ROOT / 'finalize_swe100.py')])
            subprocess.run(['tmux', 'new-session', '-d', '-s', 'continuum-swe100-finalize-20260918-r2', finalizer], check=True)
            print(json.dumps({'session': SESSION, 'controller': state, 'credential_saved_to_file': False}), flush=True)
            return
        time.sleep(1)
    raise RuntimeError('Controller did not enter collection; inspect its redacted logs')


if __name__ == '__main__':
    main()
