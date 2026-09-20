import json
import subprocess
import sys
import time
from pathlib import Path

from core import Records


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/deepseek-flash-swe100-v2-20260918'


def main():
    records = Records(RUN)
    deadline = time.monotonic() + 24 * 60 * 60
    while time.monotonic() < deadline:
        state_path = RUN / 'controller-status.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('state') in ('completed', 'stopped'):
            break
        time.sleep(30)
    else:
        records.write('dataset-finalization.json', {'state': 'waiting_timeout', 'complete_100': False})
        return
    try:
        for script in ['summarize.py', 'quality.py', 'export_dataset.py']:
            with (RUN / (script[:-3] + '-final.log')).open('w') as log:
                subprocess.run([sys.executable, str(ROOT / script), '--run-dir', str(RUN)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((RUN / 'dataset/manifest.json').read_text())
        records.write('dataset-finalization.json', {
            'state': 'complete' if manifest['complete_100'] else 'partial',
            'complete_100': manifest['complete_100'], 'validated_records': manifest['validated_records'],
            'traces': str(RUN / 'dataset/traces.jsonl'), 'controller_state': state.get('state'),
        })
    except Exception as error:
        records.write('dataset-finalization.json', {'state': 'failed', 'complete_100': False,
                                                   'error_class': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
