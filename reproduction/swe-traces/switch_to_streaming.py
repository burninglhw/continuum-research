import fcntl
import json
import shutil
import time
from pathlib import Path

from core import Records, SafetyStop


def main():
    root = Path(__file__).resolve().parent
    run = root / 'runs/nowcoding-mixed-100-20260918'
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads((run / 'stream-diagnostic.json').read_text())
        if report.get('status') != 'passed':
            raise SafetyStop('Streaming diagnostic has not passed')
        old = json.loads((run / 'config.snapshot.json').read_text())
        if old.get('stream_api'):
            raise SafetyStop('Transport transition already applied')
        new = json.loads((root / 'run_config.json').read_text())
        new['local_image_manifest'] = old['local_image_manifest']
        if {**old, 'stream_api': True} != new:
            raise SafetyStop('Only the verified transport change is permitted')
        if json.loads((run / 'budget.json').read_text())['pending']:
            raise SafetyStop('Budget reservation is still pending')
        for case in (run / 'instances').iterdir():
            metrics = json.loads((case / 'metrics.json').read_text())
            if metrics['model_turns'] != 0 or (case / 'responses.jsonl').exists() or (case / 'stream-partial.jsonl').exists():
                raise SafetyStop('Cannot change transport after receiving an agent response')
        history = run / ('transport-transition-' + time.strftime('%Y%m%d-%H%M%S'))
        history.mkdir()
        shutil.copyfile(run / 'config.snapshot.json', history / 'config.before.json')
        shutil.copyfile(run / 'provenance.json', history / 'provenance.before.json')
        for case in list((run / 'instances').iterdir()):
            destination = run / 'failed-attempts' / case.name / (history.name + '-nonstream')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / 'provenance.json', case / 'campaign-provenance.json')
            case.rename(destination)
        Records(run).write('config.snapshot.json', new)
        Records(run).event('transport-changes.jsonl', {
            'old_transport': 'nonstream', 'new_transport': 'sse_with_final_usage',
            'successful_traces_before_change': 0, 'budget_ledger_reset': False,
            'reason': 'Nonstream generation failed twice; separate bounded streaming diagnostic succeeded',
            'diagnostic_request_id': report.get('request_id'),
            'model_prompt_and_generation_limits_changed': False})
        print('VERIFIED_STREAM_TRANSPORT_ENABLED_WITH_EXISTING_BUDGET')


if __name__ == '__main__':
    main()
