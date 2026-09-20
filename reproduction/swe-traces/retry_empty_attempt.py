import argparse
import fcntl
import json
import time
from pathlib import Path

from core import Records, SafetyStop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_id')
    parser.add_argument('--reason', required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = root / 'runs/nowcoding-mixed-100-20260918'
    tasks = {json.loads(line)['instance_id'] for line in (root / 'tasks-100.jsonl').read_text().splitlines()}
    if arguments.instance_id not in tasks:
        raise SafetyStop('Task not in fixed selection')
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        case = run / 'instances' / arguments.instance_id
        metrics = json.loads((case / 'metrics.json').read_text())
        if metrics['model_turns'] != 0 or (case / 'responses.jsonl').exists():
            raise SafetyStop('Only zero-response attempts can be reviewed for a clean restart')
        errors = [json.loads(line) for line in (case / 'api-errors.jsonl').read_text().splitlines()]
        if not errors or errors[-1]['http_status'] not in [502, 503, 504]:
            raise SafetyStop('Only reviewed transient gateway failures are eligible')
        budget = json.loads((run / 'budget.json').read_text())
        if budget['pending']:
            raise SafetyStop('Unsettled reservation must be reconciled first')
        archives = run / 'failed-attempts' / arguments.instance_id
        if archives.exists() and list(archives.iterdir()):
            raise SafetyStop('One reviewed retry already used; no further automatic retries')
        archives.mkdir(parents=True, exist_ok=True)
        destination = archives / time.strftime('%Y%m%d-%H%M%S')
        case.rename(destination)
        Records(run).event('reviewed-restarts.jsonl', {'instance_id': arguments.instance_id,
            'reason': arguments.reason, 'archived_attempt': str(destination),
            'charged_upper_estimate_cny_retained': budget['charged_upper_estimate_cny'],
            'output_received': False, 'maximum_reviewed_restarts_per_task': 1})
        print('EMPTY_ATTEMPT_ARCHIVED_WITH_BUDGET_RETAINED')


if __name__ == '__main__':
    main()
