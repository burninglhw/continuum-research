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
        config = json.loads((root / 'run_config.json').read_text())
        snapshot = json.loads((run / 'config.snapshot.json').read_text())
        budget = (run / 'budget.json').read_bytes()
        if json.loads(budget)['pending']:
            raise SafetyStop('Unsettled reservation')
        for settings in [config, snapshot]:
            if (settings['per_task_stop_cny'] != 12 or settings['budget_stop_cny'] != 270
                    or settings['budget_authorized_cny'] != 300):
                raise SafetyStop('Unexpected allocation; do not apply twice')
        archive = run / ('allocation-before-' + time.strftime('%Y%m%d-%H%M%S'))
        archive.mkdir()
        shutil.copyfile(root / 'run_config.json', archive / 'run_config.json')
        shutil.copyfile(run / 'config.snapshot.json', archive / 'config.snapshot.json')
        config['per_task_stop_cny'] = 30
        snapshot['per_task_stop_cny'] = 30
        Records(root).write('run_config.json', config)
        Records(run).write('config.snapshot.json', snapshot)
        Records(run).event('request-policy-changes.jsonl', {
            'per_task_stop_cny': 30, 'previous_per_task_stop_cny': 12,
            'global_budget_authorized_cny': 300, 'global_software_stop_cny': 270,
            'reason': 'Avoid repeated artificial mid-task stops; total authorization, 50 steps and 900 seconds unchanged',
            'user_total_budget_raised': False, 'old_charges_and_checkpoint_retained': True,
        })
        assert (run / 'budget.json').read_bytes() == budget
        print('TASK_ALLOCATION_UPDATED_TOTAL_BUDGET_UNCHANGED')


if __name__ == '__main__':
    main()
