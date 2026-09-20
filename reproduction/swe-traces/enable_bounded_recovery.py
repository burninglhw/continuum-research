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
        old = json.loads((run / 'config.snapshot.json').read_text())
        new = json.loads((root / 'run_config.json').read_text())
        new['local_image_manifest'] = old['local_image_manifest']
        if old.get('automatic_api_retries') != 0 or {**old, 'automatic_api_retries': 2} != new:
            raise SafetyStop('Only one bounded recovery policy transition is permitted')
        budget = json.loads((run / 'budget.json').read_text())
        if budget['pending']:
            raise SafetyStop('Unsettled budget reservation')
        backup = run / ('config-before-recovery-' + time.strftime('%Y%m%d-%H%M%S') + '.json')
        shutil.copyfile(run / 'config.snapshot.json', backup)
        Records(run).write('config.snapshot.json', new)
        Records(run).event('request-policy-changes.jsonl', {
            'maximum_retries_per_generation': 2, 'total_attempts_per_generation_cap': 3,
            'all_uncertain_attempts_charged_at_reserved_upper_bound': True,
            'global_budget_cny': 300, 'software_stop_cny': 270, 'per_task_stop_cny': 8,
            'model_prompt_output_step_and_time_limits_unchanged': True,
            'trace_history_overwritten_or_replayed': False})
        print('BOUNDED_RECOVERY_ENABLED_WITH_UNCHANGED_BUDGET')


if __name__ == '__main__':
    main()
