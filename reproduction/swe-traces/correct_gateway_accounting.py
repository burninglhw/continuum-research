import fcntl
import json
import shutil
import time
from pathlib import Path

from core import Budget, Records, SafetyStop


def main():
    root = Path(__file__).resolve().parent
    run = root / 'runs/nowcoding-mixed-100-20260918'
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'billing-adjustments.jsonl').exists():
            raise SafetyStop('Historical cache accounting has already been reconciled')
        old = json.loads((run / 'config.snapshot.json').read_text())
        new = json.loads((root / 'run_config.json').read_text())
        new['local_image_manifest'] = old['local_image_manifest']
        if {**old, 'gateway_add_cached_input': True, 'per_task_stop_cny': 12} != new:
            raise SafetyStop('Unexpected settings change')
        records = Records(run)
        budget = Budget(records, new)
        if budget.state['pending']:
            raise SafetyStop('Pending request cannot be reconciled')
        backup = run / ('accounting-before-' + time.strftime('%Y%m%d-%H%M%S'))
        backup.mkdir()
        shutil.copyfile(run / 'budget.json', backup / 'budget.json')
        shutil.copyfile(run / 'config.snapshot.json', backup / 'config.json')
        total_adjustment = 0.0
        for path in sorted((run / 'instances').glob('*/responses.jsonl')):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            difference = sum(budget.cost(budget.input_tokens(row['usage']), row['usage']['completion_tokens'])
                             - row['conservative_cost_cny'] for row in rows)
            if difference < -1e-9:
                raise SafetyStop('Refusing an unexplained downward accounting adjustment')
            records.event('billing-adjustments.jsonl', {'task_id': path.parent.name,
                'amount_cny': difference, 'responses_reconciled': len(rows),
                'reason': 'Gateway reports cached input separately; add all cache buckets at full input rate',
                'raw_usage_records_modified': False, 'actual_provider_bill_known': False})
            total_adjustment += difference
        budget.state['charged_upper_estimate_cny'] += total_adjustment
        budget.save()
        records.write('config.snapshot.json', new)
        records.event('request-policy-changes.jsonl', {'per_task_stop_cny': 12,
            'global_budget_authorized_cny': 300, 'global_software_stop_cny': 270,
            'reason': 'Bounded per-task allocation widened after transport-debug reservations; global authorization unchanged'})
        print(json.dumps({'historical_cache_correction_cny': total_adjustment,
                          'charged_upper_estimate_cny': budget.state['charged_upper_estimate_cny']}))


if __name__ == '__main__':
    main()
