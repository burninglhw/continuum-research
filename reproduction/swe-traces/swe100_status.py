import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / 'runs/deepseek-flash-swe100-v2-20260918'


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    metrics = [read_json(path) for path in (ROOT / 'instances').glob('*/metrics.json')]
    state = read_json(ROOT / 'status.json') or {}
    active = ROOT / 'instances' / state.get('task', '_none')
    activity = {}
    if active.exists():
        for name in ['responses.jsonl', 'response-validation.jsonl', 'retry-events.jsonl']:
            path = active / name
            activity[name] = sum(1 for line in path.open()) if path.exists() else 0
    budget = read_json(ROOT / 'budget.json') or {}
    result = {
        'run_dir': str(ROOT), 'target': 100,
        'completed': sum(bool(entry.get('collection_completed')) for entry in metrics),
        'recorded_incomplete': sum(not entry.get('collection_completed') for entry in metrics),
        'controller': read_json(ROOT / 'controller-status.json'), 'collector': state,
        'active_case_events': activity,
        'new_run_conservative_cost_cny': budget.get('charged_upper_estimate_cny', 0) - budget.get('prior_provider_estimate_cny', 0),
        'all_runs_conservative_cost_cny': budget.get('charged_upper_estimate_cny'),
        'pending_request': budget.get('pending'),
        'pilot_acceptance': read_json(ROOT / 'pilot-acceptance.json'),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
