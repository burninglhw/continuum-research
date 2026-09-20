import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_events(path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def snapshot(run):
    status = read_json(run / 'status.json', {})
    metrics = [read_json(path) for path in sorted((run / 'instances').glob('*/metrics.json'))]
    completed = [row for row in metrics if row.get('collection_completed')
                 and row.get('model_turns', 0) > 0 and row.get('tool_calls', 0) > 0]
    active = run / 'instances' / status['task'] if 'task' in status else None
    responses = read_events(active / 'responses.jsonl') if active else []
    tools = read_events(active / 'tools.jsonl') if active else []
    locked = False
    if (run / 'run.lock').exists():
        with (run / 'run.lock').open('r') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                locked = True
    budget = read_json(run / 'budget.json', {})
    return {
        'collector_lock_held': locked,
        'completed_traces': len(completed),
        'target_traces': 100,
        'submitted_traces': sum(row.get('submitted', False) for row in completed),
        'status': status,
        'current_successful_model_turns': len(responses),
        'current_agent_tool_calls': sum(event.get('phase') == 'agent' for event in tools),
        'last_response_at': responses[-1].get('recorded_at') if responses else None,
        'charged_upper_estimate_cny': budget.get('charged_upper_estimate_cny'),
        'pending_reservation': budget.get('pending'),
        'free_gib': round(shutil.disk_usage(run).free / 1024 ** 3, 2),
        'incomplete_task_ids': [row['instance_id'] for row in metrics
                                if not row.get('collection_completed')],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'runs/nowcoding-mixed-100-20260918')
    parser.add_argument('--watch-seconds', type=int, default=0)
    parser.add_argument('--interval', type=int, default=30)
    arguments = parser.parse_args()
    deadline = time.monotonic() + arguments.watch_seconds
    previous = None
    while True:
        result = snapshot(arguments.run_dir)
        content = json.dumps(result, ensure_ascii=False, sort_keys=True)
        if content != previous:
            print(content, flush=True)
            previous = content
        if not result['collector_lock_held'] or result['completed_traces'] == 100 or time.monotonic() >= deadline:
            return
        time.sleep(min(max(arguments.interval, 1), max(deadline - time.monotonic(), 0)))


if __name__ == '__main__':
    main()
