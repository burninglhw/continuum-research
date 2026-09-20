import fcntl
import json
import sys
import time
from pathlib import Path

import requests

from core import Budget, Records, SafetyStop, prompt_token_upper_bound


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'run_config.json').read_text())
    run = root / 'runs/nowcoding-mixed-100-20260918'
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('AUTH_CHECK_READY', flush=True)
        key = sys.stdin.readline().strip()
        if not key:
            raise SafetyStop('Missing credential')
        records = Records(run, key)
        budget = Budget(records, config)
        messages = [{'role': 'user', 'content': 'Reply with exactly OK.'}]
        budget.reserve(prompt_token_upper_bound(messages), 128, 0, 'stream-transport-diagnostic')
        result = {'purpose': 'tiny streaming transport diagnostic; NOT a SWE-bench trace',
                  'requested_model': config['requested_model'], 'model_output_cap': 128}
        started = time.monotonic()
        chunks = []
        usage = None
        completed = False
        try:
            with requests.post('https://nowcoding.ai/v1/chat/completions',
                               headers={'Authorization': 'Bearer ' + key},
                               json={'model': config['requested_model'], 'messages': messages,
                                     'max_tokens': 128, 'stream': True, 'stream_options': {'include_usage': True}},
                               stream=True, timeout=(10, 45), allow_redirects=False) as response:
                result['http_status'] = response.status_code
                result['request_id'] = response.headers.get('x-oneapi-request-id') or response.headers.get('x-request-id')
                if response.status_code != 200:
                    result['body_excerpt'] = response.text[:2000]
                    raise SafetyStop('Diagnostic HTTP ' + str(response.status_code))
                for raw in response.iter_lines(chunk_size=1):
                    if time.monotonic() - started > 90:
                        raise SafetyStop('Diagnostic time limit reached')
                    if not raw.startswith(b'data:'):
                        continue
                    value = raw[5:].strip()
                    if value == b'[DONE]':
                        completed = True
                        break
                    event = json.loads(value)
                    records.event('stream-diagnostic-events.jsonl', {'event': event})
                    result.setdefault('first_event_seconds', time.monotonic() - started)
                    result['reported_model'] = event.get('model', result.get('reported_model'))
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event.get('choices', []):
                        content = choice.get('delta', {}).get('content')
                        if isinstance(content, str):
                            chunks.append(content)
                        if choice.get('finish_reason'):
                            result['finish_reason'] = choice['finish_reason']
            result.update(completed=completed, output=''.join(chunks), usage=usage)
            if not completed or result.get('finish_reason') != 'stop' or not chunks or usage is None:
                raise SafetyStop('Incomplete stream or missing usage')
            result['conservative_cost_cny'] = budget.settle(usage)
            result['status'] = 'passed'
        except Exception as error:
            result.update(status='failed', error_class=type(error).__name__, error=str(error))
        finally:
            result['elapsed_seconds'] = time.monotonic() - started
            if budget.state['pending']:
                result['reserved_as_spent_cny'] = budget.charge_unsettled_reservation('Uncertain streaming diagnostic; no retry')
            records.write('stream-diagnostic.json', result)
            print(records.encode(result), flush=True)


if __name__ == '__main__':
    main()
