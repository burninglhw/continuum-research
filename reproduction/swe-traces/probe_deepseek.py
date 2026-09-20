import fcntl
import json
import os
import sys
import time
from pathlib import Path

import requests

from core import Budget, Records, SafetyStop, prompt_token_upper_bound
from gateway_stream import read_completion


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/deepseek-flash-100-20260918'


def main():
    os.umask(0o077)
    config = json.loads((ROOT / 'deepseek_config.json').read_text())
    if config['api_base'] != 'https://api.deepseek.com' or config['requested_model'] != 'deepseek-flash':
        raise SafetyStop('Unexpected DeepSeek endpoint or model')
    print('AUTH_CHECK_READY', flush=True)
    key = sys.stdin.readline().strip()
    if not key:
        raise SafetyStop('Missing credential')
    records = Records(RUN, key)
    with (RUN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN / 'interface-smoke.json').exists():
            raise SafetyStop('Smoke result already exists; review before repeating a paid request')
        previous = ROOT / 'runs/nowcoding-mixed-100-20260918'
        with (previous / 'run.lock').open('a') as previous_lock:
            fcntl.flock(previous_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prior_budget = json.loads((previous / 'budget.json').read_text())
            if prior_budget['pending'] or prior_budget['charged_upper_estimate_cny'] != config['prior_cost_carryover_cny']:
                raise SafetyStop('Previous budget changed or has an unsettled request')
            if not (RUN / 'budget.json').exists():
                records.write('budget.json', {'charged_upper_estimate_cny': prior_budget['charged_upper_estimate_cny'],
                    'pending': None, 'requests': 0, 'prior_provider_estimate_cny': prior_budget['charged_upper_estimate_cny']})
                records.write('budget-carryover.json', {'source': str(previous / 'budget.json'),
                    'source_budget': prior_budget, 'actual_old_provider_bill_verified': False})
        budget = Budget(records, config)
        messages = [{'role': 'user', 'content': 'Reply with exactly one bash code block containing: printf DEEPSEEK_OK'}]
        payload = {'model': config['requested_model'], 'messages': messages, 'max_tokens': 2048,
                   'thinking': config['thinking'], 'reasoning_effort': config['reasoning_effort'],
                   'stream': True, 'stream_options': {'include_usage': True}}
        budget.reserve(prompt_token_upper_bound(messages), payload['max_tokens'], 0, 'deepseek-interface-smoke')
        records.write('interface-smoke-request.json', payload)
        result = {'api_base': config['api_base'], 'requested_model': config['requested_model'],
                  'purpose': 'interface smoke; not a SWE-bench trace'}
        try:
            started = time.perf_counter()
            with requests.post(config['api_base'] + '/chat/completions',
                               headers={'Authorization': 'Bearer ' + key}, json=payload,
                               timeout=(10, 90), allow_redirects=False, stream=True) as response:
                result['http_status'] = response.status_code
                if response.status_code != 200:
                    result['error_excerpt'] = response.text[:2000]
                    raise SafetyStop('DeepSeek HTTP ' + str(response.status_code))
                data = read_completion(response, records, 1, started, 90)
            result['reported_model'] = data.get('model')
            result['usage'] = data.get('usage') or {}
            result['conservative_cost_cny'] = budget.settle(result['usage'])
            result['response'] = data
            result['finish_reason'] = data['choices'][0]['finish_reason']
            result['nonempty_content'] = bool(data['choices'][0]['message']['content'].strip())
            if (not result['nonempty_content'] or result['finish_reason'] != 'stop'
                    or result['reported_model'] not in config['accepted_reported_models']):
                raise SafetyStop('DeepSeek model or completion validation failed')
            result['validated'] = True
        except Exception as error:
            result['validated'] = False
            result['error_class'] = type(error).__name__
            result['error'] = str(error)
        finally:
            if budget.state['pending']:
                result['uncertain_cost_cny'] = budget.charge_unsettled_reservation('Smoke outcome uncertain; no automatic retry')
            records.write('interface-smoke.json', result)
        print(records.encode({name: value for name, value in result.items() if name != 'response'}), flush=True)
        if not result.get('validated'):
            raise SystemExit(2)


if __name__ == '__main__':
    main()
