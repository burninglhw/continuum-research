import fcntl
import json
import sys
from pathlib import Path

import requests

from core import Budget, Records, prompt_token_upper_bound


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'run_config.json').read_text())
    print('AUTH_CHECK_READY', flush=True)
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit('Missing key')
    records = Records(root / 'runs/nowcoding-mixed-100-20260918', key)
    lock = (records.root / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget = Budget(records, config, allow_pending=True)
    if budget.state['pending']:
        budget.charge_unsettled_reservation('Carry forward previous request at its full reserved upper estimate')
    messages = [{'role': 'user', 'content': 'Reply with exactly OK.'}]
    budget.reserve(prompt_token_upper_bound(messages), 128, 0, 'interface-smoke')
    result = {'requested_model': config['requested_model'], 'purpose': 'interface smoke; not a SWE-bench trace'}
    try:
        response = requests.post(
            'https://nowcoding.ai/v1/chat/completions',
            headers={'Authorization': 'Bearer ' + key},
            json={'model': config['requested_model'], 'messages': messages, 'max_tokens': 128, 'stream': False},
            timeout=(10, 60), allow_redirects=False,
        )
        result['http_status'] = response.status_code
        payload = response.json()
        if response.status_code == 200:
            result['reported_model'] = payload.get('model')
            result['usage'] = payload.get('usage', {})
            result['conservative_cost_cny'] = budget.settle(result['usage'])
            result['finish_reason'] = payload.get('choices', [{}])[0].get('finish_reason')
            result['nonempty_content'] = bool(payload.get('choices', [{}])[0].get('message', {}).get('content'))
        else:
            error = payload.get('error', {})
            if isinstance(error, dict):
                result['error_type'] = error.get('type')
                result['error_code'] = error.get('code')
                result['error_message'] = str(error.get('message', ''))[:600]
            result['action'] = 'STOP: requested model not verified; no fallback to another model'
    except Exception as error:
        result['error_class'] = type(error).__name__
        result['action'] = 'STOP: retain reservation; reconcile before retry'
    if budget.state['pending']:
        result['reserved_as_spent_cny'] = budget.charge_unsettled_reservation('Failed smoke request; actual charge unknown')
    records.write('interface-smoke.json', result)
    print(records.encode(result), flush=True)
    lock.close()


if __name__ == '__main__':
    main()
