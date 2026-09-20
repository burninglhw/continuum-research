import json
import sys
from pathlib import Path

import requests

from core import Records


def main():
    print('AUTH_CHECK_READY', flush=True)
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit('Missing credential')
    root = Path(__file__).resolve().parent / 'runs/nowcoding-mixed-100-20260918'
    records = Records(root, key)
    response = requests.get('https://nowcoding.ai/v1/models',
                            headers={'Authorization': 'Bearer ' + key},
                            timeout=(10, 45), allow_redirects=False)
    result = {'purpose': 'model metadata only; no generation request', 'http_status': response.status_code}
    if response.status_code == 200:
        result['models'] = [row.get('id') for row in response.json().get('data', [])]
    else:
        result['body_excerpt'] = response.text[:2000]
    records.event('gateway-metadata-checks.jsonl', result)
    print(records.encode(result))


if __name__ == '__main__':
    main()
