import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from core import Budget, Records, SafetyStop, prompt_token_upper_bound
from gateway_stream import read_completion
from trace_protocol import response_problem


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/deepseek-flash-swe100-v2-20260918'
CONFIG = ROOT / 'deepseek_swe100_config.json'
PREVIOUS_RUN = ROOT / 'runs/deepseek-flash-100-20260918'


def initialize_budget(records, config):
    with (PREVIOUS_RUN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((PREVIOUS_RUN / 'budget.json').read_text())
        if previous['pending'] or previous['charged_upper_estimate_cny'] != config['prior_cost_carryover_cny']:
            raise SafetyStop('Previous-run accounting changed; reconcile before creating the new budget')
        if not (RUN / 'budget.json').exists():
            records.write('budget.json', {'charged_upper_estimate_cny': previous['charged_upper_estimate_cny'],
                'pending': None, 'requests': 0, 'prior_provider_estimate_cny': previous['charged_upper_estimate_cny']})
            records.write('budget-carryover.json', {'source': str(PREVIOUS_RUN / 'budget.json'),
                'source_state': previous, 'actual_provider_bill_verified': False})


def account_snapshot(key, records, name):
    with requests.get('https://api.deepseek.com/user/balance',
                      headers={'Authorization': 'Bearer ' + key}, timeout=(10, 30),
                      allow_redirects=False) as response:
        if response.status_code != 200:
            raise SafetyStop('Balance query failed with HTTP ' + str(response.status_code))
        value = response.json()
    records.write(name, value)
    if not value.get('is_available'):
        raise SafetyStop('DeepSeek account balance is unavailable or exhausted')
    return value


def smoke(key, records, config):
    path = RUN / 'interface-smoke.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value.get('validated') and value.get('api_base') == config['api_base'] and value.get('requested_model') == config['requested_model']:
            return
        raise SafetyStop('Previous interface smoke failed; review before another paid attempt')
    budget = Budget(records, config)
    messages = [{'role': 'user', 'content': 'Return exactly one bash code block containing printf DEEPSEEK_OK. Stop after its closing code fence.'}]
    payload = {'model': config['requested_model'], 'messages': messages,
               'thinking': config['thinking'], 'reasoning_effort': config['reasoning_effort'],
               'stop': config['stop_sequences'], 'max_tokens': 2048, 'stream': True,
               'stream_options': {'include_usage': True}}
    budget.reserve(prompt_token_upper_bound(messages), 2048, 0, 'interface-smoke-v2')
    records.write('interface-smoke-request.json', payload)
    result = {'api_base': config['api_base'], 'requested_model': config['requested_model'], 'validated': False}
    try:
        started = time.perf_counter()
        with requests.post(config['api_base'] + '/chat/completions',
                           headers={'Authorization': 'Bearer ' + key}, json=payload,
                           timeout=(10, 90), allow_redirects=False, stream=True) as response:
            result['http_status'] = response.status_code
            if response.status_code != 200:
                result['error_excerpt'] = response.text[:2000]
                raise SafetyStop('Interface smoke HTTP ' + str(response.status_code))
            data = read_completion(response, records, 0, started, 90, save_events=True)
        result['response'] = data
        result['conservative_cost_cny'] = budget.settle(data['usage'])
        result['reported_model'] = data.get('model')
        choice = data['choices'][0]
        result['nonempty_content'] = bool(choice['message']['content'].strip())
        problem = response_problem(choice['message']['content'], choice['finish_reason'])
        if problem or data.get('model') not in config['accepted_reported_models']:
            raise SafetyStop(problem or 'Unexpected reported model')
        result['validated'] = True
    finally:
        if budget.state['pending']:
            result['reserved_as_spent_cny'] = budget.charge_unsettled_reservation('Smoke failure preserved; no automatic smoke retry')
        records.write('interface-smoke.json', result)


def collect_stage(key, records, count, release=False, resume_reviewed=False, budget_change_reason=None):
    records.write('controller-status.json', {'state': 'collecting', 'target': count})
    command = [sys.executable, '-u', str(ROOT / 'collect.py'), '--config', str(CONFIG),
               '--run-dir', str(RUN), '--limit', str(count), '--prepare-on-demand', '--key-stdin']
    if release:
        command.extend(['--group-by-dependencies', '--release-temporary-images'])
    if resume_reviewed:
        command.append('--resume-incomplete')
    if budget_change_reason:
        command.extend(['--reviewed-budget-change', budget_change_reason])
    with (RUN / 'collector-console.log').open('a') as log:
        outcome = subprocess.run(command, input=key + '\n', text=True, stdout=log, stderr=subprocess.STDOUT)
    if outcome.returncode:
        raise SafetyStop('Collector stopped; see status.json and collector-console.log')


def verify_pilot():
    selected = [json.loads(line) for line in (ROOT / 'tasks-100.jsonl').read_text().splitlines()][:3]
    for task in selected:
        case = RUN / 'instances' / task['instance_id']
        metrics = json.loads((case / 'metrics.json').read_text())
        responses = {event['turn']: event for event in map(json.loads, (case / 'responses.jsonl').read_text().splitlines())}
        validations = [json.loads(line) for line in (case / 'response-validation.jsonl').read_text().splitlines()]
        if not metrics.get('collection_completed') or not metrics.get('non_submission_tool_calls'):
            raise SafetyStop('Pilot did not produce a completed tool-using trace')
        for event in validations:
            response = responses[event['turn']]
            if event['accepted'] and response_problem(response['content'], response['finish_reason']):
                raise SafetyStop('An accepted pilot response failed the protocol audit')


def summarize_run():
    for script in ['summarize.py', 'quality.py']:
        with (RUN / (script[:-3] + '-console.json')).open('w') as log:
            subprocess.run([sys.executable, str(ROOT / script), '--run-dir', str(RUN)],
                           stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-reviewed', action='store_true')
    parser.add_argument('--reviewed-budget-change')
    arguments = parser.parse_args()
    os.umask(0o077)
    print('AUTH_CHECK_READY', flush=True)
    key = sys.stdin.readline().strip()
    if not key:
        raise SafetyStop('Missing credential')
    config = json.loads(CONFIG.read_text())
    if config['api_base'] != 'https://api.deepseek.com' or config['requested_model'] != 'deepseek-flash':
        raise SafetyStop('Unexpected endpoint/model')
    if ((config['budget_authorized_cny'] is not None and config['budget_authorized_cny'] > 300)
            or (config['budget_stop_cny'] is not None and config['budget_stop_cny'] > 270)):
        raise SafetyStop('Budget authorization exceeded')
    records = Records(RUN, key)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {'state': 'stopped', 'started_at_unix': time.time()}
        try:
            balance = account_snapshot(key, records, 'account-at-start.json')
            print(records.encode({'account': balance, 'target': 100, 'run_dir': str(RUN)}), flush=True)
            initialize_budget(records, config)
            records.write('launch-config.json', config)
            records.write('selection.json', {'task_count': 100, 'seed': 42,
                'tasks_sha256': hashlib.sha256((ROOT / 'tasks-100.jsonl').read_bytes()).hexdigest(),
                'replaces_failed_samples': False, 'paper_generator': 'gpt-5', 'actual_generator': 'deepseek-flash'})
            smoke(key, records, config)
            collect_stage(key, records, 3, resume_reviewed=arguments.resume_reviewed,
                          budget_change_reason=arguments.reviewed_budget_change)
            verify_pilot()
            records.write('pilot-acceptance.json', {'passed': True, 'count': 3,
                'all_accepted_responses_validated': True, 'swebench_gold_tests_run': False})
            print('PILOT_PASSED: collecting the fixed 100-task sample.', flush=True)
            collect_stage(key, records, 100, release=True, resume_reviewed=arguments.resume_reviewed,
                          budget_change_reason=arguments.reviewed_budget_change)
            result['state'] = 'completed'
        except Exception as error:
            result.update(error_class=type(error).__name__, reason=str(error))
            print(records.encode(result), flush=True)
        finally:
            summarize_run()
            try:
                account_snapshot(key, records, 'account-at-stop.json')
            except Exception as error:
                records.write('account-query-error.json', {'error_class': type(error).__name__})
            result['finished_at_unix'] = time.time()
            records.write('controller-status.json', result)
            print(records.encode(result), flush=True)
        if result['state'] != 'completed':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
