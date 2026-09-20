import argparse
import concurrent.futures
import fcntl
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['MSWEA_SILENT_STARTUP'] = '1'
os.environ['MSWEA_GLOBAL_CONFIG_DIR'] = str(ROOT / 'agent-config')
os.environ['HF_HUB_OFFLINE'] = '1'

import requests
import yaml
from minisweagent.agents.default import DefaultAgent, JobTimeoutError, LimitsExceeded, NonTerminatingException, TerminatingException
from minisweagent.environments.docker import DockerEnvironment
from transformers import AutoTokenizer

from core import Budget, Records, SafetyStop, TransientGatewayError, prompt_token_upper_bound, record_configuration, summarize, timestamp
from local_images import inspect_image, load_manifest
from gateway_stream import read_completion
from trace_protocol import PROTOCOL, SYSTEM_RULES, query_with_recovery, response_problem


def checked(command, timeout=60):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise SafetyStop('Command failed: ' + shlex.join(command[:3]) + '; ' + result.stderr[-600:])
    return result.stdout.strip()


def disk_guard(config):
    free = shutil.disk_usage(ROOT).free
    if config.get('disk_reserve_checks_enabled', True) and free < config['minimum_free_gib'] * 1024 ** 3:
        raise SafetyStop('Shared disk reserve reached; no further image pulls or API requests')
    return free


def ensure_image(task, records, config):
    disk_guard(config)
    entry, details = inspect_image(task, load_manifest(config['local_image_manifest']))
    records.write('image.json', {'reference': entry['reference'], 'image_id': details['Id'],
                               'repo_digests': details.get('RepoDigests', []),
                               'size_bytes': details.get('Size'), 'newly_pulled': False,
                               'source': 'self_built', 'validation': entry})
    return entry['reference'], details['Id'], False


def prepare_task_image(task, config):
    script = ROOT.parent / 'swe-own-image/pipeline_images.py'
    command = [sys.executable, str(script), task['instance_id']]
    if not config.get('disk_reserve_checks_enabled', True):
        command.append('--disable-disk-reserve-checks')
    time_limits_enabled = config.get('image_preparation_time_limits_enabled', True)
    if not time_limits_enabled:
        command.append('--disable-build-timeout')
    subprocess.run(command,
                   stdin=subprocess.DEVNULL, check=True, timeout=3900 if time_limits_enabled else None)


def grouped_tasks(tasks):
    audit = json.loads((ROOT.parent / 'swe-own-image/provenance/dependency-groups-retry.json').read_text())
    if audit['source_dataset_sha256'] != hashlib.sha256((ROOT / 'tasks-100.jsonl').read_bytes()).hexdigest():
        raise SafetyStop('Dependency groups do not match the fixed task selection')
    membership = {task['instance_id']: task['dependency_group'] for task in audit['tasks']}
    grouped = {}
    for task in tasks:
        grouped.setdefault(membership[task['instance_id']], []).append(task)
    return [task for members in grouped.values() for task in members], grouped, membership


def release_image_cache(instance_id, group_id=None, run_dir=None):
    script = ROOT.parent / 'swe-own-image/pipeline_images.py'
    command = [sys.executable, str(script), instance_id, '--release-task']
    if group_id:
        command.extend(['--release-group', group_id])
    if run_dir:
        command.extend(['--run-dir', str(run_dir)])
    subprocess.run(command, stdin=subprocess.DEVNULL, check=True, timeout=240)


class TimedDocker(DockerEnvironment):
    def __init__(self, records, run_config, **kwargs):
        self.records = records
        self.run_config = run_config
        self.tool_events = []
        self.phase = 'setup'
        self.donor = None
        math_threads = str(max(1, int(run_config['container_cpus'])))
        kwargs['env'] = {**kwargs.get('env', {}), **dict.fromkeys([
            'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
        ], math_threads)}
        super().__init__(**kwargs)

    def _start_container(self):
        label = 'continuum-trace-' + uuid.uuid4().hex[:12]
        self.container_id = checked([
            'docker', 'run', '-d', '--rm', '--pull=never', '--name', label,
            '--label', 'continuum.trace.owner=ext.luohaowen1',
            '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges',
            '--cpus', str(self.run_config['container_cpus']),
            '--memory', str(self.run_config['container_memory_gib']) + 'g',
            '--memory-swap', str(self.run_config['container_memory_gib']) + 'g',
            '--pids-limit', str(self.run_config['container_pids']),
            '--tmpfs', '/testbed:rw,exec,nosuid,nodev,size=' + str(self.run_config['container_testbed_tmpfs_gib']) + 'g',
            '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
            '--tmpfs', '/root/.cache:rw,nosuid,nodev,size=256m',
            '--env', 'NVIDIA_VISIBLE_DEVICES=void',
            '--entrypoint', '/bin/sleep', self.config.image, 'infinity',
        ])
        try:
            self.donor = checked(['docker', 'create', '--network', 'none', '--read-only',
                                  '--label', 'continuum.trace.owner=ext.luohaowen1',
                                  '--entrypoint', '/bin/true', self.config.image])
            with tempfile.TemporaryFile() as error_stream:
                producer = subprocess.Popen(['docker', 'cp', self.donor + ':/testbed/.', '-'],
                                            stdout=subprocess.PIPE, stderr=error_stream)
                consumer = subprocess.Popen(['docker', 'exec', '-i', self.container_id,
                                             'tar', '--extract', '--file=-', '--directory=/testbed',
                                             '--no-same-owner'],
                                            stdin=producer.stdout, stdout=error_stream, stderr=error_stream)
                producer.stdout.close()
                try:
                    consumer.wait(timeout=180)
                    producer.wait(timeout=30)
                    if consumer.returncode or producer.returncode:
                        error_stream.seek(0)
                        detail = error_stream.read().decode('utf-8', errors='replace')[-3000:]
                        raise SafetyStop('Unable to copy pristine testbed into bounded tmpfs: ' + detail)
                finally:
                    for process in (producer, consumer):
                        if process.poll() is None:
                            process.kill()
                            process.wait()
            checked(['docker', 'rm', self.donor])
            self.donor = None
            state = json.loads(checked(['docker', 'inspect', self.container_id]))[0]
            host = state['HostConfig']
            assert host['NetworkMode'] == 'none' and host['ReadonlyRootfs']
            assert not host.get('Binds') and not host.get('Privileged')
            assert not host.get('DeviceRequests') and not host.get('Devices')
            self.records.write('isolation.json', {
                'container_id': self.container_id, 'network': host['NetworkMode'],
                'read_only_rootfs': host['ReadonlyRootfs'], 'bind_mounts': host.get('Binds'),
                'memory_bytes': host['Memory'], 'nano_cpus': host['NanoCpus'],
                'tmpfs': host['Tmpfs'], 'gpu_device_requests': host.get('DeviceRequests'),
                'host_api_key_forwarded': False,
                'tool_environment': self.config.env,
            })
        except BaseException:
            self.cleanup()
            raise

    def execute(self, command, cwd='', *, timeout=None):
        if not self.container_id:
            raise SafetyStop('Container is not running')
        disk_guard(self.run_config)
        duration_limit = timeout or self.run_config['tool_timeout_seconds']
        marker = '__CONTINUUM_TIMER_' + uuid.uuid4().hex + '__'
        wrapper = (
            'import json,subprocess,sys,time; '
            'command=sys.stdin.read(); '
            'started=time.monotonic(); '
            'result=subprocess.run(["timeout","-k","5",sys.argv[1],"bash","-lc",command],stdin=subprocess.DEVNULL); '
            'elapsed=(time.monotonic()-started)*1000; '
            'print("\\n"+sys.argv[2]+json.dumps({"duration_ms":elapsed,"returncode":result.returncode}),flush=True)'
        )
        invocation = 'python -u -c ' + shlex.quote(wrapper)
        invocation += ' ' + shlex.quote(str(duration_limit)) + ' ' + shlex.quote(marker)
        docker_command = ['docker', 'exec', '-i', '-w', cwd or '/testbed']
        for name, value in self.config.env.items():
            docker_command.extend(['-e', name + '=' + value])
        docker_command.extend([self.container_id, 'bash', '-lc', invocation])
        wall_start = time.perf_counter()
        with tempfile.TemporaryFile(dir=self.records.root) as input_stream, tempfile.TemporaryFile(dir=self.records.root) as output_stream:
            input_stream.write(command.encode('utf-8'))
            input_stream.seek(0)
            process = subprocess.Popen(docker_command, stdin=input_stream, stdout=output_stream, stderr=subprocess.STDOUT)
            failed = None
            while process.poll() is None:
                if os.fstat(output_stream.fileno()).st_size > self.run_config['max_tool_output_bytes']:
                    failed = 'tool_output_size_limit'
                    break
                if time.perf_counter() - wall_start > duration_limit + 15:
                    failed = 'docker_exec_wall_timeout'
                    break
                time.sleep(0.02)
            if failed:
                process.kill()
                process.wait()
            output_stream.seek(0)
            text = output_stream.read(self.run_config['max_tool_output_bytes']).decode('utf-8', errors='replace')
        wall_ms = (time.perf_counter() - wall_start) * 1000
        raw, separator, timing = text.rpartition('\n' + marker)
        if not separator or failed:
            self.records.event('tools.jsonl', {'phase': self.phase, 'command': command,
                               'raw_output': text, 'duration_ms': None, 'docker_wall_ms': wall_ms,
                               'error': failed or 'missing_internal_timer'})
            raise SafetyStop(failed or 'Internal tool timer unavailable; do not substitute fabricated timing')
        timing_data = json.loads(timing.strip())
        event = {'phase': self.phase, 'command': command, 'raw_output': raw, 'output_truncated': False,
                 'duration_ms': timing_data['duration_ms'], 'docker_wall_ms': wall_ms,
                 'returncode': timing_data['returncode'], 'timeout': timing_data['returncode'] == 124}
        self.records.event('tools.jsonl', event)
        if self.phase == 'agent':
            self.tool_events.append(event)
        if timing_data['returncode'] == 124:
            raise subprocess.TimeoutExpired(command, duration_limit, output=raw.encode())
        return {'output': raw, 'returncode': timing_data['returncode']}

    def cleanup(self):
        for attribute in ('container_id', 'donor'):
            container = getattr(self, attribute, None)
            if container:
                try:
                    subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=45)
                except Exception:
                    pass
                setattr(self, attribute, None)


class GatewayModel:
    def __init__(self, key, config, budget, records, tokenizer, task_id):
        self.key = key
        self.settings = config
        self.config = SimpleNamespace(model_name=config['requested_model'])
        self.budget = budget
        self.records = records
        self.tokenizer = tokenizer
        self.task_id = task_id
        self.n_calls = 0
        self.request_attempts = 0
        uncertain_path = budget.records.root / 'uncertain-costs.jsonl'
        previous_uncertain = [json.loads(line) for line in uncertain_path.read_text().splitlines()] if uncertain_path.exists() else []
        self.prior_uncertain_cost = sum(event['reservation']['amount_cny'] for event in previous_uncertain
                                        if event['reservation']['task_id'] == task_id)
        adjustment_path = budget.records.root / 'billing-adjustments.jsonl'
        adjustments = [json.loads(line) for line in adjustment_path.read_text().splitlines()] if adjustment_path.exists() else []
        self.historical_billing_adjustment = sum(event['amount_cny'] for event in adjustments if event['task_id'] == task_id)
        carryover_path = budget.records.root / 'attempt-cost-carryovers.jsonl'
        carryovers = [json.loads(line) for line in carryover_path.read_text().splitlines()] if carryover_path.exists() else []
        self.prior_invalid_response_cost = sum(event['completed_response_cost_cny'] for event in carryovers
                                               if event['task_id'] == task_id)
        self.cost = self.prior_uncertain_cost + self.historical_billing_adjustment + self.prior_invalid_response_cost
        self.calls = []
        self.remaining_seconds = config['job_timeout_seconds'] or None

    def get_template_vars(self):
        return {'model_name': self.config.model_name, 'n_calls': self.n_calls, 'cost': self.cost}

    def query(self, messages):
        if self.settings.get('collector_protocol') == PROTOCOL:
            return query_with_recovery(self, messages)
        query_started = time.perf_counter()
        initial_remaining = self.remaining_seconds
        retry_limit = self.settings.get('automatic_api_retries', 0)
        if retry_limit not in [0, 1, 2]:
            raise SafetyStop('Retry limit exceeds the bounded recovery policy')
        retry_history_path = self.records.root / 'retry-events.jsonl'
        previous_failures = 0
        if retry_history_path.exists():
            history = [json.loads(line) for line in retry_history_path.read_text().splitlines()]
            previous_failures = sum(event.get('turn') == self.n_calls + 1 for event in history)
            exhausted = [event for event in history if event.get('turn') == self.n_calls + 1
                         and event.get('retry_limit') == retry_limit and event.get('will_retry') is False]
            if exhausted or previous_failures > retry_limit:
                raise SafetyStop('This query already exhausted its retry policy; review gateway health before continuing')
        for retry in range(previous_failures, retry_limit + 1):
            self.remaining_seconds = initial_remaining - (time.perf_counter() - query_started)
            if self.remaining_seconds <= 10:
                raise JobTimeoutError('Agent time limit reached before another API attempt')
            try:
                return self._query_once(messages)
            except TransientGatewayError as error:
                charged = self.budget.charge_unsettled_reservation('Transient generation failure reserved in full before any retry')
                self.cost += charged
                self.prior_uncertain_cost += charged
                self.records.event('retry-events.jsonl', {'turn': self.n_calls + 1, 'retry_index': retry,
                    'retry_limit': retry_limit,
                    'reason': str(error), 'reserved_as_spent_cny': charged,
                    'will_retry': retry < retry_limit, 'sampling_parameters_changed': False})
                if retry == retry_limit:
                    raise SafetyStop(str(error) + '; bounded retry limit reached') from None
                time.sleep(2 ** retry)

    def _query_once(self, messages):
        disk_guard(self.settings)
        wire = [{'role': message['role'], 'content': message['content']} for message in messages]
        llama_prompt = len(self.tokenizer.apply_chat_template(wire, tokenize=True, add_generation_prompt=True))
        completion_limit = self.settings['max_completion_tokens']
        output_bound = completion_limit if completion_limit is not None else self.settings['api_max_completion_tokens']
        if llama_prompt + (completion_limit or 0) > self.settings['llama_context_tokens']:
            raise SafetyStop('Replay context guard reached; no forced truncation')
        prompt_bound = prompt_token_upper_bound(wire)
        self.budget.reserve(prompt_bound, output_bound, self.cost, self.task_id)
        self.request_attempts += 1
        payload = {'model': self.config.model_name, 'messages': wire, 'stream': False}
        if completion_limit is not None:
            payload['max_tokens'] = completion_limit
        if self.settings.get('thinking'):
            payload['thinking'] = self.settings['thinking']
            payload['reasoning_effort'] = self.settings['reasoning_effort']
        if self.settings.get('stop_sequences'):
            payload['stop'] = self.settings['stop_sequences']
        streaming = self.settings.get('stream_api', False)
        if streaming:
            payload.update(stream=True, stream_options={'include_usage': True})
        self.records.event('requests.jsonl', {'turn': self.n_calls + 1, 'payload': payload,
                                            'request_attempt': self.request_attempts,
                                            'llama_prompt_tokens': llama_prompt})
        start = time.perf_counter()
        duration_limit = self.settings['api_timeout_seconds']
        if self.remaining_seconds is not None:
            duration_limit = min(duration_limit, max(1, self.remaining_seconds - 10))
        try:
            response = requests.post(
                self.settings['api_base'] + '/chat/completions',
                headers={'Authorization': 'Bearer ' + self.key}, json=payload,
                timeout=(10, duration_limit), allow_redirects=False, **({'stream': True} if streaming else {}),
            )
        except requests.RequestException as error:
            raise TransientGatewayError('API transport error ' + type(error).__name__) from None
        if response.status_code != 200:
            self.records.event('api-errors.jsonl', {
                'http_status': response.status_code, 'turn': self.n_calls + 1,
                'elapsed_seconds': time.perf_counter() - start,
                'body_excerpt': getattr(response, 'text', '')[:3000],
                'request_id': getattr(response, 'headers', {}).get('x-oneapi-request-id') or getattr(response, 'headers', {}).get('x-request-id'),
                'retry_after': getattr(response, 'headers', {}).get('retry-after'),
            })
            if streaming:
                response.close()
            if response.status_code in [502, 503, 504]:
                raise TransientGatewayError('API HTTP ' + str(response.status_code))
            raise SafetyStop('API HTTP ' + str(response.status_code) + '; no model fallback')
        try:
            data = read_completion(response, self.records, self.n_calls + 1, start, duration_limit,
                                   save_events=self.settings.get('save_stream_events', False)) if streaming else response.json()
        except requests.RequestException as error:
            raise TransientGatewayError('Streaming transport error ' + type(error).__name__ + '; partial response preserved') from None
        finally:
            if streaming:
                response.close()
        usage = data.get('usage') or {}
        charge = self.budget.settle(usage)
        self.cost += charge
        self.n_calls += 1
        choices = data.get('choices') or [{}]
        content = choices[0].get('message', {}).get('content')
        reasoning = choices[0].get('message', {}).get('reasoning_content') or ''
        if not isinstance(reasoning, str):
            raise SafetyStop('Non-text reasoning response')
        call = {'turn': self.n_calls, 'requested_model': self.config.model_name,
                'reported_model': data.get('model'), 'response_id': data.get('id'),
                'usage': usage, 'conservative_cost_cny': charge,
                'billing_input_tokens_conservative': self.budget.input_tokens(usage),
                'api_wall_ms': (time.perf_counter() - start) * 1000,
                'llama_prompt_tokens': llama_prompt, 'finish_reason': choices[0].get('finish_reason'),
                'transport': data.get('transport', {'kind': 'nonstream'}),
                'reasoning_content': reasoning,
                'llama_reasoning_tokens': len(self.tokenizer.encode(reasoning, add_special_tokens=False)),
                'content': content, 'llama_completion_tokens': len(self.tokenizer.encode(content, add_special_tokens=False))
                if isinstance(content, str) else None}
        self.calls.append(call)
        self.records.event('responses.jsonl', call)
        expected_models = self.settings.get('accepted_reported_models')
        if expected_models and call['reported_model'] not in expected_models:
            raise SafetyStop('Unexpected reported model; response preserved and collection stopped')
        if self.settings.get('collector_protocol') == PROTOCOL:
            if isinstance(content, str) and not content.strip() and call['finish_reason'] == 'stop':
                raise TransientGatewayError('Empty completed answer; response and billed usage preserved')
            if isinstance(content, str) and call['finish_reason'] in ('stop', 'length'):
                if llama_prompt + call['llama_completion_tokens'] > self.settings['llama_context_tokens']:
                    raise SafetyStop('Actual replay request exceeds context')
                return {'content': content}
        if not isinstance(content, str) or not content.strip() or call['finish_reason'] != 'stop':
            raise SafetyStop('Empty/non-text/incomplete response; preserve raw response and stop')
        if llama_prompt + call['llama_completion_tokens'] > self.settings['llama_context_tokens']:
            raise SafetyStop('Actual replay request exceeds context; preserve and mark, do not truncate')
        return {'content': content}


class TracingAgent(DefaultAgent):
    def has_finished(self, output):
        text = str(output.get('output', '')).lstrip()
        if text.startswith(('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', 'MINI_SWE_AGENT_FINAL_OUTPUT')):
            substantive = [event for event in self.env.tool_events
                           if not str(event.get('raw_output', '')).lstrip().startswith(
                               ('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', 'MINI_SWE_AGENT_FINAL_OUTPUT'))]
            if not substantive:
                raise NonTerminatingException(
                    'Submission rejected: no task inspection or modification command has actually run. '
                    'Commands shown in a rejected response were NOT executed, and any claimed results in '
                    'that response are not real observations. Return exactly ONE next bash command to '
                    'inspect /testbed, then wait for the actual tool output. Do not simulate later turns.')
        return super().has_finished(output)

    def run_continuation(self, task, messages, elapsed_seconds):
        self.extra_template_vars |= {'task': task}
        self.messages = list(messages)
        self.job_start_time = time.time() - elapsed_seconds
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            while True:
                try:
                    self.step()
                except NonTerminatingException as error:
                    self.add_message('user', str(error))
                except TerminatingException as error:
                    self.add_message('user', str(error))
                    return type(error).__name__, str(error)
        finally:
            self._executor.shutdown(wait=False)

    def add_message(self, role, content, **kwargs):
        super().add_message(role, content, **kwargs)
        self.model.records.event('messages.jsonl', {'role': role, 'content': content})

    def query(self):
        if self.config.step_limit > 0 and self.model.n_calls >= self.config.step_limit:
            raise LimitsExceeded('Configured safety step limit reached, not a paper-derived turn count')
        remaining = self._get_remaining_time()
        if remaining is not None and remaining <= 0:
            raise JobTimeoutError('Agent wall-time limit reached')
        self.model.remaining_seconds = remaining
        response = self.model.query(self.messages)
        if self.model.settings.get('collector_protocol') == PROTOCOL:
            call = self.model.calls[-1]
            problem = response_problem(response['content'], call['finish_reason'])
            self.model.records.event('response-validation.jsonl', {
                'turn': call['turn'], 'accepted': problem is None, 'reason': problem,
                'commands_executed_before_validation': False,
                'rejected_response_added_to_context': False,
            })
            if problem:
                self.consecutive_invalid_responses = getattr(self, 'consecutive_invalid_responses', 0) + 1
                correction_limit = self.model.settings.get('max_format_corrections', 2)
                if correction_limit is not None and self.consecutive_invalid_responses > correction_limit:
                    raise SafetyStop('Consecutive invalid responses exceeded the correction limit')
                raise NonTerminatingException(
                    'Your last response was rejected and NONE of its commands were executed. ' + problem
                    + ' Return a short THOUGHT and exactly ONE complete bash code block containing only your next action.'
                    + ' Stop immediately after the closing fence. Do not output observations or future conversation turns.')
            self.consecutive_invalid_responses = 0
        self.add_message('assistant', **response)
        return response

    def execute_action(self, action):
        remaining = self._get_remaining_time()
        if remaining is not None and remaining <= 0:
            raise JobTimeoutError('Agent wall-time limit reached before tool execution')
        return super().execute_action(action)


def run_task(task, config, run_records, budget, tokenizer, key, *, resume=False):
    case_path = run_records.root / 'instances' / task['instance_id']
    previous_metrics = None
    previous_messages = None
    previous_tools = []
    if case_path.exists() and not resume:
        raise SafetyStop('Existing incomplete case must be inspected before rerunning')
    if resume:
        previous_metrics = json.loads((case_path / 'metrics.json').read_text())
        if previous_metrics.get('collection_completed'):
            raise SafetyStop('Completed traces must not be resumed')
        previous_messages = json.loads((case_path / 'trajectory.json').read_text())['messages']
        prior_tools_path = case_path / 'tools.jsonl'
        previous_tools = [json.loads(line) for line in (prior_tools_path.read_text().splitlines() if prior_tools_path.exists() else [])
                          if json.loads(line)['phase'] == 'agent']
        history = case_path / ('before-resume-' + uuid.uuid4().hex[:8])
        history.mkdir()
        for name in ['metrics.json', 'trajectory.json', 'baseline.json', 'execution-provenance.json']:
            if (case_path / name).exists():
                shutil.copyfile(case_path / name, history / name)
    records = Records(case_path, key)
    records.write('task.json', task)
    records.write('execution-provenance.json', {'config': config,
        'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stream_reader_sha256': hashlib.sha256((ROOT / 'gateway_stream.py').read_bytes()).hexdigest()})
    model = GatewayModel(key, config, budget, records, tokenizer, task['instance_id'])
    if resume:
        response_path = case_path / 'responses.jsonl'
        model.calls = [json.loads(line) for line in (response_path.read_text().splitlines() if response_path.exists() else [])]
        model.n_calls = len(model.calls)
        model.cost += sum(call['conservative_cost_cny'] for call in model.calls)
        request_path = case_path / 'requests.jsonl'
        model.request_attempts = len(request_path.read_text().splitlines()) if request_path.exists() else 0
    environment = None
    agent = None
    created_image = False
    reference = None
    status = 'SetupError'
    result = ''
    failure = None
    started = time.perf_counter()
    try:
        reference, image_id, created_image = ensure_image(task, records, config)
        source_config = yaml.safe_load((ROOT / 'source/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml').read_text())
        environment = TimedDocker(records, config, image=image_id, cwd='/testbed',
                                  env=source_config['environment']['env'])
        if resume:
            from task_checkpoint import restore_checkpoint
            restore_checkpoint(environment, records, task, image_id, previous_tools)
            environment.tool_events = previous_tools
        commit = environment.execute('git rev-parse HEAD')['output'].strip()
        ancestor = environment.execute('git merge-base --is-ancestor ' + shlex.quote(task['base_commit']) + ' HEAD')
        if ancestor['returncode'] != 0:
            raise SafetyStop('Image does not descend from dataset base_commit')
        baseline = environment.execute('git diff --binary ' + shlex.quote(task['base_commit']) + ' HEAD')['output']
        setup_worktree_diff = environment.execute('git diff HEAD --binary')['output']
        records.write('resume-baseline.json' if resume else 'baseline.json', {'dataset_base_commit': task['base_commit'], 'image_head': commit,
                                        'image_setup_diff_from_dataset': baseline,
                                        'image_setup_worktree_diff': setup_worktree_diff,
                                        'exported_patch_includes_setup_changes': bool(setup_worktree_diff)})
        environment.phase = 'agent'
        agent_config = source_config['agent']
        agent_config.update(step_limit=config['max_steps'], cost_limit=0, job_timeout=config['job_timeout_seconds'])
        if config.get('collector_protocol') == PROTOCOL:
            agent_config['system_template'] += '\n' + SYSTEM_RULES
        agent_config['format_error_template'] = (
            'Your last response contained {{ actions|length }} bash blocks instead of exactly one and was rejected. '
            'NONE of the commands in that response were executed. Any tool results, code edits, tests or '
            'submission described inside that response did not happen. The workspace is unchanged by that '
            'response. Return exactly ONE bash code block containing only your NEXT action, then stop and '
            'wait for the real tool output. Do not simulate future conversation turns or observations. '
            'Only submit after you have actually inspected and worked on the task through accepted commands.')
        records.write('agent-policy.json', {'policy_version': config.get('collector_protocol', 'real-tool-feedback-v2'),
                      'agent_config': agent_config, 'source_config_sha256': hashlib.sha256(
                          (ROOT / 'source/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml').read_bytes()).hexdigest(),
                      'reason': 'Explicitly distinguish rejected, unexecuted model code from real tool actions'})
        agent = TracingAgent(model, environment, **agent_config)
        if resume and previous_messages:
            status, result = agent.run_continuation(task['problem_statement'], previous_messages,
                previous_metrics.get('agent_active_seconds', previous_metrics['wall_seconds_including_setup']))
        else:
            status, result = agent.run(task['problem_statement'])
        environment.phase = 'patch-export'
        patch = environment.execute('git diff HEAD --binary')['output']
        (case_path / 'patch.diff').write_text(patch.replace(key, '[REDACTED]'))
    except Exception as error:
        status = type(error).__name__
        result = str(error)
        failure = error
    finally:
        messages = agent.messages if agent else (previous_messages or [])
        records.write('trajectory.json', {'instance_id': task['instance_id'], 'exit_status': status,
                                        'result': result, 'messages': messages})
        if failure and environment and agent:
            from task_checkpoint import save_checkpoint
            save_checkpoint(environment, records, task, image_id)
        calls = model.calls
        tools = environment.tool_events if environment else []
        substantive_tools = [event for event in tools
                             if not str(event.get('raw_output', '')).lstrip().startswith(
                                 ('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', 'MINI_SWE_AGENT_FINAL_OUTPUT'))]
        metrics = {
            'instance_id': task['instance_id'], 'exit_status': status,
            'collection_completed': failure is None and model.n_calls > 0 and len(substantive_tools) > 0,
            'non_submission_tool_calls': len(substantive_tools),
            'agent_policy_version': config.get('collector_protocol', 'real-tool-feedback-v2'),
            'submitted': status == 'Submitted', 'swebench_resolved': None,
            'model_turns': model.n_calls, 'tool_calls': len(tools),
            'api_request_attempts': model.request_attempts,
            'agent_active_seconds': time.time() - agent.job_start_time if agent and agent.job_start_time else 0,
            'resumed_from_checkpoint': resume,
            'tool_time_ms': summarize([event['duration_ms'] for event in tools]),
            'tool_durations_ms': [event['duration_ms'] for event in tools],
            'api_prompt_tokens_sum': sum(call['usage']['prompt_tokens'] for call in calls),
            'api_conservative_input_tokens_sum': sum(budget.input_tokens(call['usage']) for call in calls),
            'api_completion_tokens_sum': sum(call['usage']['completion_tokens'] for call in calls),
            'api_cached_prompt_tokens_sum': sum(call['usage'].get('prompt_cache_hit_tokens',
                (call['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0)) for call in calls),
            'api_reasoning_tokens_sum': sum((call['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0) for call in calls),
            'llama_request_tokens_sum': sum(call['llama_prompt_tokens'] + (call['llama_completion_tokens'] or 0) for call in calls),
            'llama_reasoning_tokens_sum': sum(call.get('llama_reasoning_tokens', 0) for call in calls),
            'llama_decode_tokens_including_reasoning_sum': sum((call['llama_completion_tokens'] or 0)
                + call.get('llama_reasoning_tokens', 0) for call in calls),
            'reasoning_in_next_request': False,
            'llama_final_transcript_tokens': len(tokenizer.apply_chat_template(messages, tokenize=True)) if messages else 0,
            'conservative_cost_cny': model.cost, 'prior_uncertain_cost_cny': model.prior_uncertain_cost,
            'historical_billing_adjustment_cny': model.historical_billing_adjustment,
            'prior_invalid_response_cost_cny': model.prior_invalid_response_cost,
            'wall_seconds_including_setup': time.perf_counter() - started,
            'reported_models': sorted({str(call['reported_model']) for call in calls}),
        }
        validation_path = case_path / 'response-validation.jsonl'
        validations = [json.loads(line) for line in validation_path.read_text().splitlines()] if validation_path.exists() else []
        metrics['accepted_model_turns'] = sum(event['accepted'] for event in validations) if validations else len(calls)
        metrics['rejected_response_turns'] = [event['turn'] for event in validations if not event['accepted']]
        metrics['recovery_api_responses'] = len(calls) - len(validations) if validations else 0
        records.write('metrics.json', metrics)
        if environment:
            environment.cleanup()
        if reference and created_image:
            cleanup = subprocess.run(['docker', 'image', 'rm', reference], capture_output=True, text=True, timeout=90)
            records.write('image-cleanup.json', {'reference': reference, 'returncode': cleanup.returncode,
                                               'forced': False})
    if failure:
        raise SafetyStop(str(failure)) from None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=3, choices=range(1, 101))
    parser.add_argument('--config', type=Path, default=ROOT / 'run_config.json')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/nowcoding-mixed-100-20260918')
    parser.add_argument('--key-stdin', action='store_true')
    parser.add_argument('--local-images', type=Path,
                        default=ROOT.parent / 'swe-own-image/provenance/validated-images.json')
    parser.add_argument('--check-images-only', action='store_true')
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--reviewed-budget-change')
    parser.add_argument('--reviewed-runtime-change')
    parser.add_argument('--instance-id')
    parser.add_argument('--prepare-on-demand', action='store_true')
    parser.add_argument('--group-by-dependencies', action='store_true')
    parser.add_argument('--release-temporary-images', action='store_true')
    arguments = parser.parse_args()
    os.umask(0o077)
    config = json.loads(arguments.config.read_text())
    config['local_image_manifest'] = str(arguments.local_images.resolve())
    authorized_pairs = {('https://nowcoding.ai/v1', 'claude-sonnet-4-6'),
                        ('https://api.deepseek.com', 'deepseek-flash')}
    if (config['api_base'], config['requested_model']) not in authorized_pairs:
        raise SafetyStop('Only the explicitly authorized gateway/model is permitted')
    if ((config['budget_stop_cny'] is not None and config['budget_stop_cny'] > 270)
            or (config['budget_authorized_cny'] is not None and config['budget_authorized_cny'] > 300)
            or config['workers'] != 1):
        raise SafetyStop('Budget/concurrency authorization mismatch')
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.run_dir / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        smoke_path = arguments.run_dir / 'interface-smoke.json'
        if not smoke_path.exists():
            raise SafetyStop('Run interface smoke test before starting paid collection')
        smoke = json.loads(smoke_path.read_text())
        if (smoke.get('http_status') != 200 or not smoke.get('nonempty_content')
                or smoke.get('requested_model') != config['requested_model']
                or (config['api_base'] == 'https://api.deepseek.com'
                    and (smoke.get('api_base') != config['api_base'] or not smoke.get('validated')))):
            raise SafetyStop('Interface gate failed; verify token/model before collection')
        selected = [json.loads(line) for line in (ROOT / 'tasks-100.jsonl').read_text().splitlines()][:arguments.limit]
        if arguments.instance_id:
            selected = [task for task in selected if task['instance_id'] == arguments.instance_id]
            if not selected:
                raise SafetyStop('Requested instance is not in the selected fixed sample')
        tasks = selected
        groups = {}
        membership = {}
        if arguments.group_by_dependencies:
            tasks, groups, membership = grouped_tasks(selected)
        if arguments.release_temporary_images and not arguments.group_by_dependencies:
            raise SafetyStop('Cache release requires deterministic dependency-group scheduling')
        manifest = load_manifest(config['local_image_manifest'])
        if arguments.prepare_on_demand:
            for task in tasks:
                existing = arguments.run_dir / 'instances' / task['instance_id'] / 'metrics.json'
                if existing.exists() and json.loads(existing.read_text()).get('collection_completed'):
                    continue
                prepare_task_image(task, config)
                break
        else:
            for task in tasks:
                inspect_image(task, manifest)
        if arguments.check_images_only:
            print(json.dumps({'validated_local_images': arguments.limit if not arguments.prepare_on_demand else 'first_pending', 'api_called': False}))
            return
        print('AUTH_CHECK_READY', flush=True)
        key = sys.stdin.readline().strip() if arguments.key_stdin else getpass.getpass('Model API key: ')
        if not key:
            raise SafetyStop('Missing credential')
        records = Records(arguments.run_dir, key)
        records.event('execution-plans.jsonl', {'selection_ids': [task['instance_id'] for task in selected],
            'execution_ids': [task['instance_id'] for task in tasks],
            'grouped_by_dependencies': arguments.group_by_dependencies,
            'prepare_images_on_demand': arguments.prepare_on_demand,
            'release_own_temporary_images': arguments.release_temporary_images,
            'sample_replacements': 0})
        budget = Budget(records, config)
        record_configuration(records, config, arguments.reviewed_budget_change, arguments.reviewed_runtime_change)
        records.write('provenance.json', {
            'started_at': timestamp(), 'user_claimed_upstream': config['upstream_model_claim'],
            'upstream_independently_verified': False,
            'paper_model': config['paper_model'], 'sample_sha256': hashlib.sha256((ROOT / 'tasks-100.jsonl').read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'image_manifest_sha256': hashlib.sha256(arguments.local_images.read_bytes()).hexdigest(),
            'supporting_sha256': {filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
                                 for filename in ['core.py', 'local_images.py', 'gateway_stream.py', 'trace_protocol.py', 'task_checkpoint.py', 'pip-freeze.txt', 'dataset-manifest.json',
                                                  'source/mini-swe-agent/src/minisweagent/agents/default.py',
                                                  'source/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml']},
        })
        tokenizer_path = '/export/home/ext.luohaowen1/.cache/modelscope/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
        try:
            for index, task in enumerate(tasks):
                metrics_path = arguments.run_dir / 'instances' / task['instance_id'] / 'metrics.json'
                if metrics_path.exists():
                    previous = json.loads(metrics_path.read_text())
                    if not previous.get('collection_completed') and not arguments.resume_incomplete:
                        raise SafetyStop('Previous zero-response task needs review; will not silently skip it')
                    if previous.get('collection_completed'):
                        if arguments.release_temporary_images:
                            release_image_cache(task['instance_id'], run_dir=arguments.run_dir)
                        continue
                if arguments.prepare_on_demand:
                    records.write('status.json', {'state': 'preparing_image', 'task': task['instance_id'], 'position': index + 1})
                    prepare_task_image(task, config)
                print(f"TASK_START {index + 1}/{arguments.limit} {task['instance_id']}", flush=True)
                records.write('status.json', {'state': 'running', 'task': task['instance_id'], 'position': index + 1})
                metrics = run_task(task, config, records, budget, tokenizer, key,
                                   resume=metrics_path.exists() and arguments.resume_incomplete)
                if not metrics.get('collection_completed'):
                    raise SafetyStop('Task ended without a complete replayable agent trace')
                print(records.encode({'event': 'TASK_END', **metrics}), flush=True)
                if arguments.release_temporary_images:
                    group = membership[task['instance_id']]
                    group_done = all((arguments.run_dir / 'instances' / member['instance_id'] / 'metrics.json').exists()
                                     and json.loads((arguments.run_dir / 'instances' / member['instance_id'] / 'metrics.json').read_text()).get('collection_completed')
                                     for member in groups[group])
                    release_image_cache(task['instance_id'], group if group_done else None, arguments.run_dir)
            records.write('status.json', {'state': 'stage_finished', 'requested_count': arguments.limit})
        except Exception as error:
            if budget.state['pending']:
                budget.charge_unsettled_reservation('Collection stopped after uncertain API outcome; no retry')
            records.write('status.json', {'state': 'stopped', 'reason': str(error), 'error_class': type(error).__name__})
            print(records.encode({'state': 'stopped', 'reason': str(error)}), flush=True)
            raise SystemExit(2) from None


if __name__ == '__main__':
    main()
