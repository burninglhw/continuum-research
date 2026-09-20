import json
import math
import os
import time
from pathlib import Path


class SafetyStop(RuntimeError):
    pass


class TransientGatewayError(SafetyStop):
    pass


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Records:
    def __init__(self, root, secret=''):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.secret = secret

    def encode(self, value):
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return serialized.replace(self.secret, '[REDACTED]') if self.secret else serialized

    def write(self, name, value):
        destination = self.root / name
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        with temporary.open('w') as stream:
            stream.write(self.encode(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)

    def event(self, name, value):
        with (self.root / name).open('a') as stream:
            stream.write(self.encode({'recorded_at': timestamp(), **value}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def record_configuration(records, config, budget_change_reason=None, runtime_change_reason=None):
    snapshot = records.root / 'config.snapshot.json'
    if snapshot.exists():
        previous = json.loads(snapshot.read_text())
        changes = {name for name in previous.keys() | config.keys()
                   if name not in previous or name not in config or previous[name] != config[name]}
        if changes:
            runtime_settings = {'max_completion_tokens': None, 'api_max_completion_tokens': 393216,
                                'max_format_corrections': None, 'max_steps': 0,
                                'job_timeout_seconds': 0, 'api_timeout_seconds': 600,
                                'disk_reserve_checks_enabled': False,
                                'image_preparation_time_limits_enabled': False}
            if (runtime_change_reason and changes <= runtime_settings.keys()
                    and all(config.get(name) == runtime_settings[name] for name in changes)):
                records.event('runtime-policy-changes.jsonl', {
                    'reason': runtime_change_reason, 'previous_config': previous,
                    'updated_config': config, 'api_sampling_changed': 'max_completion_tokens' in changes,
                    'task_selection_changed': False, 'previous_attempts_preserved': True,
                })
                records.write('config.snapshot.json', config)
                return
            budget_fields = {'per_task_stop_cny', 'budget_stop_cny', 'budget_authorized_cny'}
            unlimited = all(name in config and config[name] is None for name in budget_fields)
            limit = config.get('per_task_stop_cny')
            bounded_change = (changes == {'per_task_stop_cny'}
                              and type(limit) in (int, float) and math.isfinite(limit)
                              and config.get('budget_stop_cny') is not None
                              and 0 < limit <= min(8, config['budget_stop_cny']))
            if (not changes <= budget_fields or not budget_change_reason
                    or not (unlimited or bounded_change)):
                raise SafetyStop('Configuration changed; do not mix incompatible traces')
            records.event('budget-policy-changes.jsonl', {
                'reason': budget_change_reason, 'previous_config': previous,
                'updated_config': config, 'api_sampling_changed': False,
                'task_selection_changed': False, 'prior_costs_reset': False,
                'spending_caps_disabled': unlimited,
            })
    records.write('config.snapshot.json', config)


class Budget:
    def __init__(self, records, config, *, allow_pending=False):
        self.records = records
        self.config = config
        self.path = records.root / 'budget.json'
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['pending'] and not allow_pending:
                raise SafetyStop('Unsettled request reservation: manual billing reconciliation required')
        else:
            self.state = {'charged_upper_estimate_cny': 0.0, 'pending': None, 'requests': 0}
            self.save()

    def save(self):
        self.records.write('budget.json', self.state)

    def cost(self, prompt_tokens, completion_tokens):
        return (prompt_tokens * self.config['price_input_cny_per_million']
                + completion_tokens * self.config['price_output_cny_per_million']) / 1_000_000

    def input_tokens(self, usage):
        prompt = usage.get('prompt_tokens')
        if type(prompt) is not int or prompt < 0:
            raise SafetyStop('Invalid prompt usage')
        if not self.config.get('gateway_add_cached_input', False):
            return prompt
        cache_reads = [
            (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0),
            (usage.get('input_tokens_details') or {}).get('cached_tokens', 0),
            usage.get('cache_read_input_tokens', 0),
        ]
        cache_creation = [usage.get('claude_cache_creation_5_m_tokens', 0),
                          usage.get('claude_cache_creation_1_h_tokens', 0)]
        values = cache_reads + cache_creation + [usage.get('cache_creation_input_tokens', 0)]
        if any(type(value) is not int or value < 0 for value in values):
            raise SafetyStop('Invalid cached-input usage')
        return prompt + max(cache_reads) + max(sum(cache_creation), usage.get('cache_creation_input_tokens', 0))

    def reserve(self, prompt_bound, output_bound, task_spend, task_id):
        if self.state['pending']:
            raise SafetyStop('A previous request remains unsettled')
        amount = self.cost(prompt_bound, output_bound)
        global_limit = self.config['budget_stop_cny']
        task_limit = self.config['per_task_stop_cny']
        if global_limit is not None and self.state['charged_upper_estimate_cny'] + amount > global_limit:
            raise SafetyStop('Global budget guard reached before request')
        if task_limit is not None and task_spend + amount > task_limit:
            raise SafetyStop('Per-task budget guard reached before request')
        self.state['pending'] = {'task_id': task_id, 'amount_cny': amount,
                                 'prompt_bound': prompt_bound, 'output_bound': output_bound,
                                 'started_at': timestamp()}
        self.save()
        return amount

    def settle(self, usage):
        pending = self.state['pending']
        if pending is None:
            raise SafetyStop('No reservation to settle')
        counts = [usage.get('prompt_tokens'), usage.get('completion_tokens')]
        if any(type(value) is not int or value < 0 for value in counts):
            raise SafetyStop('Missing/invalid usage: reservation retained, batch stopped')
        prompt_tokens, completion_tokens = counts
        prompt_tokens = self.input_tokens(usage)
        if prompt_tokens > pending['prompt_bound'] or completion_tokens > pending['output_bound']:
            raise SafetyStop('Usage exceeded reservation assumptions; reconcile provider bill before continuing')
        amount = self.cost(prompt_tokens, completion_tokens)
        self.state['charged_upper_estimate_cny'] += amount
        self.state['requests'] += 1
        self.state['pending'] = None
        self.save()
        return amount

    def charge_unsettled_reservation(self, reason):
        pending = self.state['pending']
        if pending is None:
            return 0.0
        amount = pending['amount_cny']
        self.records.event('uncertain-costs.jsonl', {'reservation': pending, 'reason': reason,
                                                  'counted_as_spent_for_safety': True,
                                                  'actual_provider_charge_known': False})
        self.state['charged_upper_estimate_cny'] += amount
        self.state['pending'] = None
        self.save()
        return amount


def prompt_token_upper_bound(messages):
    return 2048 + sum(len(message['content'].encode('utf-8')) + 128 for message in messages)


def summarize(values):
    if not values:
        return {'n': 0, 'mean': None, 'std_population': None}
    mean = sum(values) / len(values)
    return {'n': len(values), 'mean': mean,
            'std_population': math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))}
