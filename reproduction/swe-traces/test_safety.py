import json
from pathlib import Path

import pytest

from core import Budget, Records, SafetyStop, prompt_token_upper_bound, summarize


@pytest.fixture
def config():
    settings = json.loads((Path(__file__).parent / 'run_config.json').read_text())
    settings['stream_api'] = False
    settings['automatic_api_retries'] = 0
    settings['gateway_add_cached_input'] = False
    return settings


def test_secret_is_redacted(tmp_path):
    records = Records(tmp_path, 'test-private-token')
    records.write('sample.json', {'value': 'test-private-token'})
    records.event('sample.jsonl', {'value': 'test-private-token'})
    assert 'test-private-token' not in (tmp_path / 'sample.json').read_text()
    assert 'test-private-token' not in (tmp_path / 'sample.jsonl').read_text()


def test_budget_stops_before_request(tmp_path, config):
    config['budget_stop_cny'] = 0.001
    budget = Budget(Records(tmp_path), config)
    with pytest.raises(SafetyStop, match='Global budget'):
        budget.reserve(1000, 8192, 0, 'sample')
    assert budget.state['pending'] is None


def test_task_budget_stops_before_request(tmp_path, config):
    config['per_task_stop_cny'] = 0.001
    budget = Budget(Records(tmp_path), config)
    with pytest.raises(SafetyStop, match='Per-task'):
        budget.reserve(1000, 8192, 0, 'sample')


@pytest.mark.parametrize('usage', [{}, {'prompt_tokens': -1, 'completion_tokens': 3},
                                   {'prompt_tokens': True, 'completion_tokens': 3},
                                   {'prompt_tokens': 100, 'completion_tokens': 9999}])
def test_invalid_usage_keeps_reservation(tmp_path, config, usage):
    budget = Budget(Records(tmp_path), config)
    budget.reserve(1000, 8192, 0, 'sample')
    with pytest.raises(SafetyStop):
        budget.settle(usage)
    assert budget.state['pending'] is not None
    with pytest.raises(SafetyStop):
        Budget(Records(tmp_path), config)


def test_uncertain_call_counted_conservatively(tmp_path, config):
    budget = Budget(Records(tmp_path), config)
    reserve = budget.reserve(1000, 8192, 0, 'sample')
    assert budget.charge_unsettled_reservation('test timeout') == reserve
    resumed = Budget(Records(tmp_path), config)
    assert resumed.state['charged_upper_estimate_cny'] == reserve
    assert resumed.state['pending'] is None


def test_usage_and_cache_not_double_counted(tmp_path, config):
    budget = Budget(Records(tmp_path), config)
    budget.reserve(1000, 8192, 0, 'sample')
    charged = budget.settle({'prompt_tokens': 1000, 'completion_tokens': 100,
                            'prompt_tokens_details': {'cached_tokens': 900}})
    assert charged == pytest.approx((1000 * 17.5 + 100 * 105) / 1e6)
    assert budget.state['requests'] == 1
    assert budget.state['pending'] is None


def test_gateway_separate_cache_tokens_are_charged_in_full(tmp_path, config):
    config['gateway_add_cached_input'] = True
    budget = Budget(Records(tmp_path), config)
    budget.reserve(2000, 8192, 0, 'sample')
    usage = {'prompt_tokens': 100, 'completion_tokens': 20,
             'prompt_tokens_details': {'cached_tokens': 900}}
    assert budget.input_tokens(usage) == 1000
    charged = budget.settle(usage)
    assert charged == pytest.approx((1000 * 17.5 + 20 * 105) / 1e6)


def test_utf8_upper_bound():
    messages = [{'role': 'user', 'content': '你好'}]
    assert prompt_token_upper_bound(messages) >= len('你好'.encode())


def test_empty_metrics_are_not_fake_zero_measurements():
    assert summarize([]) == {'n': 0, 'mean': None, 'std_population': None}
    assert summarize([1, 3]) == {'n': 2, 'mean': 2, 'std_population': 1}


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def encode(self, content, **kwargs):
        return list(content)


def test_gateway_records_usage_and_actual_reported_name(tmp_path, config, monkeypatch):
    import collect
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return type('Response', (), {'status_code': 200, 'json': lambda self: {
            'model': 'gateway-reported-name', 'usage': {'prompt_tokens': 100, 'completion_tokens': 50},
            'choices': [{'finish_reason': 'stop', 'message': {'content': '```bash\necho ok\n```'}}],
        }})()

    monkeypatch.setattr(collect.requests, 'post', post)
    monkeypatch.setattr(collect, 'disk_guard', lambda settings: None)
    records = Records(tmp_path, 'test-private-token')
    budget = Budget(records, config)
    model = collect.GatewayModel('test-private-token', config, budget, records, FakeTokenizer(), 'sample')
    model.query([{'role': 'user', 'content': 'test'}])
    assert model.n_calls == 1
    assert len(calls) == 1
    assert calls[0]['allow_redirects'] is False
    assert calls[0]['json']['max_tokens'] == 8192
    assert model.calls[0]['reported_model'] == 'gateway-reported-name'
    for path in tmp_path.glob('*.json*'):
        assert 'test-private-token' not in path.read_text()


def test_api_failure_never_retries_or_falls_back(tmp_path, config, monkeypatch):
    import collect
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return type('Response', (), {'status_code': 503})()

    monkeypatch.setattr(collect.requests, 'post', post)
    monkeypatch.setattr(collect, 'disk_guard', lambda settings: None)
    records = Records(tmp_path)
    budget = Budget(records, config)
    model = collect.GatewayModel('fake', config, budget, records, FakeTokenizer(), 'sample')
    with pytest.raises(SafetyStop, match='503'):
        model.query([{'role': 'user', 'content': 'test'}])
    assert len(calls) == 1
    assert budget.state['pending'] is None
    assert budget.state['charged_upper_estimate_cny'] > 0
