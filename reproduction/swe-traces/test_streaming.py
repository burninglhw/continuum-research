import json
import time
from pathlib import Path

import pytest

from core import Budget, Records, SafetyStop
from gateway_stream import read_completion


class Response:
    status_code = 200

    def __init__(self, events):
        self.events = events
        self.closed = False

    def iter_lines(self, **kwargs):
        for event in self.events:
            yield b'data: ' + (event.encode() if isinstance(event, str) else json.dumps(event).encode())

    def close(self):
        self.closed = True


def events():
    return [
        {'id': 'sample', 'model': 'reported-model', 'choices': [{'index': 0, 'delta': {'content': '```bash\n'}}]},
        {'choices': [{'index': 0, 'delta': {'content': 'echo ok\n```'}, 'finish_reason': 'stop'}]},
        {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}, '[DONE]',
    ]


def test_stream_content_and_usage_are_assembled(tmp_path):
    result = read_completion(Response(events()), Records(tmp_path), 1, time.perf_counter(), 60)
    assert result['choices'][0]['message']['content'] == '```bash\necho ok\n```'
    assert result['usage'] == {'prompt_tokens': 100, 'completion_tokens': 10}
    assert result['model'] == 'reported-model'


def test_deepseek_reasoning_is_saved_separately(tmp_path):
    stream = [{'choices': [{'delta': {'reasoning_content': 'Inspect the repository first.'}}]}] + events()
    result = read_completion(Response(stream), Records(tmp_path), 1, time.perf_counter(), 60)
    assert result['choices'][0]['message']['reasoning_content'] == 'Inspect the repository first.'
    assert result['choices'][0]['message']['content'] == '```bash\necho ok\n```'


def test_partial_reasoning_is_preserved_and_redacted(tmp_path):
    stream = [{'choices': [{'delta': {'reasoning_content': 'fake-secret'}}]}]
    with pytest.raises(SafetyStop):
        read_completion(Response(stream), Records(tmp_path, 'fake-secret'), 1, time.perf_counter(), 60)
    partial = json.loads((tmp_path / 'stream-partial.jsonl').read_text())
    assert partial['reasoning_content'] == '[REDACTED]'


def test_deepseek_usage_includes_reasoning_without_double_counting_cache(tmp_path):
    settings = json.loads((Path(__file__).parent / 'deepseek_config.json').read_text())
    budget = Budget(Records(tmp_path), settings)
    budget.reserve(2000, 16384, 0, 'sample')
    charge = budget.settle({'prompt_tokens': 1000, 'prompt_cache_hit_tokens': 900,
                           'prompt_cache_miss_tokens': 100, 'completion_tokens': 500,
                           'completion_tokens_details': {'reasoning_tokens': 400}})
    assert charge == pytest.approx((1000 * 2 + 500 * 8) / 1e6)


def test_deepseek_request_and_response_preserve_separate_reasoning(tmp_path, monkeypatch):
    import collect

    settings = json.loads((Path(__file__).parent / 'deepseek_config.json').read_text())
    stream = [{'id': 'deepseek-test', 'model': 'deepseek-flash',
               'choices': [{'delta': {'reasoning_content': 'Inspect first.'}}]},
              {'choices': [{'delta': {'content': '```bash\nls\n```'}, 'finish_reason': 'stop'}]},
              {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 30}}, '[DONE]']
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response(stream)

    tokenizer = type('Tokenizer', (), {
        'apply_chat_template': lambda self, *args, **kwargs: [1, 2],
        'encode': lambda self, content, **kwargs: list(content),
    })()
    monkeypatch.setattr(collect.requests, 'post', post)
    monkeypatch.setattr(collect, 'disk_guard', lambda config: None)
    records = Records(tmp_path, 'fake-private-token')
    model = collect.GatewayModel('fake-private-token', settings, Budget(records, settings), records, tokenizer, 'sample')
    result = model.query([{'role': 'assistant', 'content': 'Prior action', 'reasoning_content': 'Prior reasoning'},
                          {'role': 'user', 'content': 'Observation'}])
    assert calls[0][0] == 'https://api.deepseek.com/chat/completions'
    assert calls[0][1]['json']['thinking'] == {'type': 'enabled'}
    assert calls[0][1]['json']['reasoning_effort'] == 'high'
    assert all('reasoning_content' not in message for message in calls[0][1]['json']['messages'])
    assert result == {'content': '```bash\nls\n```'}
    assert model.calls[0]['reasoning_content'] == 'Inspect first.'
    assert model.calls[0]['llama_reasoning_tokens'] == len('Inspect first.')


@pytest.mark.parametrize('broken', [events()[:-1], events()[:2] + ['[DONE]'],
                                  [{'error': {'message': 'upstream failed'}}]])
def test_incomplete_stream_is_saved_and_stops(tmp_path, broken):
    with pytest.raises(SafetyStop):
        read_completion(Response(broken), Records(tmp_path), 1, time.perf_counter(), 60)
    assert (tmp_path / 'stream-partial.jsonl').exists()


def test_streaming_gateway_settles_once(tmp_path, monkeypatch):
    import collect

    settings = json.loads((Path(__file__).parent / 'run_config.json').read_text())
    settings['stream_api'] = True
    response = Response(events())
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return response

    tokenizer = type('Tokenizer', (), {
        'apply_chat_template': lambda self, *args, **kwargs: [1, 2, 3],
        'encode': lambda self, content, **kwargs: list(content),
    })()
    monkeypatch.setattr(collect.requests, 'post', post)
    monkeypatch.setattr(collect, 'disk_guard', lambda config: None)
    records = Records(tmp_path, 'fake-private-token')
    budget = Budget(records, settings)
    model = collect.GatewayModel('fake-private-token', settings, budget, records, tokenizer, 'sample')
    model.query([{'role': 'user', 'content': 'test'}])
    assert len(calls) == 1 and calls[0]['stream'] is True
    assert calls[0]['json']['stream_options'] == {'include_usage': True}
    assert response.closed and budget.state['requests'] == 1 and budget.state['pending'] is None


def test_previous_uncertain_task_cost_is_not_forgotten(tmp_path):
    import collect

    settings = json.loads((Path(__file__).parent / 'run_config.json').read_text())
    records = Records(tmp_path)
    records.event('uncertain-costs.jsonl', {'reservation': {'task_id': 'sample', 'amount_cny': 1.0}})
    budget = Budget(records, settings)
    model = collect.GatewayModel('fake', settings, budget, records, None, 'sample')
    assert model.cost == 1.0 and model.prior_uncertain_cost == 1.0


def test_transient_retry_reserves_failure_before_next_call(tmp_path, monkeypatch):
    import collect

    settings = json.loads((Path(__file__).parent / 'run_config.json').read_text())
    settings.update(stream_api=True, automatic_api_retries=2)
    failed = Response([])
    failed.status_code = 502
    calls = []
    records = Records(tmp_path)
    budget = Budget(records, settings)

    def post(url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            assert budget.state['charged_upper_estimate_cny'] > 0
        return failed if len(calls) == 1 else Response(events())

    tokenizer = type('Tokenizer', (), {
        'apply_chat_template': lambda self, *args, **kwargs: [1, 2],
        'encode': lambda self, content, **kwargs: list(content),
    })()
    monkeypatch.setattr(collect.requests, 'post', post)
    monkeypatch.setattr(collect, 'disk_guard', lambda config: None)
    monkeypatch.setattr(collect.time, 'sleep', lambda duration: None)
    model = collect.GatewayModel('fake', settings, budget, records, tokenizer, 'sample')
    model.query([{'role': 'user', 'content': 'test'}])
    assert len(calls) == 2 and model.n_calls == 1 and model.request_attempts == 2
    assert budget.state['pending'] is None
    assert calls[0]['json'] == calls[1]['json']


def test_retry_limit_is_finite_and_all_failures_are_reserved(tmp_path, monkeypatch):
    import collect

    settings = json.loads((Path(__file__).parent / 'run_config.json').read_text())
    settings.update(stream_api=True, automatic_api_retries=2)
    failed = Response([])
    failed.status_code = 502
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs)
        return failed

    tokenizer = type('Tokenizer', (), {'apply_chat_template': lambda self, *args, **kwargs: [1]})()
    monkeypatch.setattr(collect.requests, 'post', post)
    monkeypatch.setattr(collect, 'disk_guard', lambda config: None)
    monkeypatch.setattr(collect.time, 'sleep', lambda duration: None)
    records = Records(tmp_path)
    budget = Budget(records, settings)
    model = collect.GatewayModel('fake', settings, budget, records, tokenizer, 'sample')
    with pytest.raises(SafetyStop, match='retry limit'):
        model.query([{'role': 'user', 'content': 'test'}])
    assert len(calls) == 3 and model.n_calls == 0
    assert budget.state['pending'] is None
    assert len((tmp_path / 'uncertain-costs.jsonl').read_text().splitlines()) == 3
