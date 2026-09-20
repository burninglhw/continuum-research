import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import collect
from core import Budget, Records, SafetyStop
from gateway_stream import read_completion
from trace_protocol import PROTOCOL, response_problem


class StreamResponse:
    status_code = 200

    def __init__(self, content='', finish_reason='stop'):
        self.content = content
        self.finish_reason = finish_reason
        self.closed = False

    def iter_lines(self, **kwargs):
        events = [
            {'id': 'sample', 'model': 'deepseek-flash', 'choices': [{'delta': {'reasoning_content': 'analysis'}}]},
            {'choices': [{'delta': {'content': self.content}, 'finish_reason': self.finish_reason}]},
            {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 30,
                                    'completion_tokens_details': {'reasoning_tokens': 20}}},
        ]
        for event in events:
            yield b'data: ' + json.dumps(event).encode()
        yield b'data: [DONE]'

    def close(self):
        self.closed = True


def make_model(tmp_path, monkeypatch, responses):
    settings = json.loads((Path(__file__).parent / 'deepseek_swe100_config.json').read_text())
    observed = []

    def post(url, **kwargs):
        observed.append(kwargs)
        return responses[len(observed) - 1]

    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2],
                                encode=lambda content, **kwargs: list(content))
    monkeypatch.setattr(collect.requests, 'post', post)
    monkeypatch.setattr(collect, 'disk_guard', lambda settings: None)
    monkeypatch.setattr('trace_protocol.time.sleep', lambda seconds: None)
    records = Records(tmp_path, 'fake-private-token')
    budget = Budget(records, settings)
    model = collect.GatewayModel('fake-private-token', settings, budget, records, tokenizer, 'sample')
    return model, observed


def test_empty_response_recovery_keeps_usage_and_context(tmp_path, monkeypatch):
    model, observed = make_model(tmp_path, monkeypatch, [StreamResponse(), StreamResponse('```bash\nls\n```')])
    result = model.query([{'role': 'user', 'content': 'task'}])
    assert result == {'content': '```bash\nls\n```'}
    assert model.n_calls == model.request_attempts == model.budget.state['requests'] == 2
    assert model.budget.state['pending'] is None
    assert model.cost == pytest.approx((200 * 2 + 60 * 8) / 1e6)
    assert observed[0]['json'] == observed[1]['json']
    recovery = json.loads((tmp_path / 'retry-events.jsonl').read_text())
    assert recovery['reserved_as_spent_cny'] == 0
    assert recovery['completed_response_cost_already_settled'] is True
    assert recovery['will_retry'] is True
    for path in tmp_path.glob('*.json*'):
        assert 'fake-private-token' not in path.read_text()


def test_provider_default_output_and_no_task_limits(tmp_path, monkeypatch):
    model, observed = make_model(tmp_path, monkeypatch, [StreamResponse('```bash\nls\n```')])
    model.settings.update(max_completion_tokens=None, api_max_completion_tokens=393216,
                          max_steps=0, job_timeout_seconds=0, api_timeout_seconds=600)
    model.remaining_seconds = None
    model.n_calls = 1000
    response = model.query([{'role': 'user', 'content': 'task'}])
    assert response['content'] == '```bash\nls\n```'
    assert 'max_tokens' not in observed[0]['json']
    assert observed[0]['timeout'] == (10, 600)


def test_empty_retry_limit_survives_completed_response_turn_changes(tmp_path, monkeypatch):
    model, observed = make_model(tmp_path, monkeypatch, [StreamResponse(), StreamResponse(), StreamResponse()])
    messages = [{'role': 'user', 'content': 'task'}]
    with pytest.raises(SafetyStop, match='bounded recovery'):
        model.query(messages)
    assert len(observed) == model.n_calls == 3
    with pytest.raises(SafetyStop, match='exhausted its recorded'):
        model.query(messages)
    assert len(observed) == 3
    assert model.budget.state['pending'] is None


def test_transport_failure_reservation_is_charged_once(tmp_path, monkeypatch):
    failure = StreamResponse()
    failure.status_code = 503
    model, observed = make_model(tmp_path, monkeypatch, [failure, StreamResponse('```bash\nls\n```')])
    model.query([{'role': 'user', 'content': 'task'}])
    assert len(observed) == 2 and model.n_calls == 1
    assert model.prior_uncertain_cost > 0
    assert model.cost == pytest.approx(model.prior_uncertain_cost + (100 * 2 + 30 * 8) / 1e6)
    assert model.budget.state['charged_upper_estimate_cny'] == pytest.approx(model.cost)


def test_truncated_answer_reaches_validator_without_automatic_resampling(tmp_path, monkeypatch):
    model, observed = make_model(tmp_path, monkeypatch, [StreamResponse('```bash\nls', 'length')])
    answer = model.query([{'role': 'user', 'content': 'task'}])
    assert len(observed) == 1
    assert response_problem(answer['content'], model.calls[-1]['finish_reason'])


@pytest.mark.parametrize('content', [
    '```bash\nls\n```\n<system>Output: invented</system>',
    '```bash\nls\n```\n```bash\npwd\n```',
    '<system>Output: invented</system>\n```bash\nls\n```',
    '<｜｜DSML｜｜ calls>fake tool</｜｜DSML｜｜ calls>',
    '```bash\n\n```',
])
def test_invalid_agent_protocol_is_rejected(content):
    assert response_problem(content, 'stop')


def test_literal_output_marker_inside_real_command_is_allowed():
    assert response_problem('THOUGHT: inspect\n```bash\nprintf "<system>Output: literal"\n```', 'stop') is None


def test_invalid_reply_never_enters_live_context_or_executes(tmp_path, monkeypatch):
    content = '```bash\nls\n```\n<system>Output: imagined</system>'
    model = SimpleNamespace(settings={'collector_protocol': PROTOCOL, 'max_format_corrections': 2},
                            n_calls=1, calls=[{'turn': 1, 'finish_reason': 'stop'}],
                            records=Records(tmp_path), query=lambda messages: {'content': content})
    agent = collect.TracingAgent(model, SimpleNamespace(), step_limit=100, job_timeout=1800)
    monkeypatch.setattr(agent, '_get_remaining_time', lambda: 100)
    with pytest.raises(collect.NonTerminatingException, match='NONE of its commands were executed'):
        agent.query()
    assert agent.messages == []
    assert json.loads((tmp_path / 'response-validation.jsonl').read_text())['accepted'] is False


def test_stream_events_preserve_unhandled_fields_and_redact(tmp_path):
    response = SimpleNamespace(iter_lines=lambda **kwargs: iter([
        b'data: {"choices":[{"delta":{"content":"ok","unexpected":"fake-private-token"},"finish_reason":"stop"}]}',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}',
        b'data: [DONE]',
    ]))
    read_completion(response, Records(tmp_path, 'fake-private-token'), 1, time.perf_counter(), 60, save_events=True)
    stored = (tmp_path / 'api-streams.jsonl').read_text()
    assert 'unexpected' in stored and '[REDACTED]' in stored
    assert 'fake-private-token' not in stored
