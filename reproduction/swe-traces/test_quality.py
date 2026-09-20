import json

from quality import audit_case
import pytest


def test_multiple_model_blocks_are_not_counted_as_tool_calls(tmp_path):
    (tmp_path / 'metrics.json').write_text(json.dumps({
        'instance_id': 'sample', 'collection_completed': True,
        'model_turns': 2, 'tool_calls': 1, 'exit_status': 'Submitted',
    }))
    responses = [
        {'content': '```bash\nls\n```\n```bash\npwd\n```'},
        {'content': '```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```'},
    ]
    (tmp_path / 'responses.jsonl').write_text('\n'.join(json.dumps(event) for event in responses))
    (tmp_path / 'tools.jsonl').write_text(json.dumps({
        'phase': 'agent', 'raw_output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n',
    }))
    (tmp_path / 'patch.diff').write_text('')
    result = audit_case(tmp_path)
    assert result['actual_agent_tool_calls'] == 1
    assert result['bash_blocks_per_response'] == [2, 1]
    assert result['non_submission_tool_calls'] == 0
    assert result['swebench_resolved'] is None
    assert set(result['quality_flags']) == {
        'response_format_errors', 'empty_patch', 'no_non_submission_tool_call',
    }


def test_submission_without_real_work_is_rejected():
    from types import SimpleNamespace
    from collect import NonTerminatingException, TracingAgent
    output = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'
    agent = TracingAgent(SimpleNamespace(), SimpleNamespace(tool_events=[{'raw_output': output}]))
    with pytest.raises(NonTerminatingException, match='NOT executed'):
        agent.has_finished({'output': output})


def test_submission_after_actual_task_command_keeps_original_semantics():
    from types import SimpleNamespace
    from collect import TerminatingException, TracingAgent
    agent = TracingAgent(SimpleNamespace(), SimpleNamespace(tool_events=[{'raw_output': '/testbed\n'}]))
    with pytest.raises(TerminatingException):
        agent.has_finished({'output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'})


def test_archived_response_cost_retained_without_double_charging(tmp_path):
    from types import SimpleNamespace
    from collect import GatewayModel
    from core import Budget, Records
    config = {'requested_model': 'test', 'job_timeout_seconds': 900}
    records = Records(tmp_path)
    records.event('attempt-cost-carryovers.jsonl', {
        'task_id': 'sample', 'completed_response_cost_cny': 1.5,
    })
    budget = Budget(records, config)
    model = GatewayModel('fake', config, budget, records, SimpleNamespace(), 'sample')
    assert model.cost == 1.5
    assert budget.state['charged_upper_estimate_cny'] == 0


def test_retry_count_survives_process_restart(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import collect
    from core import Budget, Records, SafetyStop, TransientGatewayError
    config = {'requested_model': 'test', 'job_timeout_seconds': 900, 'automatic_api_retries': 2}
    records = Records(tmp_path)
    records.event('retry-events.jsonl', {'turn': 1, 'retry_index': 0, 'retry_limit': 2, 'will_retry': True})
    model = collect.GatewayModel('fake', config, Budget(records, config), records, SimpleNamespace(), 'sample')
    attempts = []

    def fail(messages):
        attempts.append(1)
        raise TransientGatewayError('test timeout')

    monkeypatch.setattr(model, '_query_once', fail)
    monkeypatch.setattr(collect.time, 'sleep', lambda seconds: None)
    with pytest.raises(SafetyStop, match='retry limit'):
        model.query([])
    assert len(attempts) == 2


def test_stream_wall_timeout_uses_bounded_transient_policy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import gateway_stream
    from core import Records, TransientGatewayError
    monkeypatch.setattr(gateway_stream.time, 'perf_counter', lambda: 200)
    response = SimpleNamespace(iter_lines=lambda **kwargs: iter([b'data: {}']))
    with pytest.raises(TransientGatewayError, match='time limit'):
        gateway_stream.read_completion(response, Records(tmp_path), 1, 0, 180)
