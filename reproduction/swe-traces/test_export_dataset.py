import json

import pytest

from export_dataset import build_trace


def write_lines(path, values):
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def make_case(tmp_path):
    first = '```bash\nls\n```'
    last = '```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```'
    initial = [{'role': 'user', 'content': 'task'}]
    next_messages = initial + [{'role': 'assistant', 'content': first}, {'role': 'user', 'content': 'actual output'}]
    corrected = next_messages + [{'role': 'user', 'content': 'Invalid reply rejected; no command executed.'}]
    requests = [{'turn': turn, 'request_attempt': index + 1,
                 'payload': {'messages': messages}, 'llama_prompt_tokens': 10}
                for index, (turn, messages) in enumerate([(1, initial), (2, next_messages),
                                                          (2, next_messages), (3, corrected), (4, corrected)])]
    responses = [{'turn': turn, 'content': content, 'finish_reason': 'stop'}
                 for turn, content in [(1, first), (2, 'invalid'), (3, ''), (4, last)]]
    tools = [{'phase': 'agent', 'command': command, 'duration_ms': duration}
             for command, duration in [('ls', 10), ('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', 5)]]
    (tmp_path / 'metrics.json').write_text(json.dumps({'model_turns': 4, 'tool_calls': 2}))
    (tmp_path / 'task.json').write_text(json.dumps({'instance_id': 'sample'}))
    write_lines(tmp_path / 'requests.jsonl', requests)
    write_lines(tmp_path / 'responses.jsonl', responses)
    write_lines(tmp_path / 'tools.jsonl', tools)
    write_lines(tmp_path / 'response-validation.jsonl', [
        {'turn': 1, 'accepted': True}, {'turn': 2, 'accepted': False}, {'turn': 4, 'accepted': True}])
    return requests, tools


def test_retry_attempts_and_rejected_outputs_remain_distinct(tmp_path):
    make_case(tmp_path)
    trace = build_trace(tmp_path)
    assert len(trace['events']) == 5
    assert trace['events'][1]['response'] is None
    assert trace['events'][2]['response']['content'] == 'invalid'
    assert trace['events'][3]['response']['content'] == ''
    assert [event['tool_duration_ms'] for event in trace['events']] == [10, None, None, None, 5]
    assert [event['accepted_into_context'] for event in trace['events']] == [True, False, False, False, True]


def test_invented_tool_mapping_is_rejected(tmp_path):
    requests, tools = make_case(tmp_path)
    tools[0]['command'] = 'a command the model never requested'
    write_lines(tmp_path / 'tools.jsonl', tools)
    with pytest.raises(ValueError, match='does not match'):
        build_trace(tmp_path)


def test_dropped_accepted_response_is_rejected(tmp_path):
    requests, tools = make_case(tmp_path)
    requests[1]['payload']['messages'] = [{'role': 'user', 'content': 'task'}]
    write_lines(tmp_path / 'requests.jsonl', requests)
    with pytest.raises(ValueError, match='absent from the next request'):
        build_trace(tmp_path)


def test_environment_snapshot_is_exported_with_its_trace(tmp_path):
    make_case(tmp_path)
    environment = {'instance_id': tmp_path.name, 'image_id': 'sha256:123'}
    (tmp_path / 'environment.json').write_text(json.dumps(environment))
    (tmp_path / 'image.json').write_text(json.dumps({'image_id': 'sha256:123'}))
    assert build_trace(tmp_path)['environment'] == environment


def test_environment_from_another_image_is_rejected(tmp_path):
    make_case(tmp_path)
    (tmp_path / 'environment.json').write_text(json.dumps({'instance_id': tmp_path.name, 'image_id': 'sha256:other'}))
    (tmp_path / 'image.json').write_text(json.dumps({'image_id': 'sha256:123'}))
    with pytest.raises(ValueError, match='environment archive identity'):
        build_trace(tmp_path)


def test_observer_disconnects_and_workspace_resumes_are_not_api_retries(tmp_path):
    make_case(tmp_path)
    observer = {'event': 'observer_ssh_disconnect', 'workload_restarted': False}
    resume = {'basis': 'integrity_checked_workspace_archive'}
    write_lines(tmp_path / 'observer-events.jsonl', [observer])
    write_lines(tmp_path / 'resume-events.jsonl', [resume])
    trace = build_trace(tmp_path)
    assert trace['observer_events'] == [observer]
    assert trace['resume_events'] == [resume]
    assert trace['recovery_events'] == []
