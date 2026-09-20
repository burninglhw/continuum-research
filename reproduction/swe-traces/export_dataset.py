import argparse
import hashlib
import json
import re
from pathlib import Path

from core import Records
from trace_protocol import response_problem


ROOT = Path(__file__).resolve().parent


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def build_trace(case):
    metrics = json.loads((case / 'metrics.json').read_text())
    environment_path = case / 'environment.json'
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else None
    if environment is not None:
        recorded_image = json.loads((case / 'image.json').read_text())
        if environment['instance_id'] != case.name or environment['image_id'] != recorded_image['image_id']:
            raise ValueError('Trace environment archive identity mismatch')
    requests = events(case / 'requests.jsonl')
    responses = events(case / 'responses.jsonl')
    validations = events(case / 'response-validation.jsonl')
    tools = [event for event in events(case / 'tools.jsonl') if event.get('phase') == 'agent']
    by_turn = {response['turn']: response for response in responses}
    accepted = {event['turn'] for event in validations if event.get('accepted')}
    last_attempt = {request['turn']: index for index, request in enumerate(requests)}
    if len(by_turn) != len(responses) or len(responses) != metrics['model_turns'] or len(tools) != metrics['tool_calls']:
        raise ValueError('Duplicate response turns or event/metric count mismatch')
    if len(accepted) != len(tools):
        raise ValueError('Accepted model actions and executed tools do not correspond one to one')
    timeline = []
    tool_index = 0
    previous_messages = []
    expected_assistant = None
    for index, request in enumerate(requests):
        messages = request['payload']['messages']
        if messages[:len(previous_messages)] != previous_messages:
            raise ValueError('Request history is not append-only')
        if expected_assistant is not None:
            if len(messages) <= len(previous_messages) or messages[len(previous_messages)] != expected_assistant:
                raise ValueError('Accepted response is absent from the next request context')
        previous_messages = messages
        response = by_turn.get(request['turn']) if index == last_attempt[request['turn']] else None
        tool = None
        is_accepted = response is not None and response['turn'] in accepted
        if is_accepted:
            problem = response_problem(response['content'], response['finish_reason'])
            if problem:
                raise ValueError('Accepted response failed protocol validation: ' + problem)
            command = re.search(r'```bash\s*\n(.*?)\n```', response['content'], re.S).group(1).strip()
            tool = tools[tool_index]
            tool_index += 1
            if command != tool['command'].strip():
                raise ValueError('Recorded command does not match its accepted model action')
            if not isinstance(tool.get('duration_ms'), (int, float)) or tool['duration_ms'] < 0:
                raise ValueError('Missing or invalid tool duration')
        timeline.append({'request_attempt': request['request_attempt'], 'response_turn': request['turn'],
                         'request': request['payload'], 'llama_prompt_tokens': request['llama_prompt_tokens'],
                         'response': response, 'accepted_into_context': is_accepted,
                         'tool': tool, 'tool_duration_ms': tool['duration_ms'] if tool else None})
        expected_assistant = {'role': 'assistant', 'content': response['content']} if is_accepted else None
    return {'schema_version': 1, 'instance_id': case.name, 'metrics': metrics,
            'task': json.loads((case / 'task.json').read_text()), 'environment': environment, 'events': timeline,
            'recovery_events': events(case / 'retry-events.jsonl'),
            'resume_events': events(case / 'resume-events.jsonl'),
            'observer_events': events(case / 'observer-events.jsonl'),
            'validation_events': validations, 'reasoning_in_next_request': False,
            'runtime_replay_validated': False,
            'tool_timing_scope': 'container command including shell startup; Docker transport excluded'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/deepseek-flash-swe100-v2-20260918')
    arguments = parser.parse_args()
    selected = [json.loads(line) for line in (ROOT / 'tasks-100.jsonl').read_text().splitlines()]
    run = arguments.run_dir.resolve()
    destination = run / 'dataset'
    records = Records(destination)
    coverage = []
    traces = []
    for task in selected:
        case = run / 'instances' / task['instance_id']
        entry = {'instance_id': task['instance_id'], 'state': 'not_completed'}
        if (case / 'metrics.json').exists():
            metrics = json.loads((case / 'metrics.json').read_text())
            entry.update(exit_status=metrics['exit_status'], submitted=metrics['submitted'])
            if metrics.get('collection_completed'):
                try:
                    trace = build_trace(case)
                    traces.append(trace)
                    entry.update(state='validated_record', model_responses=metrics['model_turns'],
                                 accepted_model_turns=metrics['accepted_model_turns'],
                                 tools=metrics['tool_calls'], swebench_resolved=metrics['swebench_resolved'])
                except (KeyError, ValueError) as error:
                    entry.update(state='validation_failed', error=str(error))
            else:
                entry['state'] = 'incomplete_attempt'
        coverage.append(entry)
    temporary = destination / 'traces.jsonl.tmp'
    with temporary.open('w') as stream:
        for trace in traces:
            stream.write(json.dumps(trace, ensure_ascii=False) + '\n')
    temporary.replace(destination / 'traces.jsonl')
    records.write('coverage.json', coverage)
    manifest = {'schema_version': 1, 'target_records': 100, 'validated_records': len(traces),
                'complete_100': len(traces) == len(selected) == 100,
                'environment_records': sum(trace['environment'] is not None for trace in traces),
                'environment_records_captured_from_live_image': sum(
                    (trace['environment'] or {}).get('image_capture_status') == 'captured' for trace in traces),
                'sample_seed': 42, 'sample_replacements': 0,
                'task_sample_sha256': hashlib.sha256((ROOT / 'tasks-100.jsonl').read_bytes()).hexdigest(),
                'traces_sha256': hashlib.sha256((destination / 'traces.jsonl').read_bytes()).hexdigest(),
                'generator': 'deepseek-flash', 'protocol': 'deepseek-bash-v2',
                'paper_generator': 'gpt-5', 'exact_paper_dataset': False,
                'paper_targets': json.loads((ROOT / 'dataset-manifest.json').read_text())['paper_targets'],
                'runtime_replay_validated': False, 'swebench_gold_tests_run': False,
                'includes_failed_api_attempts_within_exported_traces': True,
                'incomplete_programs_exported': False,
                'null_tool_duration_means_no_tool_executed': True}
    records.write('manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
