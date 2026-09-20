import argparse
import json
import re
from collections import Counter
from pathlib import Path

from core import Records


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def audit_case(case):
    metrics = json.loads((case / 'metrics.json').read_text())
    responses = events(case / 'responses.jsonl')
    validations = events(case / 'response-validation.jsonl')
    accepted_turns = {event['turn'] for event in validations if event.get('accepted')}
    accepted_responses = [event for event in responses if event['turn'] in accepted_turns] if validations else responses
    tools = [event for event in events(case / 'tools.jsonl') if event.get('phase') == 'agent']
    blocks = [len(re.findall(r'```bash\s*\n(.*?)\n```', response.get('content') or '', re.S))
              for response in accepted_responses]
    patch = case / 'patch.diff'
    flags = []
    if not metrics.get('collection_completed'):
        flags.append('incomplete_attempt')
    if any(count != 1 for count in blocks):
        flags.append('response_format_errors')
    if patch.exists() and patch.stat().st_size == 0:
        flags.append('empty_patch')
    substantive_tools = [event for event in tools
                         if not str(event.get('raw_output', '')).lstrip().startswith(
                             ('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', 'MINI_SWE_AGENT_FINAL_OUTPUT'))]
    if not substantive_tools:
        flags.append('no_non_submission_tool_call')
    if len(responses) != metrics['model_turns'] or len(tools) != metrics['tool_calls']:
        flags.append('event_count_mismatch')
    return {
        'instance_id': metrics['instance_id'],
        'collection_completed': metrics.get('collection_completed', False),
        'exit_status': metrics['exit_status'],
        'model_turns': len(responses),
        'accepted_model_turns': len(accepted_responses),
        'rejected_response_turns': [event['turn'] for event in validations if not event.get('accepted')],
        'responses_without_agent_validation': [event.get('turn') for event in responses if validations and event.get('turn') not in {entry['turn'] for entry in validations}],
        'bash_blocks_per_response': blocks,
        'actual_agent_tool_calls': len(tools),
        'non_submission_tool_calls': len(substantive_tools),
        'patch_bytes': patch.stat().st_size if patch.exists() else None,
        'swebench_resolved': metrics.get('swebench_resolved'),
        'quality_flags': flags,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'runs/nowcoding-mixed-100-20260918')
    arguments = parser.parse_args()
    cases = [audit_case(path.parent) for path in sorted((arguments.run_dir / 'instances').glob('*/metrics.json'))]
    report = {
        'recorded_attempts': len(cases),
        'completed_attempts': sum(case['collection_completed'] for case in cases),
        'flag_counts': dict(Counter(flag for case in cases for flag in case['quality_flags'])),
        'samples_replaced': 0,
        'submitted_is_not_test_pass': True,
        'unexecuted_model_code_blocks_are_not_tool_events': True,
        'cases': cases,
    }
    Records(arguments.run_dir).write('quality.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
