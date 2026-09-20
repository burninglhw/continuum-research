import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from core import Records, summarize


def aggregate(rows):
    return {
        'traces': len(rows),
        'model_turns': summarize([row['model_turns'] for row in rows]),
        'accepted_model_turns': summarize([row.get('accepted_model_turns', row['model_turns']) for row in rows]),
        'pooled_agent_tool_time_ms': summarize([duration for row in rows for duration in row['tool_durations_ms']]),
        'api_tokens_per_program_sum_prompt_and_completion': summarize([
            row['api_prompt_tokens_sum'] + row['api_completion_tokens_sum'] for row in rows]),
        'api_tokens_per_program_with_reported_cache': summarize([
            row.get('api_conservative_input_tokens_sum', row['api_prompt_tokens_sum'] + row['api_cached_prompt_tokens_sum'])
            + row['api_completion_tokens_sum'] for row in rows]),
        'llama_request_tokens_per_program_sum': summarize([row['llama_request_tokens_sum'] for row in rows]),
        'llama_final_transcript_tokens': summarize([row['llama_final_transcript_tokens'] for row in rows]),
        'llama_reasoning_tokens_per_program': summarize([row.get('llama_reasoning_tokens_sum', 0) for row in rows]),
        'llama_decode_tokens_including_reasoning_per_program': summarize([
            row.get('llama_decode_tokens_including_reasoning_sum', 0) for row in rows]),
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=root / 'runs/nowcoding-mixed-100-20260918')
    arguments = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted((arguments.run_dir / 'instances').glob('*/metrics.json'))]
    manifest = json.loads((root / 'dataset-manifest.json').read_text())
    completed = [row for row in rows if row.get('collection_completed') and row['model_turns'] > 0 and row['tool_calls'] > 0]
    report = {
        'target_tasks': 100,
        'completed_attempts': len(completed),
        'completed_traces': len(completed),
        'recorded_attempts': len(rows),
        'incomplete_attempts': len(rows) - len(completed),
        'attempts_with_model_response': sum(row['model_turns'] > 0 for row in rows),
        'submitted_count': sum(row['submitted'] for row in rows),
        'submission_does_not_mean_swebench_test_pass': True,
        'exit_statuses': dict(Counter(row['exit_status'] for row in rows)),
        'all_attempts': aggregate(rows),
        'completed_traces_only': aggregate(completed),
        'submitted_only': aggregate([row for row in rows if row['submitted']]),
        'paper_targets': manifest['paper_targets'],
        'not_an_exact_paper_replication': 'User-approved substitute model vs paper GPT-5; original IDs and token aggregation undefined',
        'budget': json.loads((arguments.run_dir / 'budget.json').read_text()) if (arguments.run_dir / 'budget.json').exists() else None,
    }
    Records(arguments.run_dir).write('summary.json', report)
    with (arguments.run_dir / 'metrics.csv').open('w') as stream:
        fields = ['instance_id', 'exit_status', 'collection_completed', 'model_turns', 'tool_calls', 'api_prompt_tokens_sum',
                  'api_conservative_input_tokens_sum',
                  'api_completion_tokens_sum', 'api_cached_prompt_tokens_sum', 'api_reasoning_tokens_sum',
                  'llama_reasoning_tokens_sum', 'llama_decode_tokens_including_reasoning_sum',
                  'llama_request_tokens_sum', 'llama_final_transcript_tokens', 'conservative_cost_cny']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
