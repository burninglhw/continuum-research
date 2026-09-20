import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


TOKEN_FIELDS = ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens')


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')) if value else None


def statistics(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {'n': 0}
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {'n': len(values), 'min': values[0], 'p50': percentile(0.5), 'p90': percentile(0.9),
            'p95': percentile(0.95), 'p99': percentile(0.99), 'max': values[-1],
            'mean': sum(values) / len(values)}


def content_items(record):
    content = (record.get('message') or {}).get('content', [])
    return content if isinstance(content, list) else []


def human_candidate(record):
    return (record.get('type') == 'user' and not record.get('isMeta') and not record.get('isSidechain')
            and not any(item.get('type') == 'tool_result' for item in content_items(record)))


def verify_source(root, tree_file):
    tree = json.loads(tree_file.read_text())
    assert tree.get('truncated') is False
    verified = 0
    for entry in tree['tree']:
        if entry['type'] != 'blob':
            continue
        data = (root / entry['path']).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != entry['sha']:
            raise ValueError('Source hash mismatch: ' + entry['path'])
        verified += 1
    return {'tree_api_reported_sha': tree['sha'], 'verified_git_blobs': verified, 'all_blobs_matched': True}


def audit_session(path, source_root):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    results = defaultdict(list)
    counts = Counter()
    types = Counter(record.get('type', 'missing') for record in records)
    node_ids = {record['uuid'] for record in records if record.get('uuid')}
    segment = 0
    last_timestamp = None
    for line_number, record in enumerate(records, 1):
        current_timestamp = timestamp(record.get('timestamp'))
        if current_timestamp and last_timestamp and current_timestamp < last_timestamp:
            counts['raw_timestamp_regressions'] += 1
        if current_timestamp:
            last_timestamp = current_timestamp
        if record.get('parentUuid') and record['parentUuid'] not in node_ids:
            counts['missing_parent_reference_records'] += 1
        if human_candidate(record):
            segment += 1
            counts['candidate_human_messages'] += 1
        if record.get('isSidechain'):
            counts['sidechain_records'] += 1
        for item in content_items(record):
            if item.get('type') == 'tool_result':
                results[item.get('tool_use_id')].append((line_number, record, item))
        if record.get('type') != 'assistant':
            continue
        message = record['message']
        if record.get('requestId'):
            key = 'request:' + record['requestId']
        elif message.get('id'):
            key = 'message:' + message['id']
            counts['assistant_records_without_request_id'] += 1
        else:
            key = 'unresolved-record:' + str(line_number)
            counts['assistant_records_without_any_request_identity'] += 1
        groups[key].append((line_number, segment, record))
    normalized = []
    for request_key, snapshots in groups.items():
        snapshots.sort(key=lambda entry: (timestamp(entry[2]['timestamp']), entry[0]))
        final_record = snapshots[-1][2]
        final_usage = final_record['message'].get('usage') or {}
        usage_values = {field: [((record['message'].get('usage') or {}).get(field))
                               for _, _, record in snapshots] for field in TOKEN_FIELDS}
        prompt_parts = [final_usage.get(field) for field in TOKEN_FIELDS[:3]]
        prompt_length = sum(prompt_parts) if all(type(value) is int and value >= 0 for value in prompt_parts) else None
        flags = []
        if prompt_length is None:
            flags.append('missing_or_invalid_prompt_usage')
        if any(len(set(usage_values[field])) > 1 for field in TOKEN_FIELDS[:3]):
            flags.append('conflicting_prompt_usage_within_request')
        message_ids = sorted({record['message'].get('id', '') for _, _, record in snapshots})
        if len(message_ids) != 1:
            flags.append('multiple_message_ids_for_request')
        segment_ids = sorted({entry[1] for entry in snapshots})
        if len(segment_ids) > 1:
            flags.append('request_spans_candidate_human_boundary')
        if any(record.get('isSidechain') for _, _, record in snapshots):
            flags.append('sidechain_request')
        if final_record['message'].get('stop_reason') is None:
            flags.append('final_snapshot_has_no_stop_reason')
        tools = {}
        for line_number, _, record in snapshots:
            for item in content_items(record):
                if item.get('type') != 'tool_use':
                    continue
                identity = item['id']
                if identity in tools:
                    counts['duplicate_tool_use_blocks'] += 1
                    continue
                matches = results.get(identity, [])
                distinct_result_times = sorted({result[1].get('timestamp') for result in matches
                                                if result[1].get('timestamp')})
                result_time = distinct_result_times[0] if len(distinct_result_times) == 1 else None
                observed_latency = ((timestamp(result_time) - timestamp(record['timestamp'])).total_seconds() * 1000
                                    if result_time else None)
                tools[identity] = {'tool_use_id': identity, 'tool_name': item.get('name'),
                    'emit_timestamp': record['timestamp'], 'result_timestamp': result_time,
                    'result_record_count': len(matches), 'distinct_result_timestamps': len(distinct_result_times),
                    'observed_emit_to_result_ms': observed_latency,
                    'source_line': line_number}
                if not matches:
                    flags.append('missing_tool_result')
                if len(distinct_result_times) > 1:
                    flags.append('ambiguous_tool_result_time')
                if observed_latency is not None and observed_latency < 0:
                    flags.append('negative_tool_latency')
        final_tool_ids = {item['id'] for item in content_items(final_record) if item.get('type') == 'tool_use'}
        if set(tools) - final_tool_ids:
            flags.append('last_event_only_would_drop_tools')
        nonmissing_outputs = [value for value in usage_values['output_tokens'] if type(value) is int]
        if nonmissing_outputs and final_usage.get('output_tokens') != max(nonmissing_outputs):
            flags.append('last_output_usage_not_maximum')
        normalized.append({'source_file': str(path.relative_to(source_root)), 'project_id': path.parent.name,
            'session_id': final_record.get('sessionId'), 'request_key': request_key,
            'message_ids': message_ids, 'snapshot_count': len(snapshots),
            'source_lines': [entry[0] for entry in snapshots],
            'candidate_human_segment_ids': segment_ids,
            'first_assistant_timestamp': snapshots[0][2]['timestamp'],
            'last_assistant_timestamp': final_record['timestamp'],
            'model': final_record['message'].get('model'), 'reported_prompt_tokens': prompt_length,
            'final_reported_usage': {field: final_usage.get(field) for field in TOKEN_FIELDS},
            'output_usage_values': sorted(set(nonmissing_outputs)),
            'tools': list(tools.values()), 'quality_flags': sorted(set(flags)),
            'exact_prefix_token_ids_available': False, 'ready_for_gpu_replay': False})
    normalized.sort(key=lambda record: (record['first_assistant_timestamp'], record['source_lines'][0]))
    for previous, current in zip(normalized, normalized[1:]):
        if timestamp(previous['last_assistant_timestamp']) > timestamp(current['first_assistant_timestamp']):
            counts['overlapping_request_snapshot_intervals'] += 1
        if (previous['reported_prompt_tokens'] is not None and current['reported_prompt_tokens'] is not None
                and current['reported_prompt_tokens'] < previous['reported_prompt_tokens']):
            counts['prompt_length_decreases_adjacent_requests'] += 1
    counts['raw_records'] = len(records)
    counts['assistant_snapshot_records'] = types['assistant']
    counts['request_groups'] = len(normalized)
    counts['merged_away_assistant_snapshots'] = types['assistant'] - len(normalized)
    return normalized, counts, types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-output', type=Path)
    arguments = parser.parse_args()
    verification = verify_source(arguments.source, arguments.tree)
    arguments.output.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    raw_types = Counter()
    all_requests = []
    per_session = []
    for path in sorted((arguments.source / 'raw_traces').glob('*/*.jsonl')):
        requests, counts, types = audit_session(path, arguments.source)
        totals.update(counts)
        raw_types.update(types)
        all_requests.extend(requests)
        segments = Counter(tuple(request['candidate_human_segment_ids']) for request in requests)
        per_session.append({'source_file': str(path.relative_to(arguments.source)), **dict(counts),
                            'candidate_segments_with_requests': len(segments),
                            'candidate_segments_with_two_or_more_requests': sum(count >= 2 for count in segments.values())})
    parsed = []
    for path in sorted((arguments.source / 'parsed_traces').glob('*/*/events.json')):
        parsed.extend(json.loads(path.read_text()))
    parsed_types = Counter(event['event_type'] for event in parsed)
    all_tools = [tool for request in all_requests for tool in request['tools']]
    tools_by_name = defaultdict(list)
    for tool in all_tools:
        tools_by_name[tool['tool_name']].append(tool['observed_emit_to_result_ms'])
    manifest = json.loads((arguments.source / 'parsed_traces/manifest.json').read_text())
    assert len(per_session) == manifest['num_sessions']
    assert parsed_types['llm_call'] == manifest['total_llm_calls']
    assert parsed_types['tool_call'] == manifest['total_tool_calls']
    report = {'source_commit': '181c435a090d328d00bbbee4c8eeb27d32f3abd2',
        'source_verification': verification, 'sessions': len(per_session),
        'empty_raw_sessions': sum(record['raw_records'] == 0 for record in per_session),
        'sessions_with_request_groups': sum(record['request_groups'] > 0 for record in per_session),
        'sessions_with_two_or_more_request_groups': sum(record['request_groups'] >= 2 for record in per_session),
        'projects': len({request['project_id'] for request in all_requests}),
        'raw_counts': dict(totals), 'raw_record_types': dict(raw_types),
        'parsed_event_counts': dict(parsed_types),
        'request_quality_flags': dict(Counter(flag for request in all_requests for flag in request['quality_flags'])),
        'requests_with_multiple_tools': sum(len(request['tools']) > 1 for request in all_requests),
        'unique_tool_uses_after_request_merge': len(all_tools),
        'requests_with_reported_output_one': sum(request['final_reported_usage']['output_tokens'] == 1 for request in all_requests),
        'requests_with_reported_output_one_and_tools': sum(request['final_reported_usage']['output_tokens'] == 1 and bool(request['tools']) for request in all_requests),
        'reported_prompt_tokens': statistics([request['reported_prompt_tokens'] for request in all_requests]),
        'reported_output_tokens': statistics([request['final_reported_usage']['output_tokens'] for request in all_requests]),
        'requests_over_128k_prompt': sum((request['reported_prompt_tokens'] or 0) > 131072 for request in all_requests),
        'requests_over_128k_prompt_plus_output': sum((request['reported_prompt_tokens'] or 0) + (request['final_reported_usage']['output_tokens'] or 0) > 131072 for request in all_requests),
        'tool_observed_latency_ms': statistics([tool['observed_emit_to_result_ms'] for tool in all_tools]),
        'tool_latency_by_name': {name: statistics(values) for name, values in sorted(tools_by_name.items())},
        'candidate_human_segments': sum(record['candidate_segments_with_requests'] for record in per_session),
        'candidate_human_segments_with_two_or_more_requests': sum(record['candidate_segments_with_two_or_more_requests'] for record in per_session),
        'model_counts_by_request': dict(Counter(request['model'] for request in all_requests)),
        'request_identity_sources': dict(Counter(request['request_key'].split(':')[0] for request in all_requests)),
        'synthetic_model_groups_with_zero_prompt': sum(request['model'] == '<synthetic>' and request['reported_prompt_tokens'] == 0 for request in all_requests),
        'synthetic_model_groups_with_tools': sum(request['model'] == '<synthetic>' and bool(request['tools']) for request in all_requests),
        'per_session': per_session,
        'measurement_warning': 'Tool duration is emission-to-result timestamp difference, not an independently instrumented CPU execution timer.',
        'prefix_warning': 'Reported provider cache hits do not recover prompt token identities or inter-request longest common prefixes.',
        'output_warning': 'Final snapshot usage is retained as reported, not certified as final complete generated-token usage.',
        'human_boundary_warning': 'User-message candidates are not a finalized parent/branch-aware episode segmentation.',
        'gpu_experiment_run': False, 'paid_api_called': False, 'normalized_metadata_is_replay_ready': False}
    (arguments.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    metadata_path = arguments.metadata_output or arguments.output / 'request-metadata.jsonl'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open('w') as stream:
        for request in all_requests:
            stream.write(json.dumps(request, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('per_session', 'tool_latency_by_name')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
