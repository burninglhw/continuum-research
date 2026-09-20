import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit import TOKEN_FIELDS, content_items, statistics, timestamp, verify_source


SOURCE_COMMIT = '181c435a090d328d00bbbee4c8eeb27d32f3abd2'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, values):
    with path.open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')


def boundary_kind(record):
    if record.get('type') == 'system' and record.get('subtype') in ('compact_boundary', 'local_command'):
        return record['subtype']
    if record.get('isCompactSummary'):
        return None
    if record.get('isVisibleInTranscriptOnly') or record.get('isMeta'):
        return 'metadata_boundary'
    if record.get('type') == 'user' and not any(item.get('type') == 'tool_result' for item in content_items(record)):
        return 'sidechain_input' if record.get('isSidechain') else 'observed_user_input'
    return None


def graph_index(records):
    nodes = {record['uuid']: record for record in records}
    if len(nodes) != len(records):
        raise ValueError('Duplicate node identities require manual reconciliation')
    resolved = {}
    for start in nodes:
        pending = []
        seen = set()
        current = start
        while current not in resolved:
            if current in seen:
                anchor = 'cycle:' + min(seen)
                resolved[current] = (anchor, 'parent_cycle', 0)
                break
            seen.add(current)
            record = nodes.get(current)
            if record is None:
                resolved[current] = ('missing:' + str(current), 'missing_parent_root', 0)
                break
            kind = boundary_kind(record)
            if kind:
                resolved[current] = (current, kind, 0)
                break
            if record.get('parentUuid') is None:
                resolved[current] = (current, 'unanchored_root', 0)
                break
            pending.append(current)
            current = record['parentUuid']
        for identity in reversed(pending):
            if identity in resolved:
                continue
            anchor, kind, depth = resolved[nodes[identity]['parentUuid']]
            resolved[identity] = (anchor, kind, depth + 1)
    return nodes, resolved


def identity(record):
    message = record.get('message') or {}
    return record.get('requestId') or '', message.get('id') or 'node:' + record['uuid']


def elapsed_us(start, finish):
    delta = timestamp(finish) - timestamp(start)
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def convert_session(path, source_root, split, config):
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            record['_line'] = number
            records.append(record)
    nodes, graph = graph_index(records)
    source_file = str(path.relative_to(source_root))
    source_session = path.parent.name + '/' + path.stem
    episodes = defaultdict(list)
    request_messages = defaultdict(set)
    request_anchors = defaultdict(set)
    message_requests = defaultdict(set)
    tool_owners = defaultdict(set)
    result_nodes = defaultdict(list)
    for record in records:
        anchor, kind, depth = graph[record['uuid']]
        branch = 'sidechain' if record.get('isSidechain') else 'main'
        record['_depth'] = depth
        record['_anchor'] = anchor
        record['_branch'] = branch
        episodes[(anchor, kind, branch)].append(record)
        if record.get('type') == 'assistant':
            request_id, message_id = identity(record)
            if request_id:
                request_messages[request_id].add(message_id)
                request_anchors[request_id].add((anchor, branch))
            message_requests[message_id].add(request_id)
        for item in content_items(record):
            if record.get('type') == 'assistant' and item.get('type') == 'tool_use':
                tool_owners[item['id']].add((identity(record), anchor, branch))
            if item.get('type') == 'tool_result':
                result_nodes[item.get('tool_use_id')].append(record)
    all_requests = []
    decisions = []
    accepted = []
    for (anchor, start_kind, branch), episode_nodes in sorted(episodes.items()):
        episode_nodes.sort(key=lambda record: (record['_depth'], record['_line']))
        episode_id = source_session + '/' + branch + '/' + anchor
        fatal = set()
        excluded = set()
        warnings = set()
        if branch != 'main':
            fatal.add('sidechain_not_replayed')
        if start_kind != 'observed_user_input':
            fatal.add('no_observed_user_start:' + start_kind)
        episode_node_ids = {record['uuid'] for record in episode_nodes}
        child_counts = Counter(record.get('parentUuid') for record in episode_nodes if record.get('parentUuid') in episode_node_ids)
        if any(count > 1 for count in child_counts.values()):
            fatal.add('branching_parent_chain')
        if any(not record.get('isSidechain') and nodes.get(record.get('parentUuid'), {}).get('isSidechain') for record in episode_nodes):
            fatal.add('main_chain_crosses_sidechain')
        if any(record.get('type') == 'system' and record.get('subtype') == 'api_error' for record in episode_nodes):
            fatal.add('api_error_inside_episode')
        grouped = defaultdict(list)
        request_order = []
        for record in episode_nodes:
            if record.get('type') != 'assistant':
                continue
            key = identity(record)
            grouped[key].append(record)
            if not request_order or request_order[-1] != key:
                if key in request_order:
                    fatal.add('noncontiguous_request_fragments')
                request_order.append(key)
        ordered_keys = list(dict.fromkeys(request_order))
        turns = []
        matched_results = set()
        for turn_index, key in enumerate(ordered_keys):
            snapshots = grouped[key]
            final = snapshots[-1]
            first = snapshots[0]
            request_id, message_id = key
            request_key = episode_id + '/' + (request_id or 'missing-request') + '/' + message_id
            problems = set()
            request_warnings = set()
            if request_id and len(request_messages[request_id]) != 1:
                problems.add('request_id_message_collision')
            if request_id and len(request_anchors[request_id]) != 1:
                problems.add('request_spans_boundaries_or_branches')
            if len(message_requests[message_id]) != 1:
                problems.add('message_id_request_collision')
            if not request_id:
                request_warnings.add('message_id_fallback')
            if message_id.startswith('node:'):
                problems.add('missing_message_id')
            models = {record['message'].get('model') for record in snapshots}
            if '<synthetic>' in models or any(record.get('isApiErrorMessage') for record in snapshots):
                problems.add('synthetic_or_api_error_message')
            if len(models) != 1:
                problems.add('model_changes_within_request')
            usage = final['message'].get('usage') or {}
            usages = [record['message'].get('usage') or {} for record in snapshots]
            if any(type(values.get(field)) is not int or values[field] < 0 for values in usages for field in TOKEN_FIELDS):
                problems.add('invalid_token_usage')
            if any(len({values.get(field) for values in usages}) != 1 for field in TOKEN_FIELDS[:3]):
                problems.add('conflicting_prompt_usage')
            prompt = sum(usage[field] for field in TOKEN_FIELDS[:3] if type(usage.get(field)) is int)
            output = usage.get('output_tokens') if type(usage.get('output_tokens')) is int else 0
            if prompt <= 0 or output <= 0:
                problems.add('nonpositive_token_length')
            output_values = [values.get('output_tokens', 0) for values in usages]
            numeric_outputs = [value for value in output_values if type(value) is int]
            if any(current < previous for previous, current in zip(numeric_outputs, numeric_outputs[1:])):
                problems.add('output_usage_regression')
            if final['message'].get('stop_reason') is None:
                request_warnings.add('no_final_stop_reason')
            if output <= 2:
                request_warnings.add('very_small_reported_output')
            if any(elapsed_us(previous['timestamp'], current['timestamp']) < 0 for previous, current in zip(snapshots, snapshots[1:])):
                problems.add('timestamp_regression_within_request')
            next_first = grouped[ordered_keys[turn_index + 1]][0] if turn_index + 1 < len(ordered_keys) else None
            tools = {}
            for record in snapshots:
                for item in content_items(record):
                    if item.get('type') != 'tool_use':
                        continue
                    tool_id = item['id']
                    if len(tool_owners[tool_id]) != 1:
                        problems.add('tool_id_reused_across_requests')
                    if not isinstance(item.get('name'), str) or not item['name']:
                        problems.add('invalid_tool_name')
                    if tool_id in tools:
                        tools[tool_id]['duplicate_source_lines'].append(record['_line'])
                        if tools[tool_id]['tool_name'] != item.get('name') or tools[tool_id]['emit_timestamp'] != record['timestamp']:
                            problems.add('conflicting_tool_reemission')
                        continue
                    candidates = [result for result in result_nodes.get(tool_id, [])
                                  if result['_anchor'] == anchor and result['_branch'] == branch
                                  and result['_depth'] > record['_depth']
                                  and (next_first is None or result['_depth'] < next_first['_depth'])]
                    candidate_ids = {result['uuid'] for result in candidates}
                    result = candidates[0] if len(candidate_ids) == 1 else None
                    if result is None:
                        problems.add('missing_causal_tool_result' if not candidates else 'ambiguous_causal_tool_result')
                    latency = elapsed_us(record['timestamp'], result['timestamp']) if result else None
                    if result:
                        matched_results.add((result['uuid'], tool_id))
                        if latency < 0:
                            problems.add('negative_tool_latency')
                        if result['_depth'] <= final['_depth'] or elapsed_us(final['timestamp'], result['timestamp']) < 0:
                            problems.add('streaming_tool_overlap')
                        if next_first and elapsed_us(result['timestamp'], next_first['timestamp']) < 0:
                            problems.add('next_request_precedes_tool_result')
                    tools[tool_id] = {
                        'tool_use_id': tool_id, 'tool_name': item.get('name'),
                        'emit_timestamp': record['timestamp'], 'emit_node_id': record['uuid'], 'emit_source_line': record['_line'],
                        'result_timestamp': result['timestamp'] if result else None,
                        'result_node_id': result['uuid'] if result else None,
                        'result_source_line': result['_line'] if result else None,
                        'causal_result_candidate_nodes': sorted(candidate_ids),
                        'other_result_occurrences': len(result_nodes.get(tool_id, [])) - len(candidates),
                        'observed_delay_us': latency, 'duplicate_source_lines': []}
            if len(tools) > 1:
                excluded.add('multi_tool_episode')
            if any(tool['tool_name'] in ('Task', 'ExitPlanMode', 'AskUserQuestion') for tool in tools.values()):
                excluded.add('nested_agent_or_interactive_tool')
            if any(tool['observed_delay_us'] is not None and tool['observed_delay_us'] > config['max_tool_delay_us'] for tool in tools.values()):
                excluded.add('tool_delay_above_pilot_limit')
            if prompt + output > config['max_model_len']:
                excluded.add('context_above_model_limit')
            if next_first and not tools:
                problems.add('non_tool_intermediate_transition')
            if final['message'].get('stop_reason') in ('max_tokens', 'pause_turn'):
                problems.add('truncated_or_paused_response')
            turns.append({
                'source_commit': SOURCE_COMMIT, 'source_file': source_file, 'source_session': source_session,
                'project_id': path.parent.name, 'session_id': path.stem, 'branch_id': branch,
                'episode_id': episode_id, 'request_key': request_key, 'request_id': request_id or None,
                'message_id': message_id, 'turn_index': turn_index,
                'source_lines': [record['_line'] for record in snapshots], 'node_ids': [record['uuid'] for record in snapshots],
                'first_timestamp': first['timestamp'], 'last_timestamp': final['timestamp'],
                'first_parent_uuid': first.get('parentUuid'), 'snapshot_count': len(snapshots),
                'reported_prompt_tokens': prompt, 'reported_output_tokens': output,
                'reported_usage': {field: usage.get(field) for field in TOKEN_FIELDS},
                'output_usage_values': output_values, 'output_usage_status': 'as_reported_not_certified_complete',
                'last_stop_reason': final['message'].get('stop_reason'),
                'contains_text_block': any(item.get('type') == 'text' for record in snapshots for item in content_items(record)),
                'source_model': final['message'].get('model'), 'tool_calls': list(tools.values()),
                'quality_errors': sorted(problems), 'quality_warnings': sorted(request_warnings),
                'split': split, 'exact_original_prefix_available': False, 'gpu_replay_ready': False})
            fatal.update(problems)
            warnings.update(request_warnings)
        result_ids = {(record['uuid'], item.get('tool_use_id')) for record in episode_nodes for item in content_items(record) if item.get('type') == 'tool_result'}
        if result_ids - matched_results:
            fatal.add('unmatched_tool_result_nodes')
        if not turns:
            excluded.add('no_assistant_requests')
        elif turns[-1]['tool_calls']:
            fatal.add('episode_ends_with_tool_not_terminal_reply')
        elif not turns[-1]['contains_text_block']:
            fatal.add('no_observed_terminal_text_reply')
        if not config['min_turns'] <= len(turns) <= config['max_turns']:
            excluded.add('turn_count_outside_pilot_range')
        end_kind = 'observed_terminal_reply_not_task_success' if turns and not turns[-1]['tool_calls'] else 'unresolved_end'
        status = 'quarantine' if fatal else 'excluded' if excluded else 'accepted'
        for turn in turns:
            turn['episode_status'] = status
        decision = {
            'episode_id': episode_id, 'source_file': source_file, 'source_session': source_session,
            'branch_id': branch, 'split': split, 'status': status, 'start_kind': start_kind,
            'start_node_id': anchor, 'end_kind': end_kind, 'turns': len(turns),
            'source_node_ids': [record['uuid'] for record in episode_nodes],
            'quality_errors': sorted(fatal), 'cohort_exclusions': sorted(excluded), 'quality_warnings': sorted(warnings)}
        decisions.append(decision)
        all_requests.extend(turns)
        if status == 'accepted':
            accepted.append({**{key: decision[key] for key in ('episode_id', 'source_session', 'source_file', 'split', 'start_node_id', 'start_kind', 'end_kind', 'quality_warnings')},
                'source_commit': SOURCE_COMMIT, 'branch_id': branch,
                'phase_model': 'nonstreaming_llm_then_observed_single_tool_delay',
                'output_length_policy': 'as_reported_not_certified_complete',
                'completion_scope': 'observed_user_to_terminal_reply_episode_not_verified_task_success',
                'source_history_length_preserved': True, 'cpu_metadata_ready': True,
                'gpu_replay_ready': False, 'turns': turns})
    return all_requests, decisions, accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--max-model-len', type=int, default=131072)
    arguments = parser.parse_args()
    verification = verify_source(arguments.source, arguments.tree)
    arguments.output.mkdir(parents=True, exist_ok=True)
    config = {'seed': arguments.seed, 'min_turns': 2, 'max_turns': 20,
              'max_tool_delay_us': 60000000, 'max_model_len': arguments.max_model_len,
              'train_fraction': 0.8, 'pilot_target_episodes': 8,
              'single_tool_only': True, 'gpu_enabled': False}
    paths = sorted((arguments.source / 'raw_traces').glob('*/*.jsonl'))
    tree = json.loads(arguments.tree.read_text())
    expected_raw_paths = {entry['path'] for entry in tree['tree'] if entry['type'] == 'blob' and entry['path'].startswith('raw_traces/') and entry['path'].endswith('.jsonl')}
    if {str(path.relative_to(arguments.source)) for path in paths} != expected_raw_paths:
        raise ValueError('Raw trace inventory differs from the verified source tree')
    ordered_sessions = sorted(paths, key=lambda path: digest(str(arguments.seed) + ':' + str(path.relative_to(arguments.source))))
    cutoff = len(paths) * 4 // 5
    splits = {str(path.relative_to(arguments.source)): 'train' if index < cutoff else 'test'
              for index, path in enumerate(ordered_sessions)}
    requests = []
    decisions = []
    accepted = []
    for path in paths:
        session_requests, session_decisions, session_accepted = convert_session(path, arguments.source, splits[str(path.relative_to(arguments.source))], config)
        requests.extend(session_requests)
        decisions.extend(session_decisions)
        accepted.extend(session_accepted)
    test_by_session = defaultdict(list)
    for episode in accepted:
        if episode['split'] == 'test':
            test_by_session[episode['source_session']].append(episode)
    pilot = []
    for source_session in sorted(test_by_session, key=lambda value: digest(str(arguments.seed) + ':pilot-session:' + value))[:config['pilot_target_episodes']]:
        pilot.append(min(test_by_session[source_session], key=lambda value: digest(str(arguments.seed) + ':pilot-episode:' + value['episode_id'])))
    history = defaultdict(list)
    for episode in accepted:
        if episode['split'] != 'train':
            continue
        for turn in episode['turns'][:-1]:
            tool = turn['tool_calls'][0]
            history[tool['tool_name']].append({'delay_us': tool['observed_delay_us'], 'episode_id': episode['episode_id'],
                                             'request_key': turn['request_key'], 'source_session': episode['source_session']})
    write_jsonl(arguments.output / 'requests.jsonl', requests)
    write_jsonl(arguments.output / 'episode_decisions.jsonl', decisions)
    write_jsonl(arguments.output / 'quarantine_episodes.jsonl', [decision for decision in decisions if decision['status'] == 'quarantine'])
    write_jsonl(arguments.output / 'excluded_episodes.jsonl', [decision for decision in decisions if decision['status'] == 'excluded'])
    write_jsonl(arguments.output / 'accepted_episodes.jsonl', accepted)
    write_jsonl(arguments.output / 'pilot_episodes.jsonl', pilot)
    write_json(arguments.output / 'split_manifest.json', {'seed': arguments.seed, 'assignment_before_conversion': True, 'sessions': splits})
    write_json(arguments.output / 'training_history.json', {'origin': 'accepted_train_episodes_only', 'test_future_access': False, 'samples_by_tool': dict(history)})
    write_json(arguments.output / 'config.json', config)
    report = {
        'source_commit': SOURCE_COMMIT, 'source_verification': verification,
        'converter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sessions': len(paths), 'split_source_sessions': dict(Counter(splits.values())),
        'assistant_snapshot_records': sum(request['snapshot_count'] for request in requests),
        'request_groups_using_request_and_message_id': len(requests),
        'episode_status_counts': dict(Counter(decision['status'] for decision in decisions)),
        'quality_errors_by_episode': dict(Counter(reason for decision in decisions for reason in decision['quality_errors'])),
        'cohort_exclusions_by_episode': dict(Counter(reason for decision in decisions for reason in decision['cohort_exclusions'])),
        'accepted_by_split': dict(Counter(episode['split'] for episode in accepted)),
        'accepted_source_sessions_by_split': {split: len({episode['source_session'] for episode in accepted if episode['split'] == split}) for split in ('train', 'test')},
        'accepted_requests': sum(len(episode['turns']) for episode in accepted),
        'accepted_turn_counts': statistics([len(episode['turns']) for episode in accepted]),
        'accepted_prompt_tokens': statistics([turn['reported_prompt_tokens'] for episode in accepted for turn in episode['turns']]),
        'accepted_request_warning_counts': dict(Counter(warning for episode in accepted for turn in episode['turns'] for warning in turn['quality_warnings'])),
        'pilot_episodes': len(pilot), 'pilot_source_sessions': len({episode['source_session'] for episode in pilot}),
        'pilot_requests': sum(len(episode['turns']) for episode in pilot),
        'pilot_available_concurrencies': [count for count in (1, 2, 4, 8) if count <= len(pilot)],
        'pilot_shortfall': max(0, config['pilot_target_episodes'] - len(pilot)),
        'training_tool_samples': {name: len(samples) for name, samples in history.items()},
        'gpu_run': False, 'paid_api_calls': 0, 'performance_results_available': False,
        'cpu_metadata_ready': True, 'gpu_replay_ready': False,
        'limitations': ['reported_output_lengths_not_certified', 'synthetic_prefix_required',
                       'observed_tool_delay_phase_remapping', 'pilot_cohort_selection_bias',
                       'terminal_reply_not_semantic_task_success', 'server_metadata_adapter_not_implemented']}
    write_json(arguments.output / 'conversion_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
