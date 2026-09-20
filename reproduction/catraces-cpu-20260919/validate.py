import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from audit import content_items
from convert import boundary_kind, convert_session, elapsed_us, graph_index, write_json
from synthesize import SCENARIOS, materialize_episode, read_jsonl, token_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_timestamp(milliseconds):
    return (datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=milliseconds)).isoformat()


def edge_checks():
    config = {'min_turns': 2, 'max_turns': 20, 'max_tool_delay_us': 60000000, 'max_model_len': 131072}
    user = {'uuid': 'user', 'parentUuid': 'old-unavailable-history', 'type': 'user',
            'timestamp': source_timestamp(0), 'message': {'content': '<redacted>'}}

    def assistant(node_id, parent_id, request_id, content, milliseconds):
        return {'uuid': node_id, 'parentUuid': parent_id, 'type': 'assistant', 'requestId': request_id,
                'timestamp': source_timestamp(milliseconds),
                'message': {'id': request_id + '-message', 'model': 'fixture-model', 'content': content,
                            'stop_reason': None, 'usage': {'input_tokens': 16384, 'cache_creation_input_tokens': 0,
                                                          'cache_read_input_tokens': 0, 'output_tokens': 4}}}

    def result(node_id, parent_id, tool_id, milliseconds):
        return {'uuid': node_id, 'parentUuid': parent_id, 'type': 'user', 'timestamp': source_timestamp(milliseconds),
                'message': {'content': [{'type': 'tool_result', 'tool_use_id': tool_id}]}}

    base = [user, assistant('first', 'user', 'request-1', [{'type': 'tool_use', 'id': 'tool-1', 'name': 'Read'}], 1),
            result('result', 'first', 'tool-1', 10),
            assistant('final', 'result', 'request-2', [{'type': 'text', 'text': '<redacted>'}], 20)]
    checks = []
    with tempfile.TemporaryDirectory(prefix='catraces-edge-') as folder:
        root = Path(folder)
        path = root / 'project' / 'session.jsonl'
        path.parent.mkdir()

        def run(records):
            path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')
            return convert_session(path, root, 'test', config)

        requests, decisions, accepted = run(base)
        require(len(accepted) == 1 and requests[0]['tool_calls'][0]['observed_delay_us'] == 9000, 'Simple causal chain rejected')
        checks.append('simple_chain_and_preserved_historical_prompt')
        extra_result = copy.deepcopy(base)
        extra_result[2]['message']['content'].append({'type': 'tool_result', 'tool_use_id': 'unmatched'})
        require(not run(extra_result)[2], 'Unmatched result block silently ignored')
        checks.append('unknown_result_in_matched_node_rejected')
        collision = copy.deepcopy(base)
        collision[-1]['requestId'] = 'request-1'
        require(not run(collision)[2], 'Request/message collision accepted')
        checks.append('request_message_identity_collision_rejected')
        parallel = copy.deepcopy(base)
        parallel.insert(2, assistant('second-tool', 'first', 'request-1', [{'type': 'tool_use', 'id': 'tool-2', 'name': 'Read'}], 2))
        parallel[3]['parentUuid'] = 'second-tool'
        parallel.insert(4, result('result-2', 'result', 'tool-2', 11))
        parallel[-1]['parentUuid'] = 'result-2'
        requests, decisions, accepted = run(parallel)
        require(len(requests[0]['tool_calls']) == 2 and not accepted, 'Multiple tool blocks lost or accepted into single-tool cohort')
        checks.append('tool_blocks_merged_without_last_event_loss')
        fallback = copy.deepcopy(base)
        for record in fallback:
            record.pop('requestId', None)
        require(len(run(fallback)[2]) == 1, 'Message-ID fallback failed')
        checks.append('message_id_fallback')
        compact = copy.deepcopy(base)
        compact[0] = {'uuid': 'user', 'type': 'system', 'subtype': 'compact_boundary', 'parentUuid': None, 'timestamp': source_timestamp(0)}
        compact.insert(1, {'uuid': 'summary', 'type': 'user', 'parentUuid': 'user', 'timestamp': source_timestamp(0),
                           'isCompactSummary': True, 'isVisibleInTranscriptOnly': True, 'message': {'content': '<redacted>'}})
        compact[2]['parentUuid'] = 'summary'
        requests, decisions, accepted = run(compact)
        require(not accepted and all(decision['start_kind'] == 'compact_boundary' for decision in decisions), 'Compaction summary mislabeled as human input')
        checks.append('compaction_summary_not_human_boundary')
        fork = copy.deepcopy(base)
        fork.append(assistant('branch', 'first', 'request-3', [{'type': 'text'}], 12))
        require(not run(fork)[2], 'Fork flattened into a sequential episode')
        checks.append('fork_quarantined')
        cycle = [assistant('cycle-a', 'cycle-b', 'request-1', [{'type': 'text'}], 1),
                 assistant('cycle-b', 'cycle-a', 'request-2', [{'type': 'text'}], 2)]
        require(all(value[1] == 'parent_cycle' for value in graph_index(cycle)[1].values()), 'Cycle not detected')
        checks.append('parent_cycle_detected')
        long_tool = copy.deepcopy(base)
        long_tool[2]['timestamp'] = source_timestamp(61001)
        long_tool[3]['timestamp'] = source_timestamp(62000)
        requests, decisions, accepted = run(long_tool)
        require(not accepted and requests[0]['tool_calls'][0]['observed_delay_us'] == 61000000, 'Long tool delay clipped or accepted')
        checks.append('long_tool_excluded_without_clipping')
        too_long = copy.deepcopy(base)
        too_long[1]['message']['usage']['input_tokens'] = 131072
        require(not run(too_long)[2], 'Context overflow accepted')
        checks.append('context_overflow_rejected')
        invalid_usage = copy.deepcopy(base)
        invalid_usage[1]['message']['usage']['output_tokens'] = None
        require(not run(invalid_usage)[2], 'Invalid usage accepted')
        checks.append('invalid_usage_quarantined')
        sides = copy.deepcopy(base)
        for record in sides:
            record['isSidechain'] = True
        require(not run(sides)[2], 'Sidechain accepted into main cohort')
        checks.append('sidechain_excluded')
    fixture_domain = {'token_id_low_inclusive': 256, 'token_id_high_exclusive': 32000,
                      'max_position_embeddings': 131072, 'tokenizer_sha256': 'fixture-not-a-real-tokenizer'}
    fixture_episode = {'episode_id': 'fixture/episode', 'turns': [
        {'request_key': 'fixture/request-' + str(index), 'turn_index': index,
         'reported_prompt_tokens': prompt, 'reported_output_tokens': output}
        for index, (prompt, output) in enumerate(((100, 8), (112, 8), (40, 4), (80, 4)))]}
    append = list(materialize_episode(fixture_episode, 'append_only', 7, fixture_domain))
    require(append[1][1][:108] == append[0][1] + append[0][2], 'Previous output not included in append-only prefix')
    require(append[2][0]['retained_prefix_tokens'] == 0 and append[2][0]['prefix_reset_reason'] == 'insufficient_room_for_previous_input_and_output', 'Context shrink failed to reset')
    require(append[3][0]['retained_prefix_tokens'] == 44, 'Post-reset history not resumed correctly')
    checks.append('append_prefix_includes_decode_and_resets_on_shrink')
    partial = list(materialize_episode(fixture_episode, 'reuse_50', 7, fixture_domain))
    zero = list(materialize_episode(fixture_episode, 'reuse_0', 7, fixture_domain))
    require(partial[1][0]['retained_prefix_tokens'] == 54 and all(entry[0]['retained_prefix_tokens'] == 0 for entry in zero), 'Prefix sensitivity scenarios incorrect')
    checks.append('partial_and_zero_reuse_prefix_scenarios')
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repeat-dir', type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads((arguments.run / 'config.json').read_text())
    domain = json.loads((arguments.run / 'token_domain.json').read_text())
    splits = json.loads((arguments.run / 'split_manifest.json').read_text())['sessions']
    requests = read_jsonl(arguments.run / 'requests.jsonl')
    decisions = read_jsonl(arguments.run / 'episode_decisions.jsonl')
    accepted = read_jsonl(arguments.run / 'accepted_episodes.jsonl')
    pilot = read_jsonl(arguments.run / 'pilot_episodes.jsonl')
    raw = {}
    assistant_lines = set()
    raw_tools = set()
    raw_tool_blocks = 0
    raw_nodes = set()
    for path in sorted((arguments.source / 'raw_traces').glob('*/*.jsonl')):
        relative = str(path.relative_to(arguments.source))
        records = {}
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record['_line'] = number
            records[record['uuid']] = record
            raw_nodes.add((relative, record['uuid']))
            if record.get('type') == 'assistant':
                assistant_lines.add((relative, number))
                raw_tools.update((relative, item['id']) for item in content_items(record) if item.get('type') == 'tool_use')
                raw_tool_blocks += sum(item.get('type') == 'tool_use' for item in content_items(record))
        raw[relative] = records
    mapped_lines = [(request['source_file'], number) for request in requests for number in request['source_lines']]
    require(len(mapped_lines) == len(set(mapped_lines)) and set(mapped_lines) == assistant_lines, 'Assistant source coverage is not exact')
    mapped_tools = {(request['source_file'], tool['tool_use_id']) for request in requests for tool in request['tool_calls']}
    require(mapped_tools == raw_tools, 'Tool identities lost during normalization')
    mapped_nodes = [(decision['source_file'], identity) for decision in decisions for identity in decision['source_node_ids']]
    require(len(mapped_nodes) == len(set(mapped_nodes)) and set(mapped_nodes) == raw_nodes, 'Graph partition drops or duplicates source nodes')
    for request in requests:
        records = raw[request['source_file']]
        actual_tools = {item['id'] for node in request['node_ids'] for item in content_items(records[node]) if item.get('type') == 'tool_use'}
        require(actual_tools == {tool['tool_use_id'] for tool in request['tool_calls']}, 'Request-specific tool union differs from source')
        require(request['split'] == splits[request['source_file']], 'Source session appears in multiple splits')
    decision_index = {decision['episode_id']: decision for decision in decisions}
    expected_history = {}
    accepted_index = {episode['episode_id']: episode for episode in accepted}
    for episode in accepted:
        decision = decision_index[episode['episode_id']]
        require(not decision['quality_errors'] and not decision['cohort_exclusions'], 'Accepted episode has exclusions')
        require(episode['split'] == splits[episode['source_file']], 'Split changed after filtering')
        require(config['min_turns'] <= len(episode['turns']) <= config['max_turns'], 'Turn-count condition violated')
        records = raw[episode['source_file']]
        chain = []
        current = episode['turns'][-1]['node_ids'][-1]
        seen = set()
        while True:
            require(current in records and current not in seen, 'Broken or cyclic accepted ancestry')
            seen.add(current)
            node = records[current]
            require(not node.get('isSidechain'), 'Accepted episode traverses sidechain')
            chain.append(current)
            if current == episode['start_node_id']:
                break
            current = node.get('parentUuid')
        chain.reverse()
        require(set(chain) == set(decision['source_node_ids']), 'Accepted episode is not a single complete parent path')
        require(boundary_kind(records[chain[0]]) == 'observed_user_input', 'Invalid human boundary')
        positions = {node: index for index, node in enumerate(chain)}
        for before, after in zip(chain, chain[1:]):
            require(elapsed_us(records[before]['timestamp'], records[after]['timestamp']) >= 0, 'Accepted chain has timestamp regression')
        for turn_index, turn in enumerate(episode['turns']):
            last_raw = records[turn['node_ids'][-1]]
            usage = last_raw['message']['usage']
            require(turn['reported_prompt_tokens'] == usage['input_tokens'] + usage['cache_creation_input_tokens'] + usage['cache_read_input_tokens'], 'Prompt lengths changed')
            require(turn['reported_output_tokens'] == usage['output_tokens'], 'Output lengths changed')
            require(turn['reported_prompt_tokens'] + turn['reported_output_tokens'] <= config['max_model_len'], 'Context overflow')
            require(turn['turn_index'] == turn_index and not turn['quality_errors'], 'Invalid accepted turn')
            if turn_index == len(episode['turns']) - 1:
                require(not turn['tool_calls'] and turn['contains_text_block'], 'Missing observed terminal reply')
                continue
            require(len(turn['tool_calls']) == 1, 'Non-single-tool transition accepted')
            tool = turn['tool_calls'][0]
            emit = records[tool['emit_node_id']]
            result = records[tool['result_node_id']]
            next_turn = episode['turns'][turn_index + 1]
            require(positions[emit['uuid']] <= positions[turn['node_ids'][-1]] < positions[result['uuid']] < positions[next_turn['node_ids'][0]], 'Tool causality violated')
            require(any(item.get('type') == 'tool_result' and item.get('tool_use_id') == tool['tool_use_id'] for item in content_items(result)), 'Mismatched tool result')
            require(tool['observed_delay_us'] == elapsed_us(emit['timestamp'], result['timestamp']), 'Tool delay modified')
            require(0 <= tool['observed_delay_us'] <= config['max_tool_delay_us'], 'Tool-delay condition violated')
            if episode['split'] == 'train':
                expected_history[(episode['episode_id'], turn['request_key'], tool['tool_name'])] = tool['observed_delay_us']
    history = json.loads((arguments.run / 'training_history.json').read_text())['samples_by_tool']
    actual_history = {(sample['episode_id'], sample['request_key'], name): sample['delay_us'] for name, samples in history.items() for sample in samples}
    require(actual_history == expected_history, 'Predictor history leaks test data or omits training observations')
    require(sum(map(len, history.values())) == len(actual_history), 'Duplicate predictor training samples')
    require(len({episode['source_session'] for episode in pilot}) == len(pilot), 'Pilot contains dependent episodes from the same source session')
    require(all(episode['split'] == 'test' and episode == accepted_index[episode['episode_id']] for episode in pilot), 'Pilot not drawn strictly from accepted test cohort')
    metadata = read_jsonl(arguments.run / 'scheduler_metadata.jsonl')
    controller = read_jsonl(arguments.run / 'controller_transitions.jsonl')
    allowed_keys = {'delivery_event', 'episode_id', 'turn_index', 'current_tool_name', 'terminal_reply_observed'}
    require(all(set(row) == allowed_keys and row['delivery_event'] == 'current_response_finished' for row in metadata), 'Future-duration field exposed in scheduler schema')
    current_turns = {(episode['episode_id'], turn['turn_index']): turn for episode in pilot for turn in episode['turns']}
    require(len(metadata) == len(current_turns), 'Scheduler metadata coverage mismatch')
    for row in metadata:
        turn = current_turns[(row['episode_id'], row['turn_index'])]
        tool_name = turn['tool_calls'][0]['tool_name'] if turn['tool_calls'] else None
        require(row['current_tool_name'] == tool_name and row['terminal_reply_observed'] == (tool_name is None), 'Scheduler event describes a different turn')
    require(len(controller) == sum(len(episode['turns']) - 1 for episode in pilot), 'Controller transition count mismatch')
    for row in controller:
        turn = current_turns[(row['episode_id'], row['from_turn'])]
        require(row['to_turn'] == row['from_turn'] + 1 and row['delay_after_response_us'] == turn['tool_calls'][0]['observed_delay_us'], 'Incorrect replay-controller transition')
    recipes = read_jsonl(arguments.run / 'token_recipes.jsonl')
    recipe_index = {(recipe['scenario'], recipe['episode_id'], recipe['turn_index']): recipe for recipe in recipes}
    require(len(recipe_index) == len(recipes) == len(current_turns) * len(SCENARIOS), 'Token recipe coverage mismatch')
    expected_outputs = {}
    token_count = 0
    for scenario in SCENARIOS:
        namespace_owner = {}
        for episode in pilot:
            previous = []
            for regenerated, prompt, output in materialize_episode(episode, scenario, config['seed'], domain):
                recipe = recipe_index[(scenario, episode['episode_id'], regenerated['turn_index'])]
                require(recipe == regenerated, 'Token regeneration is not deterministic')
                require(len(prompt) == recipe['prompt_tokens'] and len(output) == recipe['output_tokens'], 'Materialized token lengths differ')
                token_count += len(prompt) + len(output)
                require(min(prompt + output) >= domain['token_id_low_inclusive'] and max(prompt + output) < domain['token_id_high_exclusive'], 'Token ID out of verified domain')
                overlap = 0
                while overlap < min(len(previous), len(prompt)) and previous[overlap] == prompt[overlap]:
                    overlap += 1
                require(overlap == recipe['retained_prefix_tokens'], 'Independent prefix check failed')
                require(prompt[:overlap] == previous[:overlap], 'Retained history differs')
                require(token_hash(prompt) == recipe['prompt_token_sha256'], 'Prompt hash differs')
                output_key = (episode['episode_id'], recipe['turn_index'])
                previous_output_hash = expected_outputs.setdefault(output_key, token_hash(output))
                require(previous_output_hash == token_hash(output), 'Output stream differs across prefix scenarios')
                block = tuple(prompt[:16])
                require(namespace_owner.setdefault(block, episode['episode_id']) == episode['episode_id'], 'Cross-program cache block collision')
                previous = prompt + output
    fixture_results = edge_checks()
    scripts = Path(__file__).parent
    subprocess.run([sys.executable, str(scripts / 'convert.py'), '--source', str(arguments.source), '--tree', str(arguments.tree),
                    '--output', str(arguments.repeat_dir), '--seed', str(config['seed']), '--max-model-len', str(config['max_model_len'])], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(scripts / 'synthesize.py'), '--run', str(arguments.repeat_dir), '--token-domain', str(arguments.run / 'token_domain.json')], check=True, stdout=subprocess.DEVNULL)
    compared = []
    for artifact in sorted(arguments.repeat_dir.glob('*.json*')):
        original = arguments.run / artifact.name
        require(original.read_bytes() == artifact.read_bytes(), 'Repeated run differs: ' + artifact.name)
        compared.append(artifact.name)
    report = {
        'status': 'PASS', 'source_assistant_records_accounted_once': len(assistant_lines),
        'source_graph_nodes_accounted_once': len(raw_nodes),
        'raw_tool_use_block_occurrences': raw_tool_blocks,
        'distinct_file_tool_id_pairs_preserved': len(raw_tools),
        'accepted_episodes_checked': len(accepted), 'pilot_episodes_checked': len(pilot),
        'token_ids_regenerated_and_checked': token_count, 'edge_case_checks': fixture_results,
        'accepted_parent_chains_verified': True, 'tool_result_causality_verified': True,
        'source_lengths_and_delays_preserved': True, 'train_test_session_isolation_verified': True,
        'training_history_no_test_observations': True, 'scheduler_schema_has_no_future_duration': True,
        'server_runtime_leakage_audited': False, 'server_adapter_exists': False,
        'deterministic_repeated_artifacts': compared,
        'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gpu_used': False, 'paid_api_called': False, 'performance_results_available': False}
    write_json(arguments.run / 'validation_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
