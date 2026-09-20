import argparse
import hashlib
import json
import sys
from array import array
from pathlib import Path

from convert import write_json, write_jsonl


SCENARIOS = {'reuse_0': (0, 1), 'reuse_50': (1, 2), 'append_only': (1, 1)}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def token_bytes(tokens):
    values = array('I', tokens)
    if values.itemsize != 4:
        raise ValueError('A 32-bit unsigned integer representation is required')
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def token_hash(tokens):
    return hashlib.sha256(token_bytes(tokens)).hexdigest()


def deterministic_tokens(key, count, domain):
    raw = array('I')
    raw.frombytes(hashlib.shake_256(key.encode()).digest(count * 4))
    if raw.itemsize != 4:
        raise ValueError('A 32-bit unsigned integer representation is required')
    if sys.byteorder != 'little':
        raw.byteswap()
    low = domain['token_id_low_inclusive']
    width = domain['token_id_high_exclusive'] - low
    return [low + value % width for value in raw]


def longest_common_prefix(left, right):
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def materialize_episode(episode, scenario, seed, domain):
    numerator, denominator = SCENARIOS[scenario]
    previous_history = []
    for turn in episode['turns']:
        prompt_count = turn['reported_prompt_tokens']
        output_count = turn['reported_output_tokens']
        if prompt_count < 16 or output_count < 1:
            raise ValueError('Pilot token generation requires prompt >=16 and output >=1')
        if prompt_count + output_count > domain['max_position_embeddings']:
            raise ValueError('Token budget exceeds the verified model context limit')
        reset_reason = None
        if not previous_history:
            retained = 0
            reset_reason = 'episode_start'
        elif prompt_count < len(previous_history):
            retained = 0
            reset_reason = 'insufficient_room_for_previous_input_and_output'
        else:
            retained = len(previous_history) * numerator // denominator
            if retained == 0:
                reset_reason = 'zero_reuse_scenario'
        token_key = str(seed) + ':' + episode['episode_id'] + ':' + str(turn['turn_index'])
        suffix = deterministic_tokens(token_key + ':prompt:' + scenario, prompt_count - retained, domain)
        if retained < len(previous_history) and suffix and suffix[0] == previous_history[retained]:
            low = domain['token_id_low_inclusive']
            width = domain['token_id_high_exclusive'] - low
            suffix[0] = low + (suffix[0] - low + 1) % width
        prompt = previous_history[:retained] + suffix
        output = deterministic_tokens(token_key + ':output', output_count, domain)
        actual_lcp = longest_common_prefix(previous_history, prompt)
        if actual_lcp != retained:
            raise ValueError('Prefix relation differs from its declared scenario')
        recipe = {
            'episode_id': episode['episode_id'], 'request_key': turn['request_key'],
            'turn_index': turn['turn_index'], 'scenario': scenario, 'seed': seed,
            'generator': 'shake256-modulo-u32le-v1',
            'prompt_tokens': prompt_count, 'output_tokens': output_count,
            'retained_prefix_tokens': retained, 'actual_lcp_tokens': actual_lcp,
            'prefix_reset_reason': reset_reason,
            'prompt_token_sha256': token_hash(prompt), 'output_token_sha256': token_hash(output),
            'namespace_first_block': prompt[:16],
            'tokenizer_sha256': domain['tokenizer_sha256'],
            'lengths_are_as_reported': True, 'original_prefix_recovered': False,
            'gpu_replay_ready': False}
        yield recipe, prompt, output
        previous_history = prompt + output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--token-domain', type=Path, required=True)
    arguments = parser.parse_args()
    domain = json.loads(arguments.token_domain.read_text())
    config = json.loads((arguments.run / 'config.json').read_text())
    if domain.get('all_ids_present_and_non_special') is not True:
        raise ValueError('The token domain must be verified against the actual tokenizer')
    pilot = read_jsonl(arguments.run / 'pilot_episodes.jsonl')
    recipes = []
    for scenario in SCENARIOS:
        namespace_owners = {}
        for episode in pilot:
            for recipe, prompt, output in materialize_episode(episode, scenario, config['seed'], domain):
                block = tuple(prompt[:16])
                owner = namespace_owners.setdefault(block, episode['episode_id'])
                if owner != episode['episode_id']:
                    raise ValueError('Unexpected cross-episode cache-block sharing')
                recipes.append(recipe)
    scheduler_metadata = []
    controller_transitions = []
    for episode in pilot:
        for turn in episode['turns']:
            tool = turn['tool_calls'][0] if turn['tool_calls'] else None
            scheduler_metadata.append({
                'delivery_event': 'current_response_finished', 'episode_id': episode['episode_id'],
                'turn_index': turn['turn_index'], 'current_tool_name': tool['tool_name'] if tool else None,
                'terminal_reply_observed': tool is None})
            if tool:
                controller_transitions.append({
                    'episode_id': episode['episode_id'], 'from_turn': turn['turn_index'],
                    'to_turn': turn['turn_index'] + 1, 'delay_after_response_us': tool['observed_delay_us'],
                    'visibility': 'replay_controller_only_not_scheduler',
                    'phase_mapping': 'response_complete_then_recorded_observed_delay',
                    'source_tool_use_id': tool['tool_use_id']})
    write_json(arguments.run / 'token_domain.json', domain)
    write_jsonl(arguments.run / 'token_recipes.jsonl', recipes)
    write_jsonl(arguments.run / 'scheduler_metadata.jsonl', scheduler_metadata)
    write_jsonl(arguments.run / 'controller_transitions.jsonl', controller_transitions)
    report = {
        'scenarios': list(SCENARIOS), 'pilot_episodes': len(pilot), 'recipes': len(recipes),
        'tokens_materialized_and_checked': sum(recipe['prompt_tokens'] + recipe['output_tokens'] for recipe in recipes),
        'seed': config['seed'], 'generator': 'shake256-modulo-u32le-v1',
        'source_file_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'token_domain_verified': True, 'prefix_lcp_checked': True,
        'cross_episode_cache_block_isolation_checked': True,
        'cpu_only': True, 'model_weights_loaded': False, 'gpu_replay_ready': False,
        'fixed_output_token_engine_adapter_implemented': False,
        'note': 'Recipes reproduce token IDs; they are not measurements or a running vLLM workload.'}
    write_json(arguments.run / 'synthesis_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
