import hashlib
import json
import re
import time

from core import SafetyStop, TransientGatewayError


PROTOCOL = 'deepseek-bash-v2'
SYSTEM_RULES = '''
Execution protocol for this experiment:
Return a short THOUGHT followed by exactly one complete ```bash code block.
After the closing code fence, stop generating immediately.
The evaluator executes that single action and supplies the real observation in the next user message.
Never generate observations, terminal output, role tags, XML tool calls, or imagined future conversation turns.
Do not write any text after the closing code fence. Do not simulate the results of commands.
If a reply is rejected, its commands were not executed. Continue from the last actual observation.
'''


def response_problem(content, finish_reason):
    if finish_reason != 'stop':
        return 'The response ended with ' + str(finish_reason) + ' instead of completing one action.'
    if not isinstance(content, str) or not content.strip():
        return 'The response contained no executable answer.'
    blocks = list(re.finditer(r'```bash\s*\n(.*?)\n```', content, re.S))
    if len(blocks) != 1:
        return 'Expected exactly one bash block, received ' + str(len(blocks)) + '.'
    block = blocks[0]
    if not block.group(1).strip():
        return 'The bash action was empty.'
    if content[block.end():].strip():
        return 'Text after the bash block is not allowed; observations must come from the evaluator.'
    prefix = content[:block.start()]
    if re.search(r'<\/?(?:system|user|assistant|tool)\b|<｜｜DSML｜｜|(?:^|\n)\s*(?:Observation|Output)\s*:', prefix, re.I):
        return 'The response attempted to simulate a role or tool observation.'
    return None


def query_with_recovery(model, messages):
    wire = [{'role': message['role'], 'content': message['content']} for message in messages]
    identity = hashlib.sha256(json.dumps(wire, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    limit = model.settings.get('automatic_api_retries', 0)
    if type(limit) is not int or limit not in range(3):
        raise SafetyStop('Retry limit must be between zero and two')
    history_path = model.records.root / 'retry-events.jsonl'
    history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    previous = [event for event in history if event.get('query_id') == identity]
    if any(not event.get('will_retry') for event in previous) or len(previous) > limit:
        raise SafetyStop('This request already exhausted its recorded recovery attempts')
    started = time.perf_counter()
    initial_remaining = model.remaining_seconds
    for retry_index in range(len(previous), limit + 1):
        model.remaining_seconds = initial_remaining - (time.perf_counter() - started) if initial_remaining is not None else None
        if ((model.remaining_seconds is not None and model.remaining_seconds <= 10)
                or (model.settings['max_steps'] > 0 and model.n_calls >= model.settings['max_steps'])):
            raise SafetyStop('Task time or step budget exhausted before recovery request')
        try:
            return model._query_once(messages)
        except TransientGatewayError as error:
            reserved = model.budget.charge_unsettled_reservation('Uncertain API outcome counted before bounded recovery')
            model.cost += reserved
            model.prior_uncertain_cost += reserved
            model.records.event('retry-events.jsonl', {
                'query_id': identity, 'turn': model.n_calls, 'request_attempt': model.request_attempts,
                'retry_index': retry_index, 'retry_limit': limit,
                'reason': str(error), 'reserved_as_spent_cny': reserved,
                'completed_response_cost_already_settled': reserved == 0,
                'will_retry': retry_index < limit, 'sampling_parameters_changed': False,
            })
            if retry_index == limit:
                raise SafetyStop(str(error) + '; bounded recovery attempts exhausted') from None
            time.sleep(2 ** retry_index)
