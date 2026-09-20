import json
import time

from core import SafetyStop, TransientGatewayError


def read_completion(response, records, turn, started, duration_limit, *, save_events=False):
    fragments = []
    reasoning_fragments = []
    usage = None
    model = None
    response_id = None
    finish_reason = None
    completed = False
    event_count = 0
    first_event_ms = None
    first_content_ms = None
    raw_events = []
    try:
        for raw in response.iter_lines(chunk_size=1):
            elapsed = time.perf_counter() - started
            if elapsed > duration_limit:
                raise TransientGatewayError('Streaming request exceeded its time limit')
            if not raw.startswith(b'data:'):
                continue
            value = raw[5:].strip()
            if value == b'[DONE]':
                completed = True
                break
            event = json.loads(value)
            if save_events:
                raw_events.append(event)
            event_count += 1
            if first_event_ms is None:
                first_event_ms = elapsed * 1000
            if event.get('error'):
                raise SafetyStop('Gateway returned an error inside the event stream')
            model = event.get('model', model)
            response_id = event.get('id', response_id)
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('index', 0) != 0:
                    raise SafetyStop('Unexpected multiple streaming choices')
                content = choice.get('delta', {}).get('content')
                reasoning = choice.get('delta', {}).get('reasoning_content')
                if reasoning is not None:
                    if not isinstance(reasoning, str):
                        raise SafetyStop('Non-text streaming reasoning')
                    reasoning_fragments.append(reasoning)
                if content is not None:
                    if not isinstance(content, str):
                        raise SafetyStop('Non-text streaming content')
                    fragments.append(content)
                    if first_content_ms is None and content:
                        first_content_ms = elapsed * 1000
                if choice.get('finish_reason'):
                    finish_reason = choice['finish_reason']
        if not completed or usage is None or finish_reason is None:
            raise TransientGatewayError('Incomplete event stream or missing usage')
    except Exception:
        records.event('stream-partial.jsonl', {'turn': turn, 'content': ''.join(fragments),
                      'reasoning_content': ''.join(reasoning_fragments),
                      'reported_model': model, 'usage': usage, 'finish_reason': finish_reason,
                      'event_count': event_count, 'completed': completed})
        raise
    finally:
        if save_events:
            records.event('api-streams.jsonl', {'turn': turn, 'events': raw_events, 'completed': completed})
    return {'id': response_id, 'model': model, 'usage': usage,
            'choices': [{'message': {'content': ''.join(fragments),
                                    'reasoning_content': ''.join(reasoning_fragments)},
                         'finish_reason': finish_reason}],
            'transport': {'kind': 'sse', 'event_count': event_count,
                          'first_event_ms': first_event_ms, 'first_content_ms': first_content_ms}}
