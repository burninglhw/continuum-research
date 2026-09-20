import argparse
import hashlib
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def describe(values):
    ordered = sorted(values)
    return {
        "count": len(values), "min": min(values), "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], "max": max(values),
    } if values else {"count": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=8192)
    arguments = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    headers = {role: encode(f"<|im_start|>{role}\n") for role in ("system", "user", "assistant")}
    trailer = encode("<|im_end|>\n")
    traces = []
    for path in sorted((arguments.data / "normalized").glob("*.json")):
        trajectory = json.loads(path.read_text())
        tokens = []
        offsets = []
        for message in trajectory["messages"]:
            offsets.append(len(tokens))
            tokens.extend(headers[message["role"]])
            tokens.extend(encode(message["content"]))
            tokens.extend(trailer)
        offsets.append(len(tokens))
        turns = []
        previous_end = 0
        for turn in trajectory["turns"]:
            index = turn["assistant_message_index"]
            prompt = offsets[index] + len(headers["assistant"])
            end = offsets[index + 1]
            turns.append({
                **turn, "prompt_tokens": prompt, "end_tokens": end,
                "output_tokens": end - prompt, "reusable_prefix_tokens": previous_end,
                "new_input_tokens": prompt - previous_end,
            })
            previous_end = end
        pilot_turns = []
        stop_reason = "maximum_six_turns"
        for turn in turns[:arguments.max_turns]:
            if turn["end_tokens"] > arguments.max_tokens:
                stop_reason = "next_complete_turn_exceeds_context_budget"
                break
            pilot_turns.append(turn)
        if len(turns) < arguments.max_turns:
            stop_reason = "source_ended"
        metadata = {key: trajectory[key] for key in ("instance_id", "repo", "split", "exit_status", "sha256")}
        trace = {
            **metadata, "turns": turns, "pilot_turns": pilot_turns,
            "pilot_stop_reason": stop_reason, "total_serialized_tokens": len(tokens),
            "pilot_eligible": len(pilot_turns) >= 3,
        }
        save_json(arguments.output / "token_ids" / (trajectory["instance_id"] + ".json"), {"instance_id": trajectory["instance_id"], "token_ids": tokens})
        traces.append(trace)
    all_turns = [turn for trace in traces for turn in trace["turns"]]
    pilot_turns = [turn for trace in traces if trace["pilot_eligible"] for turn in trace["pilot_turns"]]
    report = {
        "model_path": str(arguments.model), "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_json_sha256": hashlib.sha256((arguments.model / "tokenizer.json").read_bytes()).hexdigest(),
        "serialization": "Fixed ChatML: separately tokenize header, visible content, and im_end-newline; concatenate. No hidden thinking; no history rewriting; not a generation-quality evaluation or original Devstral tokenization.",
        "all_prompt_tokens": describe([turn["prompt_tokens"] for turn in all_turns]),
        "all_completion_tokens": describe([turn["output_tokens"] for turn in all_turns]),
        "action_observation_gaps_s": describe([turn["action_feedback_gap_s"] for turn in all_turns if turn["feedback_kind"] == "action_observation"]),
        "format_feedback_gaps_s": describe([turn["action_feedback_gap_s"] for turn in all_turns if turn["feedback_kind"] == "format_error_feedback"]),
        "pilot_rule": {"max_turns": arguments.max_turns, "max_end_tokens": arguments.max_tokens, "minimum_turns": 3, "truncation": "stop before oversized turn; never trim prompt or output"},
        "pilot_sessions": sum(trace["pilot_eligible"] for trace in traces),
        "pilot_turns": len(pilot_turns),
        "pilot_prompt_tokens": describe([turn["prompt_tokens"] for turn in pilot_turns]),
        "pilot_completion_tokens": describe([turn["output_tokens"] for turn in pilot_turns]),
        "traces": traces,
    }
    save_json(arguments.output / "workload.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "traces"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
