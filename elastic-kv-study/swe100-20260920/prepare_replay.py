import argparse
import gzip
import hashlib
import json
import pickle
import statistics
from pathlib import Path

from transformers import AutoTokenizer


EXPECTED_SHA = "420e77f513f6676cb1db8ae14c6773a3752ce73416a75e65886024547b369d19"


def common_prefix(left, right):
    for position, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return position
    return min(len(left), len(right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.file_digest(args.traces.open("rb"), "sha256").hexdigest()
    assert digest == EXPECTED_SHA, digest
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    programs = []
    no_response = []
    counts = {"accepted": 0, "rejected_or_recovery": 0, "prompt_count_mismatches": 0}
    for line in args.traces.open():
        trace = json.loads(line)
        steps = []
        previous_completed = []
        for event in trace["events"]:
            response = event["response"]
            if response is None:
                no_response.append({"instance_id": trace["instance_id"], "attempt": event["request_attempt"]})
                continue
            messages = event["request"]["messages"]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            completed = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": response["content"]}],
                tokenize=True, add_generation_prompt=False,
            )
            assert completed[:len(prompt)] == prompt, (trace["instance_id"], response["turn"], "template boundary")
            output = completed[len(prompt):]
            assert output and len(completed) <= 131072, (trace["instance_id"], response["turn"], len(completed))
            if len(prompt) != event["llama_prompt_tokens"]:
                counts["prompt_count_mismatches"] += 1
            tool = event["tool"]
            counts["accepted" if event["accepted_into_context"] else "rejected_or_recovery"] += 1
            steps.append({
                "turn": response["turn"], "attempt": event["request_attempt"],
                "prompt": prompt, "output": output,
                "tool": tool["command"].strip().split()[0] if tool else None,
                "gap": event["tool_duration_ms"] / 1000 if tool else 0.0,
                "accepted": event["accepted_into_context"],
                "previous_lcp": common_prefix(previous_completed, prompt),
                "previous_computed_lcp": common_prefix(previous_completed[:-1], prompt),
                "prompt_sha256": hashlib.sha256(pickle.dumps(prompt, protocol=4)).hexdigest(),
            })
            previous_completed = completed
        assert steps
        programs.append({"instance_id": trace["instance_id"], "steps": steps})
        print(json.dumps({"prepared": len(programs), "instance_id": trace["instance_id"], "steps": len(steps)}), flush=True)
    assert len(programs) == 100
    with gzip.open(args.output / "replay.pkl.gz", "wb", compresslevel=3) as stream:
        pickle.dump(programs, stream, protocol=5)
    flat = [step for program in programs for step in program["steps"]]
    manifest = {
        "trace_sha256": digest, "programs": len(programs), "responses": len(flat),
        "no_response_http_attempts_excluded": no_response, "counts": counts,
        "prompt_tokens": sum(len(step["prompt"]) for step in flat),
        "forced_decode_tokens": sum(len(step["output"]) for step in flat),
        "max_total_length": max(len(step["prompt"]) + len(step["output"]) for step in flat),
        "mean_prompt_tokens": statistics.mean(len(step["prompt"]) for step in flat),
        "tool_seconds": sum(step["gap"] for step in flat),
        "exact_template_prefix_checked": True, "reasoning_included": False,
        "tokenizer_file_hashes": {path.name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                                  for path in args.tokenizer.glob("*.json")},
        "replay_sha256": hashlib.file_digest((args.output / "replay.pkl.gz").open("rb"), "sha256").hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    compact = [{"instance_id": program["instance_id"], "steps": [
        {**{key: value for key, value in step.items() if key not in ("prompt", "output")},
         "input_tokens": len(step["prompt"]), "output_tokens": len(step["output"])}
        for step in program["steps"]]} for program in programs]
    (args.output / "lengths.json").write_text(json.dumps(compact))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
