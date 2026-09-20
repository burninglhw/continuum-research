import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16",
                max_model_len=131072, max_num_batched_tokens=2048,
                max_num_seqs=64, enable_prefix_caching=True,
                kv_cache_memory_bytes=24 * 1024**3, gpu_memory_utilization=0.5,
                enforce_eager=True, disable_log_stats=True,
                logits_processors=["forced_tokens:ForcedTokens"])
    records = []
    prompt = [128000] + [1000 + index % 1000 for index in range(510)]
    target = [3000 + index for index in range(33)]
    params = SamplingParams(temperature=0, max_tokens=len(target), ignore_eos=True,
                            detokenize=False, extra_args={"job_id": "smoke", "is_last_step": True,
                                                         "forced_token_ids": target})
    for turn in range(2):
        started = time.perf_counter()
        result = model.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0]
        actual = list(result.outputs[0].token_ids)
        assert actual == target, (actual, target)
        record = {"turn": turn, "seconds": time.perf_counter() - started,
                  "input_tokens": len(prompt), "forced_tokens_correct": True,
                  "cached_tokens": result.num_cached_tokens}
        records.append(record)
        print(json.dumps(record), flush=True)
        prompt = prompt + target + [8000] * 17
    assert records[1]["cached_tokens"] >= 528, records
    (args.output / "smoke.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
