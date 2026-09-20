import argparse
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extend-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv"], text=True)
    (args.output / "gpu-before.txt").write_text(gpu)
    model = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16",
                max_model_len=131072, max_num_batched_tokens=2048,
                max_num_seqs=64, enable_prefix_caching=True,
                kv_cache_memory_bytes=24 * 1024**3, gpu_memory_utilization=0.5,
                enforce_eager=True, disable_log_stats=True,
                logits_processors=["forced_tokens:ForcedTokens"])
    lengths = [1024, 4096, 16384, 32768, 65536, 98304, 126976, 131056]
    fractions = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    rng = random.Random(20260920)
    prompt = [128000] + [rng.randrange(1000, 30000) for _ in range(max(lengths))]
    trial = 0

    def generate(tokens):
        nonlocal trial
        trial += 1
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False,
                                extra_args={"job_id": f"profile-{trial}", "is_last_step": True,
                                            "forced_token_ids": [128009]})
        started = time.perf_counter()
        result = model.generate([{"prompt_token_ids": tokens}], params, use_tqdm=False)[0]
        return time.perf_counter() - started, result.num_cached_tokens

    for _ in range(5):
        generate(prompt[:2048])
    experiments = [(length, fraction, repeat) for length in lengths
                   for fraction in fractions for repeat in range(3)]
    rng.shuffle(experiments)
    raw = [json.loads(line) for line in (args.extend_from / "raw.jsonl").read_text().splitlines()] if args.extend_from else []
    existing = {(record["total_tokens"], record["fraction"], record["repeat"]) for record in raw}
    with (args.output / "raw.jsonl").open("w") as stream:
        for record in raw:
            stream.write(json.dumps(record) + "\n")
        for length, fraction, repeat in experiments:
            if (length, fraction, repeat) in existing:
                continue
            model.reset_prefix_cache()
            cached = min(length - 16, int(length * fraction) // 16 * 16)
            if cached:
                generate(prompt[:cached] + [50000])
            seconds, hit = generate(prompt[:length])
            assert hit == cached, (length, fraction, cached, hit)
            record = {"total_tokens": length, "fraction": fraction, "cached_tokens": cached,
                      "repeat": repeat, "wall_seconds": seconds, "observed_hit_tokens": hit}
            raw.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps({"completed": len(raw), "total": len(experiments), **record}), flush=True)
    curves = []
    for length in lengths:
        rows = []
        previous = 0.0
        for fraction in reversed(fractions):
            subset = [record for record in raw if record["total_tokens"] == length and record["fraction"] == fraction]
            median = statistics.median(record["wall_seconds"] for record in subset)
            previous = max(previous, median)
            rows.append({"cached": subset[0]["cached_tokens"], "seconds": previous, "raw_median": median})
        rows.reverse()
        curves.append({"total": length, "points": rows})
    (args.output / "curve.json").write_text(json.dumps({"measurement": "H200 real Llama forward, eager, isolated request wall time",
        "repeats": 3, "chunk": 2048, "kv_gib": 24, "curves": curves,
        "monotonicization": "cumulative maximum from highest cached prefix toward zero"}, indent=2))
    model.llm_engine.engine_core.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
