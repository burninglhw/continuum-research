import argparse
import gzip
import hashlib
import heapq
import json
import os
import pickle
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (index - lower) * (values[upper] - values[lower])


def create_model(model_path, kv_gib):
    return LLM(model=model_path, tokenizer=model_path, dtype="bfloat16",
               max_model_len=131072, max_num_batched_tokens=2048,
               max_num_seqs=64, enable_prefix_caching=True, block_size=16,
               kv_cache_memory_bytes=int(kv_gib * 1024**3), gpu_memory_utilization=0.5,
               enforce_eager=True, disable_cascade_attn=True, disable_log_stats=True,
               scheduling_policy="fcfs" if os.environ["ELASTIC_MODE"] == "vllm-fcfs" else "continuum",
               scheduler_cls="elastic_scheduler.ElasticScheduler",
               logits_processors=["forced_tokens:ForcedTokens"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["vllm-fcfs", "continuum-public", "continuum-paper", "elastic",
                                         "static-25", "static-50", "static-75", "conditional-whole"], required=True)
    parser.add_argument("--cost-profile", type=Path)
    parser.add_argument("--jps", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-gib", type=float, default=24)
    parser.add_argument("--pilot-programs", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["ELASTIC_MODE"] = args.mode
    os.environ["RUN_OUTPUT_DIR"] = str(args.output.resolve())
    if args.cost_profile:
        os.environ["ELASTIC_COST_PROFILE"] = str(args.cost_profile.resolve())
    with gzip.open(args.data, "rb") as stream:
        programs = pickle.load(stream)
    if args.pilot_programs:
        programs = programs[:args.pilot_programs]
    else:
        assert len(programs) == 100
    rng = random.Random(args.seed)
    offset = 0.0
    arrivals = []
    for index, program in enumerate(programs):
        if index:
            offset += rng.expovariate(args.jps)
        arrivals.append(offset)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(program_ids=[program["instance_id"] for program in programs], arrival_offsets=arrivals,
                  source_sha256={path.name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                                 for path in Path(__file__).parent.glob("*.py")},
                  actual_engine="vllm 0.10.2 Continuum fork, forced-token real model forward",
                  reasoning_included=False, tool_timing="original command duration, no transport/retry sleep",
                  cascade_attention=False, gpu_index=2, status="initializing")
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "gpu-before.txt").write_text(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv"], text=True))
    model = create_model(args.model, args.kv_gib)
    engine = model.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    warm_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, detokenize=False,
                                 extra_args={"job_id": "warmup", "is_last_step": True,
                                             "forced_token_ids": [1000] * 16})
    model.generate([{"prompt_token_ids": [128000] + [1000] * 2047}], warm_params, use_tqdm=False)
    model.reset_prefix_cache()
    scheduler.metrics.clear()
    scheduler.admissions.clear()
    scheduler.schedule_samples.clear()
    scheduler.request_stats.clear()
    scheduler.memory_series.clear()
    scheduler.history.completed_lengths.clear()
    config["status"] = "running"
    config["device_properties"] = str(torch.cuda.get_device_properties(0))
    config["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    assert "0847e256-7dc7-41a9-b506-8f9e637d6d2f" in config["gpu_uuid"].lower(), config["gpu_uuid"]
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    start = time.time()
    scheduler.last_memory_time = start
    queue = [(start + offset, index, 0) for index, offset in enumerate(arrivals)]
    heapq.heapify(queue)
    in_flight = {}
    jobs = []
    turns = []
    with (args.output / "turns.jsonl").open("w") as turn_stream, (args.output / "jobs.jsonl").open("w") as job_stream:
        while queue or in_flight:
            now = time.time()
            while queue and queue[0][0] <= now:
                due, program_index, step_index = heapq.heappop(queue)
                program = programs[program_index]
                if step_index == len(program["steps"]):
                    record = {"instance_id": program["instance_id"], "arrival": arrivals[program_index],
                              "finished": due - start, "jct_seconds": due - start - arrivals[program_index],
                              "responses": len(program["steps"])}
                    jobs.append(record)
                    job_stream.write(json.dumps(record) + "\n")
                    job_stream.flush()
                    print(json.dumps({"completed_jobs": len(jobs), "total_jobs": len(programs), **record}), flush=True)
                    continue
                step = program["steps"][step_index]
                request_id = f"p{program_index}-s{step_index}"
                params = SamplingParams(temperature=0, max_tokens=len(step["output"]), ignore_eos=True,
                                        detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY,
                                        extra_args={"job_id": program["instance_id"],
                                                    "is_last_step": step_index == len(program["steps"]) - 1,
                                                    "forced_token_ids": step["output"], "trace_tool": step["tool"],
                                                    "previous_computed_lcp": step["previous_computed_lcp"]})
                submitted = time.time()
                engine.add_request(request_id, {"prompt_token_ids": step["prompt"]}, params,
                                   arrival_time=submitted)
                in_flight[request_id] = (program_index, step_index, due, submitted)
            if in_flight:
                for result in engine.step():
                    if not result.finished:
                        continue
                    finished = time.time()
                    program_index, step_index, due, submitted = in_flight.pop(result.request_id)
                    program = programs[program_index]
                    step = program["steps"][step_index]
                    assert list(result.outputs[0].token_ids) == step["output"], (program["instance_id"], step_index, "forced token mismatch")
                    record = {"instance_id": program["instance_id"], "step": step_index,
                              "request_id": result.request_id, "due": due - start,
                              "submitted": submitted - start, "finished": finished - start,
                              "arrival_dispatch_lag": submitted - due, "request_seconds": finished - due,
                              "prompt_tokens": len(step["prompt"]), "output_tokens": len(step["output"]),
                              "hit_tokens": result.num_cached_tokens, "forced_tokens_verified": True,
                              "tool_gap_seconds": step["gap"]}
                    turns.append(record)
                    turn_stream.write(json.dumps(record) + "\n")
                    turn_stream.flush()
                    heapq.heappush(queue, (finished + step["gap"], program_index, step_index + 1))
            elif queue:
                scheduler.unpin_requests_regular()
                next_event = queue[0][0]
                deadlines = [deadline for _, deadline in scheduler.pinned_requests if deadline > time.time()]
                if deadlines:
                    next_event = min(next_event, min(deadlines))
                time.sleep(max(0.0, next_event - time.time()))
        finish = time.time()
    stats = scheduler.export_metrics()
    durations = [job["jct_seconds"] for job in jobs]
    schedule_times = stats["scheduler_seconds"]
    summary = {"complete": len(jobs) == len(programs), "programs": len(jobs), "responses": len(turns),
               "all_forced_tokens_verified": all(turn["forced_tokens_verified"] for turn in turns),
               "mean_jct": statistics.mean(durations), "p50_jct": percentile(durations, .5),
               "p95_jct": percentile(durations, .95), "makespan": finish - start,
               "throughput_jobs_per_second": len(jobs) / (finish - start),
               "mean_scheduler_ms": statistics.mean(schedule_times) * 1000,
               "p95_scheduler_ms": percentile(schedule_times, .95) * 1000,
               "max_dispatch_lag": max(turn["arrival_dispatch_lag"] for turn in turns),
               "mean_dispatch_lag": statistics.mean(turn["arrival_dispatch_lag"] for turn in turns),
               "historical_prefix_recomputed_tokens": sum(row["historical_prefix_recomputed_tokens"] for row in stats["requests"].values()),
               "prefill_tokens": sum(row["prefill_tokens"] for row in stats["requests"].values()),
               "pinned_gib_seconds": stats["metrics"].get("pinned_block_seconds", 0) / 512,
               "tool_waiting_pinned_gib_seconds": stats["metrics"].get("tool_waiting_pinned_block_seconds", 0) / 512,
               "physical_eviction_blocks": stats["metrics"].get("physical_eviction_blocks", 0),
               "reclamation_events": len(stats["reclamation"])}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    config["status"] = "completed"
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    model.llm_engine.engine_core.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
