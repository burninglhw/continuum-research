import argparse
import json
import os
import time
from pathlib import Path

import torch
from vllm import SamplingParams

from run_replay import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-profile", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["RUN_OUTPUT_DIR"] = str(args.output.resolve())
    os.environ["ELASTIC_MODE"] = "continuum-public"
    os.environ["ELASTIC_COST_PROFILE"] = str(args.cost_profile.resolve())
    model = create_model(args.model, 24)
    scheduler = model.llm_engine.engine_core.engine_core.scheduler
    records = []
    prompt = [128000] + [1000 + index % 1000 for index in range(1023)]
    target = [3000 + index for index in range(32)]
    next_prompt = prompt + target + [4000] * 16
    probe = [5000 + index for index in range(8)]

    def generate(tokens, output, job, final):
        params = SamplingParams(temperature=0, max_tokens=len(output), ignore_eos=True, detokenize=False,
                                logprobs=1, extra_args={"job_id": job, "is_last_step": final,
                                                      "trace_tool": None if final else "unit",
                                                      "forced_token_ids": output})
        result = model.generate([{"prompt_token_ids": tokens}], params, use_tqdm=False)[0]
        assert list(result.outputs[0].token_ids) == output
        logprobs = [row[token].logprob for row, token in zip(result.outputs[0].logprobs, output)]
        return result.num_cached_tokens, logprobs

    def reset():
        for owner, deadline in list(scheduler.pinned_requests):
            scheduler.unpin_request(owner, deadline)
        assert not scheduler.manager.req_to_blocks, scheduler.manager.req_to_blocks.keys()
        assert scheduler.pool.get_num_free_blocks() == scheduler.pool.num_gpu_blocks - 1
        model.reset_prefix_cache()

    generate(prompt[:256], [3000], "warmup", True)
    reset()
    final_params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False,
                                  extra_args={"job_id": "causality-test", "is_last_step": True,
                                              "trace_tool": None, "forced_token_ids": [3000]})
    engine = model.llm_engine
    engine.add_request("causality-test", {"prompt_token_ids": prompt[:256]}, final_params)
    assert scheduler.requests["causality-test"].is_last_step is False
    assert scheduler.requests["causality-test"].this_func_call is None
    while engine.has_unfinished_requests():
        engine.step()
    assert not scheduler.pinned_requests
    records.append({"test": "terminal_marker_hidden_until_response_completion", "passed": True})
    reset()
    _, cold_logprobs = generate(next_prompt, probe, "cold", True)
    reset()
    generate(prompt, target, "resident", False)
    owner, _ = scheduler.pinned_requests[0]
    assert scheduler.tool_waiting_pin_count == len(scheduler.manager.req_to_blocks[owner.request_id])
    records.append({"test": "finished_request_is_counted_as_tool_waiting", "passed": True})
    scheduler.pinned_requests[0] = (owner, time.time() + 3600)
    scheduler.shrink(owner, 32)
    hit, resident_logprobs = generate(next_prompt, probe, "resident", True)
    error = max(abs(left - right) for left, right in zip(cold_logprobs, resident_logprobs))
    assert hit == 1040, hit
    assert error < 0.05, error
    records.append({"test": "unpin_not_eviction_and_raw_logprob_equivalence", "hit": hit,
                    "protected_tokens": 512, "max_raw_logprob_error": error})
    reset()
    generate(prompt, target, "evicted", False)
    owner, _ = scheduler.pinned_requests[0]
    scheduler.pinned_requests[0] = (owner, time.time() + 3600)
    scheduler.shrink(owner, 32)
    held = scheduler.pool.get_new_blocks(scheduler.pool.get_num_free_blocks())
    scheduler.pool.free_blocks(reversed(held))
    hit, recomputed_logprobs = generate(next_prompt, probe, "evicted", True)
    error = max(abs(left - right) for left, right in zip(cold_logprobs, recomputed_logprobs))
    assert hit == 512, hit
    assert error < 0.05, error
    records.append({"test": "suffix_eviction_prefix_recompute_raw_logprob_equivalence", "hit": hit,
                    "max_raw_logprob_error": error})
    reset()
    generate(prompt, target, "shared-a", False)
    generate(prompt, target, "shared-b", False)
    first, _ = scheduler.pinned_requests[0]
    second, _ = scheduler.pinned_requests[1]
    second_blocks = scheduler.manager.req_to_blocks[second.request_id].copy()
    assert any(block.ref_cnt >= 2 for block in second_blocks)
    hashes = [block.block_hash for block in second_blocks]
    scheduler.shrink(first, 0)
    assert all(block.ref_cnt >= 1 for block in second_blocks)
    assert [block.block_hash for block in second_blocks] == hashes
    records.append({"test": "shared_reference_survives_other_owner_unpin", "passed": True})
    reset()
    generate(prompt, target, "pressure-owner", False)
    owner, _ = scheduler.pinned_requests[0]
    scheduler.pinned_requests[0] = (owner, time.time() + 3600)
    original = len(scheduler.manager.req_to_blocks[owner.request_id])
    held = scheduler.pool.get_new_blocks(scheduler.pool.get_num_free_blocks() - 64)
    scheduler.mode = "elastic"
    generate([128000] + [7000] * 1535, [9000], "pressure-new", True)
    event = scheduler.reclamation[-1]
    assert event["raf"] == 1.0 and event["remaining_deficit"] == 0, event
    assert len(scheduler.manager.req_to_blocks[owner.request_id]) == original - event["revoked_blocks"]
    scheduler.pool.free_blocks(reversed(held))
    records.append({"test": "admission_targeted_boundary_reclamation", **event})
    reset()
    scheduler.mode = "continuum-public"
    generate(prompt, target, "return-owner", False)
    generate([128000] + [7100] * 1535, target, "other-owner", False)
    scheduler.pinned_requests = [(owner, time.time() + 3600) for owner, _ in scheduler.pinned_requests]
    held = scheduler.pool.get_new_blocks(scheduler.pool.get_num_free_blocks())
    scheduler.mode = "elastic"
    first_event = len(scheduler.reclamation)
    generate(next_prompt, probe, "return-owner", True)
    pressure_events = scheduler.reclamation[first_event:]
    assert pressure_events and all("return-owner" not in event["victim_jobs"] for event in pressure_events)
    assert all(event["net_raf"] == 1 for event in pressure_events), pressure_events
    scheduler.pool.free_blocks(reversed(held))
    records.append({"test": "return_owner_handoff_is_not_a_reclamation_victim", "events": pressure_events})
    reset()
    records.append({"test": "all_references_released_no_leak", "passed": True})
    (args.output / "invariants.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2), flush=True)
    model.llm_engine.engine_core.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
