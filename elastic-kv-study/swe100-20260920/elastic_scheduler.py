import heapq
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from vllm.v1.core.estimate_with_func import Continuum_Recorder
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from policy import CostCurve, OnlineHistory, release_suffix, return_probability


class ReplayEstimator:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.departures = {}
        self.turns = defaultdict(int)

    def request_arrives(self, request):
        previous = self.departures.get(request.job_id)
        if previous is not None:
            tool, departed = previous
            self.scheduler.history.observe_tool(tool, time.time() - departed)

    def request_finished(self, request):
        request.is_last_step = request.sampling_params.extra_args.get("is_last_step", False)
        request.this_func_call = request.sampling_params.extra_args.get("trace_tool")
        self.departures[request.job_id] = (request.this_func_call, time.time())
        self.turns[request.job_id] += 1
        if request.is_last_step:
            self.scheduler.history.completed_lengths.append(self.turns[request.job_id])

    def set_up_pin(self, request):
        cost = self.scheduler.cost.latency(0, request.num_computed_tokens) if self.scheduler.cost else 0.0
        return self.scheduler.history.ttl(request.this_func_call, cost, self.scheduler.mode)


class Recorder(Continuum_Recorder):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def request_waiting_to_running(self, request, prompt_length, hit_length=0):
        super().request_waiting_to_running(request, prompt_length, hit_length)
        self.scheduler.record_admission(request, hit_length, False)

    def request_evicted_to_running(self, request, prompt_length, hit_length):
        super().request_evicted_to_running(request, prompt_length, hit_length)
        self.scheduler.record_admission(request, hit_length, True)

    def request_evicted_from_running_queue(self, request):
        super().request_evicted_from_running_queue(request)
        self.scheduler.lost_pin_jobs.add(request.job_id)
        self.scheduler.metrics["preemption_events"] += 1
        self.scheduler.preempted_at[request.request_id] = time.time()


class ElasticScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = os.environ.get("ELASTIC_MODE", "vllm-fcfs")
        self.output = Path(os.environ["RUN_OUTPUT_DIR"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.cost = CostCurve(os.environ["ELASTIC_COST_PROFILE"]) if os.environ.get("ELASTIC_COST_PROFILE") else None
        self.history = OnlineHistory()
        self.tool_call_estimator = ReplayEstimator(self)
        self.continuum_recorder = Recorder(self)
        self.metrics = defaultdict(float)
        self.admissions = []
        self.reclamation = []
        self.request_stats = defaultdict(lambda: defaultdict(float))
        self.lost_pin_jobs = set()
        self.preempted_at = {}
        self.pin_started = {}
        self.pin_initial_blocks = {}
        self.last_memory_time = time.time()
        self.pin_count = 0
        self.tool_waiting_pin_count = 0
        self.memory_series = []
        self.last_series_time = 0.0
        self.schedule_samples = []
        assert len(self.kv_cache_manager.coordinator.single_type_managers) == 1
        assert self.connector is None, "Primary partial protection implementation does not support offload"
        self.manager = self.kv_cache_manager.coordinator.single_type_managers[0]
        self.pool = self.kv_cache_manager.block_pool
        self.original_allocate = self.kv_cache_manager.allocate_slots
        self.kv_cache_manager.allocate_slots = self.allocate_slots
        original_evict = self.pool._maybe_evict_cached_block

        def record_evict(block):
            result = original_evict(block)
            self.metrics["physical_eviction_blocks"] += int(result)
            return result

        self.pool._maybe_evict_cached_block = record_evict

    def integrate_memory(self):
        now = time.time()
        elapsed = now - self.last_memory_time
        self.metrics["pinned_block_seconds"] += self.pin_count * elapsed
        self.metrics["tool_waiting_pinned_block_seconds"] += self.tool_waiting_pin_count * elapsed
        self.last_memory_time = now

    def recount_memory(self):
        self.integrate_memory()
        live_jobs = {request.job_id for request in self.running}
        live_jobs.update(request.job_id for request in self.waiting)
        pinned = set()
        tool_waiting = set()
        for request, _ in self.pinned_requests:
            ids = {block.block_id for block in self.manager.req_to_blocks.get(request.request_id, [])}
            pinned.update(ids)
            if request.job_id not in live_jobs:
                tool_waiting.update(ids)
        self.pin_count = len(pinned)
        self.tool_waiting_pin_count = len(tool_waiting)
        self.metrics["max_pinned_blocks"] = max(self.metrics["max_pinned_blocks"], self.pin_count)

    def record_admission(self, request, hit_length, resumed):
        now = time.time()
        queued = now - self.preempted_at.pop(request.request_id, request.arrival_time)
        self.admissions.append({"request_id": request.request_id, "job_id": request.job_id,
                               "timestamp": now, "queue_seconds": queued, "hit_tokens": hit_length,
                               "prompt_tokens": request.num_prompt_tokens, "resumed": resumed})
        expected_reuse = request.sampling_params.extra_args.get("previous_computed_lcp", 0) // self.block_size * self.block_size
        if resumed or (expected_reuse > 0 and hit_length < expected_reuse):
            self.history.evicted_queue.append(queued)
        self.lost_pin_jobs.discard(request.job_id)
        self.metrics["admission_queue_seconds"] += queued
        self.metrics["admission_hit_tokens"] += hit_length
        for previous, deadline in list(self.pinned_requests):
            if previous.job_id == request.job_id:
                self.unpin_request(previous, deadline)

    def add_request(self, request):
        self.integrate_memory()
        request.is_last_step = False
        super().add_request(request)
        self.recount_memory()

    def pin_request(self, request, length_of_pin):
        self.integrate_memory()
        super().pin_request(request, length_of_pin)
        self.pin_started[request.request_id] = time.time()
        self.pin_initial_blocks[request.request_id] = len(self.manager.req_to_blocks[request.request_id])
        if self.mode.startswith("static-"):
            fraction = int(self.mode.split("-")[1]) / 100
            self.shrink(request, int(self.pin_initial_blocks[request.request_id] * fraction))
        self.recount_memory()

    def shrink(self, request, keep, recount=True):
        if recount:
            self.integrate_memory()
        revoked, reclaimable = release_suffix(self.manager, request.request_id, keep)
        self.metrics["protection_revoked_blocks"] += revoked
        self.metrics["became_reclaimable_blocks"] += reclaimable
        if keep == 0:
            match = next(((owner, deadline) for owner, deadline in self.pinned_requests
                          if owner.request_id == request.request_id), None)
            if match:
                self.pinned_requests.remove(match)
                self.continuum_recorder.request_unpinned(request)
            self.manager.req_to_blocks.pop(request.request_id, None)
            self.manager.num_cached_block.pop(request.request_id, None)
        if recount:
            self.recount_memory()
        return revoked, reclaimable

    def unpin_request(self, request, end_time):
        self.integrate_memory()
        super().unpin_request(request, end_time)
        self.recount_memory()

    def unpin_requests_regular(self):
        waiting = {request.job_id for request in self.waiting}
        for request, deadline in list(self.pinned_requests):
            if request.job_id not in waiting and time.time() >= deadline:
                self.metrics["ttl_expirations"] += 1
                self.unpin_request(request, deadline)

    def _free_blocks(self, request):
        self.integrate_memory()
        for previous, deadline in list(self.pinned_requests):
            if previous.job_id == request.job_id:
                self.unpin_request(previous, deadline)
        if self.policy == SchedulingPolicy.CONTINUUM and not request.is_last_step:
            ttl = self.tool_call_estimator.set_up_pin(request)
            if ttl > 0.01:
                self.pin_request(request, ttl)
                del self.requests[request.request_id]
                return
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]
        self.recount_memory()

    def pop_running_request_based_on_last_step(self, request):
        if len(self.running) <= 1 and self.pinned_requests:
            if self.mode == "continuum-public":
                victim, deadline = max(self.pinned_requests, key=lambda entry: entry[1])
            else:
                victim, deadline = max(self.pinned_requests,
                                       key=lambda entry: self.running_job_id_first_entry_time[entry[0].job_id])
            self.pinned_requests.remove((victim, deadline))
            self.lost_pin_jobs.add(victim.job_id)
            return victim, True
        candidates = [candidate for candidate in self.running if not candidate.is_last_step] or self.running
        if not candidates:
            raise RuntimeError("No running or pinned request can relieve pressure")
        victim = max(candidates, key=lambda candidate: self.running_job_id_first_entry_time[candidate.job_id])
        self.running.remove(victim)
        return victim, False

    def allocation_deficit(self, request, count, new_count, new_blocks, lookahead, encoder):
        blocks = new_blocks.blocks if new_blocks is not None else ([],)
        required = min(request.num_computed_tokens + new_count + count + lookahead, self.max_model_len)
        needed = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request.request_id, required, blocks, encoder)
        return max(0, needed - self.pool.get_num_free_blocks())

    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                       num_lookahead_tokens=0, delay_cache_blocks=False, num_encoder_tokens=0):
        arguments = (request, num_new_tokens, num_new_computed_tokens, new_computed_blocks,
                     num_lookahead_tokens, delay_cache_blocks, num_encoder_tokens)
        result = self.original_allocate(*arguments)
        can_reclaim = (request in self.running and len(self.running) <= 1) or not self.running
        if result is not None or not can_reclaim or not self.pinned_requests:
            return result

        def deficit():
            return self.allocation_deficit(request, num_new_tokens, num_new_computed_tokens,
                                           new_computed_blocks, num_lookahead_tokens, num_encoder_tokens)

        initial = deficit()
        if initial == 0:
            return result
        started = time.perf_counter()
        self.integrate_memory()
        now = time.time()
        revoked = 0
        freed = 0
        net_revoked = 0
        victim_details = defaultdict(lambda: defaultdict(int))
        victim_jobs = set()
        boundaries = []
        owners = {owner.request_id: owner for owner, _ in self.pinned_requests}
        reserved = {block.block_id for block in new_computed_blocks.blocks[0]} if new_computed_blocks else set()
        ready_jobs = {waiting.job_id for waiting in self.waiting}

        def boundary_value(owner):
            blocks = self.manager.req_to_blocks.get(owner.request_id, [])
            if not blocks:
                return float("inf")
            elapsed = now - self.pin_started[owner.request_id]
            probability = return_probability(self.history, owner.this_func_call, elapsed,
                                             owner.job_id in ready_jobs)
            if blocks[-1].block_hash is None:
                return 0.0
            return probability * self.cost.marginal(len(blocks), owner.num_computed_tokens)

        if self.mode == "elastic":
            for owner in owners.values():
                if owner.job_id != request.job_id:
                    heapq.heappush(boundaries, (boundary_value(owner), owner.request_id))
        while deficit() > 0 and self.pinned_requests:
            if self.mode == "elastic":
                if not boundaries:
                    break
                _, request_id = heapq.heappop(boundaries)
                owner = owners[request_id]
                blocks = self.manager.req_to_blocks.get(request_id, [])
                if not blocks:
                    continue
                endangered = int(blocks[-1].ref_cnt == 1 and blocks[-1].block_id not in reserved)
                released, available = self.shrink(owner, len(blocks) - 1, recount=False)
                if self.manager.req_to_blocks.get(request_id):
                    heapq.heappush(boundaries, (boundary_value(owner), request_id))
            else:
                if self.mode == "conditional-whole":
                    def value(entry):
                        candidate = entry[0]
                        blocks = self.manager.req_to_blocks[candidate.request_id]
                        reclaimable = sum(block.ref_cnt == 1 and block.block_id not in reserved for block in blocks)
                        probability = return_probability(self.history, candidate.this_func_call,
                                                         now - self.pin_started[candidate.request_id],
                                                         candidate.job_id in ready_jobs)
                        saved = self.cost.latency(0, candidate.num_computed_tokens) - self.cost.latency(len(blocks) * 16, candidate.num_computed_tokens)
                        return probability * saved / max(1, reclaimable)
                    eligible = [entry for entry in self.pinned_requests if entry[0].job_id != request.job_id]
                    if not eligible:
                        break
                    owner, _ = min(eligible, key=value)
                elif self.mode == "continuum-public":
                    owner, _ = max(self.pinned_requests, key=lambda entry: entry[1])
                else:
                    owner, _ = max(self.pinned_requests,
                                   key=lambda entry: self.running_job_id_first_entry_time[entry[0].job_id])
                endangered = sum(block.ref_cnt == 1 and block.block_id not in reserved
                                 for block in self.manager.req_to_blocks[owner.request_id])
                released, available = self.shrink(owner, 0, recount=False)
            revoked += released
            freed += available
            net_revoked += endangered
            victim_details[owner.job_id]["owner_references_revoked"] += released
            victim_details[owner.job_id]["net_endangered_blocks"] += endangered
            victim_details[owner.job_id]["became_free_blocks"] += available
            victim_details[owner.job_id]["remaining_protected_blocks"] = len(self.manager.req_to_blocks.get(owner.request_id, []))
            victim_jobs.add(owner.job_id)
            self.lost_pin_jobs.add(owner.job_id)
        remaining = deficit()
        self.recount_memory()
        self.reclamation.append({"timestamp": now, "request_id": request.request_id,
                                 "needed_blocks": initial, "revoked_blocks": revoked,
                                 "became_free_blocks": freed, "remaining_deficit": remaining,
                                 "raf": revoked / initial, "victim_jobs": sorted(victim_jobs),
                                 "net_endangered_blocks": net_revoked, "net_raf": net_revoked / initial,
                                 "victim_details": dict(victim_details),
                                 "seconds": time.perf_counter() - started})
        return self.original_allocate(*arguments)

    def schedule(self):
        started = time.perf_counter()
        previous_running_count = len(self.running)
        result = super().schedule()
        self.schedule_samples.append(time.perf_counter() - started)
        for request_id, scheduled in result.num_scheduled_tokens.items():
            request = self.requests[request_id]
            end = request.num_computed_tokens
            begin = end - scheduled
            reused = request.sampling_params.extra_args.get("previous_computed_lcp", 0)
            stats = self.request_stats[request_id]
            stats["scheduled_tokens"] += scheduled
            stats["prefill_tokens"] += max(0, min(end, request.num_prompt_tokens) - min(begin, request.num_prompt_tokens))
            stats["historical_prefix_recomputed_tokens"] += max(0, min(end, reused) - min(begin, reused))
        if (result.scheduled_new_reqs or any(result.scheduled_cached_reqs.resumed_from_preemption)
                or len(self.running) != previous_running_count):
            self.recount_memory()
        now = time.time()
        if now - self.last_series_time >= 1.0:
            self.memory_series.append({"timestamp": now, "protected_blocks": self.pin_count,
                                       "tool_waiting_protected_blocks": self.tool_waiting_pin_count,
                                       "running": len(self.running), "waiting": len(self.waiting),
                                       "free_blocks": self.pool.get_num_free_blocks()})
            self.last_series_time = now
        return result

    def update_from_output(self, scheduler_output, model_runner_output):
        previous_running_count = len(self.running)
        result = super().update_from_output(scheduler_output, model_runner_output)
        if len(self.running) != previous_running_count:
            self.recount_memory()
        return result

    def export_metrics(self):
        self.recount_memory()
        result = {"mode": self.mode, "metrics": dict(self.metrics), "admissions": self.admissions,
                  "reclamation": self.reclamation, "requests": dict(self.request_stats),
                  "scheduler_seconds": self.schedule_samples, "memory_series": self.memory_series,
                  "eta_final": self.history.eta(), "observed_tool_samples": {
                      tool: len(values) for tool, values in self.history.durations.items()},
                  "block_size": self.block_size, "gpu_blocks": self.pool.num_gpu_blocks}
        (self.output / "scheduler-metrics.json").write_text(json.dumps(result))
        return result

    def shutdown(self):
        self.export_metrics()
        super().shutdown()
