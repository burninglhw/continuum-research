import bisect
import json
import math
from collections import defaultdict, deque
from pathlib import Path


class CostCurve:
    def __init__(self, path):
        self.curves = json.loads(Path(path).read_text())["curves"]
        self.lengths = [curve["total"] for curve in self.curves]

    @staticmethod
    def interpolate(position, positions, values):
        if position <= positions[0]:
            return values[0]
        if position >= positions[-1]:
            return values[-1]
        upper = bisect.bisect_right(positions, position)
        lower = upper - 1
        fraction = (position - positions[lower]) / (positions[upper] - positions[lower])
        return values[lower] * (1 - fraction) + values[upper] * fraction

    def latency(self, cached, total):
        fraction = min(1.0, max(0.0, cached / max(1, total)))
        costs = []
        for curve in self.curves:
            positions = [point["cached"] / curve["total"] for point in curve["points"]]
            values = [point["seconds"] for point in curve["points"]]
            costs.append(self.interpolate(fraction, positions, values))
        cost = self.interpolate(total, self.lengths, costs)
        if total < self.lengths[0]:
            cost *= total / self.lengths[0]
        return cost

    def marginal(self, boundary, total, block_size=16):
        return max(0.0, self.latency((boundary - 1) * block_size, total)
                   - self.latency(boundary * block_size, total))


class OnlineHistory:
    def __init__(self):
        self.durations = defaultdict(list)
        self.evicted_queue = deque(maxlen=128)
        self.completed_lengths = []

    def observe_tool(self, tool, seconds):
        if tool is not None:
            bisect.insort(self.durations[tool], max(0.0, seconds))

    def eta(self):
        count = sum(max(0, length - 1) for length in self.completed_lengths)
        if count < 2:
            return 0.0
        sum_done = sum(length * (length - 1) / 2 for length in self.completed_lengths)
        sum_square = sum((length - 1) * length * (2 * length - 1) / 6 for length in self.completed_lengths)
        cross = sum(length * length * (length - 1) / 2
                    - (length - 1) * length * (2 * length - 1) / 6 for length in self.completed_lengths)
        variance = sum_square - sum_done**2 / count
        if variance <= 0:
            return 0.0
        return min(1.0, max(0.0, -(cross - sum_done**2 / count) / variance))

    def ttl(self, tool, full_prefill, mode):
        if tool is None:
            return 0.0
        observations = self.durations[tool]
        if mode == "continuum-public":
            return 2.0 if not observations or sum(observations) / len(observations) <= 2.0 else 0.0
        if not observations:
            return 2.0
        queue = sum(self.evicted_queue) / len(self.evicted_queue) if self.evicted_queue else 0.0
        benefit = full_prefill + queue * self.eta()
        candidates = [0.0] + sorted(set(observations))
        return max(candidates, key=lambda duration: (bisect.bisect_right(observations, duration)
                   / len(observations) * benefit - duration, -duration))

    def conditional(self, tool, elapsed, horizon=1.0):
        observations = self.durations[tool]
        if not observations:
            return 1.0 - math.exp(-horizon / 2.0)
        survived = len(observations) - bisect.bisect_right(observations, elapsed)
        if survived == 0:
            return 0.0
        upcoming = bisect.bisect_right(observations, elapsed + horizon) - bisect.bisect_right(observations, elapsed)
        return upcoming / survived


def release_suffix(manager, request_id, keep_blocks):
    blocks = manager.req_to_blocks[request_id]
    assert 0 <= keep_blocks <= len(blocks)
    released = blocks[keep_blocks:]
    hashes = [block.block_hash for block in released]
    before = manager.block_pool.get_num_free_blocks()
    manager.block_pool.free_blocks(reversed(released))
    del blocks[keep_blocks:]
    manager.num_cached_block[request_id] = min(manager.num_cached_block.get(request_id, 0), keep_blocks)
    assert hashes == [block.block_hash for block in released], "unpin must not evict"
    assert all(block.ref_cnt >= 0 for block in released), "negative block reference"
    return len(released), manager.block_pool.get_num_free_blocks() - before


def return_probability(history, tool, elapsed, ready, horizon=1.0):
    if ready:
        return 1.0
    return history.conditional(tool, elapsed, horizon)
