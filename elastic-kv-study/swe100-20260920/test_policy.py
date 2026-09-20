import unittest
from types import SimpleNamespace

from policy import CostCurve, OnlineHistory, release_suffix, return_probability


class Pool:
    def __init__(self, blocks):
        self.blocks = blocks

    def get_num_free_blocks(self):
        return sum(block.ref_cnt == 0 for block in self.blocks)

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref_cnt -= 1


class PolicyTests(unittest.TestCase):
    def test_prefix_closure_and_shared_references(self):
        blocks = [SimpleNamespace(ref_cnt=value, block_hash=str(index))
                  for index, value in enumerate([1, 1, 2, 1])]
        pool = Pool(blocks)
        manager = SimpleNamespace(req_to_blocks={"request": blocks.copy()}, block_pool=pool,
                                  num_cached_block={"request": 4})
        self.assertEqual(release_suffix(manager, "request", 2), (2, 1))
        self.assertEqual(manager.req_to_blocks["request"], blocks[:2])
        self.assertEqual([block.ref_cnt for block in blocks], [1, 1, 1, 0])
        self.assertEqual([block.block_hash for block in blocks], ["0", "1", "2", "3"])
        self.assertEqual(release_suffix(manager, "request", 0), (2, 2))

    def test_conditional_and_no_future_data(self):
        history = OnlineHistory()
        history.observe_tool("test", 1.0)
        history.observe_tool("test", 3.0)
        self.assertEqual(history.conditional("test", 1.5, 2.0), 1.0)
        self.assertEqual(history.conditional("test", 3.0, 2.0), 0.0)
        self.assertEqual(history.ttl("new", 20.0, "continuum-paper"), 2.0)
        self.assertEqual(history.ttl("test", 10.0, "continuum-paper"), 3.0)
        self.assertEqual(history.ttl("test", 0.1, "continuum-paper"), 0.0)

    def test_eta_fixed_length(self):
        history = OnlineHistory()
        history.completed_lengths = [10, 10, 10]
        self.assertAlmostEqual(history.eta(), 1.0)

    def test_returned_request_is_ready_not_censored(self):
        history = OnlineHistory()
        history.observe_tool("test", 1.0)
        self.assertEqual(return_probability(history, "test", 3.0, ready=False), 0.0)
        self.assertEqual(return_probability(history, "test", 3.0, ready=True), 1.0)

    def test_interpolation(self):
        self.assertEqual(CostCurve.interpolate(0.5, [0, 1], [4, 2]), 3)


if __name__ == "__main__":
    unittest.main()
