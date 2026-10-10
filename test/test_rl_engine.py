import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.engine import Engine
from src.mem_cache.allocator import PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.mem_cache.radix_cache import RadixCache


def make_engine(do_sample=False, scores=(0, 0)):
    pool = ReqToTokenPool(4, 128, "cpu")
    allocator = PagedTokenToKVPoolAllocator(4, 32)
    cache = RadixCache(4, pool, allocator)
    runner = SimpleNamespace(
        device="cpu", runtime=None, req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator, radix_cache=cache,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=2)),
        model=SimpleNamespace(load_weights=Mock()),
    )

    def forward(ids, positions, context):
        logits = torch.full((len(ids), 16), -torch.inf)
        logits[:, 7] = scores[0]
        logits[:, 9] = scores[1]
        return logits

    runner.forward = forward
    with patch("src.engine.ModelRunner", return_value=runner):
        engine = Engine("unused", device="cpu", do_sample=do_sample)
    return engine


class RLEngineTests(unittest.TestCase):
    def test_default_generation_remains_greedy(self):
        engine = make_engine()
        results = engine.generate_batch([[3, 4, 5]] * 4, 5)
        self.assertEqual([r.output_ids for r in results], [[7] * 5] * 4)

    def test_random_generation_samples_distinct_responses(self):
        torch.manual_seed(7)
        engine = make_engine(do_sample=True)
        results = engine.generate_batch([[3, 4, 5]] * 4, 5)
        self.assertGreater(len({tuple(r.output_ids) for r in results}), 1)
        self.assertEqual({token for r in results for token in r.output_ids}, {7, 9})
        self.assertEqual(engine.scheduler.requests, {})
        self.assertEqual(engine.runner.req_to_token_pool.available_slots(), 4)

    def test_log_probs_match_sampled_tokens(self):
        torch.manual_seed(7)
        engine = make_engine(do_sample=True, scores=(0, math.log(3)))
        results = engine.generate_batch([[3, 4, 5]] * 4, 5)
        expected = {7: math.log(0.25), 9: math.log(0.75)}
        for request in results:
            self.assertEqual(len(request.output_log_probs), len(request.output_ids))
            for token, log_prob in zip(request.output_ids, request.output_log_probs):
                self.assertAlmostEqual(log_prob, expected[token], places=6)

    def test_step_returns_log_prob_for_eos(self):
        engine = make_engine()
        engine.scheduler.eos_token_ids = {7}
        request_id = engine.add_request([3, 4, 5], 5)
        event = engine.step()[0]
        request = engine.free_request(request_id)
        self.assertEqual(request.output_ids, [7])
        self.assertEqual(request.finish_reason, "eos")
        self.assertEqual(request.output_log_probs, [event["log_prob"]])
        self.assertAlmostEqual(event["log_prob"], -math.log(2), places=6)

    def test_weight_update_clears_cached_prefix_and_commits_version(self):
        engine = make_engine()
        old_result = engine.generate([3, 4, 5, 6, 7], 1)
        cache = engine.runner.radix_cache
        prefix, _ = cache._match_prefix([3, 4, 5, 6])
        self.assertEqual(prefix.numel(), 4)
        weights = [("weight", torch.ones(1))]
        engine.update_weights(weights, version=3)
        engine.runner.model.load_weights.assert_called_once_with(weights)
        self.assertEqual(cache.root_node.children, {})
        self.assertEqual(cache.evictable_nodes, set())
        allocator = engine.runner.token_to_kv_pool_allocator
        self.assertEqual(sorted(allocator.free_pages), list(range(1, 33)))
        self.assertEqual(engine.installed_version, 3)
        self.assertEqual(old_result.weight_version, 0)
        new_result = engine.generate([3, 4, 5, 6, 7], 1)
        self.assertEqual(new_result.weight_version, 3)

    def test_weight_update_rejects_waiting_and_running_requests(self):
        engine = make_engine()
        request_id = engine.add_request([3, 4, 5], 5)
        with self.assertRaises(RuntimeError):
            engine.update_weights([], version=1)
        engine.step()
        with self.assertRaises(RuntimeError):
            engine.update_weights([], version=1)
        engine.runner.model.load_weights.assert_not_called()
        self.assertEqual(engine.installed_version, 0)
        engine.abort(request_id)
        engine.free_request(request_id)
        engine.update_weights([], version=1)
        self.assertEqual(engine.installed_version, 1)

    def test_weight_update_failure_keeps_version(self):
        engine = make_engine()
        engine.runner.model.load_weights.side_effect = ValueError("Missing weights")
        with self.assertRaisesRegex(ValueError, "Missing weights"):
            engine.update_weights([], version=1)
        self.assertEqual(engine.installed_version, 0)


if __name__ == "__main__":
    unittest.main()
