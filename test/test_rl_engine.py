import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.engine import Engine
from src.mem_cache.allocator import PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.mem_cache.radix_cache import RadixCache


def make_engine(do_sample=False):
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
        logits[:, 7] = 0
        logits[:, 9] = 0
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


if __name__ == "__main__":
    unittest.main()
