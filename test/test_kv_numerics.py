"""Cache correctness under controlled arithmetic, separate from BF16 stability.

Original BF16 application-level tests remain unchanged and may fail on near ties.
"""
import json
import unittest
from pathlib import Path

import torch

from scripts.diagnose_kv_numerics import control, replay
from src.model_executor.model_runner import ModelRunner

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class KVNumericalCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configs=list((ROOT/'models').rglob('config.json'))
        if not configs:raise unittest.SkipTest('Local model not available')
        cls.runner=ModelRunner(configs[0].parent)
        cls.fixture=json.loads((ROOT/'test/fixtures/kv_chat_tokens.json').read_text())

    def assert_fp32(self, rows):
        # Absolute budget 2e-4; observed calibration maximum was 7.42e-5.
        # No relative allowance to hide errors near zero or large logits.
        for row in rows:
            self.assertLessEqual(row['max_abs'],2e-4,row)
            # If reference margin exceeds twice max error, top-1 must be stable.
            if row['reference_margin']>2*row['max_abs']:
                self.assertTrue(row['greedy_match'],row)

    def test_fp32_fixed_three_turns(self):
        with control(self.runner,'fp32_math'):
            self.assert_fp32(replay(self.runner,self.fixture))

    def test_fp32_held_out_lengths_and_rewind(self):
        generator=torch.Generator().manual_seed(8231)
        with control(self.runner,'fp32_math'):
            for prefix in [1,2,3,4,5,7,16,31,64,128]:
                tokens=torch.randint(100,40000,(prefix+9,),generator=generator).tolist()
                with self.subTest(prefix=prefix):
                    ops=[dict(label='prefill',kind='prompt',tokens=tokens[:prefix]),
                         dict(label='decode',kind='decode',tokens=tokens[prefix:prefix+1]),
                         dict(label='extend',kind='prompt',tokens=tokens[:prefix+5]),
                         dict(label='rewind',kind='prompt',tokens=tokens[:max(1,prefix-1)]+tokens[-3:])]
                    self.assert_fp32(replay(self.runner,ops))

    def test_bf16_canonical_arithmetic_exact(self):
        with control(self.runner,'bf16_canonical'):
            ops=[dict(label='prefill',kind='prompt',tokens=[1,2,3,4,5,6,7]),
                 dict(label='rewind',kind='prompt',tokens=[1,2,3,9,10])]
            for row in replay(self.runner,ops)+replay(self.runner,self.fixture):
                self.assertEqual(row['max_abs'],0.,row)
                self.assertTrue(row['greedy_match'],row)

    def test_fp32_canonical_page_sizes_and_recovery(self):
        # Pool geometry is varied while model weights and arithmetic stay fixed.
        from src.mem_cache.allocator import (
            MHATokenToKVPool,
            PagedTokenToKVPoolAllocator,
        )
        pool=self.runner.token_to_kv_pool
        allocator=self.runner.token_to_kv_pool_allocator
        group=self.runner.runtime.groups[0]
        try:
            for page_size in [1,4,16]:
                new=MHATokenToKVPool(pool.num_layers,256,pool.num_kv_heads,pool.head_dim,
                                    torch.float32,pool.device,page_size)
                alloc=PagedTokenToKVPoolAllocator(page_size,256//page_size)
                self.runner.token_to_kv_pool=new
                self.runner.token_to_kv_pool_allocator=alloc
                group.kv_pool=new;group.allocator=alloc
                group.backend.token_to_kv_pool=new
                # Exact arithmetic-shape control tests the page mapping, without
                # broadening tolerances for ill-conditioned special-token inputs.
                with control(self.runner,'fp32_canonical'):
                    results = replay(self.runner,[
                        dict(label='prefill',kind='prompt',tokens=[1,2,3]),
                        dict(label='tail',kind='decode',tokens=[4]),
                        dict(label='cross',kind='decode',tokens=[5]),
                    ])
                    for result in results:
                        self.assertEqual(result['max_abs'], 0., result)
                self.assertEqual(len(alloc.free_pages),256//page_size)
                self.assertEqual(self.runner.req_to_token_pool.available_slots(),4)
        finally:
            self.runner.token_to_kv_pool=pool
            self.runner.token_to_kv_pool_allocator=allocator
            group.kv_pool=pool;group.allocator=allocator
            group.backend.token_to_kv_pool=pool


if __name__=='__main__':unittest.main()
