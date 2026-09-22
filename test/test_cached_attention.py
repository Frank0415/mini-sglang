"""Acceptance tests for cached attention; missing implementation must FAIL.

No backend monkey-patching. History buffers are seeded only in read-path tests
so writing, reading and offset masking have independent checks.
"""
import unittest

import torch
from torch.nn import functional as F

from src.layers.attention.torch_backend import TorchAttnBackend
from src.mem_cache.allocator import MHATokenToKVPool, PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PagedKVBatch,
)


@unittest.skipUnless(torch.cuda.is_available(), 'Current backend requires CUDA flash SDPA')
class CachedAttentionTests(unittest.TestCase):
    def setup_case(self, page_size=4, kv_heads=2):
        torch.manual_seed(42)
        rows = ReqToTokenPool(2, 128, torch.device('cuda'))
        pool = MHATokenToKVPool(2, page_size * 32, kv_heads, 16,
                               torch.bfloat16, torch.device('cuda'), page_size)
        pool.k_buffer.zero_()
        pool.v_buffer.zero_()
        alloc = PagedTokenToKVPoolAllocator(page_size, 32)
        # Exercise non-contiguous pages as well as contiguous positions inside a page.
        alloc.free_pages[:3] = [1, 3, 2]
        return rows, pool, alloc, TorchAttnBackend(rows, pool)

    def test_prefill_writes_only_selected_layer_and_slots(self):
        rows, pool, alloc, backend = self.setup_case()
        row = rows.alloc_rows(1)[0]
        slots = torch.tensor(alloc.alloc_extend(0, 3, -1, 3), device='cuda')
        rows.write((row, slice(0, 3)), slots)
        q = torch.randn(1, 4, 3, 16, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 2, 3, 16, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        info = PagedKVBatch(torch.tensor([row], device='cuda'), slots)
        batch = ForwardBatch(ForwardMode.EXTEND, torch.tensor([3]), torch.tensor([3]), {0: info})
        expected_k = pool.k_buffer.clone()
        expected_v = pool.v_buffer.clone()
        expected_k[1, slots] = k[0].transpose(0, 1)
        expected_v[1, slots] = v[0].transpose(0, 1)
        with torch.inference_mode():
            backend(q, k, v, 1, batch, info)
        torch.testing.assert_close(pool.k_buffer, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(pool.v_buffer, expected_v, rtol=0, atol=0)

    def test_uncached_control(self):
        _, _, _, backend = self.setup_case()
        q = torch.randn(1, 4, 5, 16, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, 2, 5, 16, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        with torch.inference_mode():
            actual = backend(q, k, v, 0, None, None)
            reference = F.scaled_dot_product_attention(q.float().cpu(), k.float().cpu(),
                v.float().cpu(), is_causal=True, enable_gqa=True)
        torch.testing.assert_close(actual.float().cpu(), reference, rtol=.02, atol=.02)


def make_read_case(page_size, kv_heads, case):
    def test(self):
        rows, pool, alloc, backend = self.setup_case(page_size, kv_heads)
        if case == 'decode_tail':
            prefix, count = max(1, page_size - 1), 1
        elif case == 'decode_new_page':
            prefix, count = page_size, 1
        else:
            prefix, count = page_size + 1, 3
        total = prefix + count
        row = rows.alloc_rows(1)[0]
        old = alloc.alloc_extend(0, prefix, -1, prefix)
        new = alloc.alloc_extend(prefix, total, old[-1], count)
        slots = torch.tensor(old + new, device='cuda')
        rows.write((row, slice(0, total)), slots)
        # Other rows point elsewhere. Correct reads must use the selected row.
        other = rows.alloc_rows(1)[0]
        rows.write((other, slice(0, total)), torch.zeros(total, device='cuda', dtype=torch.int64))
        q = torch.randn(1, 4, total, 16, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, kv_heads, total, 16, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        pool.set_kv_buffer(0, slots, k[0].transpose(0, 1), v[0].transpose(0, 1))
        mode = ForwardMode.DECODE if count == 1 else ForwardMode.EXTEND
        info = PagedKVBatch(torch.tensor([row], device='cuda'), torch.tensor(new, device='cuda'))
        batch = ForwardBatch(mode, torch.tensor([total]), torch.tensor([count]), {0: info})
        with torch.inference_mode():
            actual = backend(q[:, :, prefix:], k[:, :, prefix:], v[:, :, prefix:], 0, batch, info)
            reference = F.scaled_dot_product_attention(q.float().cpu(), k.float().cpu(),
                v.float().cpu(), is_causal=True, enable_gqa=True)[:, :, prefix:]
        torch.testing.assert_close(actual.float().cpu(), reference, rtol=.02, atol=.02)
    return test


for p in (1, 4, 16):
    for h in (1, 2, 4):  # MQA, GQA, MHA, with four query heads.
        for case in ('decode_tail', 'decode_new_page', 'extend_prefix'):
            setattr(CachedAttentionTests, f'test_p{p}_kv{h}_{case}', make_read_case(p, h, case))


if __name__ == '__main__':
    unittest.main()
