"""Checks that failed boundary assertions do not partially change pool state."""
import unittest
import torch
from src.mem_cache.allocator import KVPoolConfig, MHATokenToKVPool, PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPoolConfig, ReqToTokenPool


class AssertionBoundaryTests(unittest.TestCase):
    def test_context_overflow_preserves_mapping(self):
        pool = ReqToTokenPool(ReqToTokenPoolConfig(1, 4, torch.device('cpu')))
        row = pool.alloc_rows(1)[0]
        pool.write((row, slice(0, 4)), torch.tensor([0, 1, 2, 3]))
        before = pool.req_to_token.clone()
        with self.assertRaises(AssertionError):
            pool.write((row, slice(4, 5)), torch.tensor([4]))
        self.assertTrue(torch.equal(pool.req_to_token, before))

    def test_mapping_does_not_broadcast(self):
        pool = ReqToTokenPool(ReqToTokenPoolConfig(1, 4, torch.device('cpu')))
        before = pool.req_to_token.clone()
        with self.assertRaises(AssertionError):
            pool.write((1, slice(0, 3)), torch.tensor([7]))
        self.assertTrue(torch.equal(pool.req_to_token, before))

    def test_capacity_mismatch_preserves_kv(self):
        allocator = PagedTokenToKVPoolAllocator(4, 2)
        slots = torch.tensor(allocator.alloc_extend(0, 8, -1, 8))
        pool = MHATokenToKVPool(KVPoolConfig(1, 7, 1, 2, torch.float32, torch.device('cpu')))
        pool.k_buffer.zero_()
        pool.v_buffer.zero_()
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer(0, slots, torch.ones(8, 1, 2), torch.ones(8, 1, 2))
        self.assertEqual(pool.k_buffer.count_nonzero().item(), 0)
        self.assertEqual(pool.v_buffer.count_nonzero().item(), 0)

    def test_invalid_v_does_not_write_k(self):
        pool = MHATokenToKVPool(KVPoolConfig(1, 8, 3, 4, torch.float32, torch.device('cpu')))
        pool.k_buffer.zero_()
        pool.v_buffer.zero_()
        with self.assertRaises(AssertionError):
            pool.set_kv_buffer(0, torch.tensor([1, 2]), torch.ones(2, 3, 4), torch.ones(1, 3, 4))
        self.assertEqual(pool.k_buffer.count_nonzero().item(), 0)
        self.assertEqual(pool.v_buffer.count_nonzero().item(), 0)

    def test_mixed_double_free_preserves_pages(self):
        pool = PagedTokenToKVPoolAllocator(4, 3)
        slots = pool.alloc_extend(0, 8, -1, 8)
        pool.free(slots[:4])
        before = pool.free_pages.copy()
        with self.assertRaises(AssertionError):
            pool.free(slots)
        self.assertEqual(pool.free_pages, before)

    def test_mixed_double_free_preserves_rows(self):
        pool = ReqToTokenPool(ReqToTokenPoolConfig(3, 8, torch.device('cpu')))
        rows = pool.alloc_rows(2)
        pool.free_rows(rows[:1])
        before = pool.free_slots.copy()
        with self.assertRaises(AssertionError):
            pool.free_rows(rows)
        self.assertEqual(pool.free_slots, before)
