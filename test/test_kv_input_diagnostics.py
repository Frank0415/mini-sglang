"""Optional input-safety diagnostics, NOT SGLang API guarantees.

These tests ask whether malformed calls fail clearly without corrupting state.
They are deliberately separate from valid-input correctness tests. A failure
here can instead be addressed by an explicit caller-side precondition.
All malformed indexing probes run on CPU, never CUDA.
"""
import unittest
import torch

from src.mem_cache.allocator import KVPoolConfig, MHATokenToKVPool, PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool, ReqToTokenPoolConfig


class InputDiagnostics(unittest.TestCase):
    def test_negative_request_count(self):
        pool = ReqToTokenPool(ReqToTokenPoolConfig(3, 8, torch.device('cpu')))
        before = pool.free_slots.copy()
        with self.assertRaises((ValueError, AssertionError)):
            pool.alloc_rows(-1)
        self.assertEqual(pool.free_slots, before)

    def test_double_free_request_row(self):
        pool = ReqToTokenPool(ReqToTokenPoolConfig(3, 8, torch.device('cpu')))
        row = pool.alloc_rows(1)
        pool.free_rows(row)
        before = pool.free_slots.copy()
        try:
            pool.free_rows(row)
        except (ValueError, AssertionError):
            pass
        self.assertEqual(pool.free_slots, before, 'Double free corrupts free-row list')

    def test_free_reserved_request_row(self):
        pool = ReqToTokenPool(ReqToTokenPoolConfig(3, 8, torch.device('cpu')))
        before = pool.free_slots.copy()
        try:
            pool.free_rows([0])
        except (ValueError, AssertionError):
            pass
        self.assertEqual(pool.free_slots, before)

    def test_invalid_extend_lengths(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        with self.assertRaises((ValueError, AssertionError)):
            pool.alloc_extend(3, 7, 14, 2)  # 7 - 3 != 2

    def test_negative_extension(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        with self.assertRaises((ValueError, AssertionError)):
            pool.alloc_extend(3, 2, 14, -1)

    def test_nonempty_prefix_requires_valid_last_loc(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        with self.assertRaises((ValueError, AssertionError)):
            pool.alloc_extend(3, 4, -1, 1)

    def test_wrong_tail_offset(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        with self.assertRaises((ValueError, AssertionError)):
            pool.alloc_extend(3, 4, 12, 1)  # Prefix of 3 must end at page offset 2.

    def test_double_free_page(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        pool.free_pages = []
        pool.free([12, 13])
        before = pool.free_pages.copy()
        try:
            pool.free([14, 15])  # Same page, a second call.
        except (ValueError, AssertionError):
            pass
        self.assertEqual(pool.free_pages, before)

    def test_free_negative_slot(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        before = pool.free_pages.copy()
        try:
            pool.free([-1])
        except (ValueError, AssertionError):
            pass
        self.assertEqual(pool.free_pages, before)

    def test_out_of_bounds_storage_read_rejected(self):
        pool = MHATokenToKVPool(KVPoolConfig(1, 8, 1, 4, torch.float32, torch.device('cpu')))
        with self.assertRaises((IndexError, RuntimeError, ValueError, AssertionError)):
            pool.get_key_buffer(0, torch.tensor([8]))

    def test_wrong_kv_shape_rejected(self):
        pool = MHATokenToKVPool(KVPoolConfig(1, 8, 3, 4, torch.float32, torch.device('cpu')))
        with self.assertRaises((RuntimeError, ValueError, AssertionError)):
            pool.set_kv_buffer(0, torch.tensor([1, 2]), torch.zeros(2, 2, 4), torch.zeros(2, 2, 4))

    def test_out_of_bounds_request_row_rejected(self):
        pool = ReqToTokenPool(ReqToTokenPoolConfig(3, 8, torch.device('cpu')))
        with self.assertRaises((IndexError, RuntimeError, ValueError, AssertionError)):
            pool.write((4, slice(0, 1)), torch.tensor([12]))


def bad_constructor(page_size, num_pages):
    def test(self):
        with self.assertRaises((ValueError, AssertionError)):
            PagedTokenToKVPoolAllocator(page_size, num_pages)
    return test


for page_size, num_pages in [(0, 4), (-1, 4), (4, -1)]:
    setattr(InputDiagnostics, f'test_bad_config_p{page_size}_n{num_pages}',
            bad_constructor(page_size, num_pages))
