"""Behavior tests for existing KV APIs. No production code is patched.

Tests use the current project APIs and accept zero-based or one-based pages.
They do not require upstream method names or auxiliary interfaces.
"""
import math
import random
import unittest

import torch

from src.mem_cache.allocator import (
    MHATokenToKVPool,
    PagedTokenToKVPoolAllocator,
)
from src.mem_cache.memory_pool import ReqToTokenPool


def request_pool(rows=3, length=16):
    return ReqToTokenPool(rows, length, torch.device('cpu'))


def extend(allocator, previous, count):
    prefix = len(previous)
    return allocator.alloc_extend(prefix, prefix + count,
                                  previous[-1] if previous else -1, count)


class RequestPoolTests(unittest.TestCase):
    def test_shape_dtype_reserved_row(self):
        pool = request_pool()
        self.assertEqual(tuple(pool.req_to_token.shape), (4, 16))
        self.assertEqual(pool.req_to_token.dtype, torch.int32)
        self.assertTrue(torch.equal(pool.req_to_token[0], torch.zeros(16, dtype=torch.int32)))

    def test_allocation_consumes_capacity(self):
        pool = request_pool()
        self.assertEqual(len(pool.alloc_rows(2)), 2)
        self.assertEqual(pool.available_slots(), 1)

    def test_live_requests_have_distinct_rows(self):
        pool = request_pool()
        first = pool.alloc_rows(1)
        second = pool.alloc_rows(1)
        self.assertTrue(set(first).isdisjoint(second), (first, second))

    def test_exhaustion_then_reuse(self):
        pool = request_pool()
        rows = pool.alloc_rows(3)
        self.assertIsNone(pool.alloc_rows(1))
        pool.free_rows([rows[1]])
        self.assertEqual(pool.alloc_rows(1), [rows[1]])

    def test_empty_allocation_is_noop(self):
        pool = request_pool()
        before = list(pool.free_slots)
        self.assertEqual(pool.alloc_rows(0), [])
        self.assertEqual(pool.free_slots, before)

    def test_failed_allocation_is_noop(self):
        pool = request_pool()
        before = list(pool.free_slots)
        self.assertIsNone(pool.alloc_rows(4))
        self.assertEqual(pool.free_slots, before)

    def test_free_restores_capacity_exactly(self):
        pool = request_pool()
        rows = pool.alloc_rows(2)
        pool.free_rows(rows)
        self.assertEqual(pool.available_slots(), 3)
        self.assertEqual(len(set(pool.free_slots)), 3)

    def test_empty_free_is_noop(self):
        pool = request_pool()
        before = list(pool.free_slots)
        pool.free_rows([])
        self.assertEqual(pool.free_slots, before)

    def test_write_append_and_preserve_neighbors(self):
        pool = request_pool()
        pool.write((1, slice(0, 3)), torch.tensor([12, 13, 14]))
        pool.write((1, slice(3, 5)), torch.tensor([15, 28]))
        self.assertEqual(pool.req_to_token[1, :5].tolist(), [12, 13, 14, 15, 28])
        self.assertEqual(pool.req_to_token[1, 5:].count_nonzero().item(), 0)
        self.assertEqual(pool.req_to_token[2].count_nonzero().item(), 0)
        self.assertEqual(pool.req_to_token[0].count_nonzero().item(), 0)

    def test_write_last_column(self):
        pool = request_pool(length=4)
        pool.write((1, slice(3, 4)), torch.tensor([999]))
        self.assertEqual(pool.req_to_token[1].tolist(), [0, 0, 0, 999])

    def test_empty_write_is_noop(self):
        pool = request_pool()
        before = pool.req_to_token.clone()
        pool.write((1, slice(0, 0)), torch.empty(0, dtype=torch.int64))
        self.assertTrue(torch.equal(before, pool.req_to_token))

    def test_clear_after_all_requests_finish(self):
        pool = request_pool()
        rows = pool.alloc_rows(3)
        pool.free_rows(rows)
        pool.clear()
        self.assertEqual(pool.available_slots(), 3)
        self.assertEqual(sorted(pool.free_slots), [1, 2, 3])
        # clear need not erase old mapping values; callers track valid lengths.

    def test_interleaved_request_writes_do_not_alias(self):
        pool = request_pool()
        a = pool.alloc_rows(1)[0]
        b = pool.alloc_rows(1)[0]
        pool.write((a, slice(0, 2)), torch.tensor([12, 13]))
        pool.write((b, slice(0, 2)), torch.tensor([28, 29]))
        self.assertEqual(pool.req_to_token[a, :2].tolist(), [12, 13])
        self.assertEqual(pool.req_to_token[b, :2].tolist(), [28, 29])


class AllocationTests(unittest.TestCase):
    def check_allocation(self, pool, old, count, result, free_before):
        p = pool.page_size
        needed = math.ceil((len(old) + count) / p) - math.ceil(len(old) / p)
        if needed > len(free_before):
            self.assertIsNone(result)
            self.assertEqual(pool.free_pages, free_before)
            return
        self.assertIsNotNone(result)
        self.assertEqual(len(result), count, 'Must return one slot per new token')
        self.assertEqual(len(set(old + result)), len(old) + count, 'Token slots overlap')
        all_slots = old + result
        for start in range(0, len(all_slots), p):
            chunk = all_slots[start:start + p]
            self.assertEqual(chunk[0] % p, 0)
            self.assertEqual(chunk, list(range(chunk[0], chunk[0] + len(chunk))))
        old_pages = {s // p for s in old}
        new_pages = {s // p for s in result} - old_pages
        self.assertEqual(len(new_pages), needed)
        self.assertTrue(new_pages <= set(free_before), 'Allocated a page not on the free list')
        self.assertEqual(len(pool.free_pages), len(free_before) - needed)
        self.assertEqual(set(pool.free_pages), set(free_before) - new_pages)

    def test_two_new_requests_never_overlap(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        a = extend(pool, [], 4)
        b = extend(pool, [], 4)
        self.assertTrue(set(a).isdisjoint(b), (a, b))

    def test_uses_actual_fragmented_free_page_ids(self):
        pool = PagedTokenToKVPoolAllocator(4, 10)
        # A valid fragmented state: all other pages belong to other requests.
        pool.free_pages = [7, 3]
        before = pool.free_pages.copy()
        result = extend(pool, [], 5)
        self.check_allocation(pool, [], 5, result, before)

    def test_no_free_pages_but_tail_room(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        pool.free_pages = []
        result = extend(pool, [12, 13, 14], 1)
        self.assertEqual(result, [15])
        self.assertEqual(pool.free_pages, [])

    def test_full_tail_without_free_pages_fails(self):
        pool = PagedTokenToKVPoolAllocator(4, 4)
        pool.free_pages = []
        self.assertIsNone(extend(pool, [12, 13, 14, 15], 1))
        self.assertEqual(pool.free_pages, [])

    def test_insufficient_pages_does_not_partially_allocate(self):
        pool = PagedTokenToKVPoolAllocator(4, 8)
        pool.free_pages = [7]
        before = pool.free_pages.copy()
        result = extend(pool, [12, 13, 14], 6)
        self.assertIsNone(result)
        self.assertEqual(pool.free_pages, before)

    def test_free_deduplicates_same_page_within_call(self):
        pool = PagedTokenToKVPoolAllocator(4, 8)
        pool.free_pages = []
        pool.free([12, 13, 14, 15, 28, 29, 12])
        self.assertEqual(set(pool.free_pages), {3, 7})
        self.assertEqual(len(pool.free_pages), 2)

    def test_free_empty_is_noop(self):
        pool = PagedTokenToKVPoolAllocator(4, 8)
        before = pool.free_pages.copy()
        pool.free([])
        self.assertEqual(pool.free_pages, before)

    def test_allocate_free_reallocate_restores_capacity(self):
        pool = PagedTokenToKVPoolAllocator(4, 3)
        original = set(pool.free_pages)
        slots = extend(pool, [], 9)
        self.assertEqual(pool.free_pages, [])
        pool.free(slots)
        self.assertEqual(len(pool.free_pages), 3)
        self.assertEqual(set(pool.free_pages), original)
        again = extend(pool, [], 12)
        self.assertEqual(len(again), 12)
        self.assertEqual(len(set(again)), 12)


def make_new_request_case(p, count):
    def test(self):
        pool = PagedTokenToKVPoolAllocator(p, 5)
        before = pool.free_pages.copy()
        result = extend(pool, [], count)
        self.check_allocation(pool, [], count, result, before)
    return test


for p in (1, 2, 4, 7, 16):
    for count in sorted({0, 1, p - 1, p, p + 1, 2 * p + 1, 5 * p, 5 * p + 1}):
        setattr(AllocationTests, f'test_new_p{p}_n{count}', make_new_request_case(p, count))


def make_tail_case(p, used, count):
    def test(self):
        pool = PagedTokenToKVPoolAllocator(p, 10)
        old = list(range(3 * p, 3 * p + used))
        pool.free_pages = [7, 8, 9]
        before = pool.free_pages.copy()
        result = extend(pool, old, count)
        self.check_allocation(pool, old, count, result, before)
    return test


for p in (1, 2, 4, 7, 16):
    for used in sorted({1, max(1, p - 1), p}):
        for count in sorted({0, 1, p, p + 1}):
            setattr(AllocationTests, f'test_tail_p{p}_used{used}_n{count}',
                    make_tail_case(p, used, count))


class StatefulAllocationTests(unittest.TestCase):
    check_allocation = AllocationTests.check_allocation


def make_random_case(seed):
    def test(self):
        rng = random.Random(seed)
        pool = PagedTokenToKVPoolAllocator(rng.choice([1, 2, 4, 7, 16]), 9)
        all_pages = set(pool.free_pages)
        requests = {}
        for step in range(100):
            rid = rng.randrange(6)
            if rid in requests and rng.random() < 0.3:
                pool.free(requests.pop(rid))
            else:
                old = requests.get(rid, [])
                count = rng.randrange(0, pool.page_size * 3 + 1)
                before = pool.free_pages.copy()
                result = extend(pool, old, count)
                self.check_allocation(pool, old, count, result, before)
                if result is not None:
                    requests[rid] = old + result
            owners = []
            for slots in requests.values():
                owners.extend({slot // pool.page_size for slot in slots})
            self.assertEqual(len(owners), len(set(owners)), 'Live requests share pages')
            self.assertEqual(len(pool.free_pages), len(set(pool.free_pages)))
            self.assertTrue(set(owners).isdisjoint(pool.free_pages))
            self.assertEqual(set(owners) | set(pool.free_pages), all_pages)
        for slots in requests.values():
            pool.free(slots)
        self.assertEqual(set(pool.free_pages), all_pages)
        self.assertEqual(len(pool.free_pages), len(all_pages))
    return test


for seed in range(10):
    setattr(StatefulAllocationTests, f'test_random_seed_{seed}', make_random_case(seed))


class StorageTests(unittest.TestCase):
    pass


def make_storage_case(device, dtype, heads):
    def test(self):
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        pool = MHATokenToKVPool(3, 32, heads, 8, dtype, torch.device(device))
        expected_k = torch.zeros((3, pool.capacity + pool.page_size, heads, 8), device=device, dtype=dtype)
        expected_v = torch.zeros_like(expected_k)
        # Unwritten memory has no defined value; initialize only for isolation checks.
        pool.k_buffer.zero_()
        pool.v_buffer.zero_()
        # Full-layer getters share storage; selection now belongs to the caller.
        layer_k, layer_v = pool.get_kv_buffer(1)
        self.assertEqual(tuple(layer_k.shape), (pool.capacity + pool.page_size, heads, 8))
        self.assertEqual(layer_k.data_ptr(), pool.k_buffer[1].data_ptr())
        self.assertEqual(layer_v.data_ptr(), pool.v_buffer[1].data_ptr())
        slots = torch.tensor([28, 12, 15, 13], device=device)
        for layer in range(3):
            k = (torch.arange(4 * heads * 8, device=device).reshape(4, heads, 8) % 31 + layer).to(dtype)
            v = -k - 1
            pool.set_kv_buffer(layer, slots, k, v)
            expected_k[layer, slots] = k
            expected_v[layer, slots] = v
            torch.testing.assert_close(pool.get_key_buffer(layer)[slots], k, rtol=0, atol=0)
            torch.testing.assert_close(pool.get_value_buffer(layer)[slots], v, rtol=0, atol=0)
        # Overwrite reused slots; untouched slots and layers must remain intact.
        k = torch.full((1, heads, 8), 42, dtype=dtype, device=device)
        pool.set_kv_buffer(1, slots[:1], k, -k)
        torch.testing.assert_close(layer_k[slots[:1]], k, rtol=0, atol=0)
        torch.testing.assert_close(layer_v[slots[:1]], -k, rtol=0, atol=0)
        expected_k[1, slots[:1]] = k
        expected_v[1, slots[:1]] = -k
        torch.testing.assert_close(pool.k_buffer, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(pool.v_buffer, expected_v, rtol=0, atol=0)
        empty = torch.empty(0, dtype=torch.long, device=device)
        values = torch.empty((0, heads, 8), dtype=dtype, device=device)
        pool.set_kv_buffer(0, empty, values, values)
        self.assertEqual(tuple(pool.get_key_buffer(0)[empty].shape), (0, heads, 8))
    return test


for device in ('cpu', 'cuda'):
    for dtype in (torch.float32, torch.bfloat16):
        for heads in (1, 3, 9):
            setattr(StorageTests, f'test_storage_{device}_{str(dtype).split(".")[-1]}_kvheads{heads}',
                    make_storage_case(device, dtype, heads))


class IntegrationTests(unittest.TestCase):
    def test_two_requests_do_not_overwrite_each_others_kv(self):
        rows = request_pool(2)
        alloc = PagedTokenToKVPoolAllocator(4, 4)
        kv = MHATokenToKVPool(2, 16, 3, 8, torch.float32, torch.device('cpu'), page_size=4)
        a = rows.alloc_rows(1)[0]
        b = rows.alloc_rows(1)[0]
        sa = torch.tensor(extend(alloc, [], 3))
        sb = torch.tensor(extend(alloc, [], 3))
        rows.write((a, slice(0, 3)), sa)
        rows.write((b, slice(0, 3)), sb)
        for layer in range(2):
            va = torch.full((3, 3, 8), 10.0 + layer)
            vb = torch.full((3, 3, 8), 20.0 + layer)
            kv.set_kv_buffer(layer, sa, va, -va)
            kv.set_kv_buffer(layer, sb, vb, -vb)
            torch.testing.assert_close(kv.get_key_buffer(layer)[rows.req_to_token[a, :3]], va)
            torch.testing.assert_close(kv.get_key_buffer(layer)[rows.req_to_token[b, :3]], vb)


if __name__ == '__main__':
    unittest.main()
