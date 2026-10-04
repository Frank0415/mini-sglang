"""Staged acceptance gates. Implement production code; these tests must not supply it."""

import unittest
from dataclasses import fields

import torch

from src.mem_cache.allocator import MHATokenToKVPool, PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.scheduler.batch import ScheduleBatch
from src.scheduler.request import Request, RequestStatus
from src.scheduler.scheduler import Scheduler

from .radix_support import RadixCase, ids


class Stage00Baseline(unittest.TestCase):
    def test_existing_pools_page_roundtrip(self):
        for p in (1, 4, 16):
            with self.subTest(page_size=p):
                rows = ReqToTokenPool(2, 64, torch.device("cpu"))
                allocator = PagedTokenToKVPoolAllocator(p, 4)
                pool = MHATokenToKVPool(
                    2, 4 * p, 1, 4, torch.float32, torch.device("cpu"), p
                )
                row = rows.alloc_rows(1)[0]
                slots = allocator.alloc_extend(0, p + 1, -1, p + 1)
                indices = torch.tensor(slots)
                rows.write((row, slice(0, p + 1)), indices)
                kv = torch.arange((p + 1) * 4, dtype=torch.float32).reshape(p + 1, 1, 4)
                pool.set_kv_buffer(1, indices, kv, -kv)
                torch.testing.assert_close(pool.get_key_buffer(1)[indices], kv)
                allocator.free(slots)
                rows.free_rows([row])
                self.assertEqual(set(allocator.free_pages), {1, 2, 3, 4})
                self.assertEqual(rows.available_slots(), 2)


class Stage01EmptyTree(RadixCase):
    def test_empty_root_and_miss(self):
        for p in (1, 4, 16):
            with self.subTest(page_size=p):
                r = self.make(page_size=p)
                c = r.radix_cache
                self.assertEqual(c.page_size, p)
                self.assertEqual(ids(c.root_node.key), [])
                self.assertEqual(ids(c.root_node.value), [])
                self.assertIsNone(c.root_node.parent)
                self.assertFalse(c.root_node.children)
                self.assertEqual(c.total_size(), 0)
                self.assertEqual(c.protected_size(), 0)
                self.assertEqual(c.evictable_size(), 0)
                match = c.match_prefix([11] * p)
                self.assertEqual(ids(match.device_indices), [])
                self.assertIs(match.last_device_node, c.root_node)
                self.assertEqual(len(r.token_to_kv_pool_allocator.free_pages), 16)


class Stage02OneBranch(RadixCase):
    def test_insert_and_exact_match_do_not_copy_kv(self):
        r = self.make()
        tokens = list(range(10, 18))
        slots = self.allocate(r, 8)
        result = r.radix_cache.insert(tokens, slots)
        self.assertEqual(result.prefix_len, 0)
        match = r.radix_cache.match_prefix(tokens)
        self.assertEqual(ids(match.device_indices), slots)
        self.assertIs(match.last_device_node, result.last_device_node)
        self.assertEqual(r.radix_cache.total_size(), 8)
        self.assertEqual(r.radix_cache.evictable_size(), 8)
        self.assertEqual(len(r.radix_cache.root_node.children), 1)
        self.assertEqual(ids(result.last_device_node.key), tokens)
        saved = slots.copy()
        slots.clear()
        tokens[0] = 999
        self.assertEqual(
            ids(r.radix_cache.match_prefix(list(range(10, 18))).device_indices), saved
        )
        self.account(r)


class Stage03SplitAndDuplicate(RadixCase):
    def test_branch_splits_at_common_prefix(self):
        r = self.make()
        a = list(range(12))
        b = a[:8] + [50, 51, 52, 53]
        sa, _ = self.insert(r, a)
        sb, result = self.insert(r, b)
        self.assertEqual(result.prefix_len, 8)
        self.assertEqual(ids(r.radix_cache.match_prefix(a).device_indices), sa)
        self.assertEqual(
            ids(r.radix_cache.match_prefix(b).device_indices), sa[:8] + sb[8:]
        )
        root_child = next(iter(r.radix_cache.root_node.children.values()))
        self.assertEqual(ids(root_child.key), a[:8])
        self.assertEqual(len(root_child.children), 2)
        self.assertEqual(r.radix_cache.total_size(), 16)
        self.account(r)

    def test_duplicate_insert_keeps_original_slots(self):
        r = self.make()
        tokens = list(range(8))
        original, _ = self.insert(r, tokens)
        _, result = self.insert(r, tokens)
        self.assertEqual(result.prefix_len, 8)
        self.assertEqual(
            ids(r.radix_cache.match_prefix(tokens).device_indices), original
        )
        self.assertEqual(r.radix_cache.total_size(), 8)
        self.account(r)


class Stage04PageAlignment(RadixCase):
    def test_partial_tail_not_published(self):
        r = self.make()
        tokens = list(range(10))
        slots, _ = self.insert(r, tokens)
        self.assertEqual(
            ids(r.radix_cache.match_prefix(tokens).device_indices), slots[:8]
        )
        self.assertEqual(r.radix_cache.total_size(), 8)
        self.account(r)

    def test_partial_page_match_rounds_down(self):
        for p in (1, 4, 16):
            with self.subTest(page_size=p):
                r = self.make(page_size=p)
                tokens = list(range(2 * p))
                slots, _ = self.insert(r, tokens)
                query = tokens[:p] + tokens[p : p + max(0, p - 1)] + [999]
                self.assertEqual(
                    ids(r.radix_cache.match_prefix(query).device_indices), slots[:p]
                )
                short = tokens[: max(0, p - 1)]
                self.assertEqual(
                    ids(r.radix_cache.match_prefix(short).device_indices), []
                )
                self.account(r)

    def test_query_ends_inside_compressed_edge(self):
        r = self.make()
        slots, _ = self.insert(r, list(range(12)))
        match = r.radix_cache.match_prefix(list(range(4)))
        self.assertEqual(ids(match.device_indices), slots[:4])
        self.assertEqual(
            sum(
                len(n.key)
                for n in self.path(match.last_device_node, r.radix_cache.root_node)
            ),
            4,
        )
        self.account(r)


class Stage05ReferenceLocks(RadixCase):
    def test_shared_ancestors_count_unique_protected_tokens(self):
        r = self.make()
        a = list(range(8))
        b = a[:4] + [20, 21, 22, 23]
        self.insert(r, a)
        self.insert(r, b)
        c = r.radix_cache
        na = c.match_prefix(a).last_device_node
        nb = c.match_prefix(b).last_device_node
        c.inc_lock_ref(na)
        c.inc_lock_ref(nb)
        self.assertEqual(na.lock_ref, 1)
        self.assertEqual(nb.lock_ref, 1)
        self.assertEqual(na.parent.lock_ref, 2)
        self.assertEqual(c.protected_size(), 12)
        self.assertEqual(c.evictable_size(), 0)
        c.dec_lock_ref(na)
        self.assertEqual(c.protected_size(), 8)
        self.assertEqual(c.evictable_size(), 4)
        c.dec_lock_ref(nb)
        self.assertEqual(c.protected_size(), 0)
        self.assertEqual(c.evictable_size(), 12)
        self.account(r)

    def test_split_preserves_existing_node_references(self):
        r = self.make()
        self.insert(r, list(range(8)))
        c = r.radix_cache
        original = c.match_prefix(list(range(8))).last_device_node
        c.inc_lock_ref(original)
        prefix = c.match_prefix(list(range(4))).last_device_node
        self.assertIs(c.match_prefix(list(range(8))).last_device_node, original)
        self.assertIs(original.parent, prefix)
        self.assertEqual(prefix.lock_ref, 1)
        c.inc_lock_ref(prefix)
        self.assertEqual(prefix.lock_ref, 2)
        c.dec_lock_ref(original)
        self.assertEqual(prefix.lock_ref, 1)
        self.assertEqual(c.protected_size(), 4)
        c.dec_lock_ref(prefix)
        self.account(r)


class Stage06LeafEviction(RadixCase):
    def test_lru_leaf_then_parent(self):
        r = self.make()
        a = list(range(8))
        b = a[:4] + [20, 21, 22, 23]
        self.insert(r, a)
        self.insert(r, b)
        c = r.radix_cache
        c.match_prefix(a)  # A was accessed most recently; B is the first victim.
        self.assertEqual(c.evict(4), 4)
        self.assertEqual(len(ids(c.match_prefix(a).device_indices)), 8)
        self.assertEqual(len(ids(c.match_prefix(b).device_indices)), 4)
        self.assertEqual(c.total_size(), 8)
        self.assertEqual(c.evict(100), 8)
        self.assertEqual(c.total_size(), 0)
        self.account(r)

    def test_locked_path_survives_pressure(self):
        r = self.make()
        a = list(range(8))
        b = a[:4] + [20, 21, 22, 23]
        self.insert(r, a)
        self.insert(r, b)
        c = r.radix_cache
        node = c.match_prefix(a).last_device_node
        c.inc_lock_ref(node)
        self.assertEqual(c.evict(100), 4)
        self.assertEqual(len(ids(c.match_prefix(a).device_indices)), 8)
        self.assertEqual(c.evict(100), 0)
        c.dec_lock_ref(node)
        self.assertEqual(c.evict(100), 8)
        self.account(r)

    def test_zero_target_does_not_evict(self):
        r = self.make()
        self.insert(r, list(range(8)))
        self.assertEqual(r.radix_cache.evict(0), 0)
        self.assertEqual(r.radix_cache.total_size(), 8)
        self.account(r)


class Stage07RequestOwnership(unittest.TestCase):
    def test_request_has_separate_shared_prefix_state(self):
        names = {f.name for f in fields(Request)}
        self.assertTrue(
            {"cache_protected_len", "last_node"} <= names,
            "Request 需要 cache_protected_len=0 和 last_node=None",
        )
        req = Request("r", [1, 2, 3], 4)
        self.assertEqual(req.cache_protected_len, 0)
        self.assertIsNone(req.last_node)
        req.slots = [4, 5, 6, 7]
        req.cached_len = 3
        self.assertEqual(len(req.slots), 4)
        self.assertEqual(req.cached_len, 3)
        self.assertEqual(req.cache_protected_len, 0)


class Stage08FinishedRequest(RadixCase):
    def test_finished_keeps_prompt_when_output_suffix_is_evicted(self):
        r = self.make()
        req = self.request(r, "r", list(range(12)))
        req.input_ids = list(range(8))
        req.output_ids = [8, 9, 10, 11, 99]
        r.radix_cache.cache_finished_req(req)
        prompt = next(iter(r.radix_cache.root_node.children.values()))
        self.assertEqual(ids(prompt.key), list(range(8)))
        self.assertEqual(len(prompt.children), 1)
        self.assertEqual(r.radix_cache.evict(4), 4)
        self.assertEqual(
            len(ids(r.radix_cache.match_prefix(list(range(12))).device_indices)), 8
        )
        self.account(r)

    def test_finished_keeps_complete_pages_and_frees_tail(self):
        r = self.make()
        req = self.request(r, "r", list(range(10)))
        saved = req.slots.copy()
        req.output_ids = [99]  # sampled, no KV yet
        row = req.req_pool_index
        r.radix_cache.cache_finished_req(req)
        self.assertEqual(
            ids(r.radix_cache.match_prefix(list(range(10)) + [99]).device_indices),
            saved[:8],
        )
        self.assertEqual(r.radix_cache.total_size(), 8)
        self.assertEqual(r.radix_cache.protected_size(), 0)
        self.assertEqual(
            r.req_to_token_pool.available_slots(), 7
        )  # Scheduler owns row release.
        self.assertEqual(req.req_pool_index, row)
        req.slots.clear()  # Tree must own an independent index copy.
        self.assertEqual(
            len(ids(r.radix_cache.match_prefix(list(range(10))).device_indices)), 8
        )
        self.account(r)

    def test_less_than_a_page_is_freed(self):
        r = self.make()
        req = self.request(r, "r", [10, 11, 12])
        r.radix_cache.cache_finished_req(req)
        self.assertEqual(r.radix_cache.total_size(), 0)
        self.account(r)


class Stage09FinishedDuplicates(RadixCase):
    def test_two_finished_requests_keep_one_prefix_copy(self):
        r = self.make()
        tokens = list(range(8))
        a = self.request(r, "a", tokens)
        b = self.request(r, "b", tokens)
        sa = a.slots.copy()
        sb = b.slots.copy()
        r.radix_cache.cache_finished_req(a)
        r.radix_cache.cache_finished_req(b)
        self.assertEqual(ids(r.radix_cache.match_prefix(tokens).device_indices), sa)
        self.assertTrue(
            {x // 4 for x in sb} <= set(r.token_to_kv_pool_allocator.free_pages)
        )
        self.assertEqual(r.radix_cache.total_size(), 8)
        self.account(r)

    def test_duplicates_free_only_the_private_overlap(self):
        r = self.make()
        tokens = list(range(12))
        canonical, _ = self.insert(r, tokens[:8])
        req = Request("r", tokens, 2)
        match = r.radix_cache.match_prefix(tokens[:4])
        r.radix_cache.inc_lock_ref(match.last_device_node)
        req.last_node = match.last_device_node
        req.cache_protected_len = 4
        req.req_pool_index = r.req_to_token_pool.alloc_rows(1)[0]
        private = r.token_to_kv_pool_allocator.alloc_extend(4, 12, canonical[3], 8)
        req.slots = canonical[:4] + private
        req.cached_len = 12
        r.req_to_token_pool.write(
            (req.req_pool_index, slice(0, 12)), torch.tensor(req.slots)
        )
        r.radix_cache.cache_finished_req(req)
        self.assertEqual(
            ids(r.radix_cache.match_prefix(tokens).device_indices),
            canonical + private[4:],
        )
        self.assertEqual(r.radix_cache.protected_size(), 0)
        self.account(r)


class Stage10UnfinishedRequest(RadixCase):
    def test_publish_deduplicate_rebind_and_extend(self):
        r = self.make()
        a = self.request(r, "a", list(range(6)))
        b = self.request(r, "b", list(range(6)))
        original = a.slots.copy()
        b_tail = b.slots[4:].copy()
        c = r.radix_cache
        c.cache_unfinished_req(a)
        c.cache_unfinished_req(b)
        self.assertEqual(a.cache_protected_len, 4)
        self.assertEqual(b.cache_protected_len, 4)
        self.assertEqual(b.slots, original[:4] + b_tail)
        self.assertEqual(
            r.req_to_token_pool.req_to_token[b.req_pool_index, :6].tolist(), b.slots
        )
        self.assertIs(a.last_node, b.last_node)
        self.assertEqual(a.last_node.lock_ref, 2)
        self.account(r, [a, b])
        before = set(r.token_to_kv_pool_allocator.free_pages)
        c.cache_unfinished_req(b)
        self.assertEqual(set(r.token_to_kv_pool_allocator.free_pages), before)
        self.assertEqual(b.last_node.lock_ref, 2)
        more = r.token_to_kv_pool_allocator.alloc_extend(6, 8, b.slots[-1], 2)
        b.slots.extend(more)
        b.input_ids.extend([6, 7])
        b.cached_len = 8
        r.req_to_token_pool.write((b.req_pool_index, slice(6, 8)), torch.tensor(more))
        c.cache_unfinished_req(b)
        self.assertEqual(b.cache_protected_len, 8)
        self.assertEqual(c.protected_size(), 8)
        self.account(r, [a, b])
        c.cache_finished_req(a)
        c.cache_finished_req(b)
        self.assertEqual(c.protected_size(), 0)
        self.account(r)


class Stage11SchedulerAdmission(RadixCase):
    def test_match_before_prefill_and_keep_original_positions(self):
        r = self.make()
        prefix = list(range(8))
        slots, _ = self.insert(r, prefix)
        s = Scheduler(r)
        req = Request("r", prefix + [20, 21], 2)
        s.add_request(req)
        selected = s.get_next_requests()
        self.assertEqual(req.cached_len, 8)
        self.assertEqual(req.cache_protected_len, 8)
        self.assertEqual(req.slots, slots)
        self.assertEqual(r.radix_cache.protected_size(), 8)
        self.assertEqual(s.get_next_requests(), selected)
        self.assertEqual(req.last_node.lock_ref, 1)
        batch = ScheduleBatch.prepare(selected, r)
        self.assertEqual(batch.input_ids.tolist(), [20, 21])
        self.assertEqual(batch.position_ids.tolist(), [8, 9])
        self.assertEqual(batch.context.batch.extend_lens_cpu, (2,))
        self.assertEqual(
            r.req_to_token_pool.req_to_token[req.req_pool_index, :8].tolist(), slots
        )
        self.account(r, [req])

    def test_full_hit_recomputes_last_page_without_overwriting_shared_page(self):
        r = self.make()
        tokens = list(range(8))
        old, _ = self.insert(r, tokens)
        s = Scheduler(r)
        req = Request("r", tokens, 1)
        s.add_request(req)
        batch = ScheduleBatch.prepare(s.get_next_requests(), r)
        self.assertEqual(req.cached_len, 4)
        self.assertEqual(req.cache_protected_len, 4)
        self.assertEqual(batch.input_ids.tolist(), tokens[4:])
        self.assertEqual(batch.position_ids.tolist(), [4, 5, 6, 7])
        self.assertTrue(set(req.slots[4:]).isdisjoint(old))
        self.account(r, [req])


class Stage12SchedulerRelease(RadixCase):
    def test_finish_releases_row_but_keeps_tree_pages(self):
        r = self.make()
        s = Scheduler(r)
        tokens = list(range(10))
        req = Request("r", tokens, 1)
        s.add_request(req)
        batch = ScheduleBatch.prepare(s.get_next_requests(), r)
        s.process_result(batch, [99])
        self.assertEqual(req.status, RequestStatus.FINISHED)
        self.assertEqual(req.output_ids, [99])
        self.assertEqual(req.slots, [])
        self.assertIsNone(req.req_pool_index)
        self.assertEqual(r.req_to_token_pool.available_slots(), 8)
        self.assertEqual(
            len(ids(r.radix_cache.match_prefix(tokens + [99]).device_indices)), 8
        )
        self.assertEqual(r.radix_cache.protected_size(), 0)
        self.assertIs(s.return_request_result("r"), req)
        self.assertFalse(s.requests)
        self.account(r)

    def test_abort_discards_uncommitted_suffix_and_unlocks_shared_prefix(self):
        r = self.make()
        tokens = list(range(12))
        old, _ = self.insert(r, tokens[:8])
        s = Scheduler(r)
        req = Request("r", tokens, 4)
        s.add_request(req)
        ScheduleBatch.prepare(s.get_next_requests(), r)
        self.assertEqual(req.cached_len, 8)
        self.assertEqual(len(req.slots), 12)
        s.abort("r")
        s.abort("r")
        self.assertEqual(req.status, RequestStatus.ABORTED)
        self.assertEqual(r.req_to_token_pool.available_slots(), 8)
        self.assertEqual(ids(r.radix_cache.match_prefix(tokens).device_indices), old)
        self.assertEqual(r.radix_cache.protected_size(), 0)
        self.account(r)

    def test_failed_prefill_does_not_publish_uncomputed_kv(self):
        r = self.make()
        s = Scheduler(r)
        req = Request("r", list(range(8)), 2)
        s.add_request(req)
        ScheduleBatch.prepare(s.get_next_requests(), r)
        self.assertEqual(req.cached_len, 0)
        s.abort("r")
        self.assertEqual(r.radix_cache.total_size(), 0)
        self.account(r)

    def test_successful_prefill_publishes_prefix_for_other_requests(self):
        r = self.make()
        s = Scheduler(r)
        req = Request("r", list(range(8)), 4)
        s.add_request(req)
        batch = ScheduleBatch.prepare(s.get_next_requests(), r)
        s.process_result(batch, [99])
        self.assertEqual(req.status, RequestStatus.RUNNING)
        self.assertEqual(req.cached_len, 8)
        self.assertEqual(req.cache_protected_len, 8)
        self.assertEqual(
            len(ids(r.radix_cache.match_prefix(list(range(8))).device_indices)), 8
        )
        self.assertEqual(r.radix_cache.protected_size(), 8)
        s.abort("r")
        self.assertEqual(r.radix_cache.protected_size(), 0)
        self.account(r)


class Stage13CapacityEviction(RadixCase):
    def test_no_pressure_does_not_evict(self):
        r = self.make(pages=4)
        old, _ = self.insert(r, [10, 11, 12, 13])
        s = Scheduler(r)
        req = Request("r", [20, 21, 22, 23], 1)
        s.add_request(req)
        ScheduleBatch.prepare(s.get_next_requests(), r)
        self.assertEqual(
            ids(r.radix_cache.match_prefix([10, 11, 12, 13]).device_indices), old
        )
        s.abort("r")
        self.account(r)

    def test_evict_before_allocating_new_request(self):
        r = self.make(pages=3)
        self.insert(r, list(range(8)))
        s = Scheduler(r)
        req = Request("r", list(range(20, 28)), 1)
        s.add_request(req)
        selected = s.get_next_requests()
        self.assertEqual(selected, [req])
        batch = ScheduleBatch.prepare(selected, r)
        self.assertEqual(batch.input_len, 8)
        self.assertLess(r.radix_cache.total_size(), 8)
        s.abort("r")
        self.account(r)

    def test_all_locked_keeps_waiting_request_retryable(self):
        r = self.make(pages=2)
        self.insert(r, list(range(8)))
        c = r.radix_cache
        node = c.match_prefix(list(range(8))).last_device_node
        c.inc_lock_ref(node)
        s = Scheduler(r)
        req = Request("r", [20, 21, 22, 23], 1)
        s.add_request(req)
        self.assertEqual(s.get_next_requests(), [])
        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertIsNone(req.req_pool_index)
        self.assertEqual(req.slots, [])
        self.assertEqual(c.total_size(), 8)
        self.assertEqual(c.protected_size(), 8)
        c.dec_lock_ref(node)
        self.assertEqual(s.get_next_requests(), [req])
        ScheduleBatch.prepare([req], r)
        s.abort("r")
        self.account(r)

    def test_match_is_protected_before_eviction(self):
        r = self.make(pages=2)
        old, _ = self.insert(r, list(range(8)))
        s = Scheduler(r)
        req = Request("r", [0, 1, 2, 3, 20, 21, 22, 23], 1)
        s.add_request(req)
        selected = s.get_next_requests()
        self.assertEqual(selected, [req])
        batch = ScheduleBatch.prepare(selected, r)
        self.assertEqual(req.slots[:4], old[:4])
        self.assertEqual(req.cached_len, 4)
        self.assertEqual(batch.input_ids.tolist(), [20, 21, 22, 23])
        self.assertEqual(r.radix_cache.protected_size(), 4)
        s.abort("r")
        self.account(r)


class Stage14ResetAndDisable(RadixCase):
    def test_reset_frees_cache_and_removes_old_matches(self):
        r = self.make()
        self.insert(r, list(range(12)))
        r.radix_cache.reset()
        self.assertEqual(
            ids(r.radix_cache.match_prefix(list(range(12))).device_indices), []
        )
        self.assertEqual(r.radix_cache.total_size(), 0)
        self.account(r)
        r.radix_cache.reset()
        self.account(r)

    def test_reset_rejects_live_references(self):
        r = self.make()
        self.insert(r, list(range(8)))
        c = r.radix_cache
        node = c.match_prefix(list(range(8))).last_device_node
        c.inc_lock_ref(node)
        before = set(r.token_to_kv_pool_allocator.free_pages)
        with self.assertRaises(ValueError):
            c.reset()
        self.assertEqual(set(r.token_to_kv_pool_allocator.free_pages), before)
        self.assertEqual(c.total_size(), 8)
        c.dec_lock_ref(node)
        c.reset()
        self.account(r)

    def test_disabled_cache_retains_no_pages(self):
        r = self.make(disable=True)
        req = self.request(r, "r", list(range(8)))
        r.radix_cache.cache_finished_req(req)
        self.assertEqual(r.radix_cache.total_size(), 0)
        self.assertEqual(
            ids(r.radix_cache.match_prefix(req.input_ids).device_indices), []
        )
        self.account(r)


class Stage15EngineLifecycle(RadixCase):
    def engine(self, enabled=True):
        from unittest.mock import patch

        from src.engine import Engine

        r = self.make(disable=not enabled)
        calls = []

        def forward(input_ids, position_ids=None, context=None):
            calls.append(
                {
                    "tokens": input_ids.tolist(),
                    "positions": position_ids.tolist(),
                    "lengths": context.batch.extend_lens_cpu,
                }
            )
            # CPU fixture writes distinguishable KV to real slots; this is a
            # scheduler/ownership test. Stage 16 executes the real model.
            locations = context.batch.cache_batches[0].out_cache_loc
            values = input_ids.float().reshape(-1, 1, 1).expand(-1, 1, 4)
            for layer in range(2):
                r.token_to_kv_pool.set_kv_buffer(
                    layer, locations, values + layer, -values - layer
                )
            logits = torch.zeros(input_ids.numel(), 128)
            logits[:, 99] = 1
            return logits

        r.forward = forward

        def factory(model_dir, device="cpu", *, enable_radix_cache=True):
            self.assertEqual(enable_radix_cache, enabled)
            return r

        with patch("src.engine.ModelRunner", side_effect=factory):
            engine = Engine(
                "unused", device="cpu", max_batch_size=3, enable_radix_cache=enabled
            )
        return engine, r, calls

    def test_two_turns_use_new_requests_but_reuse_prefix(self):
        e, r, calls = self.engine()
        a = e.generate(list(range(8)), 1)
        self.assertEqual(a.output_ids, [99])
        self.assertEqual(r.radix_cache.total_size(), 8)
        b = e.generate(list(range(8)) + [99, 30, 31], 1)
        self.assertNotEqual(a.request_id, b.request_id)
        self.assertEqual(calls[-1]["tokens"], [99, 30, 31])
        self.assertEqual(calls[-1]["positions"], [8, 9, 10])
        self.assertFalse(e.scheduler.requests)
        self.assertFalse(e.scheduler.running)
        self.assertEqual(r.req_to_token_pool.available_slots(), 8)
        e.flush_cache()
        self.account(r)
        self.assertEqual(r.radix_cache.total_size(), 0)

    def test_three_requests_and_idle_flush(self):
        e, r, calls = self.engine()
        inputs = [list(range(8)) + [i] for i in (30, 31, 32)]
        results = e.generate_batch(inputs, 1)
        self.assertEqual([x.input_ids for x in results], inputs)
        self.assertEqual(calls[0]["lengths"], (9, 9, 9))
        self.assertEqual(r.radix_cache.total_size(), 8)
        self.assertEqual(r.radix_cache.protected_size(), 0)
        self.account(r)
        e.flush_cache()
        self.assertEqual(r.radix_cache.total_size(), 0)
        self.account(r)

    def test_flush_requires_idle_engine_and_cache_off_still_generates(self):
        e, r, _ = self.engine()
        rid = e.add_request(list(range(8)), 2)
        with self.assertRaises(ValueError):
            e.flush_cache()
        e.abort(rid)
        e.flush_cache()
        self.account(r)
        off, runner, calls = self.engine(enabled=False)
        for _ in range(2):
            off.generate(list(range(8)), 1)
        self.assertEqual([len(c["tokens"]) for c in calls], [8, 8])
        self.assertEqual(runner.radix_cache.total_size(), 0)
        self.account(runner)


class Stage16RealModel(RadixCase):
    def test_three_conversations_two_turns(self):
        from contextlib import ExitStack
        from pathlib import Path
        from unittest.mock import patch

        from transformers import AutoTokenizer

        from scripts.diagnose_kv_numerics import control
        from src.engine import Engine

        if not torch.cuda.is_available():
            self.skipTest("需要 CUDA；本步骤尚未通过")
        root = Path(__file__).resolve().parents[2]
        models = list((root / "models").rglob("config.json"))
        if not models:
            self.skipTest("缺少本地 SmolLM2 权重；本步骤尚未通过")
        model_dir = models[0].parent
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        enabled = Engine(model_dir, max_batch_size=3, enable_radix_cache=True)
        disabled = Engine(model_dir, max_batch_size=3, enable_radix_cache=False)
        histories = [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer briefly.",
                }
            ]
            for _ in range(3)
        ]
        turns = [
            [
                "My name is Mira. Say hello.",
                "I am visiting Kyoto. Suggest one place.",
                "I am learning Python. Suggest one topic.",
            ],
            [
                "What is my name?",
                "What city am I visiting?",
                "What language am I learning?",
            ],
        ]
        try:
            with ExitStack() as stack:
                stack.enter_context(control(enabled.runner, "bf16_canonical"))
                stack.enter_context(control(disabled.runner, "bf16_canonical"))
                for turn, questions in enumerate(turns):
                    inputs = []
                    for history, question in zip(histories, questions):
                        history.append({"role": "user", "content": question})
                        inputs.append(
                            tokenizer.apply_chat_template(
                                history,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=False,
                            )
                        )
                    lengths = []
                    original = enabled.runner.forward

                    def record(
                        input_ids,
                        position_ids=None,
                        context=None,
                        _lengths=lengths,
                        _forward=original,
                    ):
                        _lengths.append(context.batch.extend_lens_cpu)
                        return _forward(input_ids, position_ids, context)

                    with patch.object(enabled.runner, "forward", side_effect=record):
                        actual = enabled.generate_batch(inputs, 8)
                    expected = disabled.generate_batch(inputs, 8)
                    self.assertEqual(
                        [r.output_ids for r in actual], [r.output_ids for r in expected]
                    )
                    if turn == 1:
                        self.assertTrue(
                            all(
                                new < len(prompt)
                                for new, prompt in zip(lengths[0], inputs)
                            ),
                            f"第二轮没有减少 prefill: {lengths[0]}",
                        )
                    self.assertFalse(enabled.scheduler.requests)
                    self.assertFalse(enabled.scheduler.running)
                    self.assertEqual(
                        enabled.runner.req_to_token_pool.available_slots(), 4
                    )
                    self.assertEqual(enabled.runner.radix_cache.protected_size(), 0)
                    for history, result in zip(histories, actual):
                        history.append(
                            {
                                "role": "assistant",
                                "content": tokenizer.decode(
                                    result.output_ids, skip_special_tokens=True
                                ),
                            }
                        )
                    print(
                        f"轮次 {turn + 1}: 完整输入={[len(x) for x in inputs]}, 实际 prefill={lengths[0]}",
                        flush=True,
                    )
                self.assertGreater(enabled.runner.radix_cache.total_size(), 0)
                enabled.flush_cache()
                self.assertEqual(enabled.runner.radix_cache.total_size(), 0)
                self.assertEqual(
                    len(enabled.runner.token_to_kv_pool_allocator.free_pages), 512
                )
        finally:
            for engine in (enabled, disabled):
                for rid, req in list(engine.scheduler.requests.items()):
                    if req.status in (RequestStatus.WAITING, RequestStatus.RUNNING):
                        engine.abort(rid)
