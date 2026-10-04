import random
import unittest

import torch

from src.mem_cache.allocator import PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.mem_cache.radix_cache import RadixCache
from src.scheduler.request import Request


class RadixCurrentTests(unittest.TestCase):
    def make(self, p=4):
        self.p = p
        self.rows = ReqToTokenPool(8, 256, "cpu")
        self.alloc = PagedTokenToKVPoolAllocator(p, 128)
        self.cache = RadixCache(p, self.rows, self.alloc)

    def allocate(self, n):
        slots = self.alloc.alloc_extend(0, n, -1, n)
        self.assertIsNotNone(slots)
        return slots

    def insert(self, tokens):
        slots = self.allocate(len(tokens))
        values = torch.tensor(slots, dtype=torch.int64)
        duplicate, node = self.cache._insert(tokens, values)
        self.alloc.free(slots[:duplicate])
        self.alloc.free(slots[len(tokens) // self.p * self.p :])
        return slots, node, values

    def request(self, tokens):
        row = self.rows.alloc_rows(1)[0]
        slots = self.allocate(len(tokens))
        req = Request(str(row), tokens.copy(), 8, cached_len=len(tokens),
                      slots=slots, req_pool_index=row)
        self.rows.write((row, slice(0, len(tokens))), torch.tensor(slots))
        return req

    def account(self, active=()):
        free = set(self.alloc.free_pages)
        self.assertEqual(len(free), len(self.alloc.free_pages))
        tree, seen, leaves = set(), set(), set()
        pending = list(self.cache.root_node.children.values())
        while pending:
            node = pending.pop()
            self.assertNotIn(node, seen)
            seen.add(node)
            self.assertGreater(len(node.key), 0)
            self.assertEqual(len(node.key) % self.p, 0)
            self.assertEqual(len(node.key), node.value.numel())
            self.assertGreaterEqual(node.lock_ref, 0)
            values = node.value.tolist()
            for i in range(0, len(values), self.p):
                page = values[i] // self.p
                self.assertEqual(values[i:i+self.p],
                                 list(range(page*self.p, (page+1)*self.p)))
                self.assertNotIn(page, tree)
                tree.add(page)
            if not node.children and node.lock_ref == 0:
                leaves.add(node)
            for key, child in node.children.items():
                self.assertIs(child.parent, node)
                self.assertEqual(key, tuple(child.key[:self.p]))
                pending.append(child)
        self.assertEqual(leaves, self.cache.evictable_nodes)
        private = set()
        for req in active:
            pages = {s // self.p for s in req.slots[req.cache_protected_len:]}
            self.assertTrue(private.isdisjoint(pages))
            private |= pages
        self.assertTrue(free.isdisjoint(tree | private))
        self.assertTrue(tree.isdisjoint(private))
        self.assertEqual(free | tree | private, set(range(1, 129)))

    def test_empty_alignment_clone_and_duplicates(self):
        for p in (1, 4, 16):
            with self.subTest(page_size=p):
                self.make(p)
                idx, node = self.cache._match_prefix([999])
                self.assertEqual(idx.numel(), 0)
                self.assertIs(node, self.cache.root_node)
                tokens = list(range(3*p + (p > 1)))
                slots, _, values = self.insert(tokens)
                values.fill_(9999)
                self.insert(tokens)
                self.assertEqual(self.cache._match_prefix(tokens)[0].tolist(), slots[:3*p])
                self.account()

    def test_split_locked_identity_and_eviction(self):
        for p in (1, 4, 16):
            with self.subTest(page_size=p):
                self.make(p)
                tokens = list(range(3*p))
                slots, tail, _ = self.insert(tokens)
                self.cache._inc_lock_ref(tail)
                _, middle = self.cache._match_prefix(tokens[:p])
                self.assertIs(tail.parent, middle)
                self.assertEqual(middle.lock_ref, 1)
                self.assertEqual(self.cache._match_prefix(tokens)[0].tolist(), slots)
                self.assertEqual(self.cache._evict(10000), 0)
                self.account()
                self.cache._dec_lock_ref(tail)
                self.assertEqual(self.cache._evict(10000), 3*p)
                self.account()

    def test_lru_touch_and_parent_cascade(self):
        self.make()
        self.insert(list(range(8)))
        self.insert(list(range(20, 28)))
        self.cache._match_prefix(list(range(8)))
        self.assertEqual(self.cache._evict(0), 0)
        self.assertEqual(self.cache._evict(4), 8)
        self.assertEqual(self.cache._match_prefix(list(range(20, 28)))[0].numel(), 0)
        self.assertEqual(self.cache._match_prefix(list(range(8)))[0].numel(), 8)
        self.account()

    def test_unfinished_dedup_repeat_finish_and_abort(self):
        for p in (1, 4, 16):
            with self.subTest(page_size=p):
                self.make(p)
                tokens = list(range(2*p + (p > 1)))
                a, b = self.request(tokens), self.request(tokens)
                self.account([a, b])
                for req in (a, b, b):
                    self.cache.cache_unfinished_requests(req)
                    self.account([a, b])
                self.assertEqual(a.last_node.lock_ref, 2)
                self.assertEqual(a.slots[:2*p], b.slots[:2*p])
                self.assertEqual(self.rows.req_to_token[b.req_pool_index, :len(tokens)].tolist(), b.slots)
                self.cache.cache_finished_requests(a)
                self.account([b])
                self.cache.cache_finished_requests(b, is_insert=False)
                self.account()
                self.cache._evict(10000)
                self.account()

    def test_finished_dedup_and_prompt_boundary(self):
        self.make()
        a = self.request(list(range(14)))
        a.input_ids = list(range(8))
        a.output_ids = list(range(8, 15))
        b = self.request(list(range(14)))
        self.cache.cache_finished_requests(a)
        self.cache.cache_finished_requests(b)
        self.account()
        self.assertEqual(self.cache._evict(4), 4)
        self.assertEqual(self.cache._match_prefix(list(range(14)))[0].numel(), 8)
        self.account()

    def test_cross_turn_manual_admission_and_growth(self):
        self.make()
        a = self.request(list(range(8)))
        original = a.slots.copy()
        self.cache.cache_finished_requests(a)
        self.rows.free_rows([a.req_pool_index])
        b = Request("next", list(range(10)), 8)
        indices, node = self.cache._match_prefix(b.input_ids[:-1])
        self.cache._inc_lock_ref(node)
        b.slots = indices.tolist()
        b.cached_len = b.cache_protected_len = indices.numel()
        b.last_node = node
        b.req_pool_index = self.rows.alloc_rows(1)[0]
        self.assertEqual(b.slots, original)
        b.slots.extend(self.alloc.alloc_extend(8, 10, b.slots[-1], 2))
        b.cached_len = 10
        self.rows.write((b.req_pool_index, slice(0, 10)), torch.tensor(b.slots))
        self.cache.cache_unfinished_requests(b)
        self.account([b])
        b.output_ids = [10, 11, 12]
        b.slots.extend(self.alloc.alloc_extend(10, 12, b.slots[-1], 2))
        b.cached_len = 12
        self.cache.cache_unfinished_requests(b)
        self.assertEqual(b.cache_protected_len, 12)
        self.account([b])
        self.cache.cache_finished_requests(b)
        self.account()

    def test_abort_uncommitted_slots(self):
        self.make()
        req = self.request(list(range(6)))
        self.cache.cache_unfinished_requests(req)
        req.slots.extend(self.alloc.alloc_extend(6, 10, req.slots[-1], 4))
        self.cache.cache_finished_requests(req, is_insert=False)
        self.assertEqual(self.cache._match_prefix(list(range(10)))[0].numel(), 4)
        self.account()

    def test_random_prefix_oracle_and_page_accounting(self):
        for p in (1, 4, 16):
            with self.subTest(page_size=p):
                self.make(p)
                rng = random.Random(731)
                known = {}
                for step in range(100):
                    if step % 10 == 9:
                        self.cache._evict(100000)
                        known.clear()
                    tokens = [v for _ in range(rng.randint(1, 5))
                              for v in [rng.randrange(4)]*p]
                    slots, _, _ = self.insert(tokens)
                    for n in range(p, len(tokens)+1, p):
                        known.setdefault(tuple(tokens[:n]), slots[n-p:n])
                    for query in (tokens, [rng.randrange(4)]*p + tokens):
                        expected = []
                        for n in range(p, len(query)+1, p):
                            value = known.get(tuple(query[:n]))
                            if value is None:
                                break
                            expected.extend(value)
                        self.assertEqual(self.cache._match_prefix(query)[0].tolist(), expected)
                    self.account()


if __name__ == "__main__":
    unittest.main()
