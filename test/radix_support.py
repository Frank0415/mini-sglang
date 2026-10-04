"""Test fixtures only. No Radix implementation or mock tree is provided."""

import unittest
from importlib import import_module
from types import SimpleNamespace

import torch

from src.mem_cache.allocator import MHATokenToKVPool, PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.scheduler.request import Request, RequestStatus


def ids(values):
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)


class RadixCase(unittest.TestCase):
    def make(self, page_size=4, pages=16, disable=False):
        rows = ReqToTokenPool(8, 256, torch.device("cpu"))
        allocator = PagedTokenToKVPoolAllocator(page_size, pages)
        pool = MHATokenToKVPool(
            2, page_size * pages, 1, 4, torch.float32, torch.device("cpu"), page_size
        )
        module = import_module("src.mem_cache.radix_cache")
        cls = getattr(module, "RadixCache", None)
        self.assertIsNotNone(
            cls,
            "请先实现 src/mem_cache/radix_cache.py 的 RadixCache；见 docs/RADIX_GUIDE.md",
        )
        cache = cls(
            req_to_token_pool=rows,
            token_to_kv_pool_allocator=allocator,
            disable=disable,
        )
        return SimpleNamespace(
            device="cpu",
            runtime=None,
            radix_cache=cache,
            req_to_token_pool=rows,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=pool,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=2)),
        )

    def allocate(self, runner, count):
        slots = runner.token_to_kv_pool_allocator.alloc_extend(0, count, -1, count)
        self.assertIsNotNone(slots, "测试夹具容量不足")
        return slots

    def insert(self, runner, tokens):
        slots = self.allocate(runner, len(tokens))
        result = runner.radix_cache.insert(tokens, slots)
        # Standalone insert transfers the new aligned suffix, but does not free
        # duplicate inputs or unaligned tails; the caller owns those portions.
        runner.token_to_kv_pool_allocator.free(slots[: result.prefix_len])
        aligned = (
            len(tokens) // runner.radix_cache.page_size * runner.radix_cache.page_size
        )
        runner.token_to_kv_pool_allocator.free(slots[aligned:])
        return slots, result

    def request(self, runner, name, tokens, cached_len=None):
        count = len(tokens) if cached_len is None else cached_len
        req = Request(name, list(tokens), 8)
        req.req_pool_index = runner.req_to_token_pool.alloc_rows(1)[0]
        req.slots = self.allocate(runner, count)
        req.cached_len = count
        req.status = RequestStatus.RUNNING
        runner.req_to_token_pool.write(
            (req.req_pool_index, slice(0, count)), torch.tensor(req.slots)
        )
        return req

    def path(self, node, root):
        result = []
        while node is not root:
            self.assertIsNotNone(node)
            result.append(node)
            node = node.parent
        return result

    def nodes(self, cache):
        result = []
        pending = list(cache.root_node.children.values())
        seen = set()
        while pending:
            node = pending.pop()
            self.assertNotIn(id(node), seen, "树出现环或同一节点有多个父节点")
            seen.add(id(node))
            result.append(node)
            for child in node.children.values():
                self.assertIs(child.parent, node)
                pending.append(child)
        return result

    def account(self, runner, active=()):
        allocator = runner.token_to_kv_pool_allocator
        p = allocator.page_size
        free = set(allocator.free_pages)
        self.assertEqual(len(free), len(allocator.free_pages), "空闲页重复")
        tree = set()
        for node in self.nodes(runner.radix_cache):
            key, value = ids(node.key), ids(node.value)
            self.assertEqual(len(key), len(value))
            self.assertGreater(len(key), 0)
            self.assertEqual(len(key) % p, 0)
            pages = set()
            for start in range(0, len(value), p):
                chunk = value[start : start + p]
                self.assertEqual(chunk[0] % p, 0)
                self.assertEqual(chunk, list(range(chunk[0], chunk[0] + p)))
                page = chunk[0] // p
                self.assertNotIn(page, pages)
                pages.add(page)
            self.assertTrue(tree.isdisjoint(pages), "树内一页被重复持有")
            tree |= pages
        private = set()
        for req in active:
            pages = {slot // p for slot in req.slots[req.cache_protected_len :]}
            self.assertTrue(private.isdisjoint(pages), "私有页被两个请求持有")
            private |= pages
        self.assertTrue(free.isdisjoint(tree | private))
        self.assertTrue(tree.isdisjoint(private))
        self.assertEqual(free | tree | private, set(range(1, allocator.num_pages + 1)))
        self.assertEqual(runner.radix_cache.total_size(), len(tree) * p)
        self.assertEqual(
            runner.radix_cache.total_size(),
            runner.radix_cache.protected_size() + runner.radix_cache.evictable_size(),
        )
