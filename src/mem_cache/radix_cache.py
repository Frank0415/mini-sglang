from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .allocator import PagedTokenToKVPoolAllocator
from .memory_pool import ReqToTokenPool
from ..scheduler.request import Request
from time import monotonic_ns
import heapq

@dataclass(eq=False)
class TreeNode:
    node_id: int
    key: list[int]
    value: torch.Tensor

    parent: TreeNode | None = None
    # We only use the 1st page of any node's key to find the child node
    children: dict[tuple[int, ...], TreeNode] = field(default_factory=dict)

    lock_ref: int = 0
    last_accessed: int = 0


class RadixCache:
    root_node: TreeNode
    page_size: int
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: PagedTokenToKVPoolAllocator
    next_node_id: int = 0

    def __init__(
        self,
        page_size: int,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: PagedTokenToKVPoolAllocator,
    ) -> None:
        self.root_node = TreeNode(node_id=0, key=(), value=torch.tensor([]))
        self.page_size = token_to_kv_pool_allocator.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.evictable_nodes: set[TreeNode] = set()

    def _insert(self, key: list[int], value: torch.Tensor) -> tuple[int, TreeNode]:
        aligned_len = len(key) // self.page_size * self.page_size
        key = key[:aligned_len]
        value = value[:aligned_len]

        indices, parent = self._match_prefix(key)
        prefix_len = indices.numel()

        if prefix_len == aligned_len:
            return prefix_len, parent

        child = TreeNode(
            node_id=self.next_node_id,
            key=key[prefix_len:].copy(),
            value=value[prefix_len:].clone(),
            parent=parent,
            last_accessed=monotonic_ns(),
        )
        self.next_node_id += 1

        parent.children[
            tuple(child.key[: self.page_size])
        ] = child

        self._refresh_leaf(parent)
        self._refresh_leaf(child)

        return prefix_len, child

    def _match_prefix(self, prefix: list[int]) -> tuple[torch.Tensor, TreeNode]:
        max_aligned_len = len(prefix) // self.page_size * self.page_size
        prefix = prefix[:max_aligned_len]

        node = self.root_node
        matched_values = []
        offset = 0
        accessed_at = monotonic_ns()
        node.last_accessed = accessed_at

        while offset < len(prefix):
            child_key = tuple(prefix[offset : offset + self.page_size])
            child = node.children.get(child_key)
            if child is None:
                break

            limit = min(len(child.key), len(prefix) - offset)
            matched_len = 0

            while matched_len < limit:
                query_page = prefix[
                    offset + matched_len: offset + matched_len + self.page_size
                ]
                child_page = child.key[matched_len : matched_len + self.page_size]
                if query_page != child_page:
                    break
                matched_len += self.page_size
            if matched_len < len(child.key):
                child = self._split_node(child, matched_len)

            child.last_accessed = accessed_at
            matched_values.append(child.value)
            offset += matched_len
            node = child

        if matched_values:
            indices = torch.cat(matched_values)
        else:
            indices = torch.empty(
                0, dtype=torch.int64, device=self.req_to_token_pool.req_to_token.device
            )

        return indices, node

    def _split_node(self, child_node: TreeNode, length: int) -> TreeNode:
        parent = child_node.parent
        old_child_key = tuple(child_node.key[:self.page_size])

        middle = TreeNode(
            node_id=self.next_node_id,
            key=child_node.key[:length],
            value=child_node.value[:length].clone(),
            parent=parent,
            children={},
            lock_ref=child_node.lock_ref,
            last_accessed=child_node.last_accessed,
        )
        self.next_node_id += 1

        child_node.key = child_node.key[length:]
        child_node.value = child_node.value[length:].clone()
        child_node.parent = middle

        middle.children[tuple(child_node.key[:self.page_size])] = child_node
        parent.children[old_child_key] = middle

        self._refresh_leaf(middle)
        self._refresh_leaf(child_node)

        return middle

    def _inc_lock_ref(self, node: TreeNode):
        while node is not None and node is not self.root_node:
            node.lock_ref += 1
            self._refresh_leaf(node)
            node = node.parent


    def _dec_lock_ref(self, node: TreeNode):
        while node is not None and node is not self.root_node:
            node.lock_ref -= 1
            self._refresh_leaf(node)
            node = node.parent

    def _refresh_leaf(self, node: TreeNode) -> None:
        if(
            node is not self.root_node and not node.children and node.lock_ref == 0
        ):
            self.evictable_nodes.add(node)
        else:
            self.evictable_nodes.discard(node)

    def _free_node(self, node: TreeNode) -> int:
        parent = node.parent
        released = len(node.key)

        self.token_to_kv_pool_allocator.free(node.value.tolist())
        del parent.children[tuple(node.key[:self.page_size])]
        self.evictable_nodes.discard(node)
        node.parent = None

        self._refresh_leaf(parent)
        return released

    def _evict(self, num_tokens: int) -> int:
        candidates = [
            (node.last_accessed,node.node_id, node) for node in self.evictable_nodes
        ]

        heapq.heapify(candidates)

        released = 0

        while candidates and released < num_tokens:
            _, _, node = heapq.heappop(candidates)
            parent = node.parent

            released += self._free_node(node)

            if parent in self.evictable_nodes:
                heapq.heappush(candidates, (parent.last_accessed, parent.node_id, parent))

        return released

    def cache_unfinished_requests(self, req: Request):
        """
        Cache the unfinished requests to the radix cache.
        This will be called when the request is not finished.
        """
        aligned_len = (
            req.cached_len // self.page_size * self.page_size
        )

        token_ids = (
            req.input_ids + req.output_ids
        )[:aligned_len]

        old_protected_len = req.cache_protected_len
        old_node = req.last_node

        values = torch.tensor(
            req.slots[:aligned_len],
            dtype=torch.int64,
            device=self.req_to_token_pool.req_to_token.device,
        )

        prefix_len, _ = self._insert(token_ids, values)
        indices, last_node = self._match_prefix(token_ids)

        duplicate_slots = req.slots[
            old_protected_len : prefix_len
        ]

        self._inc_lock_ref(last_node)

        req.slots[:aligned_len] = indices.tolist()

        self.req_to_token_pool.write(
            (req.req_pool_index, slice(0, aligned_len)),
            indices,
        )

        self._dec_lock_ref(old_node)

        self.token_to_kv_pool_allocator.free(duplicate_slots)

        req.cache_protected_len = aligned_len
        req.last_node = last_node
        
    def cache_finished_requests(self, req: Request, is_insert: bool = True):
        """
        Cache the finished requests to the radix cache.
        This will be called when the request is finished.
        This will free the protection of the tree nodes and other resources.
        """
        protected_len = req.cache_protected_len

        if is_insert:
            aligned_len = (
                req.cached_len // self.page_size * self.page_size
            )

            token_ids = (
                req.input_ids + req.output_ids
            )[:aligned_len]

            values = torch.tensor(
                req.slots[:aligned_len],
                dtype=torch.int64,
                device=self.req_to_token_pool.req_to_token.device,
            )

            prefix_len, _ = self._insert(token_ids, values)

            prompt_len = min(len(req.input_ids), aligned_len)
            prompt_len = (
                prompt_len // self.page_size * self.page_size
            )
            self._match_prefix(token_ids[:prompt_len])

            private_slots = (
                req.slots[protected_len : prefix_len]
                + req.slots[aligned_len:]
            )
        else:
            private_slots = req.slots[protected_len:]

        self.token_to_kv_pool_allocator.free(private_slots)
        self._dec_lock_ref(req.last_node)

        req.cache_protected_len = 0
        req.last_node = None