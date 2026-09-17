from dataclasses import dataclass

import torch

from ..layers.attention.torch_backend import TorchAttnBackend
from ..mem_cache.allocator import (
    MHATokenToKVPool,
    PagedTokenToKVPoolAllocator,
    TokenToKVPool,
)
from ..mem_cache.cache_spec import CacheSpec, PagedKVSpec
from ..mem_cache.memory_pool import ReqToTokenPool


@dataclass
class PagedKVRuntime:
    req_to_token_pool: ReqToTokenPool
    kv_pool: TokenToKVPool
    allocator: PagedTokenToKVPoolAllocator
    backend: TorchAttnBackend


@dataclass(frozen=True)
class LayerBinding:
    group_id: int
    local_layer_id: int


@dataclass
class ModelRuntime:
    groups: dict[int, PagedKVRuntime]
    layers: dict[int, LayerBinding]

    def find_layer(self, layer_id: int):
        "Return which group and local layer id the given layer_id belongs to"
        binding = self.layers[layer_id]
        return self.groups[binding.group_id], binding.local_layer_id

def build_runtime(
    specs: tuple[CacheSpec, ...],
    *,
    max_requests: int,
    max_context_len: int,
    page_size: int,
    num_pages: int,
    device: str,
) -> ModelRuntime:
    groups = {}
    layers = {}

    for group_id, spec in enumerate(specs):
        if not isinstance(spec, PagedKVSpec):
            raise NotImplementedError()

        rows = ReqToTokenPool(
            total_reqs=max_requests,
            max_context_len=max_context_len,
            device=torch.device(device),
        )

        pool = MHATokenToKVPool(
            num_layers=len(spec.layer_ids),
            capacity=page_size * num_pages,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            dtype=spec.dtype,
            device=torch.device(device),
            page_size=page_size,
        )

        allocator = PagedTokenToKVPoolAllocator(
            page_size=page_size,
            num_pages=num_pages,
        )

        groups[group_id] = PagedKVRuntime(
            req_to_token_pool=rows,
            kv_pool=pool,
            allocator=allocator,
            backend=TorchAttnBackend(rows, pool),
        )

        for local_id, layer_id in enumerate(spec.layer_ids):
            if layer_id in layers:
                raise ValueError()

            layers[layer_id] = LayerBinding(
                group_id=group_id,
                local_layer_id=local_id,
            )

    return ModelRuntime(groups=groups, layers=layers)