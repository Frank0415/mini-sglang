from dataclasses import dataclass

from ..layers.attention.torch_backend import TorchAttnBackend
from ..mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPool,
)
from ..mem_cache.memory_pool import ReqToTokenPool
from ..mem_cache.cache_spec import CacheSpec, PagedKVSpec

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

    def for_layer(self, layer_id: int):
        binding = self.layers[layer_id]
        return self.groups[binding.group_id], binding.local_layer_id

