from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PagedKVSpec:
    layer_ids: tuple[int, ...]
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

CacheSpec = PagedKVSpec