from dataclasses import dataclass

import torch

from ..mem_cache.cache_spec import PagedKVSpec
from .model_config import ModelConfig


@dataclass(frozen=True)
class ReqToTokenPoolConfig:
    total_reqs: int
    max_context_len: int
    device: torch.device


@dataclass(frozen=True)
class TokenToKVPoolConfig:
    num_layers: int
    capacity: int  # Usable token slots; the pool adds the reserved page.
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    page_size: int


@dataclass(frozen=True)
class PagedTokenToKVPoolAllocatorConfig:
    page_size: int
    num_pages: int  # Usable pages, excluding page 0.


def make_pool_configs(
    model_config: ModelConfig,
    *,
    page_size: int,
    num_pages: int,
    total_reqs: int,
    device: str | torch.device,
    max_context_len: int | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[
    ReqToTokenPoolConfig,
    TokenToKVPoolConfig,
    PagedTokenToKVPoolAllocatorConfig,
]:
    """Describe the current Llama-family model's uniform, all-layer paged KV.

    An explicit context limit is capped by the model's position limit.
    An explicit dtype overrides the model configuration, with FP32 as fallback.
    """
    hf_config = model_config.hf_config
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    if dtype is None:
        dtype = getattr(hf_config, "dtype", None) or torch.float32

    spec = PagedKVSpec(
        layer_ids=tuple(range(hf_config.num_hidden_layers)),
        num_kv_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    context_len = hf_config.max_position_embeddings
    if max_context_len is not None:
        context_len = min(context_len, max_context_len)
    device = torch.device(device)

    return (
        ReqToTokenPoolConfig(total_reqs, context_len, device),
        TokenToKVPoolConfig(
            num_layers=len(spec.layer_ids),
            capacity=page_size * num_pages,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            dtype=spec.dtype,
            device=device,
            page_size=page_size,
        ),
        PagedTokenToKVPoolAllocatorConfig(page_size, num_pages),
    )
