from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ..configs.cache_config import make_pool_configs
from ..configs.model_config import ModelConfig
from ..layers.attention.torch_backend import TorchAttnBackend
from ..mem_cache.allocator import (
    MHATokenToKVPool,
    PagedTokenToKVPoolAllocator,
    TokenToKVPool,
)
from ..mem_cache.memory_pool import ReqToTokenPool
from ..model_loader.loader import load_model
from .forward_batch_info import ForwardBatch, ForwardContext, ForwardMode
from .runtime import LayerBinding, ModelRuntime, PagedKVRuntime


class ModelRunner:
    model: nn.Module
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool
    token_to_kv_pool_allocator: PagedTokenToKVPoolAllocator
    attention_backend: TorchAttnBackend

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        attn_backend: TorchAttnBackend | None = None,
    ):
        model_config: ModelConfig = ModelConfig.from_pretrained(model_dir)
        self.model_config = model_config
        self.device = device
        self.model = load_model(
            model_dir, device=device, model_config=model_config,
        )
        req_config, kv_config, allocator_config = make_pool_configs(
            model_config,
            page_size=4,
            num_pages=512,
            total_reqs=4,
            max_context_len=1024,
            device=device,
            dtype=next(self.model.parameters()).dtype,
        )
        self.req_to_token_pool = ReqToTokenPool(**asdict(req_config))
        self.token_to_kv_pool = MHATokenToKVPool(**asdict(kv_config))
        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            **asdict(allocator_config)
        )
        self.attention_backend = (
            TorchAttnBackend(self.req_to_token_pool, self.token_to_kv_pool) if attn_backend is None else attn_backend
        )
        self.runtime = ModelRuntime(
            groups={
                0: PagedKVRuntime(
                    req_to_token_pool=self.req_to_token_pool,
                    kv_pool=self.token_to_kv_pool,
                    allocator=self.token_to_kv_pool_allocator,
                    backend=self.attention_backend,
                )
            },
            layers={
                i: LayerBinding(group_id=0, local_layer_id=i)
                for i in range(kv_config.num_layers)
            },
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        context: ForwardContext | None = None,
    ):
        if context is None:
            # Existing no-padding, full-recompute path; no KV slots are allocated.
            batch_size, seq_len = input_ids.shape
            lengths = torch.full(
                (batch_size,), seq_len, dtype=torch.int64, device=input_ids.device
            )
            context = ForwardContext(
                runtime=self.runtime,
                batch=ForwardBatch(
                    mode=ForwardMode.EXTEND,
                    seq_lens=lengths,
                    extend_lens=lengths.clone(),
                    cache_batches={},
                ),
            )
        return self.model(
            input_ids, position_ids, context=context
        )
        
