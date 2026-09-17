from pathlib import Path
from dataclasses import asdict

from torch import nn

from ..mem_cache.allocator import TokenToKVPool, MHATokenToKVPool, PagedTokenToKVPoolAllocator
from ..mem_cache.memory_pool import ReqToTokenPool
from ..model_loader.loader import load_model
from ..layers.attention.torch_backend import TorchAttnBackend
from ..configs.model_config import ModelConfig
from ..configs.cache_config import make_pool_configs

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
        req_config, kv_config, allocator_config = make_pool_configs(
            model_config,
            page_size=4,
            num_pages=512,
            total_reqs=4,
            max_context_len=1024,
            device=device,
        )
        self.req_to_token_pool = ReqToTokenPool(**asdict(req_config))
        self.token_to_kv_pool = MHATokenToKVPool(**asdict(kv_config))
        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            **asdict(allocator_config)
        )
        self.attention_backend = (
            TorchAttnBackend(self.req_to_token_pool, self.token_to_kv_pool) if attn_backend is None else attn_backend
        )
        self.model = load_model(
            model_dir, device=device, model_config=model_config,
            attn_backend=self.attention_backend, dtype=kv_config.dtype,
        )
        
