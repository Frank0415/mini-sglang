import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...mem_cache.memory_pool import ReqToTokenPool
from ...mem_cache.allocator import TokenToKVPool

"""
Old implementation:

groups = q.shape[1] // k.shape[1]

k = k.repeat_interleave(groups, dim=1)
v = v.repeat_interleave(groups, dim=1)

return F.scaled_dot_product_attention(
    q,
    k,
    v,
    dropout_p=0.0,
    is_causal=True,
)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=True,
    )
"""

class TorchAttnBackend(nn.Module):
    def __init__(self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: TokenToKVPool):
        super().__init__()
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q: [batch, query_heads, seq_len, head_dim]
        k: [batch, kv_heads, seq_len, head_dim]
        v: [batch, kv_heads, seq_len, head_dim]
        """
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True,
            )
        
