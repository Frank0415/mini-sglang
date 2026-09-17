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
    def __init__(
        self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: TokenToKVPool
    ):
        super().__init__()
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch,
        cache_batch,
    ):
        """
        q: [batch, query_heads, seq_len, head_dim]
        k/v: [batch, kv_heads, seq_len, head_dim]

        batch: Execution Mode, seq_lens, extend_lens
        cache_batch: indices of the req_pool and the location of the cache slot

        k,v has already applied RoPE
        Here the q,k,v is the input instead of the full kv buffer
        The k,v will be concatenated to the full kv buffer in this program
        currently the support of kv caching only batch=1
        """
        attn_mask = None
        is_causal = True

        if cache_batch is not None:
            if q.shape[0] != 1:
                raise ValueError("Cached attention currently requires batch=1")
            
            # 1. calculate the input and total length
            input_len = q.shape[2]
            total_len = int(batch.seq_lens[0].item())
            prefix_len = total_len - input_len

            # 2. update the kv buffer's size:
            # [1, heads, tokens, dim] → [tokens, heads, dim]
            new_k = k[0].transpose(0, 1)
            new_v = v[0].transpose(0, 1)

            # 3. set the kv buffer using the new k and v
            # The output cache location is the location of the cache slot in the token to kv pool
            # output cache location is passed in by cache_batch, constructed by the allocator
            self.token_to_kv_pool.set_kv_buffer(
                layer_id,
                cache_batch.out_cache_loc,
                new_k,
                new_v,
            )

            # 4. get the kv buffer from the token to kv pool
            # slots are the physical location of the kv buffer in the token to kv pool
            row = int(cache_batch.req_pool_indices[0].item())
            slots = self.req_to_token_pool.req_to_token[row, :total_len]

            # get the actual cache of the prev k and v
            k_buffer, v_buffer = self.token_to_kv_pool.get_kv_buffer(layer_id)

            
            # [tokens, heads, dim] → [1, heads, tokens, dim]
            k = k_buffer[slots].transpose(0, 1).unsqueeze(0)
            v = v_buffer[slots].transpose(0, 1).unsqueeze(0)

            if prefix_len > 0:
                is_causal = False

                # For example query positions [1,2], key [0,1,2]
                # the attn_mask is [[True, True, False], [True, True, True]] as 2 > 1
                if input_len > 1:
                    query_positions = (
                        prefix_len
                        + torch.arange(input_len, device=q.device)
                    )
                    key_positions = torch.arange(total_len, device=q.device)

                    attn_mask = (
                        key_positions[None, :] <= query_positions[:, None]
                    )

        with sdpa_kernel([
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                enable_gqa=True,
            )
