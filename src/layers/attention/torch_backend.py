import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...mem_cache.allocator import TokenToKVPool
from ...mem_cache.memory_pool import ReqToTokenPool

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
        q: [total_tokens, query_heads, seq_len, head_dim]
        k/v: [total_tokens, kv_heads, seq_len, head_dim]

        batch: Execution Mode, seq_lens, extend_lens
        cache_batch: indices of the req_pool and the location of the cache slot

        k,v has already applied RoPE
        Here the q,k,v is the input instead of the full kv buffer
        The k,v will be concatenated to the full kv buffer in this program
        currently the support of kv caching only batch=1
        """
        attn_mask = None
        is_causal = True

        # 1. set the kv buffer using the new k and v
        # The output cache location is the location of the cache slot in the token to kv pool
        # output cache location is passed in by cache_batch, constructed by the allocator
        if cache_batch is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer_id,
                cache_batch.out_cache_loc,
                k,
                v,
            )
            k_buffer, v_buffer = self.token_to_kv_pool.get_kv_buffer(layer_id)

        # 2. allocate the output tensor
        output = torch.empty_like(q)
        offsets = batch.query_offsets_cpu

        for i, total_len in enumerate(batch.seq_lens_cpu):
            start, end = offsets[i], offsets[i + 1]
            input_len = end - start
            prefix_len = total_len - input_len

            request_q = q[start:end]

            # 3. get the kv buffer from the token to kv pool
            # slots are the physical location of the kv buffer in the token to kv pool
            if cache_batch is not None:
                row = cache_batch.req_pool_indices_cpu[i]
                slots = self.req_to_token_pool.req_to_token[row, :total_len]
                request_k = k_buffer[slots]
                request_v = v_buffer[slots]
            else:
                request_k = k[start:end]
                request_v = v[start:end]

            # [tokens, heads, D] → [1, heads, tokens, D]
            request_q = request_q.transpose(0, 1).unsqueeze(0)
            request_k = request_k.transpose(0, 1).unsqueeze(0)
            request_v = request_v.transpose(0, 1).unsqueeze(0)

            attn_mask = None
            is_causal = prefix_len == 0

            if prefix_len > 0 and input_len > 1:
                query_positions = prefix_len + torch.arange(
                    input_len, device=q.device
                )
                key_positions = torch.arange(total_len, device=q.device)
                attn_mask = (
                    key_positions[None, :] <= query_positions[:, None]
                )

            with sdpa_kernel([
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]):
                request_output = F.scaled_dot_product_attention(
                    request_q,
                    request_k,
                    request_v,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    dropout_p=0.0,
                    enable_gqa=True,
                )

            output[start:end] = request_output.squeeze(0).transpose(0, 1)

        return output
