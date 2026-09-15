import torch

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply standard, non-interleaved Llama RoPE to Q and K.
    q: [batch, query_heads, seq_len, head_dim]
    k: [batch, kv_heads, seq_len, head_dim]
    position_ids: [batch, seq_len], on the same device as Q and K.
    """
    head_dim = q.shape[-1]
    dim_indices = torch.arange(
        0, head_dim, 2, device=q.device, dtype=torch.float32
    )
    inv_freq = 1.0 / (rope_theta ** (dim_indices / head_dim))
    angles = position_ids.float().unsqueeze(-1) * inv_freq
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    cos = angles.cos()
    sin = angles.sin()

    q_first, q_second = q.chunk(2, dim=-1)
    k_first, k_second = k.chunk(2, dim=-1)
    q_rotated = torch.cat((-q_second, q_first), dim=-1)
    k_rotated = torch.cat((-k_second, k_first), dim=-1)

    return (
        q * cos.to(q.dtype) + q_rotated * sin.to(q.dtype),
        k * cos.to(k.dtype) + k_rotated * sin.to(k.dtype),
    )
