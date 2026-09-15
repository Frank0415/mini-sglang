import torch
import torch.nn as nn
from transformers import LlamaConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return self.weight * hidden_states.to(input_dtype)


class Embedding(nn.Embedding):
    def __init__(self, config: LlamaConfig):
        super().__init__(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )


class LMhead(nn.Linear):
    def __init__(self, config: LlamaConfig):
        super().__init__(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )