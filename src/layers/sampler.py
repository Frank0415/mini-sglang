import torch


def greedy_sample(next_token_logits: torch.Tensor) -> torch.Tensor:
    return next_token_logits.argmax(dim=-1)
