import torch


def greedy_sample(next_token_logits: torch.Tensor) -> torch.Tensor:
    return next_token_logits.argmax(dim=-1)


def random_sample(next_token_logits: torch.Tensor) -> torch.Tensor:
    probabilities = next_token_logits.float().softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
