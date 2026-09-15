import torch.nn as nn
import torch
from ..configs.model_config import ModelConfig
from transformers import LlamaConfig
from ..layers.layers import Embedding,LMhead,RMSNorm

class LlamaForCausalLM(nn.Module):
    LlamaModel: nn.Module
    LMhead: nn.Module

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.LlamaModel = LlamaModel(config)
        self.LMhead = LMhead(config)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        position_ids: [batch_size, seq_len]
        return: [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.LlamaModel(input_ids, position_ids)
        return self.LMhead(hidden_states)


class LlamaModel(nn.Module):
    Embedding: nn.Module
    LlamaDecoderLayer: nn.Module
    FinalRMSNorm: nn.Module

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.Embedding = Embedding(config)
        self.LlamaDecoderLayer = nn.ModuleList(LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.FinalRMSNorm = RMSNorm(config)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        position_ids: [batch_size, seq_len]
        return: [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.Embedding(input_ids, position_ids)
        for layer in self.LlamaDecoderLayer:
            hidden_states = layer(hidden_states)
        hidden_states = self.FinalRMSNorm(hidden_states)
        return hidden_states
        
class LlamaDecoderLayer(nn.Module):
    RMSNorm: nn.Module
    LlamaAttention: nn.Module
    LlamaMLP: nn.Module

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
