from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig
from collections.abc import Iterable
from ..layers.attention.torch_backend import TorchAttnBackend
from ..layers.layers import Embedding, LMhead, RMSNorm
from ..layers.rotary_embedding import apply_rope

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

    @torch.no_grad()
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded_parameter_ids = set()

        for name, weight in weights:
            if name not in params:
                raise ValueError(f"Unexpected weight: {name}")

            parameter = params[name]

            if parameter.shape != weight.shape:
                raise ValueError(
                    f"{name}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(weight.shape)}"
                )

            parameter.copy_(weight)
            loaded_parameter_ids.add(id(parameter))

        missing = [
            name
            for name, parameter in self.named_parameters()
            if id(parameter) not in loaded_parameter_ids
        ]

        if missing:
            raise ValueError(f"Missing weights: {missing}")

class LlamaModel(nn.Module):
    Embedding: nn.Module
    LlamaDecoderLayer: nn.Module
    FinalRMSNorm: nn.Module

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.Embedding = Embedding(config)
        self.LlamaDecoderLayer = nn.ModuleList(
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.FinalRMSNorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        position_ids: [batch_size, seq_len]
        return: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.Embedding(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)

        return self.FinalRMSNorm(hidden_states)


class LlamaDecoderLayer(nn.Module):
    RMSNorm: nn.Module
    LlamaAttention: nn.Module
    LlamaMLP: nn.Module

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttn(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return residual + hidden_states


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SmolLM2 uses SwiGLU
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttn(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        )

        rope = getattr(config, "rope_parameters", None) or {}
        self.rope_theta = rope.get("rope_theta", getattr(config, "rope_theta", 10000.0))

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.backend = TorchAttnBackend()

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rope(q, k, position_ids, self.rope_theta)

        output = self.backend(q, k, v)
        output = output.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        return self.o_proj(output)
