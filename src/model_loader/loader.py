from pathlib import Path

import torch
from torch import nn

from ..configs.model_config import ModelConfig
from ..models.registry import ModelRegistry
from .weight_utils import iter_weights


def load_model(
    model_dir: str | Path,
    model_config: ModelConfig,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """
    Load local weights using the configured dtype unless explicitly overridden.

    If the model configuration has no dtype, use PyTorch's FP32 model precision.
    """
    if dtype is None:
        dtype = getattr(model_config.hf_config, "dtype", None)

    model_class = ModelRegistry.get_model_class(model_config.architectures)
    model = model_class(model_config.hf_config)

    model = model.to(dtype=dtype)
    model.load_weights(iter_weights(model_dir))
    model = model.to(device=device)
    model.eval()

    return model
