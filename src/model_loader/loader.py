from pathlib import Path

import torch
from torch import nn

from ..configs.model_config import ModelConfig
from ..models.registry import ModelRegistry
from .weight_utils import iter_weights


def load_model(model_dir: str | Path,    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,) -> nn.Module:
    """
    Load the model from the local directory.
    """
    model_config: ModelConfig = ModelConfig.from_pretrained(model_dir)
    model_class = ModelRegistry.get_model_class(model_config.architectures)
    model =  model_class(model_config.hf_config)

    model = model.to(dtype = dtype)
    model.load_weights(iter_weights(model_dir))

    model = model.to(device=device)
    model.eval()

    return model
