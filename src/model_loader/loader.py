from pathlib import Path
from ..configs.model_config import ModelConfig
from ..models.registry import get_model_class

def load_model(model_dir: str | Path) -> ModelConfig:
    """
    Load the model from the local directory.
    """
    model_config: ModelConfig = ModelConfig.from_pretrained(model_dir)
    model_class = get_model_class(model_config.architectures)
    return model_class(model_config.hf_config)