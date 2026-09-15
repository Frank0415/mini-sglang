from dataclasses import dataclass
from transformers import PretrainedConfig,AutoConfig
from pathlib import Path

@dataclass
class ModelConfig:
    hf_config: PretrainedConfig

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "ModelConfig":
        hf_config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        return cls(hf_config)

    @property
    def architectures(self) -> tuple[str, ...]:
        return tuple(self.hf_config.architectures or ())