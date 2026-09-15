from .llama import LlamaForCausalLM
from dataclasses import dataclass

@dataclass
class ModelRegistry:
    MODEL_REGISTRY = {
        "LlamaForCausalLM": LlamaForCausalLM,
    }

    @classmethod
    def get_model_class(self, architectures: tuple[str, ...]) -> type:
        for architecture in architectures:
            if architecture in self.MODEL_REGISTRY:
                return self.MODEL_REGISTRY[architecture]
        raise ValueError(f"Architecture {architectures} not found in registry")