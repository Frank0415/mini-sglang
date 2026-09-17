import torch
from torch import nn
from dataclasses import dataclass
from abc import ABC,abstractmethod

@dataclass
class KVPoolConfig:
    num_layers: int
    capacity: int          # How many token's KV can be stored
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

class TokenToKVPool(ABC):
    def __init__(self,config:KVPoolConfig):
        self.config = config

    @abstractmethod
    def set_kv_buffer(
        self,
        layer_id:int,
        slot_ids:torch.Tensor,
        k:torch.Tensor,
        v:torch.Tensor
    ) -> None:
        ...

    @abstractmethod
    def get_key_buffer(
        self,
        layer_id:int,
        slot_ids:torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def get_value_buffer(
        self,
        layer_id:int,
        slot_ids:torch.Tensor
    ) -> torch.Tensor:
        ...

class MHATokenToKVPool(TokenToKVPool):
    def __init__(self,config:KVPoolConfig):
        super().__init__(config)
        shape = (
            config.num_layers,
            config.capacity,
            config.num_kv_heads,
            config.head_dim
        )
        self.k_buffer = torch.empty(
            shape, dtype=config.dtype, device=config.device
        )
        self.v_buffer = torch.empty(
            shape, dtype=config.dtype, device=config.device
        )

    def set_kv_buffer(
        self,
        layer_id:int,
        slot_ids:torch.Tensor,
        k:torch.Tensor,
        v:torch.Tensor
    ) -> None:
        pass

    def get_key_buffer(
        self,
        layer_id:int,
        slot_ids:torch.Tensor
    ) -> torch.Tensor:
        pass

    def get_value_buffer(
        self,
        layer_id:int,
        slot_ids:torch.Tensor
    ) -> torch.Tensor:
        pass

class PagedTokenToKVPoolAllocator:
    def __init__(self,page_size:int, num_pages:int):
        self.page_size = page_size
        self.num_pages = num_pages
        self.free_pages = list[bool]([True] * num_pages)

    def alloc_extend():
        raise NotImplementedError("PagedTokenToKVPollAllocator.alloc_extend is not implemented")

    def alloc_decode():
        raise NotImplementedError("PagedTokenToKVPollAllocator.alloc_decode is not implemented")

    def free():
        raise NotImplementedError("PagedTokenToKVPollAllocator.free is not implemented")
