from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class KVPoolConfig:
    num_layers: int
    capacity: int  # Usable token slots, excluding reserved page 0
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    page_size: int = 1


class TokenToKVPool(ABC):
    def __init__(self, config: KVPoolConfig):
        self.config = config

    @abstractmethod
    def set_kv_buffer(
        self, layer_id: int, slot_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None: ...

    @abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor: ...

    @abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor: ...

    @abstractmethod
    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class MHATokenToKVPool(TokenToKVPool):
    def __init__(self, config: KVPoolConfig):
        assert config.num_layers > 0, "num_layers must be positive"
        assert config.capacity >= 0, "capacity must be nonnegative"
        assert config.page_size > 0, "page_size must be positive"
        assert config.capacity % config.page_size == 0, "capacity must be page-aligned"
        assert config.num_kv_heads > 0, "num_kv_heads must be positive"
        assert config.head_dim > 0, "head_dim must be positive"
        super().__init__(config)
        shape = (
            config.num_layers,
            config.capacity + config.page_size,
            config.num_kv_heads,
            config.head_dim,
        )
        self.k_buffer = torch.empty(shape, dtype=config.dtype, device=config.device)
        self.v_buffer = torch.empty(shape, dtype=config.dtype, device=config.device)

    def set_kv_buffer(
        self, layer_id: int, slot_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        assert 0 <= layer_id < self.config.num_layers, "layer_id out of range"
        assert slot_ids.ndim == 1, "slot_ids must be one-dimensional"
        # Page 0 may receive dummy writes, but is never allocated to requests.
        assert torch.all(
            (slot_ids >= 0) & (slot_ids < self.config.capacity + self.config.page_size)
        ), (
            "allocated slots exceed KV storage capacity"
        )
        expected = (slot_ids.numel(), self.config.num_kv_heads, self.config.head_dim)
        # Check both tensors before either write, including broadcastable shapes.
        assert tuple(k.shape) == expected, f"K shape must be {expected}"
        assert tuple(v.shape) == expected, f"V shape must be {expected}"
        self.k_buffer[layer_id, slot_ids, :] = k
        self.v_buffer[layer_id, slot_ids, :] = v

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Return this layer's full K buffer as a view; the backend selects slots."""
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Return this layer's full V buffer as a view; the backend selects slots."""
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)


class PagedTokenToKVPoolAllocator:
    def __init__(self, page_size: int, num_pages: int):
        assert page_size > 0, "page_size must be positive"
        assert num_pages >= 0, "num_pages must be nonnegative"
        self.page_size = page_size
        self.num_pages = num_pages
        # num_pages counts usable pages; page 0 is reserved.
        self.free_pages = list[int](range(1, num_pages + 1))

    def alloc_extend(
        self, prefix_len: int, seq_len: int, last_loc: int, extend_num_tokens: int
    ):
        """
        Allocate tokens for extending the sequence.
        Args:
            prefix_len: The length of the prefix.
            seq_len: The length of the total sequence (prefix + new tokens).
            last_loc: The physical location of the last token in the prefix.
            extend_num_tokens: The number of tokens to extend.
        Returns:
            A list of token indices that can be used to set the KV buffer.
            None if the allocation fails.
        """
        assert seq_len == prefix_len + extend_num_tokens, "sequence length mismatch"
        assert prefix_len >= 0 and extend_num_tokens >= 0, "lengths must be nonnegative"
        if prefix_len == 0:
            assert last_loc == -1, "an empty prefix must use last_loc=-1"
        else:
            assert self.page_size <= last_loc < (self.num_pages + 1) * self.page_size, (
                "last_loc must refer to a usable slot"
            )
            assert (last_loc + 1) % self.page_size == prefix_len % self.page_size, (
                "last_loc does not match the prefix page offset"
            )
        
        remain_tokens = self.page_size - last_loc % self.page_size - 1
        remain_tokens_list = list[int](range(last_loc + 1, (last_loc // self.page_size + 1) * self.page_size))
        if extend_num_tokens <= remain_tokens:
            return remain_tokens_list[:extend_num_tokens]
        else:
            extend_num_tokens -= remain_tokens
            need_pages = (extend_num_tokens + self.page_size - 1) // self.page_size
            if need_pages > len(self.free_pages):
                return None
            for i in self.free_pages[:need_pages]:
                token_list = list[int](range(i*self.page_size,i*self.page_size+min(self.page_size,extend_num_tokens)))
                remain_tokens_list.extend(token_list)
                extend_num_tokens -= len(token_list)
            self.free_pages = self.free_pages[need_pages:]
            return remain_tokens_list

    def free(self, token_list_pos: list[int]):
        """
        Free the tokens that are no longer needed.
        Args:
            token_list_pos: The physical positions of the tokens to free.
        """
        page_to_free = set[int]()
        for pos in token_list_pos:
            assert self.page_size <= pos < (self.num_pages + 1) * self.page_size, (
                "slot to free is reserved or out of range"
            )
            page_to_free.add(pos // self.page_size)
        assert page_to_free.isdisjoint(self.free_pages), "page already free"
        self.free_pages.extend(page_to_free)
