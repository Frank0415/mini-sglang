from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import torch

from .runtime import ModelRuntime


class ForwardMode(Enum):
    EXTEND = "extend"
    DECODE = "decode"


@dataclass
class PagedKVBatch:
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor

    @cached_property
    def req_pool_indices_cpu(self) -> tuple[int, ...]:
        """One CPU snapshot shared by all layers of this forward.

        Create a new batch when indices change; do not mutate this batch's tensors.
        """
        return tuple(self.req_pool_indices.tolist())


@dataclass
class RecurrentBatch:
    state_indices: torch.Tensor


@dataclass
class ForwardBatch:
    mode: ForwardMode
    seq_lens: torch.Tensor
    extend_lens: torch.Tensor
    cache_batches: dict[int, PagedKVBatch | RecurrentBatch]

    @cached_property
    def seq_lens_cpu(self) -> tuple[int, ...]:
        """Snapshot once per forward, rather than synchronizing in every layer.

        Sequence lengths are immutable during a forward; each step builds a new batch.
        """
        return tuple(self.seq_lens.tolist())

    @cached_property
    def extend_lens_cpu(self) -> tuple[int, ...]:
        return tuple(self.extend_lens.tolist())
    
    @cached_property
    def query_offsets_cpu(self) -> tuple[int, ...]:
        offsets = [0]
        for len in self.extend_lens_cpu:
            offsets.append(offsets[-1] + len)
        return tuple(offsets)


@dataclass
class ForwardContext:
    runtime: ModelRuntime
    batch: ForwardBatch
