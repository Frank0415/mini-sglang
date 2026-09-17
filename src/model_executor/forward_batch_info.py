from dataclasses import dataclass
from enum import Enum

import torch

from .runtime import ModelRuntime


class ForwardMode(Enum):
    EXTEND = "extend"
    DECODE = "decode"


@dataclass
class PagedKVBatch:
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor


@dataclass
class RecurrentBatch:
    state_indices: torch.Tensor


@dataclass
class ForwardBatch:
    mode: ForwardMode
    seq_lens: torch.Tensor
    extend_lens: torch.Tensor
    cache_batches: dict[int, PagedKVBatch | RecurrentBatch]


@dataclass
class ForwardContext:
    runtime: ModelRuntime
    batch: ForwardBatch