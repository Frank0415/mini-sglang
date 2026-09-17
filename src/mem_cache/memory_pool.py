from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ReqToTokenPoolConfig:
    total_reqs: int
    max_context_len: int
    device: torch.device


class ReqToTokenPool:
    def __init__(self,config:ReqToTokenPoolConfig):
        self.config = config

        # Only use row 1 - total_reqs, leave 0 for reserved
        self.req_to_token = torch.zeros(
            (config.total_reqs + 1, config.max_context_len),
            dtype=torch.int32,
            device=config.device,
        )
        # Row number is managed by CPU, mapping content is stored on the specified device
        self.free_slots = list(range(1, config.total_reqs + 1))

    def available_slots(self) -> int:
        return len(self.free_slots)

    def alloc_rows(self,need_size:int) -> list[int] | None:
        pass

    def write(
        self,
        indices: tuple[int, slice],
        values: torch.Tensor,
    ) -> None:
        pass

    def free_rows(self,indices:list[int]) -> None:
        pass

    def clear(self) -> None:
        pass
