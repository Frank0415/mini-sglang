from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mem_cache.radix_cache import TreeNode


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class Request:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int

    output_ids: list[int] = field(default_factory=list)
    cached_len: int = 0
    slots: list[int] = field(default_factory=list)
    req_pool_index: int | None = None
    cache_protected_len: int = 0
    last_node: TreeNode | None = None

    status: RequestStatus = RequestStatus.WAITING
    finish_reason: str | None = None
    output_log_probs: list[float] = field(default_factory=list)
