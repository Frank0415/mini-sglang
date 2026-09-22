from collections import deque
from typing import TYPE_CHECKING

import torch

from .batch import ScheduleBatch
from .request import Request, RequestStatus

if TYPE_CHECKING:
    from ..model_executor.model_runner import ModelRunner


class Scheduler:
    def __init__(self, runner: "ModelRunner"):
        self.runner = runner
        self.waiting: deque[Request] = deque()
        self.running: Request | None = None
        # Keep completed results accessible by request ID.
        self.requests: dict[str, Request] = {}
        eos = runner.model_config.hf_config.eos_token_id
        self.eos_token_ids = set(
            eos if isinstance(eos, list) else ([] if eos is None else [eos])
        )

    def add_request(self, request: Request) -> None:
        if request.request_id in self.requests:
            raise ValueError(f"Duplicate request ID: {request.request_id}")
        if request.status != RequestStatus.WAITING:
            raise ValueError("Only new waiting requests can be submitted")
        if not request.input_ids or request.max_new_tokens <= 0:
            raise ValueError("Input must be nonempty and max_new_tokens must be positive")
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def get_next_requests(self) -> list[Request]:
        if self.running is not None:
            return [self.running]
        while self.waiting:
            request = self.waiting.popleft()
            if request.status != RequestStatus.WAITING:
                continue  # A waiting request may have been cancelled.
            request.status = RequestStatus.RUNNING
            self.running = request
            return [request]
        return []

