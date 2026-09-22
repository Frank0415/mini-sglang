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

    def process_result(self, batch: ScheduleBatch, token_id: int | torch.Tensor) -> None:
        """
        Called after successful forward and sampling. 
        Do not call with raw logits.
        """
        if batch.requests != [self.running]:
            raise ValueError("Result does not belong to the running request")
        request = self.running
        if isinstance(token_id, torch.Tensor):
            token_id = int(token_id.item())
        total_len = int(batch.context.batch.seq_lens[0].item())
        if request.cached_len + batch.input_len != total_len:
            raise ValueError("Batch result is stale or has already been processed")

        request.cached_len = total_len
        request.output_ids.append(token_id)
        if token_id in self.eos_token_ids:
            request.finish_reason = "eos"
        elif len(request.output_ids) >= request.max_new_tokens:
            request.finish_reason = "length"
        else:
            return  # Keep this request running for its next decode.

        request.status = RequestStatus.FINISHED
        self._release(request)
        self.running = None

    def abort(self, request_id: str) -> None:
        request = self.requests[request_id]
        if request.status in (RequestStatus.FINISHED, RequestStatus.ABORTED):
            return
        self._release(request)
        request.status = RequestStatus.ABORTED
        request.finish_reason = "aborted"
        if self.running is request:
            self.running = None

    def _release(self, request: Request) -> None:
        """
        Only release allocated slots, 
        including an unfinished forward's slots.
        """
        if request.slots:
            if torch.device(self.runner.device).type == "cuda":
                torch.cuda.synchronize(self.runner.device)
            self.runner.token_to_kv_pool_allocator.free(request.slots)
            request.slots.clear()
        if request.req_pool_index is not None:
            self.runner.req_to_token_pool.free_rows([request.req_pool_index])
            request.req_pool_index = None
        request.cached_len = 0

    def return_request_result(self, request_id: str) -> Request:
        """
        Return the request result to the scheduler, freeing the request.
        """
        request = self.requests[request_id]
        if request.status != RequestStatus.FINISHED:
            raise ValueError("Request is not finished")
        return self.requests.pop(request_id)
