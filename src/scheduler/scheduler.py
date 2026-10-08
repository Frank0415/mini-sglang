from collections import deque
from typing import TYPE_CHECKING

import torch

from .batch import ScheduleBatch
from .request import Request, RequestStatus

if TYPE_CHECKING:
    from ..model_executor.model_runner import ModelRunner


class Scheduler:
    def __init__(self, runner: "ModelRunner", max_batch_size: int = 1):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
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
            raise ValueError(
                "Input must be nonempty and max_new_tokens must be positive"
            )
        if (
            len(request.input_ids) + request.max_new_tokens - 1
            > self.runner.req_to_token_pool.max_context_len
        ):
            raise ValueError("Requested generation exceeds max_context_len")
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def _needed_pages(self, request: Request) -> int:
        page_size = self.runner.token_to_kv_pool_allocator.page_size
        target_len = len(request.input_ids) + len(request.output_ids)
        current_pages = (request.cached_len + page_size - 1) // page_size
        target_pages = (target_len + page_size - 1) // page_size
        return target_pages - current_pages

    def get_next_requests(self) -> list[Request]:
        if self.running:
            return self.running.copy()
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting.popleft()
            if request.status != RequestStatus.WAITING:
                continue  # A waiting request may have been cancelled.
            request.status = RequestStatus.RUNNING
            self.running.append(request)
        return self.running.copy()

    def process_result(
        self, batch: ScheduleBatch, token_ids: list[int] | torch.Tensor
    ) -> None:
        """
        Called after successful forward and sampling.
        Do not call with raw logits.
        """
        if len(batch.requests) != len(self.running) or any(
            self.requests.get(request.request_id) is not request
            or request is not running
            for request, running in zip(batch.requests, self.running)
        ):
            raise ValueError("Result does not belong to the running request")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        total_lens = batch.context.batch.seq_lens_cpu
        input_lens = batch.context.batch.extend_lens_cpu
        for request, input_len, total_len in zip(
            batch.requests, input_lens, total_lens
        ):
            if request.cached_len + input_len != total_len:
                raise ValueError("Batch result is stale or has already been processed")

        for request, total_len, token_id in zip(batch.requests, total_lens, token_ids):
            request.cached_len = total_len
            request.output_ids.append(token_id)
            if token_id in self.eos_token_ids:
                request.finish_reason = "eos"
            elif len(request.output_ids) >= request.max_new_tokens:
                request.finish_reason = "length"
            else:
                self.runner.radix_cache.cache_unfinished_requests(request)
                continue  # Keep this request running for its next decode.

            request.status = RequestStatus.FINISHED
            self._release(request)
        self.running = [
            request
            for request in self.running
            if request.status == RequestStatus.RUNNING
        ]

    def abort(self, request_id: str) -> None:
        request = self.requests[request_id]
        if request.status in (RequestStatus.FINISHED, RequestStatus.ABORTED):
            return
        self._release(request)
        request.status = RequestStatus.ABORTED
        request.finish_reason = "aborted"
        self.running = [running for running in self.running if running is not request]

    def _release(self, request: Request) -> None:
        """
        Only release allocated slots,
        including an unfinished forward's slots.
        """
        if request.slots and torch.device(self.runner.device).type == "cuda":
            torch.cuda.synchronize(self.runner.device)
        self.runner.radix_cache.cache_finished_requests(
            request,
            is_insert= request.status == RequestStatus.FINISHED,
        )
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

    def _retract(self, request: Request) -> None:
        self._release(request)

        self.running = [
            item for item in self.running if item is not request
        ]

        request.status = RequestStatus.WAITING
        self.waiting.append(request)
