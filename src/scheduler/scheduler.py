from collections import deque
from typing import TYPE_CHECKING

import torch

from .batch import ScheduleBatch
from .request import Request, RequestStatus

if TYPE_CHECKING:
    from ..model_executor.model_runner import ModelRunner


class Scheduler:
    def __init__(
        self,
        runner: "ModelRunner",
        max_batch_size: int = 1,
        decode_headroom_tokens: int = 16,
    ):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.decode_headroom_tokens = decode_headroom_tokens
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
        allocator = self.runner.token_to_kv_pool_allocator
        if (
            len(request.input_ids) + request.max_new_tokens - 1
            > allocator.num_pages * allocator.page_size
        ):
            raise ValueError("Requested generation exceeds KV pool capacity")
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def _needed_pages(self, request: Request, headroom_tokens: int = 0) -> int:
        page_size = self.runner.token_to_kv_pool_allocator.page_size
        target_len = len(request.input_ids) + len(request.output_ids)
        target_len += min(
            headroom_tokens,
            max(0, request.max_new_tokens - len(request.output_ids) - 1),
        )
        current_pages = (request.cached_len + page_size - 1) // page_size
        target_pages = (target_len + page_size - 1) // page_size
        return target_pages - current_pages

    def _ensure_running_capacity(self) -> bool:
        retracted = False

        while self.running:
            needed = sum(self._needed_pages(request) for request in self.running)
            free = len(self.runner.token_to_kv_pool_allocator.free_pages)

            if needed <= free:
                return retracted

            self.runner.radix_cache._evict(
                (needed - free) * self.runner.token_to_kv_pool_allocator.page_size
            )

            free = len(self.runner.token_to_kv_pool_allocator.free_pages)
            if needed <= free:
                return retracted

            if len(self.running) == 1:
                request = self.running[0]
                self.abort(request.request_id)
                request.finish_reason = "capacity"
                return True

            self._retract(self.running[-1])
            retracted = True

        return retracted

    def _try_admit(self, request: Request) -> bool:
        if not self.runner.req_to_token_pool.available_slots():
            return False

        cache = self.runner.radix_cache
        allocator = self.runner.token_to_kv_pool_allocator
        all_ids = request.input_ids + request.output_ids
        indices, node = cache._match_prefix(all_ids[:-1])
        cache._inc_lock_ref(node)

        request.last_node = node
        admitted = False

        try:
            request.cache_protected_len = indices.numel()
            request.cached_len = indices.numel()
            request.slots = indices.tolist()
            available = len(allocator.free_pages) + cache._evictable_pages()
            needed = sum(
                self._needed_pages(item, self.decode_headroom_tokens)
                for item in self.running
            ) + self._needed_pages(request, self.decode_headroom_tokens)

            if needed > available and self.running or self._needed_pages(request) > available:
                    return False

            request.req_pool_index = self.runner.req_to_token_pool.alloc_rows(1)[0]
            self.runner.req_to_token_pool.write(
                (request.req_pool_index, slice(0, request.cached_len)), indices
            )
            admitted = True
            return True
        finally:
            if not admitted:
                self._release(request)

    def get_next_requests(self) -> list[Request]:
        if self._ensure_running_capacity():
            return self.running.copy()
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting[0]
            if request.status != RequestStatus.WAITING:
                self.waiting.popleft()
                continue  # A waiting request may have been cancelled.
            if not self._try_admit(request):
                if not self.running:
                    self.abort(request.request_id)
                    request.finish_reason = "capacity"
                    self.waiting.popleft()
                    continue
                break
            self.waiting.popleft()
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
            is_insert=request.status == RequestStatus.FINISHED,
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
        if request.status not in (RequestStatus.FINISHED, RequestStatus.ABORTED):
            raise ValueError("Request is not finished")
        return self.requests.pop(request_id)

    def _retract(self, request: Request) -> None:
        self._release(request)

        self.running = [item for item in self.running if item is not request]

        request.status = RequestStatus.WAITING
        self.waiting.append(request)
