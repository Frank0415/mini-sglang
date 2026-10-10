from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import torch

from .layers.sampler import greedy_sample, random_sample
from .model_executor.model_runner import ModelRunner
from .scheduler.batch import ScheduleBatch
from .scheduler.request import Request, RequestStatus
from .scheduler.scheduler import Scheduler


class Engine:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        max_batch_size: int = 4,
        decode_headroom_tokens: int = 16,
        do_sample: bool = False,
    ):
        self.do_sample = do_sample
        self.installed_version = 0
        self.runner = ModelRunner(model_dir, device)
        self.scheduler = Scheduler(
            self.runner,
            max_batch_size=max_batch_size,
            decode_headroom_tokens=decode_headroom_tokens,
        )

    def add_request(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> str:
        """提交请求，返回 request_id。"""
        request_id = uuid4().hex

        request = Request(
            request_id=request_id,
            input_ids=list(input_ids),
            max_new_tokens=max_new_tokens,
            weight_version=self.installed_version,
        )

        self.scheduler.add_request(request)
        return request_id

    @torch.inference_mode()
    def step(self):
        """
        Do one step of inference, return none when nothing to do.
        """
        requests = []
        try:
            requests = self.scheduler.get_next_requests()
            if not requests:
                return
            batch = ScheduleBatch.prepare(requests, self.runner)

            logits = self.runner.forward(
                batch.input_ids,
                batch.position_ids,
                context=batch.context,
            )

            # logits: (total_tokens, vocab_size)
            # logits[last_indices]: (batch_size, vocab_size)
            # token_ids: (batch_size)
            last_indices = batch.context.batch.extend_lens.cumsum(0) - 1
            sampler = random_sample if self.do_sample else greedy_sample
            next_token_logits = logits[last_indices]
            token_ids = sampler(next_token_logits)
            log_probs = next_token_logits.float().log_softmax(dim=-1)
            token_log_probs = log_probs.gather(1, token_ids[:, None]).squeeze(-1).tolist()
            token_ids = token_ids.tolist()
            self.scheduler.process_result(batch, token_ids)
            for request, log_prob in zip(requests, token_log_probs):
                request.output_log_probs.append(log_prob)

        except Exception:
            for request in requests or self.scheduler.running.copy():
                self.scheduler.abort(request.request_id)
            raise

        return [{
            "request_id": request.request_id,
            "token_id": token_id,
            "log_prob": log_prob,
            "weight_version": request.weight_version,
            "finished": request.status == RequestStatus.FINISHED,
            "finish_reason": request.finish_reason,
        } for request, token_id, log_prob in zip(requests, token_ids, token_log_probs)]

    def abort(self, request_id: str):
        self.scheduler.abort(request_id)

    def free_request(self, request_id: str):
        return self.scheduler.return_request_result(request_id)

    def update_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        version: int,
    ) -> None:
        if self.scheduler.running or any(
            request.status == RequestStatus.WAITING
            for request in self.scheduler.waiting
        ):
            raise RuntimeError("Finish pending requests before updating weights")
        self.runner.model.load_weights(weights)
        if torch.device(self.runner.device).type == "cuda":
            torch.cuda.synchronize(self.runner.device)
        self.runner.radix_cache.clear()
        self.installed_version = version

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> Request:
        """
        Generate text until the request is finished.
        """
        return self.generate_batch([input_ids], max_new_tokens)[0]

    def generate_batch(
        self,
        input_ids: list[list[int]],
        max_new_tokens: int,
    ) -> list[Request]:
        request_ids = [self.add_request(ids, max_new_tokens) for ids in input_ids]
        requests = [self.scheduler.requests[request_id] for request_id in request_ids]

        while any(request.status not in (
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
        ) for request in requests):
            self.step()

        return [self.free_request(request_id) for request_id in request_ids]
