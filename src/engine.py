from pathlib import Path
from uuid import uuid4

import torch

from .layers.sampler import greedy_sample
from .model_executor.model_runner import ModelRunner
from .scheduler.batch import ScheduleBatch
from .scheduler.request import Request, RequestStatus
from .scheduler.scheduler import Scheduler


class Engine:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
    ):
        self.runner = ModelRunner(model_dir, device)
        self.scheduler = Scheduler(self.runner)

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
        )

        self.scheduler.add_request(request)
        return request_id

    @torch.inference_mode()
    def step(self):
        """
        Do one step of inference, return none when nothing to do.
        """
        requests = self.scheduler.get_next_requests()
        if not requests:
            return
        request = requests[0]
        try:
            batch = ScheduleBatch.prepare(requests, self.runner)

            logits = self.runner.forward(
                batch.input_ids,
                batch.position_ids,
                context=batch.context,
            )

            # logits: (batch_size, num_tokens, vocab_size)
            # logits[:, -1, :]: (batch_size, vocab_size)
            # token_id: (batch_size)
            token_id = greedy_sample(logits[:, -1, :]).item()
            self.scheduler.process_result(batch, token_id)

        except Exception:
            self.scheduler.abort(request.request_id)
            raise

        return {
            "request_id": request.request_id,
            "token_id": token_id,
            "finished": request.status == RequestStatus.FINISHED,
            "finish_reason": request.finish_reason,
        }

    def abort(self, request_id: str):
        self.scheduler.abort(request_id)

    def free_request(self, request_id: str):
        return self.scheduler.return_request_result(request_id)

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> Request:
        """
        Generate text until the request is finished.
        """
        request_id = self.add_request(input_ids, max_new_tokens)
        request = self.scheduler.requests[request_id]

        while request.status not in (
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
        ):
            self.step()

        return self.free_request(request_id)
