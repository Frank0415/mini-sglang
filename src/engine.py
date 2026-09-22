from pathlib import Path
from uuid import uuid4

import torch

from .model_executor.model_runner import ModelRunner
from .scheduler.scheduler import Scheduler
from .scheduler.request import Request, RequestStatus
from .scheduler.batch import ScheduleBatch
from .layers.sampler import greedy_sample


class Engine:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
    ):
        self.runner = ModelRunner(model_dir, device)
        self.scheduler = Scheduler()

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
        requests = self.scheduler.get_next_requests()
        if not requests:
            return
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
            self.scheduler.abort(requests.request_id)
            raise

        return {
            "request_id": requests.request_id,
            "token_id": token_id,
            "finished": requests.status == RequestStatus.FINISHED,
            "finish_reason": requests.finish_reason,
        }
