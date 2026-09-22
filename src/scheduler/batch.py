from dataclasses import dataclass

import torch

from ..model_executor.forward_batch_info import (
    ForwardContext,
    ForwardMode,
    PagedKVBatch,
    ForwardBatch,
)
from .request import Request
from ..model_executor.model_runner import ModelRunner


@dataclass
class ScheduleBatch:
    requests: list[Request]
    mode: ForwardMode
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    context: ForwardContext
    input_len: int

    @classmethod
    def prepare(cls, requests: list[Request], runner: ModelRunner) -> "ScheduleBatch":
        if len(requests) != 1:
            raise ValueError("ScheduleBatch currently requires exactly one request")
        request = requests[0]
        all_ids = request.input_ids + request.output_ids

        prefix_len = request.cached_len
        new_ids = all_ids[prefix_len:]
        total_len = prefix_len + len(new_ids)
        if not new_ids:
            raise ValueError("Request has no pending input tokens")
        if total_len > runner.req_to_token_pool.max_context_len:
            raise ValueError("Request exceeds max_context_len")
        mode = (
            ForwardMode.DECODE
            if request.output_ids and prefix_len >= len(request.input_ids)
            else ForwardMode.EXTEND
        )
        if mode == ForwardMode.DECODE and len(new_ids) != 1:
            raise ValueError("Decode requires exactly one pending token")
        last_loc = request.slots[-1] if request.slots else -1

        if request.req_pool_index is None:
            row = runner.req_to_token_pool.alloc_rows(1)
            if row is None:
                raise ValueError(
                    "Failed to allocate request pool index, request pool full"
                )
            request.req_pool_index = row[0]

        new_slots = runner.token_to_kv_pool_allocator.alloc_extend(
            prefix_len=prefix_len,
            seq_len=total_len,
            last_loc=last_loc,
            extend_num_tokens=len(new_ids),
        )

        if new_slots is None:
            # The scheduler owns cleanup. In particular, do not detach an
            # existing row from a running request that still owns historical KV.
            raise ValueError(
                "KV Cache allocation failed, KV Cache insufficient free pages"
            )

        request.slots.extend(new_slots)
        device = runner.device
        locations = torch.tensor(new_slots, device=device, dtype=torch.int64)
        runner.req_to_token_pool.write(
            (request.req_pool_index, slice(prefix_len, total_len)), locations
        )

        input_ids = torch.tensor([new_ids], dtype=torch.int64, device=device)
        position_ids = torch.arange(
            prefix_len, total_len, dtype=torch.int64, device=device
        ).unsqueeze(0)

        # The positions of the KV Cache to write to
        cache_batch = PagedKVBatch(
            req_pool_indices=torch.tensor(
                [request.req_pool_index], dtype=torch.int64, device=device
            ),
            out_cache_loc=locations,
        )
        context = ForwardContext(
            runtime=runner.runtime,
            batch=ForwardBatch(
                mode=mode,
                seq_lens=torch.tensor([total_len], dtype=torch.int64, device=device),
                extend_lens=torch.tensor([len(new_ids)], dtype=torch.int64, device=device),
                cache_batches={0: cache_batch},
            ),
        )
        return cls(
            requests=requests,
            mode=mode,
            input_ids=input_ids,
            position_ids=position_ids,
            context=context,
            input_len=len(new_ids),
        )
