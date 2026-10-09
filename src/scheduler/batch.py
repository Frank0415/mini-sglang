from dataclasses import dataclass

import torch

from ..model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardContext,
    ForwardMode,
    PagedKVBatch,
)
from ..model_executor.model_runner import ModelRunner
from .request import Request


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
        if not requests:
            raise ValueError("ScheduleBatch requires at least one request")
        flat_input_ids = []
        flat_positions = []
        flat_slots = []
        request_rows = []
        seq_lens = []
        extend_lens = []
        modes = []
        device = runner.device
        for request in requests:
            all_ids = request.input_ids + request.output_ids
            if request.req_pool_index is None:
                row = runner.req_to_token_pool.alloc_rows(1)
                if row is None:
                    raise ValueError(
                        "Failed to allocate request pool index, request pool full"
                    )
                request.req_pool_index = row[0]

                indices, node = runner.radix_cache._match_prefix(
                    all_ids[:-1]
                )

                runner.radix_cache._inc_lock_ref(node)

                request.last_node = node
                request.cache_protected_len = indices.numel()
                request.cached_len = indices.numel()
                request.slots = indices.tolist()

                runner.req_to_token_pool.write(
                    (request.req_pool_index, slice(0, request.cached_len)),
                    indices,
                )
            
            prefix_len = request.cached_len
            new_ids = all_ids[prefix_len:]
            total_len = prefix_len + len(new_ids)
            last_loc = request.slots[-1] if request.slots else -1
            
            if not new_ids:
                raise ValueError("Request has no pending input tokens")
            if total_len > runner.req_to_token_pool.max_context_len:
                raise ValueError("Request exceeds max_context_len")
            mode = (
                ForwardMode.DECODE
                if request.output_ids and len(new_ids) == 1
                else ForwardMode.EXTEND
            )
            if mode == ForwardMode.DECODE and len(new_ids) != 1:
                raise ValueError("Decode requires exactly one pending token")

            new_slots = runner.token_to_kv_pool_allocator.alloc_extend(
                prefix_len=prefix_len,
                seq_len=total_len,
                last_loc=last_loc,
                extend_num_tokens=len(new_ids),
            )
            if new_slots is None:
                allocator = runner.token_to_kv_pool_allocator
                page_size = allocator.page_size

                prefix_pages = (prefix_len + page_size - 1) // page_size
                total_pages = (total_len + page_size - 1) // page_size
                needed_pages = total_pages - prefix_pages
                missing_pages = needed_pages - len(allocator.free_pages)

                runner.radix_cache._evict(missing_pages * page_size)

                new_slots = allocator.alloc_extend(
                    prefix_len=prefix_len,
                    seq_len=total_len,
                    last_loc=last_loc,
                    extend_num_tokens=len(new_ids),
                )
                if new_slots is None:
                    # Added later by debugging scheduler:
                    # The scheduler owns cleanup. In particular, do not detach an
                    # existing row from a running request that still owns historical KV.
                    raise ValueError(
                        "KV Cache allocation failed, KV Cache insufficient free pages"
                    )

            request.slots.extend(new_slots)
            locations = torch.tensor(new_slots, device=device, dtype=torch.int64)
            runner.req_to_token_pool.write(
                (request.req_pool_index, slice(prefix_len, total_len)), locations
            )
            flat_input_ids.extend(new_ids)
            flat_positions.extend(range(prefix_len, total_len))
            flat_slots.extend(new_slots)
            request_rows.append(request.req_pool_index)
            seq_lens.append(total_len)
            extend_lens.append(len(new_ids))
            modes.append(mode)

        mode = (
            ForwardMode.DECODE
            if all(mode == ForwardMode.DECODE for mode in modes)
            else ForwardMode.EXTEND
        )
        input_ids = torch.tensor(flat_input_ids, dtype=torch.int64, device=device)
        position_ids = torch.tensor(flat_positions, dtype=torch.int64, device=device)

        # The positions of the KV Cache to write to
        cache_batch = PagedKVBatch(
            req_pool_indices=torch.tensor(
                request_rows, dtype=torch.int64, device=device
            ),
            out_cache_loc=torch.tensor(flat_slots, dtype=torch.int64, device=device),
        )
        context = ForwardContext(
            runtime=runner.runtime,
            batch=ForwardBatch(
                mode=mode,
                seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=device),
                extend_lens=torch.tensor(extend_lens, dtype=torch.int64, device=device),
                cache_batches={0: cache_batch},
            ),
        )
        return cls(
            requests=requests,
            mode=mode,
            input_ids=input_ids,
            position_ids=position_ids,
            context=context,
            input_len=len(flat_input_ids),
        )
