"""Test-side caller for batch-one paged KV; not an engine/session implementation.

Exercises the real allocator, mapping pool, Runner, model and backend. It never
implements attention or writes model K/V on behalf of the backend.
"""
import torch

from src.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardContext,
    ForwardMode,
    PagedKVBatch,
)


class CacheHarness:
    def __init__(self, runner):
        self.runner = runner
        rows = runner.req_to_token_pool.alloc_rows(1)
        if rows is None:
            raise RuntimeError('No free request row')
        self.row = rows[0]
        self.tokens = []
        self.slots = []
        self.closed = False
        self.steps = []

    @torch.inference_mode()
    def append(self, tokens, mode=ForwardMode.EXTEND):
        if not tokens:
            raise ValueError('Need nonempty input')
        old = len(self.tokens)
        end = old + len(tokens)
        if end > self.runner.req_to_token_pool.max_context_len:
            raise ValueError('Context limit exceeded')
        alloc = self.runner.token_to_kv_pool_allocator
        new = alloc.alloc_extend(old, end, self.slots[-1] if old else -1, len(tokens))
        if new is None:
            raise RuntimeError('KV pages exhausted')
        self.slots.extend(new)
        device = self.runner.device
        locations = torch.tensor(new, device=device)
        self.runner.req_to_token_pool.write((self.row, slice(old, end)), locations)
        ctx = ForwardContext(self.runner.runtime, ForwardBatch(
            mode=mode,
            seq_lens=torch.tensor([end], device=device),
            extend_lens=torch.tensor([len(tokens)], device=device),
            cache_batches={0: PagedKVBatch(torch.tensor([self.row], device=device), locations)},
        ))
        logits = self.runner.forward(
            torch.tensor([tokens], device=device),
            torch.arange(old, end, device=device).unsqueeze(0),
            context=ctx,
        )
        self.tokens.extend(tokens)
        self.steps.append((mode, old, len(tokens), tuple(new)))
        return logits

    def prepare_prompt(self, target):
        """Compare canonical template tokens, preserving only their exact prefix."""
        common = 0
        for a, b in zip(self.tokens, target):
            if a != b:
                break
            common += 1
        # Recompute the last position if the entire target is cached: KV alone
        # does not provide its logits. This harness does not retain old logits.
        common = min(common, len(target) - 1)
        if common < len(self.tokens):
            alloc = self.runner.token_to_kv_pool_allocator
            retained_page_end = ((common + alloc.page_size - 1) // alloc.page_size) * alloc.page_size
            alloc.free(self.slots[retained_page_end:])
            self.slots = self.slots[:common]
            self.tokens = self.tokens[:common]
        return self.append(target[common:]), common

    def close(self):
        if not self.closed:
            if torch.device(self.runner.device).type == 'cuda':
                torch.cuda.synchronize(self.runner.device)
            self.runner.token_to_kv_pool_allocator.free(self.slots)
            self.runner.req_to_token_pool.free_rows([self.row])
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
