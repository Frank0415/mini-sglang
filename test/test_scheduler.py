"""Single-item scheduling lifecycle with real CPU pools, no model required."""
import unittest
from types import SimpleNamespace

import torch

from src.mem_cache.allocator import PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.model_executor.forward_batch_info import ForwardMode
from src.scheduler.batch import ScheduleBatch
from src.scheduler.request import Request, RequestStatus
from src.scheduler.scheduler import Scheduler


def make_runner(pages=4, context_len=32):
    return SimpleNamespace(
        device='cpu', runtime=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=2)),
        req_to_token_pool=ReqToTokenPool(2, context_len, torch.device('cpu')),
        token_to_kv_pool_allocator=PagedTokenToKVPoolAllocator(4, pages),
    )


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.runner = make_runner()
        self.scheduler = Scheduler(self.runner)

    def prepare(self):
        return ScheduleBatch.prepare(self.scheduler.get_next_requests(), self.runner)

    def assert_released(self, req):
        self.assertEqual(self.runner.req_to_token_pool.available_slots(), 2)
        self.assertEqual(set(self.runner.token_to_kv_pool_allocator.free_pages), {1,2,3,4})
        self.assertEqual(len(self.runner.token_to_kv_pool_allocator.free_pages), 4)
        self.assertIsNone(req.req_pool_index)
        self.assertEqual(req.slots, [])
        self.assertEqual(req.cached_len, 0)

    def test_prefill_decode_cross_page_finish_then_next_request(self):
        a = Request('a', [10,11,12], 3)
        b = Request('b', [30], 1)
        self.scheduler.add_request(a)
        self.scheduler.add_request(b)
        first = self.prepare()
        self.assertEqual(first.mode, ForwardMode.EXTEND)
        self.assertEqual(first.input_ids.tolist(), [[10,11,12]])
        self.assertEqual(first.position_ids.tolist(), [[0,1,2]])
        self.assertEqual(a.cached_len, 0)  # Preparation is not a successful forward.
        self.scheduler.process_result(first, 20)
        self.assertEqual(a.cached_len, 3)
        self.assertEqual(a.output_ids, [20])
        second = self.prepare()
        self.assertIs(second.requests[0], a)
        self.assertEqual(second.mode, ForwardMode.DECODE)
        self.assertEqual(second.input_ids.tolist(), [[20]])
        self.assertEqual(second.position_ids.tolist(), [[3]])
        self.assertEqual(len(self.runner.token_to_kv_pool_allocator.free_pages), 3)
        self.scheduler.process_result(second, 21)
        third = self.prepare()
        self.assertEqual(third.position_ids.tolist(), [[4]])
        self.assertEqual(len(self.runner.token_to_kv_pool_allocator.free_pages), 2)
        self.scheduler.process_result(third, 22)
        self.assertEqual(a.status, RequestStatus.FINISHED)
        self.assertEqual(a.finish_reason, 'length')
        self.assertEqual(a.output_ids, [20,21,22])
        self.assert_released(a)
        next_batch = self.prepare()
        self.assertIs(next_batch.requests[0], b)
        self.scheduler.process_result(next_batch, 2)
        self.assertEqual(b.finish_reason, 'eos')
        self.assertEqual(self.scheduler.get_next_requests(), [])
        self.assert_released(b)

    def test_abort_waiting_keeps_running_resources(self):
        a,b = Request('a',[10],5), Request('b',[20],5)
        self.scheduler.add_request(a); self.scheduler.add_request(b)
        batch = self.prepare()
        pages = self.runner.token_to_kv_pool_allocator.free_pages.copy()
        self.scheduler.abort('b')
        self.assertEqual(b.status, RequestStatus.ABORTED)
        self.assertEqual(self.runner.token_to_kv_pool_allocator.free_pages, pages)
        self.assertEqual(self.scheduler.get_next_requests(), [a])
        self.scheduler.abort('a')
        self.scheduler.abort('a')
        self.assert_released(a)
        self.assertEqual(self.scheduler.get_next_requests(), [])

    def test_allocation_failure_preserves_existing_row_until_abort(self):
        self.runner = make_runner(pages=1)
        self.scheduler = Scheduler(self.runner)
        a = Request('a',[10,11,12,13],5)
        self.scheduler.add_request(a)
        first = self.prepare(); self.scheduler.process_result(first,20)
        row = a.req_pool_index
        slots = a.slots.copy()
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(a.req_pool_index,row)
        self.assertEqual(a.slots,slots)
        self.assertEqual(a.cached_len,4)
        self.scheduler.abort('a')
        self.assertEqual(self.runner.token_to_kv_pool_allocator.free_pages,[1])
        self.assertEqual(self.runner.req_to_token_pool.available_slots(),2)

    def test_failed_forward_can_be_aborted_without_committing_cache(self):
        a=Request('a',[10,11,12,13,14],5)
        self.scheduler.add_request(a)
        self.prepare()
        # Caller aborts after runner.forward raises; allocated != committed.
        self.assertEqual(a.cached_len,0)
        self.assertEqual(len(a.slots),5)
        self.scheduler.abort('a')
        self.assert_released(a)

    def test_reject_duplicate_id(self):
        self.scheduler.add_request(Request('a',[10],1))
        with self.assertRaises(ValueError):
            self.scheduler.add_request(Request('a',[11],1))
        self.assertEqual(len(self.scheduler.requests),1)

    def test_prepare_rejects_multiple_requests(self):
        with self.assertRaises(ValueError):
            ScheduleBatch.prepare([Request('a',[10],1),Request('b',[11],1)], self.runner)
        self.assertEqual(self.runner.req_to_token_pool.available_slots(),2)

    def test_context_overflow_is_checked_before_allocation(self):
        a=Request('a',list(range(33)),1)
        self.scheduler.add_request(a)
        with self.assertRaises(ValueError):self.prepare()
        self.scheduler.abort('a')
        self.assert_released(a)

    def test_reject_result_from_another_request(self):
        a=Request('a',[10],2)
        self.scheduler.add_request(a)
        batch=self.prepare()
        batch.requests=[Request('other',[10],2)]
        with self.assertRaises(ValueError):self.scheduler.process_result(batch,20)
        self.assertEqual(a.cached_len,0)
        self.scheduler.abort('a')

    def test_reject_duplicate_result_without_advancing_request(self):
        a = Request('a', [10], 3)
        self.scheduler.add_request(a)
        batch = self.prepare()
        self.scheduler.process_result(batch, torch.tensor([20]))
        with self.assertRaises(ValueError):
            self.scheduler.process_result(batch, 21)
        self.assertEqual(a.output_ids, [20])
        self.assertEqual(a.cached_len, 1)
        self.scheduler.abort('a')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class SchedulerModelTests(unittest.TestCase):
    @torch.inference_mode()
    def test_two_requests_through_real_model_and_greedy_sampler(self):
        from pathlib import Path

        from scripts.diagnose_kv_numerics import control
        from src.layers.sampler import greedy_sample
        from src.model_executor.model_runner import ModelRunner
        model_files = list((Path(__file__).resolve().parents[1] / 'models').rglob('config.json'))
        if not model_files:
            self.skipTest('Local model unavailable')
        runner = ModelRunner(model_files[0].parent)
        scheduler = Scheduler(runner)
        requests = [Request('a', [10,11,12], 3), Request('b', [20,21], 2)]
        for request in requests:
            scheduler.add_request(request)
        seen = []
        try:
            # Control arithmetic so this test isolates scheduling, not known
            # BF16 shape-dependent numerical differences.
            with control(runner, 'bf16_canonical'):
                while current := scheduler.get_next_requests():
                    batch = ScheduleBatch.prepare(current, runner)
                    logits = runner.forward(batch.input_ids, batch.position_ids, batch.context)
                    request = current[0]
                    full = runner.forward(torch.tensor(
                        [request.input_ids + request.output_ids], device=runner.device
                    ))
                    torch.testing.assert_close(logits, full[:, -batch.input_len:], rtol=0, atol=0)
                    scheduler.process_result(batch, greedy_sample(logits[:, -1]))
                    seen.append(request.request_id)
            self.assertEqual(seen, sorted(seen))
            self.assertTrue(all(r.status == RequestStatus.FINISHED for r in requests))
            self.assertTrue(all(r.output_ids for r in requests))
            self.assertEqual(runner.req_to_token_pool.available_slots(), 4)
            self.assertEqual(len(runner.token_to_kv_pool_allocator.free_pages), 512)
        finally:
            for request in requests:
                scheduler.abort(request.request_id)


if __name__=='__main__':unittest.main()
