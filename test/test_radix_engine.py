import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

from src.engine import Engine
from src.mem_cache.allocator import PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool
from src.mem_cache.radix_cache import RadixCache
from src.model_executor.forward_batch_info import ForwardBatch, ForwardContext, ForwardMode
from src.scheduler.request import RequestStatus


def nodes(cache):
    pending = list(cache.root_node.children.values())
    while pending:
        node = pending.pop()
        yield node
        pending.extend(node.children.values())


class RadixEngineFailureTests(unittest.TestCase):
    def test_forward_failure_after_prefix_hit_preserves_tree(self):
        rows = ReqToTokenPool(4, 64, "cpu")
        allocator = PagedTokenToKVPoolAllocator(4, 16)
        cache = RadixCache(4, rows, allocator)
        slots = allocator.alloc_extend(0, 8, -1, 8)
        cache._insert(list(range(8)), torch.tensor(slots))
        runner = SimpleNamespace(
            device="cpu", runtime=None, req_to_token_pool=rows,
            token_to_kv_pool_allocator=allocator, radix_cache=cache,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=2)),
        )

        def fail(*args, **kwargs):
            raise RuntimeError("injected forward failure")

        runner.forward = fail
        with patch("src.engine.ModelRunner", return_value=runner):
            engine = Engine("unused", device="cpu")
        rid = engine.add_request(list(range(10)), 4)
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            engine.step()
        self.assertEqual(engine.scheduler.requests[rid].status, RequestStatus.ABORTED)
        self.assertEqual(rows.available_slots(), 4)
        self.assertEqual(len(allocator.free_pages), 14)
        self.assertEqual(len(set(allocator.free_pages)), 14)
        self.assertEqual(cache._match_prefix(list(range(8)))[0].tolist(), slots)
        self.assertTrue(all(n.lock_ref == 0 for n in nodes(cache)))
        waiting = engine.add_request([42], 1)
        engine.abort(waiting)
        self.assertEqual(rows.available_slots(), 4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RadixEngineGPUTests(unittest.TestCase):
    fp32_math = False

    @torch.inference_mode()
    def test_two_conversations_two_rounds_against_full_recompute(self):
        model_dir = next(Path("models").glob("**/snapshots/*/config.json")).parent
        engine = Engine(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        runner = engine.runner
        if self.fp32_math:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            runner.model.float()
            runner.token_to_kv_pool.k_buffer = runner.token_to_kv_pool.k_buffer.float()
            runner.token_to_kv_pool.v_buffer = runner.token_to_kv_pool.v_buffer.float()
            self.enterContext(patch(
                "src.layers.attention.torch_backend.sdpa_kernel",
                side_effect=lambda _: sdpa_kernel(SDPBackend.MATH),
            ))
        original_forward = runner.forward
        report = []
        disagreements = []
        for title, prompts in (
            ("Kyoto", ["I am visiting Kyoto. Suggest one place to see.",
                       "What should I do there? Give one short suggestion."]),
            ("Cooking", ["Suggest a simple breakfast with eggs.",
                         "How can I add vegetables to it? Keep it brief."]),
        ):
            messages = []
            for turn, prompt in enumerate(prompts, 1):
                messages.append({"role": "user", "content": prompt})
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=False,
                )
                rid = engine.add_request(ids, 20)
                req = engine.scheduler.requests[rid]
                record = {"conversation": title, "turn": turn,
                          "prompt_tokens": len(ids), "steps": []}

                def checked_forward(input_ids, position_ids=None, context=None):
                    full_ids = req.input_ids + req.output_ids
                    prefix_len = req.cached_len
                    self.assertEqual(input_ids.tolist(), full_ids[prefix_len:])
                    self.assertEqual(position_ids.tolist(), list(range(prefix_len, len(full_ids))))
                    cache_batch = context.batch.cache_batches[0]
                    self.assertEqual(cache_batch.out_cache_loc.tolist(), req.slots[prefix_len:])
                    self.assertEqual(
                        runner.req_to_token_pool.req_to_token[
                            req.req_pool_index, :len(full_ids)
                        ].tolist(), req.slots,
                    )
                    shared = req.slots[:req.cache_protected_len]
                    before = [(
                        runner.token_to_kv_pool.get_key_buffer(layer)[shared].clone(),
                        runner.token_to_kv_pool.get_value_buffer(layer)[shared].clone(),
                    ) for layer in range(runner.token_to_kv_pool.num_layers)]
                    logits = original_forward(input_ids, position_ids, context)
                    for layer, (k, v) in enumerate(before):
                        self.assertTrue(torch.equal(k, runner.token_to_kv_pool.get_key_buffer(layer)[shared]))
                        self.assertTrue(torch.equal(v, runner.token_to_kv_pool.get_value_buffer(layer)[shared]))
                    lengths = torch.tensor([len(full_ids)], device="cuda")
                    full_context = ForwardContext(
                        runtime=runner.runtime,
                        batch=ForwardBatch(
                            mode=ForwardMode.EXTEND, seq_lens=lengths,
                            extend_lens=lengths, cache_batches={},
                        ),
                    )
                    reference = original_forward(
                        torch.tensor(full_ids, device="cuda"),
                        torch.arange(len(full_ids), device="cuda"), full_context,
                    )[-1].float()
                    actual = logits[-1].float()
                    picked, expected = actual.argmax().item(), reference.argmax().item()
                    top = reference.topk(2).values
                    step = {"prefix_tokens": prefix_len, "input_tokens": input_ids.numel(),
                            "max_logit_diff": (actual-reference).abs().max().item(),
                            "token": picked, "reference_token": expected,
                            "reference_top2_margin": (top[0]-top[1]).item()}
                    record["steps"].append(step)
                    if picked != expected:
                        disagreements.append((title, turn, len(record["steps"])-1, step))
                    return logits

                runner.forward = checked_forward
                try:
                    while req.status == RequestStatus.WAITING or req.status == RequestStatus.RUNNING:
                        engine.step()
                finally:
                    runner.forward = original_forward
                self.assertEqual(req.status, RequestStatus.FINISHED)
                self.assertEqual(runner.req_to_token_pool.available_slots(), 4)
                self.assertTrue(all(n.lock_ref == 0 for n in nodes(runner.radix_cache)))
                tree_pages = [s // 4 for n in nodes(runner.radix_cache)
                              for s in n.value.tolist()[::4]]
                free = runner.token_to_kv_pool_allocator.free_pages
                self.assertEqual(len(tree_pages), len(set(tree_pages)))
                self.assertEqual(len(free), len(set(free)))
                self.assertTrue(set(tree_pages).isdisjoint(free))
                self.assertEqual(set(tree_pages) | set(free), set(range(1, 513)))
                if turn == 2:
                    self.assertGreater(record["steps"][0]["prefix_tokens"], 0)
                result = engine.free_request(rid)
                answer = tokenizer.decode(result.output_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": answer})
                record["answer"] = answer
                record["retained_pages"] = len(tree_pages)
                report.append(record)
                print(json.dumps({k:v for k,v in record.items() if k != "steps"}, ensure_ascii=False), flush=True)
        runner.radix_cache._evict(512*4)
        self.assertEqual(len(set(runner.token_to_kv_pool_allocator.free_pages)), 512)
        suffix = "fp32-math" if self.fp32_math else "bf16"
        output = Path(f"/tmp/mini-sglang-radix-engine-{suffix}.json")
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Detailed report: {output}", flush=True)
        self.assertEqual(disagreements, [], f"Greedy differences: {disagreements}")


class RadixEngineFP32Tests(RadixEngineGPUTests):
    fp32_math = True


if __name__ == "__main__":
    unittest.main()
