"""Real SmolLM2 acceptance test: one request, three chat-template turns.

Teacher-forcing uses full-recompute greedy tokens after comparing each cached
prediction. This keeps testing all three turns even when an early cached token
is wrong. If every next-token check passes, the greedy sequences also agree.
The allocation/turn loop lives only in test/kv_cache_harness.py, not the engine.
"""
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.model_executor.forward_batch_info import ForwardMode
from src.model_executor.model_runner import ModelRunner
from test.kv_cache_harness import CacheHarness

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / ('models/smollm2-135m-instruct/'
    'models--HuggingFaceTB--SmolLM2-135M-Instruct/'
    'snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CachedChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / 'config.json').exists():
            raise unittest.SkipTest('Local SmolLM2 snapshot is not available')
        cls.runner = ModelRunner(MODEL_DIR, device='cuda')
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    @torch.inference_mode()
    def test_three_turn_chat(self):
        runner = self.runner
        allocator = runner.token_to_kv_pool_allocator
        initial_pages = set(allocator.free_pages)
        initial_rows = runner.req_to_token_pool.available_slots()
        messages = []
        problems = []
        prompts = (
            'My name is Mira. Say hello in one short sentence.',
            'What is my name?',
            'Use my name in a short goodbye.',
        )
        eos = runner.model.config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
        previous_cached_length = 0
        with CacheHarness(runner) as cache:
            request_row = cache.row
            for turn, user_text in enumerate(prompts, 1):
                messages.append({'role': 'user', 'content': user_text})
                target = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=False)
                cached, reused = cache.prepare_prompt(target)
                if turn > 1:
                    self.assertGreater(reused, 0, 'Later turns must reuse the prior token prefix')
                    # Prompt preparation must not silently start a new request.
                    self.assertEqual(cache.row, request_row)
                generated = []
                max_error = 0.0
                prefill_tokens = len(target) - reused
                for step in range(8):
                    full = runner.forward(torch.tensor([cache.tokens], device='cuda'))
                    expected = full[:, -cached.shape[1]:]
                    error = (cached.float() - expected.float()).abs()
                    max_error = max(max_error, error.max().item())
                    if not torch.isfinite(cached).all() or not torch.isclose(
                        cached.float(), expected.float(), atol=.125, rtol=.02
                    ).all():
                        problems.append(f'turn {turn}, step {step}: logits differ (max={error.max().item():.4f})')
                    expected_token = full[:, -1].argmax(-1).item()
                    actual_token = cached[:, -1].argmax(-1).item()
                    if actual_token != expected_token:
                        problems.append(f'turn {turn}, step {step}: greedy {actual_token} != {expected_token}')
                    generated.append(expected_token)
                    used_pages = len(initial_pages) - len(allocator.free_pages)
                    required_pages = (len(cache.tokens) + allocator.page_size - 1) // allocator.page_size
                    self.assertEqual(used_pages, required_pages)
                    if expected_token in eos_ids or step == 7:
                        break  # Last sampled token has no KV yet.
                    cached = cache.append([expected_token], ForwardMode.DECODE)
                answer = self.tokenizer.decode(generated, skip_special_tokens=True)
                messages.append({'role': 'assistant', 'content': answer})
                print(f'\nTurn {turn}: prompt={len(target)}, reused={reused}, '
                      f'prefill_computed={prefill_tokens}, previously_cached={previous_cached_length}, '
                      f'max_logit_error={max_error:.6f}\nReference answer: {answer}', flush=True)
                previous_cached_length = len(cache.tokens)
            self.assertEqual(len(messages), 6)
            self.assertTrue(any(mode == ForwardMode.DECODE for mode, *_ in cache.steps))
            self.assertTrue(all(count == 1 for mode, _, count, _ in cache.steps if mode == ForwardMode.DECODE))
        self.assertEqual(set(allocator.free_pages), initial_pages)
        self.assertEqual(len(allocator.free_pages), len(initial_pages))
        self.assertEqual(runner.req_to_token_pool.available_slots(), initial_rows)
        self.assertFalse(problems, '\n'.join(problems[:12]) + f'\nTotal failed checks: {len(problems)}')

    @torch.inference_mode()
    def test_close_reclaims_pages_and_new_request_can_reuse_them(self):
        runner = self.runner
        pages = set(runner.token_to_kv_pool_allocator.free_pages)
        rows = runner.req_to_token_pool.available_slots()
        for prompt in ([1, 2, 3, 4, 5], [9, 8, 7]):
            with CacheHarness(runner) as cache:
                out = cache.append(list(prompt))
                expected = runner.forward(torch.tensor([prompt], device='cuda'))
                torch.testing.assert_close(out, expected, rtol=.02, atol=.125)
            cache.close()  # Test caller's idempotent close, not allocator double-free.
            self.assertEqual(set(runner.token_to_kv_pool_allocator.free_pages), pages)
            self.assertEqual(len(runner.token_to_kv_pool_allocator.free_pages), len(pages))
            self.assertEqual(runner.req_to_token_pool.available_slots(), rows)

    @torch.inference_mode()
    def test_changed_prompt_preserves_partial_tail_page(self):
        runner = self.runner
        pages = set(runner.token_to_kv_pool_allocator.free_pages)
        with CacheHarness(runner) as cache:
            cache.append([1, 2, 3, 4, 5, 6, 7])
            target = [1, 2, 3, 9, 10]
            out, reused = cache.prepare_prompt(target)
            self.assertEqual(reused, 3)
            self.assertEqual(cache.tokens, target)
            expected = runner.forward(torch.tensor([target], device='cuda'))[:, reused:]
            torch.testing.assert_close(out, expected, rtol=.02, atol=.125)
        self.assertEqual(set(runner.token_to_kv_pool_allocator.free_pages), pages)


if __name__ == '__main__':
    unittest.main()
