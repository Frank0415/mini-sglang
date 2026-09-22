"""Two independent two-turn conversations through the real Engine and HF.generate.

Each generate call uses batch=1. Each implementation carries its own previous
answer into round two; input equality is recorded before comparing logits.
KV is released after each generation, as required by the current Engine contract.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine import Engine
from src.scheduler.request import RequestStatus

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / ('models/smollm2-135m-instruct/'
    'models--HuggingFaceTB--SmolLM2-135M-Instruct/'
    'snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac')
REPORT_PATH = ROOT / 'docs/diagnostics/engine_conversations.json'


class EngineWiringTests(unittest.TestCase):
    def test_forward_failure_preserves_error_and_releases_resources(self):
        from test.test_scheduler import make_runner
        runner = make_runner()
        runner.forward = Mock(side_effect=RuntimeError('forward failed'))
        with patch('src.engine.ModelRunner', return_value=runner):
            engine = Engine('unused', device='cpu')
        self.assertIs(engine.scheduler.runner, runner)
        request_id = engine.add_request([10, 11, 12], 2)
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            engine.step()
        self.assertEqual(engine.scheduler.requests[request_id].status, RequestStatus.ABORTED)
        self.assertIsNone(engine.scheduler.running)
        self.assertEqual(runner.req_to_token_pool.available_slots(), 2)
        self.assertEqual(set(runner.token_to_kv_pool_allocator.free_pages), {1, 2, 3, 4})


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class EngineConversationTests(unittest.TestCase):
    report_path = REPORT_PATH

    @classmethod
    def setUpClass(cls):
        if not (MODEL_DIR / 'config.json').is_file():
            raise unittest.SkipTest('Local SmolLM2 snapshot unavailable')
        cls.engine = Engine(MODEL_DIR, device='cuda')
        cls.reference = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa',
        ).to('cuda').eval()
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        cls.report = {
            'device': torch.cuda.get_device_name(),
            'dtype': 'bfloat16', 'batch_size': 1, 'max_new_tokens': 20,
            'hf_use_cache': True, 'cross_turn_kv_reuse': False,
            'conversations': {},
        }

    @classmethod
    def tearDownClass(cls):
        cls.report_path.parent.mkdir(parents=True, exist_ok=True)
        cls.report_path.write_text(json.dumps(cls.report, indent=2, ensure_ascii=False) + '\n')

    @torch.inference_mode()
    def check_conversation(self, name, prompts):
        engine_messages, hf_messages = [], []
        runner = self.engine.runner
        original_forward = runner.forward
        failures, rounds = [], []
        pages = set(runner.token_to_kv_pool_allocator.free_pages)
        row_count = runner.req_to_token_pool.available_slots()
        self.report['conversations'][name] = rounds
        try:
            for round_no, text in enumerate(prompts, 1):
                engine_messages.append({'role': 'user', 'content': text})
                hf_messages.append({'role': 'user', 'content': text})
                local_ids = self.tokenizer.apply_chat_template(
                    engine_messages, tokenize=True, add_generation_prompt=True, return_dict=False)
                hf_ids = self.tokenizer.apply_chat_template(
                    hf_messages, tokenize=True, add_generation_prompt=True, return_dict=False)
                scores, step_shapes = [], []

                def record(input_ids, position_ids=None, context=None):
                    self.assertEqual(input_ids.shape[0], 1)
                    out = original_forward(input_ids, position_ids, context)
                    scores.append(out[:, -1].detach().float().cpu())
                    step_shapes.append(list(input_ids.shape))
                    return out

                runner.forward = record
                result = self.engine.generate(local_ids, max_new_tokens=20)
                runner.forward = original_forward
                inputs = torch.tensor([hf_ids], device='cuda')
                output = self.reference.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=False, max_new_tokens=20, use_cache=True,
                    repetition_penalty=1.0, no_repeat_ngram_size=0,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                reference_ids = output.sequences[0, len(hf_ids):].tolist()
                answer = self.tokenizer.decode(result.output_ids, skip_special_tokens=True)
                reference_answer = self.tokenizer.decode(reference_ids, skip_special_tokens=True)
                mismatch = next((i for i, (a,b) in enumerate(zip(result.output_ids, reference_ids)) if a != b), None)
                same = result.output_ids == reference_ids
                errors = []
                # Only compare distributions while both paths have identical histories.
                if local_ids == hf_ids:
                    for i in range(min(len(scores), len(output.scores))):
                        if mismatch is not None and i > mismatch:
                            break
                        a = scores[i][0]
                        b = output.scores[i][0].float().cpu()
                        top = b.topk(2)
                        errors.append({
                            'step': i, 'max_abs': (a-b).abs().max().item(),
                            'mean_abs': (a-b).abs().mean().item(),
                            'reference_top2': top.indices.tolist(),
                            'reference_scores': top.values.tolist(),
                            'reference_margin': (top.values[0]-top.values[1]).item(),
                        })
                resources_ok = (
                    set(runner.token_to_kv_pool_allocator.free_pages) == pages
                    and len(runner.token_to_kv_pool_allocator.free_pages) == len(pages)
                    and runner.req_to_token_pool.available_slots() == row_count
                    and not self.engine.scheduler.requests
                    and self.engine.scheduler.running is None
                )
                rounds.append({
                    'round': round_no, 'user': text,
                    'same_prompt_tokens': local_ids == hf_ids,
                    'engine_prompt_length': len(local_ids), 'hf_prompt_length': len(hf_ids),
                    'engine_token_ids': result.output_ids, 'hf_token_ids': reference_ids,
                    'engine_text': answer, 'hf_text': reference_answer,
                    'tokens_equal': same, 'first_token_mismatch': mismatch,
                    'engine_finish_reason': result.finish_reason,
                    'forward_shapes': step_shapes, 'resources_reclaimed': resources_ok,
                    'aligned_logit_errors': errors,
                })
                print(f'\n{name} round {round_no}: tokens_equal={same}, resources_reclaimed={resources_ok}\n'
                      f'User: {text}\nEngine: {answer}\nHF:     {reference_answer}', flush=True)
                self.assertEqual(result.status, RequestStatus.FINISHED)
                self.assertTrue(resources_ok)
                self.assertEqual(step_shapes[0], [1, len(local_ids)])
                self.assertTrue(all(shape == [1,1] for shape in step_shapes[1:]))
                if not same:
                    failures.append(f'round {round_no}: engine/HF token sequences differ at {mismatch}')
                engine_messages.append({'role': 'assistant', 'content': answer})
                hf_messages.append({'role': 'assistant', 'content': reference_answer})
            self.assertFalse(failures, '; '.join(failures))
        finally:
            runner.forward = original_forward
            for rid in list(self.engine.scheduler.requests):
                self.engine.abort(rid)

    def test_conversation_name(self):
        self.check_conversation('name', (
            'My name is Alice. Please greet me briefly.',
            'What is my name? Answer briefly.',
        ))

    def test_conversation_travel(self):
        self.check_conversation('travel', (
            'I am visiting Kyoto. Suggest one place to see.',
            'What city am I visiting? Answer briefly.',
        ))


class EngineSingleTurnTests(EngineConversationTests):
    """Two separate prompts, one turn each, using the same batch-one engine."""

    report_path = ROOT / 'docs/diagnostics/engine_single_turn.json'

    def test_conversation_name(self):
        self.check_conversation('name', (
            'My name is Alice. Please greet me briefly.',
        ))

    def test_conversation_travel(self):
        self.check_conversation('travel', (
            'I am visiting Kyoto. Suggest one place to see.',
        ))


if __name__ == '__main__':
    unittest.main()
