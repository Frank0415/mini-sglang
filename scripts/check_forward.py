import argparse
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.model_executor.model_runner import ModelRunner

DEFAULT_MODEL_DIR = PROJECT_ROOT / (
    "models/smollm2-135m-instruct/"
    "models--HuggingFaceTB--SmolLM2-135M-Instruct/"
    "snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
PROMPTS = (
    "What is gravity?",
    "Calculate 7 + 5 and give a short answer.",
    "Explain why the sky looks blue in two sentences.",
)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    # Explicit initial BF16 comparison thresholds, not bitwise equality.
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("Token count must be positive and tolerances nonnegative")
    return args


@torch.inference_mode()
def hf_forward(model, input_ids):
    # Match the local backend's flash SDPA selection; disable dropout via eval().
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return model(input_ids=input_ids, use_cache=False).logits


@torch.inference_mode()
def greedy(forward, input_ids, max_new_tokens, eos_ids):
    sequence = input_ids.clone()
    generated = []
    for _ in range(max_new_tokens):
        logits = forward(sequence)
        if not torch.isfinite(logits).all():
            raise AssertionError("Non-finite logits during generation")
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        token_id = next_token.item()
        generated.append(token_id)
        sequence = torch.cat((sequence, next_token), dim=1)
        if token_id in eos_ids:
            break
    return generated


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires CUDA for the local flash attention backend")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    runner = ModelRunner(args.model_dir, device="cuda")
    model = runner.model
    reference = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    eos = reference.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: {type(model).__name__}; layers: {len(model.model.layers)}")
    print(f"Dtype: {next(model.parameters()).dtype}; KV cache: disabled")
    print(f"BF16 tolerance: atol={args.atol}, rtol={args.rtol}")
    shared = model.lm_head.weight is model.model.embed_tokens.weight
    print(f"Shared embedding/output weights: {shared}")
    if model.config.tie_word_embeddings and not shared:
        raise AssertionError("Configured shared weights are not tied")

    failures = []
    for index, prompt in enumerate(PROMPTS, 1):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to("cuda")
        actual = runner.forward(input_ids)
        expected = hf_forward(reference, input_ids)
        shape = (*input_ids.shape, model.config.vocab_size)
        if tuple(actual.shape) != shape or tuple(expected.shape) != shape:
            raise AssertionError(f"Expected {shape}, got {actual.shape}, {expected.shape}")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError("Forward produced non-finite logits")
        error = (actual.float() - expected.float()).abs()
        close = torch.isclose(actual.float(), expected.float(), atol=args.atol, rtol=args.rtol)
        next_match = torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))
        all_top1 = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
        print(f"\n[{index}] {prompt}")
        print(f"Shape: {tuple(actual.shape)}; finite: yes")
        print(f"Logit error: max={error.max().item():.6f}, mean={error.mean().item():.6f}")
        print(f"Outside tolerance: {(~close).sum().item()}/{close.numel()}")
        print(f"All-position top-1 agreement: {all_top1:.2%}; next-token match: {next_match}")
        if not close.all():
            failures.append(f"prompt {index}: logits exceed tolerance")
        if not next_match:
            failures.append(f"prompt {index}: next token differs")
        actual_ids = greedy(runner.forward, input_ids, args.max_new_tokens, eos_ids)
        expected_ids = greedy(lambda ids: hf_forward(reference, ids), input_ids,
                              args.max_new_tokens, eos_ids)
        print(f"Local token IDs: {actual_ids}")
        print(f"HF token IDs:    {expected_ids}")
        print(f"Local text: {tokenizer.decode(actual_ids)!r}")
        print(f"HF text:    {tokenizer.decode(expected_ids)!r}")
        if actual_ids != expected_ids:
            failures.append(f"prompt {index}: greedy sequence differs")
    if failures:
        raise AssertionError("Validation failed: " + "; ".join(failures))
    print(f"\nPASS: {len(PROMPTS)} prompts; forward logits and greedy sequences agree.")


if __name__ == "__main__":
    main()
