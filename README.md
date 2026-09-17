# mini-sglang

# Loading a Model and Running Inference

This example uses `smollm2-135m-instruct` to show the model loading and inference workflow.

## Model Loading

```text
ModelConfig
├── architectures (hf_config.architectures)
└── hf_config
    ├── hidden_size
    ├── num_hidden_layers
    └── tie_word_embeddings
```

`model_loader/loader.py` loads the model and returns a model instance:

```python
model_class = ModelRegistry.get_model_class(model_config.architectures)
model = model_class(model_config.hf_config)
```

`model_loader/weight_utils.py` reads the weight files and provides an iterator over their weights. The model loads these weights through `model.load_weights`:

```python
model.load_weights(iter_weights(model_dir))
```

```text
HF model directory
→ configs/model_config.py: ModelConfig reads the complete configuration
→ models/registry.py: The registry finds the LlamaForCausalLM class
→ models/llama.py: LlamaForCausalLM creates the model layers from the configuration
```

## Model Structure

```text
LlamaForCausalLM
├── LlamaModel
│   ├── Embedding
│   ├── Multiple LlamaDecoderLayer instances
│   │   ├── Input RMSNorm
│   │   ├── LlamaAttention
│   │   ├── Residual connection
│   │   ├── Post-attention RMSNorm
│   │   ├── LlamaMLP
│   │   └── Residual connection
│   └── Final RMSNorm
└── LM head
```

## Inference Testing

Use `scripts/check_forward.py` to test whether the model loads and runs inference correctly.

# KV Cache

The KV cache design draws on SGLang's implementation. It includes a request mapping pool, `ReqToTokenPool`; a paged slot allocator, `PagedTokenToKVPoolAllocator`; and a KV storage pool, `MHATokenToKVPool`.

```text
Request index + token position in the sequence
                │
                ▼
          ReqToTokenPool
                │ Look up the physical slot
                ▼
         MHATokenToKVPool
                │
                ▼
       K and V for the corresponding layer

TokenToKVPoolAllocator
    └── Track which physical slots are available and which have been released
```

# Directory Structure

```text
src/
├── configs/
│   ├── model_config.py          # Model configuration, preserving HF architecture information
│   ├── load_config.py           # Weight paths and formats
│   ├── engine_config.py         # Device, precision, and runtime capacity
│   └── sampling_params.py       # Per-request generation parameters
├── models/
│   ├── registry.py             # Architecture names → model classes
│   └── llama.py                # Model structure, forward, and load_weights
├── model_loader/
│   ├── loader.py               # Coordinate the loading process
│   └── weight_utils.py         # Read weight files
├── model_executor/
│   ├── model_runner.py         # Prepare execution inputs and invoke the model
│   └── forward_batch_info.py   # Tensors, positions, and cache indices for the current step
├── scheduler/
│   ├── request.py              # Request state across steps
│   ├── batch.py                # Requests selected for the current step
│   ├── scheduler.py            # Admission, execution, completion, and resource reclamation
│   └── policy.py               # Request selection order
├── mem_cache/
│   ├── allocator.py            # Allocate and release slots
│   ├── memory_pool.py          # Request position mapping and physical KV storage
│   └── radix_cache.py          # Prefix matching, references, and eviction
├── layers/
│   ├── attention/
│   │   └── torch_backend.py    # Attention computation and KV access
│   ├── layers.py              # Shared computational layers
│   └── sampler.py             # Select tokens from scores
├── tokenizer/
│   └── tokenizer.py
└── engine.py
```
