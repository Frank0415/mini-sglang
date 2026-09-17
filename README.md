# mini-sglang

# Loading a Model and Running Inference

This example uses `smollm2-135m-instruct` to show the model loading and inference workflow.

## Model Initialization

```text
ModelRunner initialization
│
├── 1. Read the model configuration
│      ModelConfig.from_pretrained(model_dir)
│
├── 2. Create the model and load weights
│      ModelRegistry → LlamaForCausalLM → load_weights
│
├── 3. Generate configurations for the three components
│      make_pool_configs
│      ├── Model parameters: layer count, KV head count, and head dimension
│      ├── Model precision: actual weight dtype
│      └── Runtime settings: page size, page count, request count, and length limit
│
├── 4. Create cache components
│      ├── ReqToTokenPool
│      ├── PagedTokenToKVPoolAllocator
│      └── MHATokenToKVPool
│
├── 5. Create TorchAttnBackend
│      Reference the request mapping pool and KV storage pool
│
└── 6. Assemble ModelRuntime
       Store cache groups, backends, and layer mappings
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

## KV Cache Resource Management

```text
Q, K, V for the current step
│
├── Q ────────────────────────────────────┐
│                                         │
└── K, V                                  │
    │                                     │
    ▼                                     │
set_kv_buffer                             │
Write to the slots in out_cache_loc       │
    │                                     │
    ▼                                     │
Look up the request mapping table         │
Get the request's valid history slots     │
    │                                     │
    ▼                                     │
Read K, V from the current layer's buffer │
History + current input                   │
    │                                     │
    └──────────────────┬──────────────────┘
                       ▼
        Build the attention mask from positions
                       │
                       ▼
                  PyTorch SDPA
                       │
                       ▼
        Return results only for the current tokens
```

```text
Request A: row 2 of the mapping table
│
├── Logical position 0 → slot 12 ┐
├── Logical position 1 → slot 13 │
├── Logical position 2 → slot 14 ├── Page 3
├── Logical position 3 → slot 15 ┘
│
├── Logical position 4 → slot 28 ┐
└── Logical position 5 → slot 29 ├── Page 7
                        slot 30 │   Remaining slots reserved for request A
                        slot 31 ┘
```

## ForwardContext Resource Management

```text
ForwardContext
│
├── runtime: long-lived shared resources
│   │
│   ├── groups[group_id]
│   │   ├── req_to_token_pool
│   │   ├── kv_pool
│   │   ├── allocator
│   │   └── backend
│   │
│   └── layers[model_layer_id]
│       ├── group_id: which cache group to use
│       └── local_layer_id: which layer within the pool to use
│
└── batch: execution information for the current step
    │
    ├── mode: EXTEND / DECODE
    ├── seq_lens: total lengths including the current input
    ├── extend_lens: lengths of the new input in this step
    │
    └── cache_batches[group_id]
        ├── req_pool_indices: row indices in the request mapping table
        └── out_cache_loc: KV write slots for the current step
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
