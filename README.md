# mini-sglang

## Directory Structure

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
