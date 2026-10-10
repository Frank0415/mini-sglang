# mini-sglang

## 同步 RL 采样

创建 `Engine(model_dir, do_sample=True)`，按 softmax 概率随机采样。当前固定
temperature=1，使用完整词表。默认 `do_sample=False`，保持贪心生成。

`generate()` 和 `generate_batch()` 返回的 Request 包含 `output_ids` 与
`output_log_probs`，按生成位置一一对应，包含 EOS。`step()` 也返回本次词元的
`log_prob`。这些值来自模型的完整 softmax；启用随机采样时，它们就是采样概率的对数。

一轮采样结束后，调用 `engine.update_weights(model.state_dict().items(), version)`。
引擎加载参数、等待 GPU 写入完成、释放旧前缀缓存，再设置 `installed_version`。
有等待或运行中的请求时，更新会报错。Request 的 `weight_version` 和 `step()` 的
同名字段记录该请求实际使用的版本；初始版本为 0。

# 加载模型并进行推理

以 `smollm2-135m-instruct` 为例，展示加载并进行推理的实现。

## 模型初始化

```
ModelRunner 初始化
│
├── 1. 读取模型配置
│      ModelConfig.from_pretrained(model_dir)
│
├── 2. 创建模型并加载权重
│      ModelRegistry → LlamaForCausalLM → load_weights
│
├── 3. 生成三个组件的配置
│      make_pool_configs
│      ├── 模型参数：层数、KV 头数、每头维度
│      ├── 模型精度：实际权重 dtype
│      └── 运行设置：页大小、页数、请求数、长度上限
│
├── 4. 创建缓存组件
│      ├── ReqToTokenPool
│      ├── PagedTokenToKVPoolAllocator
│      └── MHATokenToKVPool
│
├── 5. 创建 TorchAttnBackend
│      引用请求映射池和 KV 存储池
│
└── 6. 组装 ModelRuntime
       保存缓存组、后端和层号映射
```

```
HF 模型目录
→ configs/model_config.py：ModelConfig：读取完整配置
→ models/registry.py：注册表：找到 LlamaForCausalLM 类
→ models/llama.py：LlamaForCausalLM：按配置创建模型层
```

## 模型结构

```
LlamaForCausalLM
├── LlamaModel
│   ├── Embedding
│   ├── 多个 LlamaDecoderLayer
│   │   ├── 输入 RMSNorm
│   │   ├── LlamaAttention
│   │   ├── 残差连接
│   │   ├── 后置 RMSNorm
│   │   ├── LlamaMLP
│   │   └── 残差连接
│   └── 最终 RMSNorm
└── LM head
```

## 推理测试

通过 scripts/check_forward.py 进行推理测试，检查模型是否能够正确加载和推理。

# KV Cache

KV Cache 借鉴了 sglang 的实现，包含请求映射池 `ReqToTokenPool`、槽位分配器 `PagedTokenToKVPoolAllocator`、KV 存储池 `MHATokenToKVPool`。

```
请求编号 + token 在序列中的位置
                │
                ▼
          ReqToTokenPool
                │ 查出物理槽位
                ▼
         MHATokenToKVPool
                │
                ▼
           对应层的 K、V

TokenToKVPoolAllocator
    └── 管理哪些物理槽位可以分配、哪些已经归还
```

## KV Cache 资源管理

```
本轮 Q、K、V
│
├── Q ─────────────────────────────┐
│                                  │
└── K、V                           │
    │                              │
    ▼                              │
set_kv_buffer                      │
写入 out_cache_loc 指定的槽位        │
    │                              │
    ▼                              │
查询请求映射表                       │
取得该请求的有效历史槽位               │
    │                              │
    ▼                              │
从当前层缓冲区读取 K、V               │
历史 + 本轮输入                      │
    │                              │
    └──────────────┬───────────────┘
                   ▼
          根据位置生成注意力遮罩
                   │
                   ▼
              PyTorch SDPA
                   │
                   ▼
          只返回本轮 token 的结果
```

```
请求 A：映射表第 2 行
│
├── 逻辑位置 0 → 槽位 12 ┐
├── 逻辑位置 1 → 槽位 13 │
├── 逻辑位置 2 → 槽位 14 ├── 页 3
├── 逻辑位置 3 → 槽位 15 ┘
│
├── 逻辑位置 4 → 槽位 28 ┐
└── 逻辑位置 5 → 槽位 29 ├── 页 7
                槽位 30 │   剩余位置留给请求 A
                槽位 31 ┘
```

## ForwardContext 资源管理

```
ForwardContext
│
├── runtime：长期共享的资源
│   │
│   ├── groups[group_id]
│   │   ├── req_to_token_pool
│   │   ├── kv_pool
│   │   ├── allocator
│   │   └── backend
│   │
│   └── layers[模型层号]
│       ├── group_id：使用哪个缓存组
│       └── local_layer_id：使用池内哪一层
│
└── batch：本轮执行信息
    │
    ├── mode：EXTEND / DECODE
    ├── seq_lens：包含本轮输入后的总长度
    ├── extend_lens：本轮新增输入长度
    │
    └── cache_batches[group_id]
        ├── req_pool_indices：请求映射表行号
        └── out_cache_loc：本轮 KV 写入槽位
```

# 目录结构

```
src/
├── configs/
│   ├── model_config.py          # 模型配置，保留 HF 架构信息
│   ├── load_config.py           # 权重路径、格式
│   ├── engine_config.py         # 设备、精度、运行容量
│   └── sampling_params.py       # 每个请求的生成参数
├── models/
│   ├── registry.py             # 架构名称 → 模型类
│   └── llama.py                # 模型结构、forward、load_weights
├── model_loader/
│   ├── loader.py               # 组织加载流程
│   └── weight_utils.py         # 读取权重文件
├── model_executor/
│   ├── model_runner.py         # 构造执行输入、调用模型
│   └── forward_batch_info.py   # 本轮张量、位置和缓存索引
├── scheduler/
│   ├── request.py              # 跨轮次的请求状态
│   ├── batch.py                # 本轮选中的请求
│   ├── scheduler.py            # 接纳、执行、结束和回收
│   └── policy.py               # 请求选择顺序
├── mem_cache/
│   ├── allocator.py            # 分配和归还槽位
│   ├── memory_pool.py          # 请求位置映射、实际 KV 存储
│   └── radix_cache.py          # 前缀匹配、引用和淘汰
├── layers/
│   ├── attention/
│   │   └── torch_backend.py    # 注意力计算与 KV 访问
│   ├── layers.py              # 共享计算层
│   └── sampler.py             # 从分数选择 token
├── tokenizer/
│   └── tokenizer.py
└── engine.py
```
