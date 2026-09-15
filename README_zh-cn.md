# mini-sglang

## 目录结构

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