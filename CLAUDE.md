# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

Nano-vLLM 是 vLLM 离线推理路径的约 1200 行重写：仅支持 Qwen3、仅 CUDA、强依赖 FlashAttention-2、单机 TP。对外 API 只有 `nanovllm.LLM` 与 `nanovllm.SamplingParams`。

## 常用命令

仓库无测试、无 lint、无构建脚本。`pip install -e .` 后跑 `python example.py` 或 `python bench.py`。两者都**硬编码模型路径 `~/huggingface/Qwen3-0.6B/`**，改路径直接编辑文件里的 `path = ...`。

## 架构（按阅读顺序）

1. **引擎主循环** `engine/llm_engine.py`：`generate` 驱动 `add_request → step`；每个 step = `Scheduler.schedule()` + `ModelRunner.call("run", ...)` + `Scheduler.postprocess()`。TP 下 rank 0 inline，rank ≥1 是子进程跑 `ModelRunner.loop()`。
2. **调度器 + 分页 KV** `engine/{scheduler,block_manager,sequence}.py`：严格 prefill 优先、两种 batch 互斥；单序列塞不下预算时允许 **chunked prefill**（仅限 step 内第一个序列）；decode 分不到 block 时 LIFO 抢占 running 尾部。`BlockManager` 用 xxhash64（折入前块 hash）做前缀缓存，首次 miss 会 latch 让后续块都新分配。
3. **模型执行 + attention** `engine/model_runner.py`, `layers/attention.py`, `models/qwen3.py`：`ModelRunner` 按 `total * gpu_memory_utilization - used - peak + current` 残余显存切 KV cache；非 eager 对 `[1,2,4,8] + range(16, max_bs+1, 16)` 组 decode batch size 捕获 CUDA graph（prefill 和 bs>512 走 eager）。`Attention.forward` 读 `utils/context.py` 的 `_CONTEXT` 单例，按 `is_prefill` + 是否有 `block_tables` 分派到 `flash_attn_varlen_func` / `flash_attn_with_kvcache`；Triton `store_kvcache_kernel` 用 `slot_mapping`（`-1` 跳过，服务于 graph padding）写回分页槽位。TP 在 `layers/linear.py`，`RowParallelLinear` 的 bias **仅 rank 0 加**。

## 编辑时的坑

- `Sequence.block_size` 是**类属性**，在 `LLMEngine.__init__` 时写入；更早实例化会用到默认 256。
- `Context` 是 `ModelRunner.run` 里 reset 的全局单例，不要在 `run` 外引用它的张量。
- 只有 rank 0 从 `run` 返回采样结果，tokenizer / scheduler / sampler 都只跑在 rank 0。
- `BlockManager.allocate` 的前缀缓存等值判断是整块 `token_ids` 列表比较；换 hash 算法要保留这个兜底。
- TP 启动用 `tcp://localhost:2333` 做 NCCL rendezvous，单机同时跑两个引擎会冲突。

## 相关文档

- [todo.md](./todo.md)：用户后续学习/开发计划（5 个由易到难的方向）。规划新功能前先对照这里。
- [.learn_suggest.md](./.learn_suggest.md)：用户学习笔记，记录对设计决策的理解，可作补充上下文。
