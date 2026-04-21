# nano-vllm 学习路线 TODO

最终目标：借助这个项目的迭代，理解**模型推理系统设计**与**真正的 vLLM 项目**。

下表按建议执行顺序排列（而非"从易到难"）。每项的价值、代码量、前置知识写在表下的详细说明里。

## 总览

| # | 阶段 | 内容 | 预计周期 | 代码量 | 难度 | 对目标的 ROI |
|---|---|---|---|---|---|---|
| 1 | P1 预热 | 指标收集 + 流式 generator | 2–3 天 | ~150 行 | 易 | ★★☆☆☆ |
| 2 | P2 快手项 | `free_block_ids` 改 LRU + stop strings | 2–3 天 | ~180 行 | 易 | ★★★☆☆ |
| 3 | P3 模型支持 | Llama3 **或** Qwen3-MoE | 1–2 周 | 400–800 行 | 中 | ★★★★☆（MoE 为 ★★★★★） |
| 4 | P4 核心 | 混合 prefill + decode 批 | 2–3 周 | ~500 行 | 中–难 | ★★★★★ |
| 5 | P5 进阶 | `AsyncLLMEngine` + OpenAI 兼容 HTTP server | 2 周 | ~500 行 | 中 | ★★★★★ |
| 6 | P6 大项目 | Speculative decoding | 1 个月+ | 1000+ 行 | 难 | ★★★★★ |
| — | 可选 | FP8/AWQ 量化、CPU swap、自写 PagedAttention kernel、top-k/p/penalties 采样、argparse CLI | 按兴趣 | 不等 | 不等 | ★★☆☆☆ |

**说明**：
- ROI 按"对理解 vLLM 系统设计的帮助"打分，不是对代码价值打分。
- "可选"组里的项目要么偏数值/偏 kernel（和系统设计关系弱），要么是纯工程 busywork，按需穿插即可，不占主线。
- P1/P2 存在的意义是**为 P4 铺路**——没有指标，P4 做完也评估不出收益。

---

## P1. 预热：指标 + 流式（2–3 天，~150 行）

**要做的事**

- [ ] 在 `Scheduler` / `BlockManager` 中埋点，每 step 收集：
  - 前缀缓存命中率（本 step 新 allocate 的块中 hit 的比例）
  - `prefill_tokens` / `decode_tokens` / `preempted_seqs` 计数
  - KV 利用率 = `len(used_block_ids) / total_blocks`
  - 每 step GPU 端耗时（`torch.cuda.Event` 计）
- [ ] 加一个 `--log-stats N` 开关：每 N 步打印一行汇总。
- [ ] 把 `LLMEngine.generate` 改成 generator，逐 token 产出；保留原 list 返回的包装。

**学到什么 / 为什么值得**
- 指标是 P4 的**前置条件**：没有命中率和 KV 利用率，改完混合 batch 你也说不清"到底提升了什么"。
- 流式改造会强迫你明确 rank-0 tokenizer / sampler / postprocess 的所有权边界——这就是 `AsyncLLMEngine` 的雏形。

**前置知识**：几乎没有。`torch.cuda.Event` 异步计时的基本用法。

**跳过的东西**：argparse / 去硬编码路径。纯 busywork，需要时再加 20 行就够。

---

## P2. 快手项：LRU + stop strings（2–3 天，~180 行）

**要做的事**

- [ ] 把 `BlockManager.free_block_ids` 从 FIFO 改成 LRU：
  - 现状：`deque`，`_deallocate_block` 用 `append`、`_allocate_block` 取 `[0]`，等于 **FIFO**；这意味着刚释放、前缀 hash 还热的块会被**最先覆盖**，对前缀缓存命中率不利。
  - 目标：让"最久没被用"的空闲块优先被覆盖，保留最近释放的热块给前缀 hit。
  - 用 P1 的命中率指标**前后对比**，这才是这项的意义。
- [ ] 在 `Sequence` 增加 `stop_token_ids` / `stop_strings`，每步 `postprocess` 后检查：
  - token id stop 好做；
  - string stop 必须处理**跨 token 边界**（上一步最后几字节 + 这一步新字节），且 tokenizer 可能 merge 尾部——保留一个滑窗字符串比较。

**学到什么 / 为什么值得**
- LRU 的修改**量极小**（~30 行），但揭示了一个真实的生产问题：淘汰策略直接决定前缀缓存有没有用。vLLM 的 `BlockPool` 里有完整的 LRU + evictor 结构，你改完再去看就会秒懂。
- stop strings 是 tokenization 边界问题的经典 case，在 vLLM 里是 bug 聚集地。

**前置知识**：tokenizer 的 `decode(..., skip_special_tokens=...)` 行为、BPE merge 对尾部不稳定的影响。

---

## P3. 模型支持：Llama3 **或** Qwen3-MoE（1–2 周）

**两条分叉，二选一：**

### 3a. Llama3（~400–450 行，难度中）

- [ ] 在 `models/` 新增 `llama.py`，复用 `ColumnParallelLinear` / `QKVParallelLinear` / `RowParallelLinear` / `RMSNorm`。
- [ ] 差异点：
  - RoPE 走 `rope_scaling={'rope_type': 'llama3', ...}`（frequency interpolation），需要扩展现有 rotary 模块；
  - 没有 qk-norm（Qwen3 有）；
  - `lm_head` 可能 tie / 不 tie weights；
  - bias 配置与 Qwen3 不同。
- [ ] 扩展 `Qwen3ForCausalLM.packed_modules_mapping` 的套路到 `LlamaForCausalLM`，`utils/loader.py` 按 `config.architectures` 分发。

### 3b. Qwen3-MoE（~700–800 行，难度中–难）

- [ ] 和 3a 一样需要建一个新模型，但多了 **MoE 层**：
  - Router (`gate`) → top-k → 多个 expert 的 `MergedColumnParallelLinear` + `RowParallelLinear`；
  - 需要写一个 `FusedMoE` 层：把 top-k 路由后的 token 按 expert 分组，调用 grouped gemm（可先用 loop 版 `torch.einsum` 跑通正确性，再换 Triton/cuBLAS grouped gemm）。
- [ ] TP 下还要决定是走 **TP + expert replicate** 还是 **EP（expert parallel，不同 expert 放不同卡）**。真 vLLM 是 EP，入门可先只做 TP 版。

**学到什么 / 对比**
- 3a：让你摸清 `packed_modules_mapping` + `weight_loader` 的契约，理解"一个新模型要改什么"。价值中等。
- 3b：**能直接接触 vLLM 近两年最重的工程演进**（MoE/EP）。价值很高。代价是多出一整个 FusedMoE 层的调试时间。

**建议**：如果时间紧，做 3a；如果有 2+ 周预算且想真懂 vLLM，直接挑战 3b，3a 的收获其实是 3b 的子集。

**前置知识**：HF `AutoConfig` / `AutoModel` 的权重命名规律；MoE 的话要补一下 top-k routing 和 load balance loss（推理不需要后者，但理解模型结构要）。

---

## P4. 核心：混合 prefill + decode 批（2–3 周，~500 行）

**背景**：现在 `Scheduler.schedule()` **严格 prefill 优先**，一 step 里要么全 prefill 要么全 decode。这会让已经在跑的 decode 序列被一个新 prompt 的 prefill 整体阻塞，TTFT 好看但 TPOT 尖刺很严重。真正的 vLLM 通过 chunked prefill + 混合 batch 把这两者摊平——这是它"连续批处理"的灵魂。

**要做的事**

- [ ] **Scheduler**：允许同一 step 内容纳若干 decode seq + 一个 chunked prefill。
  - `running` 的 decode 先入 batch，直到达到 `max_num_batched_tokens` 预算；
  - 预算还有剩、且 `waiting` 非空，再往里塞一个 chunked prefill（chunk 长度 = 剩余预算）。
  - 抢占逻辑要顺带改：现在是"decode 预算不够 → 抢 running 尾部"，混合后仍然适用，但要评估新的优先级。
- [ ] **`ModelRunner.prepare_*`**：合并 `prepare_prefill` / `prepare_decode` 为一个统一 `prepare_step`。
  - `cu_seqlens_q` 里 decode 段是 `+1`、prefill 段是 `+N`；
  - `slot_mapping` 一段全写（prefill），一段每条一个槽位（decode）；
  - `block_tables` 对两种都要填（prefill chunked 时后续 chunk 也要读 cache）。
- [ ] **`Attention.forward`**：统一到 `flash_attn_varlen_func(..., block_table=...)`（decode 作 `seqlen_q=1` 的特例）。
  - **好消息**：查了一下代码，现在 prefill（`attention.py:67-70`）和 decode（`:72-74`）**都已经在给 FA-2 传 `block_table`** 了。kernel 层几乎没有阻力，主要是把 dispatch 分支合并、把 decode 的 `q.unsqueeze(1)` 换成统一布局。
- [ ] **CUDA graph**：这是**真正的设计决策**。混合 batch 的 prefill 尺寸动态，没法稳定 replay。两种选择：
  - 只对"纯 decode batch"继续 capture + replay；混合 batch 走 eager；
  - 或彻底放弃 graph。
  - 用 P1 的指标评估：混合带来的 TPOT 收益 vs 丢失 graph 的 per-step 开销，哪个赢？
- [ ] **Benchmark**：在 `bench.py` 上加两种 workload：
  - 纯 decode 稳态；
  - 不断有新请求进来的"类在线"场景。
  - 对比 TTFT P50/P99、TPOT P50/P99、吞吐。

**学到什么 / 为什么是核心**
- vLLM 的 `GPUModelRunnerBase` + `Scheduler` 结构和你改完的东西**几乎一一对应**；这一步做完再去读 vLLM 源码，会有"原来如此"的快感。
- 强迫你理解 FlashAttention varlen API 的全部约束（`cu_seqlens`、`block_table` 语义、`causal` 在混合 batch 下的语义）。
- CUDA graph 的取舍是教科书不会教你的真实工程题。

**前置知识**
- FlashAttention `flash_attn_varlen_func` 所有参数含义；
- `cu_seqlens` 的累加约定；
- CUDA graph capture/replay 的限制（shape 不能变、tensor 地址要稳定）。

---

## P5. 进阶：`AsyncLLMEngine` + HTTP server（2 周，~500 行）

**背景**：nano-vllm 现在是纯离线批处理——给一个 `prompts` 列表、等它全跑完。真 vLLM 是一个**在线服务**：请求随时到、随时取消、要流式返回、要 backpressure。这是一整套 nano-vllm 完全没有的东西。

**要做的事**

- [ ] 封装一个 `AsyncLLMEngine`：
  - 用 `asyncio.Queue` 接收请求；
  - 后台有一个 task 驱动现有的 `step()` 循环；
  - 每个请求对应一个 `asyncio.Queue` 接收生成的 token（复用 P1 的流式改造）；
  - 取消：`asyncio.CancelledError` 要能沿着请求 → 引擎传播，及时从 `waiting` / `running` 里移除并 `deallocate`。
- [ ] 用 FastAPI 写一个 OpenAI 兼容的 `/v1/completions` 和 `/v1/chat/completions`：
  - `stream=True` 走 SSE；
  - `stop` 字段（复用 P2）；
  - 基础的超时、并发上限、backpressure（队列满时返回 429）。
- [ ] 简单的端到端压测：`ab` / `wrk` / 自己写的 `asyncio` client。

**学到什么**
- 请求生命周期 → 引擎状态机的完整映射；
- 取消和资源回收的正确姿势（`deallocate` 不能漏）；
- 异步并发控制、SSE 流式传输。

**前置知识**：`asyncio`、`FastAPI`、SSE 基本格式。

**依赖**：P1 的流式 generator 是前提。

---

## P6. 大项目：Speculative Decoding（1 个月+，1000+ 行）

**背景**：小 draft 模型快速生成 K 个候选，大 target 模型一次并行 verify，按概率比例接受/拒绝。是 vLLM 近两年最重要的吞吐优化之一。

**关键依赖**：**强烈建议 P4 做完再做这个**。原因：verify 阶段本质上就是"让 target 在一个 decode step 里一次吃 K 个 token"，也就是 decode batch 里每条序列 `seqlen_q = K` 而非 1——这**正是 P4 建立的能力**。P4 没做，这里会在同一类问题上踩两遍坑。

**要做的事**

- [ ] 引擎层容纳两个 `ModelRunner`（draft + target）：
  - 最简单的版本：两个模型各占一部分显存，共享或独立的 KV cache；
  - 决策题：draft 的 KV 要不要也走分页？draft 是否要 CUDA graph？
- [ ] Draft 阶段：对每条序列自回归跑 K 步。
- [ ] Verify 阶段：target 一次前向，K 个位置拿到 K 组 logits；按 rejection sampling 规则决定接受到第几个 token。
- [ ] **KV 回滚**：rejected 的 token 要把 K/V 从分页 cache 里"撤销"。要点：
  - `Sequence.num_cached_tokens` 往回调；
  - 若某 block 因此变回"未满"，其 hash 要从 `hash_to_block_id` 里摘掉（否则会污染别人的前缀缓存！）；
  - `ref_count` 要保持正确。
- [ ] 度量平均接受长度、端到端加速比。

**学到什么**
- 两阶段调度下的资源分配（哪边该多给显存？draft 多跑几步还是少跑几步？）；
- KV 一致性在非单调前进时的维护；
- 在已有引擎上做**侵入式改造**的工程方法。

**前置知识**
- Spec decoding 的数学（rejection sampling，为什么等价于从 target 分布采样）；
- 对 P4 改造后的调度器和 attention 路径了如指掌。

---

## 可选项（不占主线，按兴趣穿插）

### FP8 / AWQ 量化
教的是**数值与 kernel 集成**，不是系统设计，偏离主目标。FP8 用 `torch._scaled_mm` 能做到 ~300 行；AWQ/GPTQ 主要是接 marlin / exllama 的胶水，~800 行。**不推荐作为主线**。

### CPU swap / 分层 KV（~200 行）
把当前"抢占即丢"改成"抢占时把 KV 块搬到 CPU、重调度时搬回"。会让你理解 vLLM 的 swap-based preemption。和 P4 正交，做完 P4 再加更容易。

### 自写 PagedAttention Triton kernel（~200 行 Triton）
替换 FA 的 decode 路径。对"vLLM 为什么维护自家 kernel"这个问题有直接回答。和系统设计关系中等。

### Top-k / top-p / repetition penalty 采样（~150 行）
现在 `sampler.py` 只有 Gumbel-max + temperature。扩展它是**纯数值工作**，对理解系统设计几乎无贡献。**需要时再加**。

### argparse / 去硬编码路径
Busywork，~50 行，需要时顺手做掉，不当作学习任务。
