# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

Nano-vLLM is a from-scratch, ~1,200-line re-implementation of vLLM's offline inference path. It is intentionally narrow in scope: one model family (Qwen3, via `transformers.Qwen3Config`), CUDA-only, FlashAttention-2 required, single-node tensor parallelism. The public surface is just `nanovllm.LLM` and `nanovllm.SamplingParams` — `LLM` is a thin subclass of `LLMEngine`.

## Common commands

This repo has no test suite, linter, or build script. Development is done by running the two entry points against a local HuggingFace model checkout.

- Install for development: `pip install -e .` (requires CUDA + a working `flash-attn` build)
- Download the canonical test model:
  ```bash
  huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
    --local-dir ~/huggingface/Qwen3-0.6B/ --local-dir-use-symlinks False
  ```
- Run the example (small prompts, eager mode): `python example.py`
- Run the benchmark (256 random sequences, CUDA graph on): `python bench.py`
- Both entry points hard-code the model path to `~/huggingface/Qwen3-0.6B/`; edit the `path = ...` line to point elsewhere.

## Architecture

The inference stack is three layers deep. Reading them in this order is the fastest way to become productive.

### 1. Engine loop (`nanovllm/engine/llm_engine.py`)
`LLMEngine.generate` drives a classic `add_request` → `step` loop until all sequences finish. Each `step` calls `Scheduler.schedule()` to pick a batch, `ModelRunner.call("run", ...)` to execute it, then `Scheduler.postprocess()` to append tokens / retire sequences. The engine forks one subprocess per extra TP rank at init time — rank 0 runs inline, rank ≥ 1 runs in its own `mp.Process` and enters a blocking `ModelRunner.loop()`.

### 2. Scheduler + paged KV cache (`engine/scheduler.py`, `engine/block_manager.py`, `engine/sequence.py`)
- The scheduler strictly prefers **prefill over decode**: as long as any `waiting` sequence fits in the `max_num_batched_tokens` budget it keeps prefilling; only when `waiting` is drained (or budget is full) does it run a decode step. Mixed prefill/decode batches do not exist.
- **Chunked prefill**: if a single waiting sequence doesn't fit the remaining token budget, the scheduler splits its prefill into this step's chunk and leaves it at the head of `waiting`. This is only allowed for the first scheduled seq in the step (`if remaining < num_tokens and scheduled_seqs: break`). `seq.num_cached_tokens` tracks how much prefill has already been persisted to KV cache.
- **Preemption**: in the decode path, if we cannot append a new block for some running sequence, the scheduler preempts the *tail* of `running` (LIFO), returning those blocks and moving the seq back to `waiting`. Preempted sequences re-prefill from their cached tokens when rescheduled.
- `BlockManager` implements paged KV with **prefix caching keyed by xxhash64 over the block's token ids** (with the previous block's hash folded in as a prefix). On `allocate`, each full-sized block is looked up in `hash_to_block_id`; a hit bumps `ref_count` and advances `seq.num_cached_tokens`; the first miss forces every subsequent block to be freshly allocated (`cache_miss` latches). `may_append` is called during decode to hash a block once it becomes full.
- `Sequence.block_size` is a **class variable** set once at `LLMEngine.__init__` from `config.kvcache_block_size`. The config asserts block size is a multiple of 256.

### 3. Model runner + attention (`engine/model_runner.py`, `layers/attention.py`, `models/qwen3.py`)
- `ModelRunner` owns the GPU: it initializes NCCL, loads weights, runs a dummy `warmup_model` batch, then **sizes the KV cache from residual memory** (`total * gpu_memory_utilization - used - peak + current`) and reshapes a single flat tensor view into each `Attention` module's `k_cache` / `v_cache`. If `enforce_eager=False` it also captures CUDA graphs for a fixed set of decode batch sizes (`[1,2,4,8] + range(16, max_bs+1, 16)`) sharing a single memory pool.
- **Cross-rank dispatch**: `ModelRunner.call(method_name, *args)` on rank 0 writes `(method_name, args)` as pickled bytes into a 1 MiB `SharedMemory` segment and sets one `mp.Event` per worker; workers block in `read_shm`, deserialize, and call the same method. This is how `run` and `exit` reach every GPU. Only rank 0 runs the tokenizer, scheduler, and sampler — workers return `None` from `run`.
- **Attention contract**: `layers/attention.Attention.forward` reads metadata from a module-level `_CONTEXT` (`utils/context.py`) populated by `ModelRunner.prepare_prefill` / `prepare_decode`. The same `Attention` module handles both phases by branching on `context.is_prefill`:
  - Prefill without prefix cache: `flash_attn_varlen_func(q, k, v, …)` over the just-projected tensors.
  - Prefill with prefix cache (`context.block_tables is not None`): same call, but `k`/`v` come from the paged cache and a `block_table` is passed.
  - Decode: `flash_attn_with_kvcache` with `cache_seqlens=context_lens`.
  - Always: a small Triton kernel `store_kvcache_kernel` scatters the current step's K/V into paged slots given `slot_mapping` (entries of `-1` are skipped — this is how CUDA graph padding is handled).
- **Tensor parallelism lives in the linear layers** (`layers/linear.py`): `ColumnParallelLinear` shards output dim (`tp_dim=0`), `RowParallelLinear` shards input dim (`tp_dim=1`) and all-reduces after the matmul, `QKVParallelLinear` / `MergedColumnParallelLinear` are fused column-parallel variants with per-shard `weight_loader`s. `RowParallelLinear` applies `bias` **only on rank 0** to avoid double-counting after all-reduce.
- **Weight loading** (`utils/loader.py`): walks `*.safetensors` in the model dir and routes each tensor through the layer's own `weight_loader` when present. The `Qwen3ForCausalLM.packed_modules_mapping` dict tells the loader how to map `q_proj`/`k_proj`/`v_proj` into the fused `qkv_proj` and `gate_proj`/`up_proj` into `gate_up_proj` with the correct `shard_id`.

### CUDA graph fast path details
The captured graphs are **decode-only** and take their inputs from persistent CPU/GPU tensors stored on `ModelRunner.graph_vars`. Each decode step copies the current batch into those tensors (padding to the next-largest captured batch size), `fill_(-1)`s the `slot_mapping` tail so the Triton KV-cache writer no-ops on padded rows, calls `graph.replay()`, and slices the output back to `bs`. Prefill, the warmup batch, and any decode step with `bs > 512` fall back to eager execution.

## Things to watch out for when editing

- `Sequence.block_size` is set as a class attribute at engine construction. If you instantiate `Sequence` objects before `LLMEngine.__init__` runs (e.g., in a new test harness), they'll use the default 256.
- `Context` is a global singleton reset in `ModelRunner.run` via `reset_context()`. Don't rely on it outside a `run` call, and don't stash references to its tensors.
- Only rank 0 returns meaningful values from `ModelRunner.run` (the sampled token ids). Workers return `None`; the driver does all tokenization and sampling.
- Prefix caching equality check in `BlockManager.allocate` is `self.blocks[block_id].token_ids != token_ids` — a full list compare. Hash collisions fall back to cache-miss correctly, but changing the hashing scheme must preserve that fallback.
- The TP launch uses `tcp://localhost:2333` as the NCCL rendezvous. Running two engines on one host simultaneously will collide.
