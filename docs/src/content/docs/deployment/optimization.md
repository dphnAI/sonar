---
title: Optimize a deployment
description: Tune Sonar for latency, throughput, memory use, and startup time.
---

Optimization starts with a production-like workload. A setting that improves
one request can reduce throughput under load. A short prompt benchmark can also
hide prefill limits that appear with long prompts.

Record these values before you change the server:

- Time to first token
- Inter-token latency
- End-to-end request latency
- Requests completed per second
- Input and output tokens per second
- GPU memory use
- KV cache occupancy and request preemptions

Use [`aphrodite bench`](/deployment/benchmarking/) to measure offline execution,
online serving, startup, and parameter sweeps.

Warm the server before you record results. Use the same prompts, output lengths,
sampling parameters, and concurrency for each comparison.

## Set the GPU memory budget

`--gpu-memory-utilization` controls the memory budget for one Sonar instance.
The default is `0.92`.

The fraction applies to the GPU's maximum memory capacity. It does not apply to
the memory that happens to be free when Sonar starts. On an 80 GiB GPU, the
default requests about 73.6 GiB for the model executor.

Sonar checks the free memory during startup. Startup fails if the requested
fraction of total capacity is not free. For example, a second process that
already uses 20 GiB can make `--gpu-memory-utilization 0.92` fail on an 80 GiB
GPU. Sonar does not reinterpret `0.92` as 92 percent of the remaining 60 GiB.

Use a lower value when another process shares the GPU:

```bash
aphrodite serve MODEL --gpu-memory-utilization 0.75
```

After model loading, Sonar profiles model activations and CUDA graph memory. It
assigns the remainder of the requested budget to the KV cache. A higher value
usually increases the KV cache capacity. It also leaves less safety margin for
other CUDA allocations.

Use `--kv-cache-memory-bytes` only when you need an explicit KV cache size. This
setting overrides `--gpu-memory-utilization` for KV cache sizing.

## Choose the maximum model length

`--max-model-len` limits the combined prompt and output length. An explicit
limit prevents one request from reserving a context length that the service
does not need.

```bash
aphrodite serve MODEL --max-model-len 32768
```

Sonar also supports automatic fitting:

```bash
aphrodite serve MODEL --max-model-len auto
```

`auto` first reads the maximum context length from the model configuration. If
the full context fits, Sonar keeps it. Otherwise, Sonar finds the largest length
that fits in the available KV cache across all workers.

Automatic fitting cannot predict your preferred concurrency. A context length
that fits one request can still leave too little KV cache for many concurrent
requests. Use an explicit lower value when the service has a known context
limit.

The startup log reports when automatic fitting reduces the model length. Check
the final value before you expose the service.

## Size the scheduler

Two limits control the work in a scheduler iteration.

### `--max-num-seqs`

This option limits the active sequences in one iteration. A higher value can
increase throughput. It also increases activation memory, CUDA graph coverage,
and KV cache pressure.

Start with the expected peak concurrency. Reduce the value when startup
profiling uses too much memory or when active requests are frequently
preempted.

```bash
aphrodite serve MODEL --max-num-seqs 64
```

### `--max-num-batched-tokens`

This option limits the number of scheduled tokens in one iteration. The budget
contains prefill and decode tokens.

A larger value can improve long-prompt throughput. A smaller value gives decode
requests more frequent scheduling opportunities and can reduce inter-token
latency.

Test these cases separately:

1. Short prompts with long outputs
2. Long prompts with short outputs
3. Mixed prompt lengths at production concurrency

Change one scheduler limit at a time.

## Use chunked prefill

Chunked prefill divides a large prompt across scheduler iterations. It prevents
one long prefill from occupying the complete token budget. This can improve
decode latency when long prompts arrive while other requests generate tokens.

Compare time to first token and inter-token latency. A small token budget can
protect decode latency and make long-prefill completion slower.

## Understand KV cache pressure

The KV cache holds attention keys and values for active sequences. Its capacity
depends on:

- Model architecture and number of attention layers
- KV head count and head size
- KV cache data type
- Tensor and pipeline parallel layout
- Memory left after weights, activations, and CUDA graphs

Frequent request preemption indicates that active sequences do not fit in the
KV cache. Preemption repeats model work and increases tail latency.

Apply these changes in order:

1. Reduce `--max-num-seqs`.
2. Reduce `--max-model-len`.
3. Increase `--gpu-memory-utilization` when the GPU has safe headroom.
4. Use FP8 KV cache when the model and platform support it.
5. Add model parallelism when model weights consume most device memory.

Use FP8 KV cache with:

```bash
aphrodite serve MODEL --kv-cache-dtype fp8
```

Test output quality. Some checkpoints provide calibrated KV scales. Other
checkpoints use default scales or need scale calculation.

## Use prefix caching

Prefix caching is enabled by default. No flag is required.

It reuses KV cache blocks when requests share an identical token prefix. Common
examples include a system prompt, a document prefix, a tool definition, or a
few-shot prompt.

Prefix caching saves prefill work. It does not reduce decode work after the
shared prefix. A cache hit also requires the tokenized prefix and relevant
request properties to match.

Use repeated production prompts to measure its benefit. Disable it for a
controlled comparison:

```bash
aphrodite serve MODEL --no-enable-prefix-caching
```

Keep it enabled unless the target platform or a specific feature requires it to
be disabled.

## Quantize model weights

Quantized weights reduce model memory. The saved memory can hold a larger KV
cache or allow more model replicas.

The speed result depends on the format and GPU. A smaller weight format can be
slower when its dequantization kernel does not suit the hardware.

Use the generated [quantization matrix](/reference/quantization/) to find the
platform declaration and minimum NVIDIA capability. Then benchmark the exact
checkpoint on the target GPU.

Compare output quality with the unquantized checkpoint. Weight-only formats can
affect model quality even when the serving interface is unchanged.

### FP4 MoE on SM120 GPUs

On SM120 GPUs, use `--moe-backend b12x` for supported FP4 MoE checkpoints.
B12x is included in `flashinfer-python`. Do not install the archived standalone
`b12x` package. If it is already installed, remove it and restore the bundled
provider in the same environment as Sonar:

```bash
uv pip uninstall b12x
uv pip install --reinstall --no-deps 'flashinfer-python==0.7.2'
```

B12x supports tensor parallelism, not expert parallelism. Do not combine it with
`--enable-expert-parallel`.

MXFP4 uses MXFP8 activations by default. Set
`APHRODITE_B12X_MOE_FP4_FORCE_A16=1` to use 16-bit activations instead.
This changes activation precision. Compare output quality and performance on
your workload before you use it in production.

By default, for DeepSeek-V4.1 on SM120, Sonar uses native FP8 sparse indexers from the bundled
DeepGEMM, including 128-row KV-cache pages. If an external DeepGEMM package lacks
the required APIs or SM120 headers, Sonar uses the Triton indexer instead.
`APHRODITE_USE_DEEP_GEMM=0` forces that fallback; it does not select the MoE backend.
General DeepGEMM GEMMs remain disabled on SM120.

### Native DeepSeek-V4.1 backend on SM120

Set `APHRODITE_DSV41_SM120_NATIVE=1` and pass
`--attention-backend B12X_DSV41` when you launch DeepSeek-V4.1 to select the
experimental native B12x backend. With DSpark, also set
`"attention_backend": "B12X_DSV41"` in `--speculative-config`.

This path uses packed compressed KV caches, native sparse attention, fused
output projections, and native Engram lookup. It retains Sonar's MoE scheduling.

For BF16 query chunks, including prefill, the backend fuses query/KV RMSNorm
with MXFP8 query quantization and the KV rotary embedding. The query
projection and optional indexer reuse the quantized query with their
original prepared GEMMs. The kernel preserves BF16 normalization rounding
and keeps KV in BF16.

With 16 local query heads, target-model prefill decodes each packed K=V tile
once into shared memory for both QK and PV, mapping indexed pages during the
gather. It retains BF16 attention arithmetic and the native 64-candidate
tile and reduction order. Programs compile during preparation and reuse
caller-owned output/LSE storage across live batch sizes. Other head counts
and draft-model prefill retain the provider route.

Native clustered decode specializes its shared-memory allocation to the
actual candidate width. The eight-way partition and ordered merge are
unchanged; smaller SWA-only partitions no longer reserve indexed-key storage.

The native indexer can split prefill query rows across tensor-parallel ranks
and all-gather the selected indices before attention. Set
`--attention-config '{"indexer_tp_shard_min_rows":2048}'` to enable it for
chunks of at least 2048 rows; the default threshold is 65536. Request/page
metadata retains its global row offsets, including mixed prefill/decode
batches and uneven row splits. Pure decode and CUDA-graph capture retain
their existing path. The candidate-producing layer stays replicated to
avoid gathering its much wider candidate buffer.
The score arithmetic and production top-k policy are unchanged. The existing
top-k kernel resolves exact boundary ties by thread arrival order, so even
repeated unsharded launches can select different tied indices. Do not assume
bit-identical selected indices or final model outputs from row sharding.

The native backend also applies an identity-only missing-argument check to
the verified CuTe argument binder. Its original membership check invokes
tensor equality on a private sentinel, adding exception-handling overhead
to eager launches. The compatibility fix changes no kernel arguments or
GPU arithmetic. It checks the complete binder implementation before replacing
it; unknown implementations stay unchanged and produce a warning.

For mHC boundaries of up to 64 tokens, the backend computes the coefficients
for the next boundary on a separate CUDA stream while attention or the MoE
layer runs. The residual mix, the input collapse, and its RMSNorm stay on the
critical path. Results match the fused B12x kernels bit for bit.

DSpark projects the draft layers' context keys and values together, with one
input quantization. Drafting combines the Markov-logit addition, sampling, and
proposal-cache write when their layouts match. Sampling preserves the existing
dtype rounding, per-request seeds, and noise.

With tensor parallelism, drafting and verification sample on each rank's
vocabulary shard. Only a few values per row cross GPUs, instead of
full-vocabulary logits. A step falls back to gathered logits when any request
uses top-k, top-p, min-p, logits processors such as penalties or logit bias,
logprobs, or structured outputs. These optimizations require no additional
flags.

Use the bundled B12x provider described above. The archived standalone package
is not supported.

Do not combine this path with pipeline parallelism, context parallelism, dual
batch overlap, or LoRA. The existing backend remains the default. Remove the
environment variable and the B12x attention-backend overrides to return to it.

#### Implementation boundaries

Native model policy and preparation live in
`aphrodite/models/deepseek_v41/sm120/`; B12x worker lifecycle adapters live in
`aphrodite/v1/worker/b12x_utils.py`. V4.1 message ordering belongs to its
renderer, not the shared V4 renderer. Keep shared configuration and worker
changes limited to dispatch and lifecycle hooks when syncing upstream.

CUDA implementations and their operator registrations are colocated under
`csrc/attention/mla/`, `csrc/moe/`, `csrc/engram/`, and `csrc/collectives/`.
Engram fake implementations are registered by its feature module after the
native operators load, without adding model-specific registrations to global
operator modules.

B12x weight preparation preserves checkpoint-native `[gate; up]` ordering for
both W4A8 and W4A16; the declared `w31` layout controls provider packing.
Do not pre-swap those rows. The fused W4A8 decode kernel admits hidden sizes
divisible by 512 in the range 512–8192; other shapes retain provider kernels.

### Experimental paired PCIe all-reduce

Set `APHRODITE_ALLREDUCE_USE_PCIE_PULL=1` to select the experimental paired
all-reduce. Initial support targets the native DeepSeek-V4.1 backend on four
SM120 GPUs in one node. All selected GPU pairs must support CUDA peer reads and
writes without crossing a NUMA system link.

This path requires the V2 model runner, BF16 activations, and a model hidden
size of 5120. A speculative draft must use the same TP4 group. Keep adaptive
verification disabled. Do not combine it with pipeline, context, data, or
expert parallelism, dual batch overlap, LoRA, sleep mode, or KV offloading.
Unsupported configurations fail during startup.

The paired kernel handles contiguous two-dimensional tensors with 1–8 rows and
a width of 5120 or 6144. Other shapes retain their existing collective backend.
The reduction adds operands in rank order in FP32, then rounds once to BF16.
This uses a different rounding contract from BF16 NCCL. Do not assume that
the modes are numerically interchangeable.

The same backend also all-gathers contiguous tensors of up to 16 KiB per rank,
such as the per-row packets of vocabulary-parallel sampling. A gather copies
bytes, so its results are identical to NCCL. Larger all-gathers retain their
existing backend.

The engine checks a collective completion snapshot before it releases each
step's tokens or logprobs. With speculation, that snapshot includes both target
and draft collectives. Token copies overlap draft proposal, but output waits
for the final snapshot. A protocol failure stops the engine; it does not switch
the affected collective to NCCL or retry it. Restart the engine after such a
failure.

This option is disabled by default. Compare output quality and latency with
the option disabled before you use it in a deployment.

## Choose parallelism

Tensor parallelism splits model layers across GPUs and reduces weight memory on
each rank. It adds communication to each model step.

Pipeline parallelism splits layers into stages. It can help when tensor
parallel kernels do not scale well across the available interconnect.

Data parallelism creates independent model replicas for throughput when one
replica fits on the selected hardware.

Expert parallelism distributes MoE experts. It can reduce expert weight
pressure and can add all-to-all communication.

See [Choose parallelism](/deployment/parallelism/) for a selection procedure.
Do not add GPUs only to increase the reported batch capacity. Measure the
communication cost and per-request latency.

## Tune speculative decoding

Speculative decoding can reduce inter-token latency. It works best when a cheap
proposer has a high acceptance length.

It also allocates draft-model state and schedules extra work. High request
concurrency can already use the target GPU efficiently. Speculation can reduce
throughput in that condition.

Start with speculation disabled. Add one method and compare:

- Mean acceptance length
- Per-position acceptance rate
- Inter-token latency
- Requests per second
- GPU memory use

See the [speculative decoding guide](/features/speculative-decoding/) for MTP,
DSpark, DFlash, EAGLE, and lookup methods.

## Select compilation behavior

The default hybrid execution uses compiled graphs where they are useful and
eager execution where required. The first run can spend time compiling and
capturing CUDA graphs.

Keep the compilation cache on persistent storage. Reuse the same Sonar commit,
model configuration, PyTorch build, and GPU type to retain cache hits.

Use eager mode for a short development run:

```bash
aphrodite serve MODEL --enforce-eager
```

Eager mode reduces startup work and usually lowers steady-state performance. Do
not use it as a production default without a benchmark.

CUDA graph capture also consumes memory. Lower `--max-num-seqs` when graph
capture causes startup memory pressure. This reduces the range of captured
batch sizes.

## Optimize startup

Model download, weight loading, quantization setup, compilation, and CUDA graph
capture can each dominate startup.

Use these practices:

- Keep model files in a persistent Hugging Face cache.
- Keep the compilation cache between restarts.
- Avoid changing shape-related flags when you need cache reuse.
- Use local storage with adequate read throughput for large checkpoints.
- Warm one replacement replica before you stop the old replica.

Do not use startup time as the only reason to enable eager mode. Long-running
services usually recover compilation cost through better steady-state
performance.

## A repeatable tuning procedure

1. Set the required context length.
2. Set a safe GPU memory fraction.
3. Test the default scheduler settings.
4. Tune `--max-num-seqs` for the expected concurrency.
5. Tune `--max-num-batched-tokens` for the prompt mix.
6. Check KV cache preemption and cache hit behavior.
7. Test quantization or FP8 KV cache when memory is the limit.
8. Select parallelism when the model does not fit or one replica is too slow.
9. Add speculative decoding when inter-token latency is still the limit.
10. Repeat the complete production-like benchmark.

Keep the baseline results with the deployment configuration. This makes a
performance regression visible after a model or Sonar update.
