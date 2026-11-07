---
title: Python LLM API
description: Run batched offline inference with the LLM class.
---

Use `LLM` for local batch jobs and Python applications that do not need an HTTP
server.

## When to use `LLM`

Use the Python API for evaluation, data generation, document processing, and
other jobs where one Python process owns the engine. Use the HTTP server when
several applications must share the model or when you need request
authentication, health checks, and network-level scaling.

## Generate text

```python
from sonar import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

outputs = llm.generate(
    ["Explain continuous batching.", "Explain a KV cache."],
    params,
)

for output in outputs:
    print(output.outputs[0].text)
```

Create one `LLM` instance and reuse it. Model initialization allocates device
memory and loads the weights.

`generate()` accepts one sampling configuration for the complete batch or one
configuration per prompt. The result order matches the prompt order.

```python
prompts = ["Write a title.", "Write a summary."]
params = [
    SamplingParams(temperature=0.8, max_tokens=16),
    SamplingParams(temperature=0.2, max_tokens=128),
]
outputs = llm.generate(prompts, params)
```

Each request can return more than one candidate:

```python
params = SamplingParams(n=3, temperature=0.8, max_tokens=64)
result = llm.generate(["Name a monitoring service."], params)[0]

for candidate in result.outputs:
    print(candidate.text)
```

## Generate chat responses

```python
from sonar import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
conversations = [
    [{"role": "user", "content": "Give me two names for a monitoring service."}],
    [{"role": "user", "content": "What is tensor parallelism?"}],
]

outputs = llm.chat(
    conversations,
    SamplingParams(max_tokens=96),
)
```

The model must provide a chat template. You can pass a template when the model
repository does not include one.

Use dictionaries with `role` and `content` fields. Multimodal models also
accept structured content items. See [Multimodal inputs](/features/multimodal/)
for image, audio, and video examples.

## Stream output

`LLM` is an offline batch interface. It returns completed request outputs. Use
`AsyncLLMEngine` or the HTTP server when your application must consume tokens
as they arrive. The OpenAI-compatible server exposes streaming with
`stream=True`.

## Control memory use

Use `gpu_memory_utilization` to reserve less device memory for Sonar.

```python
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    gpu_memory_utilization=0.85,
    max_model_len=8192,
)
```

Lower `max_model_len` when the configured context length creates a KV cache
that is larger than your workload needs.

You can also let Sonar select the largest context length that fits:

```python
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    max_model_len="auto",
)
```

The default `gpu_memory_utilization` is `0.92`. It is a fraction of total GPU
memory, not currently free memory. Sonar checks free memory during startup and
fails if another process leaves less than the requested amount.

## Use more than one GPU

```python
llm = LLM(
    model="Qwen/Qwen3-32B",
    tensor_parallel_size=4,
)
```

Read [Choose parallelism](/deployment/parallelism/) before you combine
parallelism modes.

## Reuse shared prompts

Prefix caching is enabled by default. Requests can reuse KV cache blocks when
their tokenized prefixes are identical. This is useful for a shared system
prompt, a repeated document prefix, or multiple samples from one prompt.

Disable it only for a controlled comparison:

```python
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_prefix_caching=False,
)
```

## Handle outputs and errors

Inspect `finish_reason` before you store a generated answer. A value such as
`length` means the request reached its output-token limit. Treat model loading
errors as startup failures. Treat individual request validation errors as data
errors and record the source prompt.

Release the engine before another process claims the same GPUs:

```python
from sonar.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

del llm
destroy_model_parallel()
destroy_distributed_environment()
```

For normal scripts, process exit releases the resources. Explicit cleanup is
useful in test suites and notebooks that create more than one engine.

## Next steps

- Use [sampling and structured output](/guides/sampling-and-structured-output/)
  to control generation.
- Use [speculative decoding](/features/speculative-decoding/) to reduce decode
  latency.
- Run [`sonar bench`](/deployment/benchmarking/) before and after a
  performance change.
