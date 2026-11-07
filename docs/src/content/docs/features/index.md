---
title: Feature guide
description: Select Sonar features for common inference workloads.
---

## Memory and throughput

- [Prefix caching](/features/prefix-caching/) reuses shared prompt prefixes.
- [Quantization](/reference/quantization/) reduces weight or KV cache memory.
- [Speculative decoding](/features/speculative-decoding/) can reduce decode latency.
- [Deployment optimization](/deployment/optimization/) explains scheduler and
  memory controls.

## Model customization

- [LoRA adapters](/features/lora/) serve one or more adapters with a base model.
- [Tool calling](/features/tool-calling/) parses model output into API tool calls.
- [Reasoning and tool parsers](/features/reasoning-and-tools/) select a parser
  that matches the model.
- [Sampling and structured output](/guides/sampling-and-structured-output/)
  constrains generated JSON or text.

## Input types

- [Multimodal inputs](/features/multimodal/) cover image, audio, and video models.
- [FP8 vision attention](/features/fp8-vit-attention/) can accelerate Qwen3
  vision encoders on supported NVIDIA and AMD GPUs.
- Encoder-decoder models support sequence-to-sequence tasks.
- Pooling runners provide embeddings, classifications, rewards, and scores.

## Operations

- [Observability](/features/observability/) covers logs, metrics, and health checks.
- [`sonar bench`](/deployment/benchmarking/) measures latency, throughput,
  startup, and live-server load.
- [Production deployment](/deployment/production/) covers lifecycle and
  readiness.
