---
title: Deployment recipes
description: Starting configurations for common Sonar hardware layouts.
---

Replace each model name with a checkpoint that fits the device. Treat the
commands as baselines and run the [optimization procedure](/deployment/optimization/).
Measure each result with [`sonar bench`](/deployment/benchmarking/).

## One consumer NVIDIA GPU

```bash
sonar serve Qwen/Qwen3-8B-AWQ \
  --max-model-len auto \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16
```

Use a quantized checkpoint when weights leave too little KV cache.

## One datacenter GPU

```bash
sonar serve Qwen/Qwen3-32B \
  --max-model-len auto \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64
```

Increase scheduler limits only after memory profiling and a concurrency test.

## Eight-GPU NVLink server

```bash
sonar serve LARGE_MODEL \
  --tensor-parallel-size 8 \
  --max-model-len auto
```

Keep the tensor-parallel group inside the NVLink domain.

## Two eight-GPU nodes

```bash
sonar serve VERY_LARGE_MODEL \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

Start the cluster runtime on both nodes and verify the GPU resource count before
you launch Sonar. Keep tensor parallel traffic within each node.

## Mixture-of-experts model

```bash
sonar serve MOE_MODEL \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb
```

Measure all-to-all traffic and expert imbalance. Disable EPLB for a baseline.

## AMD ROCm

```bash
sonar serve MODEL \
  --max-model-len auto \
  --gpu-memory-utilization 0.90
```

Use a Sonar wheel or source build for ROCm that matches the installed ROCm
version.

## Jetson Thor

```bash
sonar serve SMALL_MODEL \
  --max-model-len auto \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 8
```

GPU memory and system memory share the same physical DRAM on Thor. Leave space
for the operating system, model download, and media preprocessing.

## CPU development

```bash
SONAR_TARGET_DEVICE=cpu sonar serve SMALL_MODEL \
  --max-model-len 4096 \
  --max-num-seqs 4
```

Set `SONAR_CPU_KVCACHE_SPACE` to a conservative GiB value. CPU serving is
useful for functional tests and low-rate workloads.
