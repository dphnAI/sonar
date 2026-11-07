---
title: Choose parallelism
description: Select tensor, pipeline, data, and expert parallelism for a deployment.
---

Start with one GPU. Add parallelism only when one GPU cannot meet the memory or
throughput requirement.

## Decision procedure

1. Measure the memory needed for weights, KV cache, activations, and
   compilation.
2. Check whether one model replica fits on one GPU.
3. Use tensor parallelism when the model does not fit.
4. Add pipeline parallelism when tensor parallelism cannot span the required
   GPUs efficiently.
5. Add data parallel replicas when one model replica fits and throughput is the
   limiting requirement.
6. Enable expert parallelism for a mixture-of-experts model after you select the
   tensor and data parallel sizes.

Do not select a layout from parameter count alone. Quantization changes weight
memory. Context length and concurrency change KV cache memory. Some kernels
also have divisibility requirements for attention heads or experts.

## Map the hardware topology

Record which GPUs share NVLink or another high-bandwidth fabric. Record the
network interface and bandwidth between nodes. On NVIDIA systems, start with:

```bash
nvidia-smi topo -m
```

Keep frequent collectives inside the fastest communication domain. Tensor
parallel workers communicate during model execution. Pipeline workers send
activations between stages. Data-parallel replicas can serve requests
independently.

## Tensor parallelism

Tensor parallelism splits each layer across GPUs. It requires fast GPU-to-GPU
links because workers communicate during each layer.

```bash
sonar serve Qwen/Qwen3-32B --tensor-parallel-size 4
```

Use tensor parallelism within one NVLink or high-bandwidth GPU domain. Avoid a
large tensor-parallel group across slow network links.

Tensor parallelism also reduces the KV cache available per rank in ways that
depend on the model architecture. Benchmark the final context length and
concurrency. A model that starts successfully can still have insufficient KV
cache for the production workload.

## Pipeline parallelism

Pipeline parallelism assigns different layer ranges to different workers. It
uses less frequent communication than tensor parallelism and can cross nodes.
Pipeline bubbles can increase latency at low request volume.

```bash
sonar serve MODEL \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

The total GPUs per replica equal the tensor-parallel size multiplied by the
pipeline-parallel size.

Pipeline parallelism can help when a model has an uneven head count that does
not divide across the desired tensor-parallel size. Select a pipeline size that
permits a valid and reasonably balanced layer partition.

## Data parallelism

Data parallelism creates model replicas. Each replica handles different
requests.

```bash
sonar serve MODEL --data-parallel-size 4
```

Use data parallelism when one replica fits and request volume is high. A data
parallel size of four needs enough memory for four copies of the model.

Data parallelism can give better throughput isolation than one large replica.
The router must distribute requests across the replicas. Use session affinity
only when an application requires it. Prefix cache entries are local to a
replica unless the deployment uses a supported external cache design.

## Expert parallelism

Expert parallelism distributes mixture-of-experts layers across the available
parallel ranks.

```bash
sonar serve MODEL \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

Use expert load balancing when request traffic sends an uneven load to experts.
Measure interconnect traffic before you use expert parallelism across nodes.

Expert parallelism affects only the MoE layers. Dense attention and other
layers still follow the selected tensor and pipeline layout. Test representative
prompts because expert routing can make synthetic workloads misleading.

## Common layouts

| Hardware | Model condition | Initial layout |
| --- | --- | --- |
| One GPU | Model fits | TP=1, PP=1, DP=1 |
| One 8-GPU NVLink node | Model needs 8 GPUs | TP=8 |
| Two 8-GPU nodes | Model needs both nodes | TP=8, PP=2 |
| Four nodes with a model that fits per node | Throughput limited | TP within each node, DP=4 |
| MoE model on one 8-GPU node | Expert compute or memory limited | TP=8 with expert parallelism |

## Multi-node launch

Each node must use the same Sonar commit, model files, tokenizer files, and
runtime configuration. The nodes must resolve each other over the selected
network interface. Open only the ports required by the distributed runtime.

See [Distributed deployment](/deployment/distributed/) for launch procedures
and environment checks.

## Validate the layout

Run one request at the maximum supported input length. Then apply a
representative concurrent load:

```bash
sonar bench serve \
  --backend openai \
  --base-url http://127.0.0.1:2242 \
  --model MODEL \
  --dataset-name random \
  --input-len 2048 \
  --output-len 256 \
  --request-rate 8 \
  --num-prompts 500 \
  --save-result
```

Compare completed throughput, time to first token, inter-token latency, and
tail request latency. Check GPU utilization and communication bandwidth during
the run. A layout that reduces weight memory can still lose performance through
collective communication or pipeline bubbles.

## Common failure patterns

| Symptom | Likely cause | First check |
| --- | --- | --- |
| Startup hangs | Rank or network mismatch | Rank count, addresses, ports, and interface selection |
| One rank runs out of memory | Uneven partition or other GPU process | Per-rank memory and pipeline partition |
| Low utilization with high latency | Slow collectives or pipeline bubbles | Topology and batch size |
| MoE throughput varies by prompt | Expert imbalance | Expert load metrics and representative prompts |
| More GPUs reduce throughput | Communication dominates | Compare a smaller TP group or add DP replicas |

Benchmark the chosen layout with production prompt lengths and output lengths.
