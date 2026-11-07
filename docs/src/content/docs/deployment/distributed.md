---
title: Distributed deployment
description: Run one Sonar model replica across GPUs or nodes.
---

Choose the parallel layout before you configure the cluster. The GPUs required
by one replica equal tensor parallel size multiplied by pipeline parallel size.
Data parallelism creates additional replicas.

## One node

Use tensor parallelism across GPUs with a fast local interconnect:

```bash
sonar serve MODEL --tensor-parallel-size 8
```

Check the visible GPU order before launch. Every worker must see the model files
and use the same Python environment.

## Multiple nodes

Sonar can start one multiprocessing executor on each host. You do not need a
Ray cluster for multi-node tensor or pipeline parallelism.

Each host runs the same `sonar serve` command with a different
`--node-rank`. Node 0 starts the API server. The other nodes use `--headless`
and run model workers only.

The total world size is:

```text
tensor_parallel_size × pipeline_parallel_size × data_parallel_size
```

`--nnodes` must divide the total world size. Each node starts
`world_size / nnodes` local workers. Make the same number of GPUs visible on
each node.

### Four-node example

This example uses four nodes with eight GPUs per node. Tensor parallelism stays
within each node. Pipeline parallelism spans the four nodes.

Run these settings on all nodes:

```bash
ulimit -n 1048576

export HEAD_IP=10.0.0.10
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

Replace `10.0.0.10` with the data-plane address of node 0. Replace `eth0` with
the interface that carries traffic between the nodes.

Start node 0:

```bash
sonar serve MODEL \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 4 \
  --nnodes 4 \
  --node-rank 0 \
  --master-addr "$HEAD_IP" \
  --master-port 29501 \
  --max-model-len auto
```

Start the worker process on node 1:

```bash
sonar serve MODEL \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 4 \
  --nnodes 4 \
  --node-rank 1 \
  --master-addr "$HEAD_IP" \
  --master-port 29501 \
  --max-model-len auto \
  --headless
```

Run the same command on nodes 2 and 3. Set `--node-rank 2` on node 2 and
`--node-rank 3` on node 3.

All nodes must use the same model, Sonar commit, command options, and
environment. A local model path must contain identical files on every node.

`sonar run` remains a compatibility alias for `sonar serve`. New
deployment instructions use `serve`.

### Select the executor

Multi-node CUDA launches select the `mp` executor automatically when
`--nnodes` is greater than one. You can make the selection explicit:

```bash
--distributed-executor-backend mp
```

Do not set the backend to `ray` for this launch method. The current
configuration validator accepts multi-node `--nnodes` with `mp`, `uni`, or an
external launcher.

## Network requirements

Every node must reach `--master-addr` on `--master-port`. NCCL can also open
additional connections between workers. Configure the host firewall and
container network for the complete data-plane network.

Set the same data-plane interface for NCCL and Gloo:

```bash
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

Replace `eth0` with the data-plane interface. Do not select a management
interface with lower bandwidth.

Set `SONAR_HOST_IP` to this node's address on the data-plane network:

```bash
export SONAR_HOST_IP=10.0.0.11
```

Use a different, reachable address on each node. `--master-addr` selects the
head node; it does not set the address that each worker advertises.
Set this variable when the default interface uses an address that other
nodes cannot reach, such as a private container bridge.

Set `NCCL_IB_DISABLE=1` only when the nodes do not have a working InfiniBand or
RoCE path:

```bash
export NCCL_IB_DISABLE=1
```

Set `NCCL_P2P_DISABLE=1` only when peer-to-peer access between local GPUs is
unavailable or unstable:

```bash
export NCCL_P2P_DISABLE=1
```

These settings can reduce performance on hardware that supports the disabled
transport. Test the default NCCL configuration first.

## Control pipeline layer placement

Sonar distributes transformer layers across pipeline stages automatically. Use
`SONAR_PP_LAYER_PARTITION` when memory use or compute time is uneven:

```bash
export SONAR_PP_LAYER_PARTITION=18,20,20,20
```

Provide one integer for each pipeline stage. The values must sum to the model's
transformer-layer count. Use the same value on every node.

Input and output embeddings can make the first and last stages use different
amounts of memory. Measure memory and stage time before you override the
automatic partition.

## Data parallel serving

```bash
sonar serve MODEL \
  --tensor-parallel-size 8 \
  --data-parallel-size 2
```

This layout creates two replicas with eight tensor-parallel ranks each. Provide
enough GPUs for both replicas.

External load-balancer mode can assign a data-parallel rank to each independent
server process. Use the generated
[server argument reference](/reference/server-arguments/) for the address, port,
rank, and local-size fields.

## Expert parallelism

```bash
sonar serve MOE_MODEL \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

Expert parallelism uses all-to-all communication. Test it on the actual
interconnect. Enable expert load balancing only after you collect an imbalance
baseline.

## Diagnose startup

Set `NCCL_DEBUG=INFO` when ranks hang during a collective. Verify GPU count,
rank count, node clocks, firewall rules, and interface selection.

Test one node first. Then test a minimal multi-node model before you allocate
the production checkpoint.

Check these conditions on every host:

```bash
python -c 'import torch; print(torch.cuda.device_count())'
ip route get "$HEAD_IP"
```

For the four-node example, the first command must report eight visible GPUs on
each node. Check that node 0 listens on the configured master port after launch.

If initialization times out, compare `HEAD_IP`, `--master-port`, `--nnodes`,
and `--node-rank` across all commands. Each rank must be unique and in the range
from zero through `nnodes - 1`.

## Failure handling

One failed rank stops its model replica. Remove the complete replica from
traffic and restart all its ranks. Data parallel replicas can continue serving
when the frontend removes the failed replica.
