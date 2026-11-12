---
title: Production deployment
description: Operate a Sonar server behind a proxy or load balancer.
---

Run one Sonar process per model replica. Put a reverse proxy or load balancer in
front of the replicas. The frontend should handle public TLS, request limits,
and traffic removal during shutdown.

## Choose a serving topology

Start with a complete model replica. Add a more complex topology only when
measurements show a specific limit.

| Topology | Use it when | Main cost |
| --- | --- | --- |
| One replica | The model fits and meets the target load | One failure removes all capacity |
| Replicated serving | One replica meets latency targets but needs more throughput | Each replica stores the model |
| Multi-node TP or PP | One replica cannot fit on one node | Cross-node communication |
| Prefill/decode disaggregation | Prefill traffic disrupts decode latency | KV transfer and routing |
| External KV cache | Prefixes must survive local cache eviction or serve several replicas | Storage and transfer latency |

See [Choose parallelism](/deployment/parallelism/) before you split one replica
across GPUs. See [Distributed deployment](/deployment/distributed/) for
multi-node launch commands.

Use the [Rust frontend](/deployment/rust-frontend/) when you want to evaluate
the experimental Rust HTTP serving layer. Check its route and option
compatibility before production use.

## Start a private server

```bash
export SONAR_API_KEY="$(openssl rand -hex 32)"

sonar serve Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 2242 \
  --served-model-name qwen3 \
  --api-key "$SONAR_API_KEY" \
  --max-model-len 32768
```

Bind to `127.0.0.1` when a proxy runs on the same host. Bind to a private
network address when a separate load balancer must reach the process.

## Replicate a model

Run one complete Sonar server for each replica. Give every replica a different
port or host address. Put a load balancer in front of their OpenAI-compatible
endpoints.

Use least-request or queue-aware routing when request lengths vary. Round-robin
routing can send several long prompts to one replica while another replica is
idle.

Prefix cache entries belong to the replica that processed the prefix. Routing
similar prefixes to the same replica can improve the hit rate. This affinity
can also produce an uneven load. Measure both effects.

## Disaggregate prefill and decode

Prefill processes all input tokens. It usually needs high compute throughput.
Decode produces new tokens one step at a time and is sensitive to scheduling
delay.

Prefill/decode disaggregation runs these phases on separate Sonar instances:

```text
client
  |
  v
P/D router ---> prefill instance
  |                 |
  |            KV cache transfer
  |                 |
  +----------> decode instance ---> response
```

The router sends the prompt to a prefill instance. The prefill instance
produces the prompt KV cache and transfers it to a decode instance. The decode
instance continues generation from that cache.

Use this topology when long prompts cause unacceptable inter-token latency for
active decode requests. It also permits separate scaling and tuning of prefill
and decode capacity. KV transfer adds work, so disaggregation does not
automatically increase total throughput.

### Use the NIXL connector

Sonar includes `NixlConnector` for asynchronous KV cache transfer. Install NIXL
and its required transport for the target network. Configure one unique side
channel address for each engine.

From a Sonar source checkout, activate the serving environment and install the
connector dependencies:

```bash
bash tools/install_kv_connectors.sh
```

The script keeps the installed PyTorch and CUDA packages. It selects the NIXL
and Mooncake distributions for that CUDA version. CUDA 13 needs
`mooncake-transfer-engine-cuda13`, not the CUDA 12 distribution.

If you use the optional KVCR offload tier, install its pinned dependencies:

```bash
bash tools/install_kv_offload.sh
```

Start a prefill instance:

```bash
CUDA_VISIBLE_DEVICES=0 \
SONAR_NIXL_SIDE_CHANNEL_HOST=10.0.0.10 \
SONAR_NIXL_SIDE_CHANNEL_PORT=14590 \
sonar serve MODEL \
  --host 0.0.0.0 \
  --port 8100 \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer",
    "engine_id": "prefill-0",
    "kv_load_failure_policy": "fail"
  }'
```

Start a decode instance:

```bash
CUDA_VISIBLE_DEVICES=1 \
SONAR_NIXL_SIDE_CHANNEL_HOST=10.0.0.11 \
SONAR_NIXL_SIDE_CHANNEL_PORT=14591 \
sonar serve MODEL \
  --host 0.0.0.0 \
  --port 8200 \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_consumer",
    "engine_id": "decode-0",
    "kv_load_failure_policy": "fail"
  }'
```

The model, tokenizer, revision, KV cache data type, block size, and parallel
layout must be compatible between the two instances. Use `kv_producer` for
prefill and `kv_consumer` for decode. The older `kv_both` role is deprecated
for `NixlConnector`.

The example uses one GPU per instance. Each side can use tensor, pipeline, or
data parallelism when the connector supports that layout.

### Put a P/D router in front

Clients must send requests to a P/D-aware router. A normal HTTP load balancer
does not coordinate the prefill request, KV metadata, and decode request.

The repository includes a development proxy:

```bash
python benchmarks/disagg_benchmarks/nixl_proxy_server.py \
  --prefill-url http://10.0.0.10:8100/v1/completions \
  --decode-url http://10.0.0.11:8200/v1/completions \
  --prefill-engine-id prefill-0 \
  --decode-engine-id decode-0 \
  --prefill-nixl-host 10.0.0.10 \
  --prefill-nixl-port 14590 \
  --decode-nixl-host 10.0.0.11 \
  --decode-nixl-port 14591 \
  --port 9000
```

Use this proxy to validate the topology. A production router must perform
health-aware P/D selection, propagate cancellations, preserve streaming, and
remove failed engines from service discovery.

Sonar also contains P2P NCCL and LMCache examples under
`examples/disaggregated_prefill/`. Treat the P2P NCCL XpYd example as
experimental. Select a connector from its tested transport, model, and
parallelism support.

### Choose a KV load failure policy

`kv_load_failure_policy` defaults to `fail`. A failed transfer then fails the
request. This protects decode latency because the decode instance does not run
an unexpected prefill.

The `recompute` policy lets the decode instance recompute missing KV blocks.
It can improve request availability, but the extra prefill can delay other
decode requests. Measure tail latency before you use it.

### Scale P and D separately

Scale prefill capacity from queued input tokens, prefill latency, and time to
first token. Scale decode capacity from running sequences, inter-token latency,
and queued decode work.

Use production prompt and output distributions to determine the P-to-D ratio.
A ratio from a short-prompt benchmark will not fit a long-context workload.
The router must stop assigning requests before it removes an engine.

### Monitor KV transfers

Collect metrics from both sides and from the router. NIXL exports transfer
duration, post duration, transferred bytes, descriptor count, failed
transfers, failed notifications, and expired requests.

Alert on transfer failures and expired requests. Compare transfer time with
prefill time and time to first token. A fast prefill stage cannot meet its
latency target when the KV path is congested.

For background and current upstream connector behavior, see the
[vLLM disaggregated-prefill guide](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
and the
[NIXL connector guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/).
Sonar's local code and examples define the supported configuration for your
installed Sonar commit.

## Reuse an external KV cache

Install the optional dependencies with `tools/install_kv_connectors.sh` from
the source checkout.

### LMCache

For CPU offloading within each engine:

```bash
PYTHONHASHSEED=0 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=5 \
sonar serve Qwen/Qwen3-0.6B \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

For a separate cache process, start the server and select `LMCacheMPConnector`:

```bash
lmcache server --l1-size-gb 5 --l1-init-size-gb 5 --eviction-policy LRU
```

Set `lmcache.mp.host` and `lmcache.mp.port` in `kv_connector_extra_config`
when the server is not at `tcp://localhost:5555`.

Use the same LMCache version for the engine and cache server.

### Mooncake

Select `MooncakeConnector` for prefill/decode transfers or
`MooncakeStoreConnector` for a shared Mooncake Store. The Store connector reads
its configuration from `MOONCAKE_CONFIG_PATH`. Start a Mooncake master before
the engine.

For TCP-only transfers, set `MC_FORCE_TCP=1` in addition to
`"mooncake_protocol": "tcp"` in the transfer connector's extra configuration,
or `"protocol": "tcp"` in the Store configuration.

## Configure the proxy

Permit streaming responses and disable response buffering. Set the upstream
read timeout above the longest permitted request duration. Limit request body
size when clients can submit images, audio, or large prompts.

Do not retry a request after response bytes reach the client. A retry can run
the same generation twice. Retries are safe only before the proxy receives
response headers.

## Health and readiness

Use `/health` for process health. Data-parallel supervisor deployments also
provide `/ready` and `/readyz`.

Keep a new replica out of service until model loading and graph capture finish.
Send a small warm-up request before you add it to the load balancer.

## Shut down without dropping requests

1. Remove the replica from load-balancer discovery.
2. Wait for active requests to finish.
3. Stop the Sonar process.
4. Start the replacement and wait for health checks.
5. Warm the replacement before you restore traffic.

Set the orchestrator termination grace period above the maximum request time.

## Protect state and caches

Mount the model cache and compilation cache on persistent storage. Give each
replica writable cache space or use a cache that supports concurrent writers.

Keep secrets outside command history. Inject API keys through a secret manager
or environment file with restricted permissions.

## Capacity and overload

Set client deadlines. Apply an admission limit at the proxy when the request
queue causes unacceptable tail latency. Watch running requests, waiting
requests, KV cache pressure, and preemptions.

Scale replicas from sustained queue depth and latency. GPU utilization alone
does not show whether prompt processing or decode latency is the limit.

## Benchmark the complete topology

Send benchmark traffic through the public routing path:

```bash
sonar bench serve \
  --backend openai \
  --base-url http://router.internal:9000 \
  --model MODEL \
  --dataset-name random \
  --input-len 2048 \
  --output-len 256 \
  --request-rate 8 \
  --num-prompts 1000 \
  --save-result
```

Compare a complete replica with the disaggregated topology. Keep the model,
prompt distribution, output distribution, and offered load fixed. Report time
to first token, inter-token latency, end-to-end latency, and completed
throughput.

## Verify a release

Test these operations before production traffic:

- Authenticated chat request
- Streaming cancellation
- Request at the maximum permitted context
- Health check during model load and shutdown
- Metrics collection
- One failed replica behind the load balancer
- Failed prefill, decode, router, and KV transfer for a P/D deployment

See [Security](/deployment/security/) and
[Observability](/features/observability/) for the related controls.
