# aphrodite-frontend-rs

This is a Rust drop-in alternative frontend for Aphrodite. The current goal is to rebuild the northbound serving layer in Rust while still talking to the core Python Aphrodite engine process(es) via ZMQ over the existing engine boundary.

It should still be considered experimental, and is not feature-complete. We are working to add more functionality from the python front-end.

See <https://github.com/Inferact/vllm-frontend-rs> for the original commit history before it was moved into the main aphrodite repo.

## Architecture

The component is organized as a Cargo workspace with several crates, layered bottom-up:

```text
┌─────────────────────────────────┐
│  aphrodite-cmd / aphrodite-rs             │  CLI entrypoint:
│                                 │  Python Aphrodite frontend subprocess
│                                 │  Rust managed-engine serve mode
│                                 │  Engine-free render mode
├─────────────────────────────────┤
│  aphrodite-server                    │  OpenAI-compatible HTTP API (axum)
├─────────────────────────────────┤
│  aphrodite-chat                      │  Chat completions: template rendering,
│                                 │  structured assistant events,
│                                 │  reasoning & tool parsing
├─────────────────────────────────┤
│  aphrodite-text                      │  Tokenizer & incremental detokenizer
├─────────────────────────────────┤
│  aphrodite-llm                       │  Thin token-in/token-out facade over
│                                 │  the engine client
├─────────────────────────────────┤
│  aphrodite-engine-core-client        │  ZMQ transport + MessagePack protocol
│                                 │  for the headless Aphrodite engine
└─────────────────────────────────┘
```

`aphrodite-rs` integrates into Python `aphrodite` as a Rust frontend subprocess.
Python owns process startup and launches the Rust API server as a Python-supervised worker, while
passing the inherited listening socket and transport addresses into `aphrodite-rs`.

For example:

```bash
APHRODITE_USE_RUST_FRONTEND=1 aphrodite serve Qwen/Qwen3-0.6B
```

To also expose the Rust frontend's gRPC inference and control services, add
`--grpc-port`. The HTTP listener remains on `--port`:

```bash
APHRODITE_USE_RUST_FRONTEND=1 aphrodite serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --grpc-port 50051
```

The gRPC listener binds on `--host` like HTTP, or on `127.0.0.1` when HTTP uses
`--uds`. When HTTP uses TCP, its port and the gRPC port must differ unless
requesting an automatically assigned port with `0`.

This also works with Python-supervised hybrid data parallelism. It requires a
frontend process and cannot be combined with `--headless` or
`--data-parallel-multi-port-external-lb`.

The separate `aphrodite serve MODEL --grpc --port 50051` command launches the Python
SMG `AphroditeEngine` gRPC server in place of HTTP. The Rust listener exposes the
`aphrodite.Inference` and `aphrodite.Control` services alongside HTTP. These are different
gRPC APIs; `--grpc` and `--grpc-port` are mutually exclusive.

### Sonar sampling and capacity

The OpenAI chat/completion endpoints and gRPC generation requests preserve Sonar
samplers, including dynamic temperature, DRY, XTC, top-a, tail-free, typical,
Mirostat, smoothing, and skew. HTTP sampling fields remain top-level JSON keys;
gRPC groups them in `GenerateRequest.aphrodite_sampling`. Omitted values use the
Python engine's defaults. DRY string breakers are tokenized before submission.
The engine's per-request phrase retry metadata remains in `extra_args`; it does
not change the positional engine-output protocol.

`GET /v1/capacity` reports initialized KV-cache and scheduler capacity for every
connected data-parallel rank. It uses the same API-key authentication as the
other inference routes, and supports the frontend's HTTP/TLS configuration.

### RL weight synchronization

With `APHRODITE_SERVER_DEV_MODE=1`, the Rust frontend supports the HTTP weight-transfer
lifecycle used by `HTTPAphroditeWeightSyncClient`: initialization, starting an update,
transferring weights, and finishing the update. It also supports draft-model
updates and `/update_weight_version` and `/weight_info` for version tracking.

For trainer-side usage, see the [IPC](../examples/rl/rlhf_http_ipc.py) and
[NCCL](../examples/rl/rlhf_http_nccl.py) examples. Configure the weight-transfer
backend on the server and pause generation during updates, as shown there.

### External Engine

`aphrodite-rs serve` can be run standalone with `--data-parallel-size-local 0` when the Python engines
are started elsewhere and this node should run only the Rust frontend. The frontend still uses
the global `--data-parallel-size` to determine how many engines it expects to join the shared handshake.

Engine membership is fixed at startup. The Rust frontend supports its managed
in-process DP coordinator and Python-supervised external coordinator wave
updates, but does not implement Elastic EP prepare/commit, rank replacement, or
runtime membership changes. Elastic EP deployments requiring those operations
must use the Python frontend. The ready-response coordination-store port and
engine-facing coordinator addresses are therefore not consumed by Rust;
its external frontend subscription uses `dp_stats_address`.

```bash
aphrodite serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1
```

Then start the Rust frontend-only server:

```bash
aphrodite-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0
```

To build the `aphrodite-rs` in isolation:

```bash
# from the local checkout
./tools/build_rust.sh
```

### Hybrid and External DP Load Balancing

Like `aphrodite serve` (see [Data Parallel Deployment](../docs/serving/data_parallel_deployment.md)), `aphrodite-rs serve` can run one Rust frontend per node that serves only the node's data parallel ranks, with an upstream load balancer spreading requests across nodes. Every node passes the global `--data-parallel-size`, the address of the node that owns rank 0 as `--data-parallel-address`, and the same explicit `--data-parallel-rpc-port`, because all engines join the handshake of the rank-0 frontend. For MoE models, the rank-0 frontend also runs the DP coordinator.

```bash
# Node 0: ranks 0-1
aphrodite-rs serve <moe-model> \
  --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-hybrid-lb \
  --data-parallel-address 10.0.0.1 --data-parallel-rpc-port 13345
# Node 1: ranks 2-3
aphrodite-rs serve <moe-model> \
  --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2 \
  --data-parallel-address 10.0.0.1 --data-parallel-rpc-port 13345
```

For external load balancing, start one `aphrodite-rs serve` per rank with `--data-parallel-rank <rank>`. A request can be pinned to one of a frontend's ranks with the `X-data-parallel-rank` header, which takes the global rank. Unlike `aphrodite serve`, `--data-parallel-rpc-port` has no default here and ranks are not inferred from `--nnodes`/`--node-rank`.

### Engine-free renderer

`aphrodite-rs render` serves text-only request preprocessing without starting or
connecting to a Python inference engine:

```bash
cargo run --manifest-path rust/Cargo.toml -p aphrodite-cmd --release -- \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768
```

It exposes `/v1/chat/completions/render` and `/v1/completions/render`. Only
tokenizer and model configuration files are loaded; model weights, PyTorch,
and Aphrodite kernels are not required.

To serve HTTPS, pass `--ssl-certfile`. If the certificate and private key are
in separate files, pass `--ssl-keyfile` for the key; otherwise the key is read from
the certificate file. For mTLS, pass `--ssl-ca-certs` with a CA
bundle and set `--ssl-cert-reqs` to `1` (optional) or `2` (required) to
verify client certificates. Use `--ssl-ciphers` to override the default
OpenSSL cipher list.

```bash
# Combined PEM (cert + key in one file):
aphrodite-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/combined.pem

# Separate cert and key files:
aphrodite-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

The render endpoints return the public token-in `GenerateRequest` consumed by
the Rust `/inference/v1/generate` endpoint. A chat render response, or one item
from a completion render response, can be submitted to that endpoint without
changing its fields.

The render and inference paths use the same `aphrodite-chat` and `aphrodite-text`
request-preparation logic; render mode stops before engine submission.
Tool-call and reasoning parsers use model-based auto-detection by default. Use
`--tool-call-parser` and `--reasoning-parser` to override either selection;
unified parsers require the same selection for both options.

```bash
curl http://127.0.0.1:8000/v1/chat/completions/render \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 16
  }'
```

### Example Request

After either full-frontend startup path, you can use any OpenAI-compatible
client against the inference endpoints:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```
