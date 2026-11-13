---
title: Rust frontend
description: Run Sonar with the experimental Rust HTTP frontend.
---

The Rust frontend provides an alternative HTTP serving layer for Sonar. The
Python supervisor still starts and manages the model engine. It passes the
listening socket and engine transport addresses to the `sonar-rs`
subprocess.

The Rust frontend is experimental. It does not implement every route or
frontend option from the Python server. Test the required API surface before
production use.

## Enable the packaged frontend

Set `SONAR_USE_RUST_FRONTEND=1`:

```bash
SONAR_USE_RUST_FRONTEND=1 \
  sonar serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3
```

Sonar searches for an executable named `sonar-rs` inside the installed
`sonar` package. Startup fails with a file-path error when the package does
not contain the binary.

Set an explicit path when you build the binary from a source checkout:

```bash
cargo build \
  --release \
  --manifest-path rust/Cargo.toml \
  --bin sonar-rs

SONAR_USE_RUST_FRONTEND=1 \
SONAR_RUST_FRONTEND_PATH="$PWD/rust/target/release/sonar-rs" \
  sonar serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3
```

Build the binary from the same commit as the Python package. The two processes
share an internal transport protocol.

## Architecture

The integrated launch has three parts:

```text
client
  |
  v
sonar-rs HTTP frontend
  |
  | ZMQ and MessagePack
  v
Python engine process
  |
  v
model workers
```

Python binds the requested HTTP socket before it starts the Rust process. The
Rust frontend inherits that socket. This keeps `--host`, `--port`, TLS, and
process supervision in the normal `sonar serve` lifecycle.

Only one Rust API process can run for one integrated launch.
`--api-server-count` values greater than one are rejected. Use complete Sonar
replicas behind a load balancer when you need more frontend capacity.

## Supported routes

The current Rust router provides these public routes:

| Route | Purpose |
| --- | --- |
| `/health` | Process and engine health |
| `/metrics` | Prometheus metrics |
| `/load` | Current frontend load |
| `/version` | Sonar and Rust frontend versions |
| `/v1/models` | Served model list |
| `/v1/completions` | OpenAI-compatible completions |
| `/v1/chat/completions` | OpenAI-compatible chat completions |
| `/v1/decisions` and `/v1/systemone` | Structured decisions |
| `/tokenize` and `/detokenize` | Tokenizer operations |
| `/inference/v1/generate` | Token-input and token-output generation |

The decision routes serve the models that read labels by generation: Qwen3,
Qwen3.5, Gemma 4, LFM2-VL and DiffusionGemma. They read the engine's
`--logprobs-mode` and `--diffusion-config` like the Python frontend. Models
that answer through a classification head (D3, Clef-Omni and Clef-Flash)
return 501 on `/v1/systemone`, as do image inputs to LFM2-VL and images for
models without a Rust image processor. Use the Python frontend for them.

Runtime LoRA routes are available when
`SONAR_ALLOW_RUNTIME_LORA_UPDATING` is enabled. Development and profiling
routes require their corresponding server settings.

The Rust frontend supports streaming chat and completion responses. It also
supports API-key authentication, CORS, request-ID headers, TLS, chat
templates, reasoning parsers, and tool parsers.

### gRPC health probes

Add `--grpc-port 50051` to expose the Rust frontend's gRPC listener. Use its
`grpc.health.v1.Health` service with `liveness` for startup/liveness probes and
`readiness` for readiness probes.

`readiness` and `sonar.Inference` report not-serving during pause, sleep, partial
wake, unfinished weight updates, shutdown, or engine failure. `liveness`
reports engine failure without treating an intentional pause as a crash.
Using readiness for a liveness probe can restart a deliberately paused engine.

The gRPC interface is for trusted callers. Its `media_io_kwargs` supports
`video.fps` without the HTTP per-request override flag; frame limits still apply.

## Check option compatibility

Run this command from a source checkout:

```bash
rust/target/release/sonar-rs serve --help
```

The help output contains a section named
`Options not implemented in Rust frontend yet`. The Rust process rejects these
options with a startup error instead of silently applying different behavior.

Current limitations include multiple frontend-owned model and tokenizer
overrides. Some multimodal controls and tracing options are also unavailable.
The exact list can change with each commit, so use the binary's help output as
the source of truth.

Engine-owned options still pass to the Python engine. Examples include
`--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization`, and
compilation settings.

## Validate a deployment

Check health, version, and model discovery:

```bash
curl --fail http://127.0.0.1:2242/health
curl --fail http://127.0.0.1:2242/version
curl --fail http://127.0.0.1:2242/v1/models
```

Send a non-streaming request:

```bash
curl --fail http://127.0.0.1:2242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "Reply with one short sentence."}
    ],
    "max_tokens": 32
  }'
```

Check the complete streaming path:

```bash
curl --fail --no-buffer \
  http://127.0.0.1:2242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "Count from one to three."}
    ],
    "max_tokens": 32,
    "stream": true
  }'
```

The response must contain one or more `data:` events and end with
`data: [DONE]`.

Run [`sonar bench serve`](/deployment/benchmarking/) through the Rust
frontend before production use. Use the same model, parsers, authentication,
and streaming mode as the target workload.

## Diagnose startup

If Sonar cannot find the binary, set `SONAR_RUST_FRONTEND_PATH` to an
absolute executable path. If the Rust process rejects an option, remove the
option or use the Python frontend for that deployment.

Check both Python and Rust log lines during startup. A successful launch reports
that the engines connected before it reports the HTTP bind address.

Unset `SONAR_USE_RUST_FRONTEND` to return to the Python frontend:

```bash
unset SONAR_USE_RUST_FRONTEND
unset SONAR_RUST_FRONTEND_PATH
sonar serve MODEL
```
