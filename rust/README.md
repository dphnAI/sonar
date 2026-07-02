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

### External Engine

`aphrodite-rs serve` can be run standalone with `--data-parallel-size-local 0` when the Python engines
are started elsewhere and this node should run only the Rust frontend. The frontend still uses
the global `--data-parallel-size` to determine how many engines it expects to join the shared handshake.

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
./build_rust.sh
```

### Example Request

After either startup path, you can use any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```
