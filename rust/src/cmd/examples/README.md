# `sonar-rs` CLI Quick Start

Start Qwen3 with one managed `sonar-rs serve` command from the repo root:

```bash
HF_HUB_OFFLINE=1 \
SONAR_CPU_KVCACHE_SPACE=2 \
SONAR_HOST_IP=127.0.0.1 \
SONAR_LOOPBACK_IP=127.0.0.1 \
cargo run --bin sonar-rs -- serve \
  Qwen/Qwen3-0.6B \
  --python ../sonar/.venv/bin/python \
  --max-model-len 512 \
  -- \
  --dtype float16
```

This launches:

- a managed headless Python `sonar` engine
- the Rust OpenAI-compatible frontend on `127.0.0.1:8000`

All Python engine arguments must be placed after `--`. Arguments before `--` are parsed by the Rust
frontend itself.

You can then send OpenAI-style requests to the Rust frontend:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```

If you already started headless `sonar` yourself, use `frontend` instead:

```bash
cargo run --bin sonar-rs -- frontend \
  --handshake-address tcp://127.0.0.1:62100 \
  Qwen/Qwen3-0.6B
```
