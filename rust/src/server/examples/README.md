# Server Smoke Test

Start a fresh headless `sonar` engine:

```bash
source ../sonar/.venv/bin/activate
HF_HUB_OFFLINE=1 \
SONAR_LOGGING_LEVEL=DEBUG \
SONAR_CPU_KVCACHE_SPACE=2 \
SONAR_HOST_IP=127.0.0.1 \
SONAR_LOOPBACK_IP=127.0.0.1 \
python3 -m sonar.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

Run the Rust server smoke test:

```bash
cargo run -p sonar-server --example external_engine_openai_qwen -- \
  --handshake-address tcp://127.0.0.1:62100
```

The example starts the Rust OpenAI-compatible server on an ephemeral local port,
connects to it via the `async-openai` Rust client, lists models, and then checks
that a streamed chat completion yields the assistant role chunk, final-answer
content chunks, and a terminal finish chunk. This example intentionally uses
`async-openai`'s standard typed `create_stream` API instead of BYOT, so it does
not inspect the nonstandard `reasoning_content` field even though the Rust
server may emit it for reasoning-capable models such as Qwen3. For reasoning
behavior itself, use the `sonar-chat` smoke test or the `sonar-server`
route tests.

IMPORTANT: Restart `sonar` each time you run the smoke test. The current headless
engine cannot safely handle frontend reconnects after the client shuts down.
