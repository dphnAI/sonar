# LLM Smoke Test

Start headless `aphrodite`:

```bash
source ../aphrodite/.venv/bin/activate
HF_HUB_OFFLINE=1 \
APHRODITE_LOGGING_LEVEL=DEBUG \
APHRODITE_CPU_KVCACHE_SPACE=2 \
APHRODITE_HOST_IP=127.0.0.1 \
APHRODITE_LOOPBACK_IP=127.0.0.1 \
python3 -m aphrodite.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

Run the Rust smoke test through the `aphrodite-llm` generate interface:

```bash
cargo run -p aphrodite-llm --example external_engine_smoke -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

IMPORTANT: You must restart `aphrodite` each time you run the smoke test, as the Aphrodite engine cannot manage frontend closures and subsequent reconnects. In other words, do not reuse existing `aphrodite` instances, if any.
