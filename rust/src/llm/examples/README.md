# LLM Smoke Test

Start headless `sonar`:

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

Run the Rust smoke test through the `sonar-llm` generate interface:

```bash
cargo run -p sonar-llm --example external_engine_smoke -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

IMPORTANT: You must restart `sonar` each time you run the smoke test, as the Sonar engine cannot manage frontend closures and subsequent reconnects. In other words, do not reuse existing `sonar` instances, if any.
