# Engine-Core Smoke Tests

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
  --enable-sleep-mode \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

Run the Rust smoke test through the `sonar-engine-core-client` utility interface:

```bash
cargo run -p sonar-engine-core-client --example external_engine_utility_call -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

If your current engine setup does not support sleep mode, skip the `sleep` / `wake_up` part of the
smoke:

```bash
cargo run -p sonar-engine-core-client --example external_engine_utility_call -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1 \
  --skip-sleep-wake
```

Run the Rust smoke test for sample logprobs decoding through the raw engine-core request path:

```bash
cargo run -p sonar-engine-core-client --example external_engine_logprobs -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

This smoke requests a small generated-token `logprobs` payload plus prompt logprobs over a much
longer prompt, so it exercises both the inline and aux-frame decode paths against a real engine.
The Rust client decodes those payloads into semantic per-position records rather than exposing the
raw ndarray/tensor wire shape.

IMPORTANT: You must restart `sonar` each time you run the smoke test, as the Sonar engine cannot manage frontend closures and subsequent reconnects. In other words, do not reuse existing `sonar` instances, if any.
