# Chat Smoke Test

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

Run the Rust chat smoke test through the `sonar-chat` interface:

```bash
cargo run -p sonar-chat --example external_engine_chat_qwen -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1 \
  --prompt 'What is the capital of France? Answer with one word.'
```

The example defaults to `Qwen/Qwen3-0.6B` and exercises text-only chat through
the Rust `tokenizers` library and standard Hugging Face model files. The chat
crate also supports tools, image/video/audio preprocessing, and native
DeepSeek V3.2/V4/V4.1, Harmony, Inkling, and Kimi K3 renderers; this example
does not supply those inputs.

Parser-built output grammars are initialized from the final prompt token IDs,
including expanded multimodal placeholders. They compose supported answer
constraints with tool-call and reasoning formats; `--tool-strict-level` sets
the tool constraint floor. The `hf` response-template parser is opt-in.
Completed chat events retain engine lifecycle timestamps and KV/encoder
transfer parameters. Multimodal stage timing is exposed through
`mm_timing_layer` and `mm_request_span`.

Sonar wheels retain the `sonar._rust_tool_parser` Python extension alongside
the Rust frontend. Its `Tool` accepts optional `strict` and `defer_loading`
fields; deferred tools remain callable and are only hidden from initial
template rendering.

IMPORTANT: Restart `sonar` each time you run the smoke test. The current headless
engine cannot safely handle frontend reconnects after the client shuts down.
