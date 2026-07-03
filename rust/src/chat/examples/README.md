# Chat Smoke Test

Start a fresh headless `aphrodite` engine:

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

Run the Rust chat smoke test through the `aphrodite-chat` interface:

```bash
cargo run -p aphrodite-chat --example external_engine_chat_qwen -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1 \
  --prompt 'What is the capital of France? Answer with one word.'
```

The example now defaults to `Qwen/Qwen3-0.6B`. The current `aphrodite-chat`
request model stays text-first and supports either plain string content or
OpenAI-style text blocks, while the output side now emits structured assistant
events and automatically separates reasoning blocks for supported models. Tool
use and multimodal inputs are still out of scope. It uses the Rust
`tokenizers` library for the tokenizer itself, plus standard Hugging Face
config files to load the chat template and EOS metadata.

IMPORTANT: Restart `aphrodite` each time you run the smoke test. The current headless
engine cannot safely handle frontend reconnects after the client shuts down.
