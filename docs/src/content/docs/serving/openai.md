---
title: OpenAI-compatible API
description: Serve models through OpenAI-compatible HTTP endpoints.
---

Start the server with a model repository or a local model path.

```bash
sonar serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3 \
  --api-key "$SONAR_API_KEY"
```

Set the client base URL to `http://HOST:2242/v1`.

The value passed to `--served-model-name` is the model name that clients send.
If you omit it, clients must use the model path or repository name that started
the server.

## Check the service

```bash
curl --fail http://127.0.0.1:2242/health
curl http://127.0.0.1:2242/v1/models \
  -H "Authorization: Bearer $SONAR_API_KEY"
```

Use `/health` for readiness checks. Use `/metrics` for Prometheus collection.
Do not expose these routes to an untrusted network without an access-control
layer.

## Chat Completions

Use `/v1/chat/completions` for instruct and chat models.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:2242/v1",
    api_key="local-key",
)
response = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Explain prefix caching."}],
    temperature=0.2,
)
print(response.choices[0].message.content)
```

Stream tokens by setting `stream=True`:

```python
stream = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Explain continuous batching."}],
    stream=True,
)

for event in stream:
    text = event.choices[0].delta.content
    if text:
        print(text, end="", flush=True)
```

## Responses

Use `/v1/responses` with clients that use the OpenAI Responses API. Model
features such as reasoning and tool calls depend on the selected parser.

Use Chat Completions when an existing client depends on its response schema.
Use Responses when the client needs the newer response-item format.

## Completions

Use `/v1/completions` for a raw prompt.

```python
response = client.completions.create(
    model="qwen3",
    prompt="A KV cache stores",
    max_tokens=64,
)
```

## Embeddings

Start Sonar with an embedding model. Then call `/v1/embeddings`.

```python
response = client.embeddings.create(
    model="intfloat/e5-small-v2",
    input=["first document", "second document"],
)
```

## Audio and multimodal requests

Supported multimodal models accept OpenAI message content with image, audio, or
video items. Transcription models expose the compatible audio endpoints. Check
the [model matrix](/reference/models/) before you deploy a model.

## Tool calls and reasoning

Start the server with parsers that match the model:

```bash
sonar serve MODEL \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1
```

Parser names and output formats are model-specific. Do not select a parser
only because the server accepts its name. Follow the model publisher's format
and test both streaming and non-streaming responses. See
[Reasoning and tool parsers](/features/reasoning-and-tools/).

## Sonar request parameters

Sonar accepts additional sampling fields through `extra_body`.

```python
response = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "min_p": 0.05,
        "repetition_penalty": 1.05,
    },
)
```

Use the generated [server argument reference](/reference/server-arguments/) for
startup options. Use the server OpenAPI schema at `/openapi.json` for the current
request and response schemas.

## Timeouts, retries, and limits

Set a client timeout that permits the longest expected prompt and output. Retry
connection failures and temporary `5xx` responses with backoff. Do not retry a
non-idempotent request automatically unless the application can accept a
duplicate generation.

Limit input and output lengths at the application boundary. A server-wide
`--max-model-len` limits the combined sequence length. Per-request token limits
still control each response.

## Measure the deployed path

Test the same endpoint, proxy, authentication, and model name that production
uses:

```bash
sonar bench serve \
  --backend openai \
  --base-url http://127.0.0.1:2242 \
  --model qwen3 \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --request-rate 4 \
  --num-prompts 500 \
  --save-result
```

Read [Benchmark with `sonar bench`](/deployment/benchmarking/) before you
compare configurations.
