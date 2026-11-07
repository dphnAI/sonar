---
title: Quickstart
description: Start a Sonar server and send your first request.
---

Install Sonar before you use this guide.

## Start a server

```bash
sonar serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 2242
```

Sonar downloads the model on the first start. It stores Hugging Face files in
the standard Hugging Face cache.

## Send a chat request

```bash
curl http://localhost:2242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Write one sentence about paged attention."}
    ]
  }'
```

## Use the Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:2242/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

Use `--api-key` when clients outside a trusted network can reach the server.
