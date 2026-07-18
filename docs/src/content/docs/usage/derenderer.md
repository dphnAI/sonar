---
title: Derenderer APIs
---

The derenderer API is the postprocessing counterpart to the render API. Where `/render` turns a request into token IDs, `/derender` turns generated token IDs back into a fully formed OpenAI-compatible response. It handles detokenization, reasoning parsing, and tool call parsing without a GPU.

This closes the loop for a token-in/token-out engine in disaggregated serving:

- **GPU-less postprocessing**: Detokenization, reasoning parsing, and tool call parsing run on the same GPU-less frontend that hosts `/render`.
- **Parser parity**: The derenderer reuses Aphrodite's tool and reasoning parsers, so a disaggregated deployment produces the same `content`, `reasoning`, and `tool_calls` split as a standard `aphrodite serve` server.
- **Non-streaming**: The endpoints expect a complete generate response with all token IDs present and perform one-shot parsing. Streaming derender would require a separate endpoint design and is not currently supported.

Both endpoints are hosted by the GPU-less rendering server started with `aphrodite launch render`, alongside the `/render` endpoints.

## Pipeline

```text
request -> render -> token_ids -> generate -> token_ids -> derender -> response
```

The derender step needs more than the engine's `token_ids`. It also consumes the original `chat_request` or `completion_request` and `prompt_tokens` carried over from the render step so the tool and reasoning parsers have the context they need.

## API Reference

- Chat Completions Derender API (`/v1/chat/completions/derender`): postprocess a single generate response into a chat completion response.
- Completions Derender API (`/v1/completions/derender`): postprocess a list of generate responses, one per prompt, into a completion response.

## Request Format

`/v1/chat/completions/derender` wraps one generate response:

```json
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "generate_response": {
    "request_id": "request-id",
    "choices": [
      {
        "index": 0,
        "token_ids": [128000, 791, 4320],
        "finish_reason": "stop"
      }
    ]
  },
  "prompt_tokens": 24,
  "chat_request": {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }
}
```

`/v1/completions/derender` wraps one generate response per prompt:

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "generate_responses": [
    {
      "request_id": "request-id",
      "choices": [
        {
          "index": 0,
          "token_ids": [128000, 791, 4320],
          "finish_reason": "stop"
        }
      ]
    }
  ],
  "prompt_tokens": [12],
  "completion_request": {
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": ["Once upon a time"]
  }
}
```

Oversized payloads are rejected with a `400` before any tokenizer decode or parser runs.

## Example

The example below drives the full render -> generate -> derender round trip for a chat request against a GPU-less render server and a token-in/token-out engine.

```py
import httpx

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RENDER = "http://localhost:8100"
ENGINE = "http://localhost:8200"

chat_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
}

with httpx.Client(timeout=60.0) as client:
    generate_request = client.post(
        f"{RENDER}/v1/chat/completions/render", json=chat_request
    ).json()
    prompt_tokens = len(generate_request["token_ids"])

    generate_response = client.post(
        f"{ENGINE}/inference/v1/generate", json=generate_request
    ).json()

    response = client.post(
        f"{RENDER}/v1/chat/completions/derender",
        json={
            "model": MODEL,
            "generate_response": generate_response,
            "prompt_tokens": prompt_tokens,
            "chat_request": chat_request,
        },
    ).json()

print(response["choices"][0]["message"]["content"])
```

Passing `chat_request` lets the derenderer run the configured tool and reasoning parsers. The returned message then carries the same `content`, `reasoning`, and `tool_calls` split a normal `aphrodite serve` server would produce. Omit `chat_request` for plain detokenization only.
