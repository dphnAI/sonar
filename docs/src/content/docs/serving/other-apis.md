---
title: Other APIs
description: Use Sonar endpoints outside the OpenAI-compatible API.
---

Sonar provides endpoint families for tasks that do not fit the OpenAI API.
Available routes depend on the model runner and server configuration.

## Pooling and scoring

Pooling models can provide embeddings, classifications, rewards, and token-level
representations. Use the task endpoint that matches the model runner.

The principal routes are:

| Route | Typical use |
| --- | --- |
| `/v1/embeddings` | OpenAI-compatible embeddings |
| `/v2/embed` | Native embedding requests |
| `/pooling` | Raw pooled model output |
| `/score` | Score one or more text pairs |
| `/rerank` and `/v2/rerank` | Rank documents against a query |
| `/generative_scoring` | Score candidates with a generation model |

Start a model with the runner and task required by that model. Check `/docs` on
a running server for the installed version's exact request schema.

## Tokenization

Use `/tokenize` to convert text to token IDs. Use `/detokenize` for the reverse
operation. These endpoints use the tokenizer loaded by the server.

## Render and derender

The render API performs request preprocessing without model execution. The
derender API converts generated token IDs into a response. These endpoints help
separate CPU preprocessing from GPU inference.

OpenAI-compatible render routes include `/v1/chat/completions/render` and
`/v1/completions/render`. Sonar also provides
`/inference/v1/generate` for token-input and token-output workflows. These APIs
are useful when a gateway performs tokenization or when CPU preprocessing runs
separately from GPU workers.

Do not assume token IDs are portable between models. The tokenizer, added
tokens, chat template, and model revision must match.

### Multimodal Render Features

Multimodal render responses include a `features` object with per-modality
hashes, placeholder ranges, and serialized processor data. When the model
exposes placeholder-metadata or `keep_on_cpu` fields (for example
`image_grid_thw`), the response also includes `mm_metadata`. Each
`mm_metadata` entry is a base64-encoded `MultiModalKwargsItem` containing
only those fields, not encoder inputs such as `pixel_values`.

The arrays in `mm_hashes`, `mm_placeholders`, `kwargs_data`, and
`mm_metadata` use the same per-modality item order. Downstream workers
should split those fields:

- Encode requests keep `kwargs_data`.
- Prefill requests may omit `kwargs_data` and send `mm_metadata` only when
  `ec_transfer_params` is also set, so embeddings are loaded by the EC
  connector. Omitting `kwargs_data` without `ec_transfer_params` is rejected.
- Legacy clients that ignore `mm_metadata` and keep sending `kwargs_data`
  continue to work.

#### Example

The example below shows how a disaggregated encode / prefill coordinator can
split a multimodal render response. The render step returns both
`kwargs_data` (encoder tensors plus metadata) and `mm_metadata` (metadata
only). Encode keeps the full payload; prefill drops `kwargs_data` after the
EC connector has published embeddings.

```python
import httpx

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
RENDER = "http://localhost:8100"  # aphrodite launch render ...
ENCODE = "http://localhost:8200"  # encode worker
PREFILL = "http://localhost:8300"  # prefill worker

chat_request = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "<data-url>"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
}

with httpx.Client(timeout=120.0) as client:
    # 1. Render: preprocess into token IDs and multimodal features.
    render_response = client.post(
        f"{RENDER}/v1/chat/completions/render", json=chat_request
    ).json()

    features = render_response["features"]
    # features["kwargs_data"]["image"][0]  -> pixel_values + image_grid_thw
    # features["mm_metadata"]["image"][0]  -> image_grid_thw only

    # 2. Encode: send full kwargs_data so the encoder can run vision towers.
    encode_response = client.post(
        f"{ENCODE}/inference/v1/generate",
        json={
            "token_ids": render_response["token_ids"],
            "features": {
                "mm_hashes": features["mm_hashes"],
                "mm_placeholders": features["mm_placeholders"],
                "kwargs_data": features["kwargs_data"],
            },
            "sampling_params": {"max_tokens": 1},
        },
    ).json()
    ec_transfer_params = encode_response["ec_transfer_params"]

    # 3. Prefill: omit kwargs_data; load embeddings via EC connector.
    prefill_response = client.post(
        f"{PREFILL}/inference/v1/generate",
        json={
            "token_ids": render_response["token_ids"],
            "features": {
                "mm_hashes": features["mm_hashes"],
                "mm_placeholders": features["mm_placeholders"],
                "mm_metadata": features["mm_metadata"],
            },
            "ec_transfer_params": ec_transfer_params,
            "sampling_params": {"max_tokens": 64},
        },
    ).json()

print(prefill_response["choices"][0]["token_ids"])
```

Single-process clients can keep passing the full render response to
`/inference/v1/generate` unchanged; `mm_metadata` is optional when `kwargs_data` is present.

#### Payload shape

##### Render response

`/v1/chat/completions/render` returns both `kwargs_data` and `mm_metadata`.
The arrays share the same per-modality item order. Base64 blobs are truncated
below for readability.

```json
{
  "token_ids": [151644, 872],
  "features": {
    "mm_hashes": {"image": ["abc123..."]},
    "mm_placeholders": {"image": [{"offset": 0, "length": 256}]},
    "kwargs_data": {
      "image": ["<base64 MultiModalKwargsItem: pixel_values + image_grid_thw>"]
    },
    "mm_metadata": {
      "image": ["<base64 MultiModalKwargsItem: image_grid_thw only>"]
    }
  }
}
```

Forward `kwargs_data` to the encode worker. Keep `mm_metadata` for prefill.

##### Prefill request

Prefill omits `kwargs_data` and sends `mm_metadata` with
`ec_transfer_params` from the encode response:

```json
{
  "token_ids": [151644, 872],
  "features": {
    "mm_hashes": {"image": ["abc123..."]},
    "mm_placeholders": {"image": [{"offset": 0, "length": 256}]},
    "mm_metadata": {
      "image": ["<base64 MultiModalKwargsItem: image_grid_thw only>"]
    }
  },
  "ec_transfer_params": {
    "ec_items": [{"mm_hash": "abc123...", "peer_host": "10.0.0.1"}]
  },
  "sampling_params": {"max_tokens": 64}
}
```

## Anthropic Messages

Sonar provides `/v1/messages` and `/v1/messages/count_tokens` for
Anthropic-compatible clients. Confirm that the selected model has a suitable
chat template. Parser support for reasoning and tool calls remains
model-specific.

## Transcription

Audio transcription models expose `/v1/audio/transcriptions`. Use multipart
form data as required by the OpenAI-compatible transcription schema. Check the
[model matrix](/reference/models/) before you select a model.

The supported response formats are `json`, `text`, `verbose_json`, and, for
models with diarization support, `diarized_json`. The diarized format returns
speaker-attributed segments and is currently supported by
`OpenMOSS-Team/MOSS-Transcribe-Diarize`:

```json
{
  "task": "transcribe",
  "duration": 6.1,
  "text": "Hello. Hi, how are you?",
  "segments": [
    {
      "type": "transcript.text.segment",
      "id": "seg_0",
      "start": 0.0,
      "end": 2.8,
      "text": "Hello.",
      "speaker": "S01"
    }
  ],
  "usage": {"type": "duration", "seconds": 7}
}
```

## Kobold API

Use `--launch-kobold-api` to enable the Kobold-compatible routes. Check the
generated [server arguments](/reference/server-arguments/) for related options.

## Health and metrics

- `/health` reports server health.
- `/metrics` exports Prometheus metrics.
- `/docs` shows the interactive OpenAPI documentation.
- `/openapi.json` returns the machine-readable schema.

Do not expose administrative and diagnostic endpoints to an untrusted network.

## Discover routes on your server

The available routes depend on the selected model and server mode. Inspect the
schema instead of relying on a route from another deployment:

```bash
curl -fsS http://127.0.0.1:2242/openapi.json > openapi.json
```

Use the schema from the same Sonar commit that runs in production. This avoids
client drift when an endpoint gains a field or changes validation.
