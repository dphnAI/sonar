---
title: LoRA adapters
description: Serve and operate LoRA adapters with a shared base model.
---

LoRA adapters provide specialized model names while the base weights remain
shared. The adapter must target the same base model architecture and compatible
weight revision.

## Load adapters at startup

```bash
sonar serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules support=/models/support legal=/models/legal
```

Use the adapter name as the API model:

```python
response = client.chat.completions.create(
    model="support",
    messages=[{"role": "user", "content": "Help me reset my account."}],
)
```

The base served model name remains available unless the deployment restricts
it at the proxy.

## Set capacity

`--max-loras` limits adapters active in one batch. `--max-cpu-loras` controls
the host adapter cache. The adapter rank must not exceed
`--max-lora-rank`.

Higher limits allocate more memory. Size them from measured concurrent adapter
use instead of the total adapter catalog.

## Load adapters at runtime

Sonar provides `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` when runtime
adapter management is enabled. Treat these routes as administrative.

Do not let untrusted clients submit arbitrary filesystem paths or repository
names. Use a service-side allowlist and pin adapter revisions.

## Route requests

Use stable adapter names in client requests. A load balancer must send the
request to a replica that has the adapter or load it before routing traffic.

For several replicas, deploy adapter changes as a coordinated operation. Check
`/v1/models` on every replica before you announce the adapter.

## Performance

Batching requests from several adapters can reduce kernel efficiency. Measure
the expected adapter mix and `--max-loras` value.

Keep frequently used adapters in the device or host cache. A cold adapter load
adds latency to the first request.

## Troubleshoot adapters

Check the base model name, module targets, rank, and tensor shapes. An adapter
trained for another base revision can load and still produce incorrect output.

Compare one deterministic prompt against the training framework before you
enable batching.
