---
title: Observability
description: Monitor Sonar with health checks, metrics, logs, and traces.
---

Monitor client latency and engine state together. A client timeout can come from
queueing, prefill, decode, media loading, or a proxy.

## Health checks

Use `/health` for server health:

```bash
curl --fail http://localhost:2242/health
```

Data-parallel supervisor deployments also provide `/ready` and `/readyz`. Give
startup probes enough time for model loading, compilation, and graph capture.

## Prometheus metrics

```bash
curl http://localhost:2242/metrics
```

Track these metric groups:

- Running and waiting requests
- Time to first token
- Inter-token latency
- End-to-end request latency
- Prompt and generation throughput
- KV cache occupancy and preemption
- Prefix-cache hits
- Speculative draft and acceptance counters

Alert on sustained queue growth and tail latency. A short GPU utilization spike
does not require an alert.

## Request correlation

Use `--enable-request-id-headers` to return `X-Request-Id`. Clients can send
their own value for correlation across a proxy and application.

Do not put user content in a request ID.

## Logs

Set `SONAR_LOGGING_LEVEL=DEBUG` for a bounded diagnostic run. Debug output
can be large and can contain workload details.

Request logging must follow the service privacy policy. Restrict access and
define retention before you enable prompt logging.

### Request completion timings

Python and Rust use the same definitions for request completion logs:

| Field | Definition |
| --- | --- |
| E2E time | Time from the frontend's recorded request arrival to completion. |
| TTFT | Time from the frontend's recorded request arrival to its first received engine token. |
| Prefill tokens/s | Prompt tokens divided by the engine interval from first scheduling to first output. This interval includes prefill preemptions. |
| Decode tokens/s | Generated tokens divided by the engine interval from first output to last output. |

Prefill throughput includes cached prompt tokens. It is an effective prompt
rate, not a measure of GPU compute throughput. Check the cache-hit count when
you compare requests.

Server-side TTFT is not client-side TTFT. Network latency and output buffering
can increase the time before a client receives its first text.

## OpenTelemetry traces

Set `--otlp-traces-endpoint` to an OTLP collector. Keep the collector on a
reliable private network and test trace export under load.

Trace sampling reduces export overhead. Preserve full traces for errors and a
small sample of successful requests.

## Dashboard layout

Put request rate, errors, and latency in the first row. Add queue depth and
engine throughput below them. Put GPU memory, KV cache, prefix cache, and
speculative metrics in diagnostic rows.

Record deployment annotations for Sonar, model, and configuration changes. This
lets an operator match a performance change to a release.
