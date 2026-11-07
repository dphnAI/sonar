---
title: Benchmark with sonar bench
description: Measure Sonar latency, throughput, startup, and online serving.
---

Use `sonar bench` for repeatable measurements. Record the command, model
revision, Sonar commit, hardware, and server configuration with each result.

List the benchmark commands in the installed version:

```bash
sonar bench --help
```

The available commands include `latency`, `throughput`, `serve`, `startup`,
`perf`, `mm-processor`, and `sweep`.

## Measure one offline batch

`latency` loads the model in the benchmark process. It measures repeated
execution of one fixed batch after warm-up.

```bash
sonar bench latency \
  --model Qwen/Qwen3-8B \
  --input-len 512 \
  --output-len 128 \
  --batch-size 8 \
  --num-iters-warmup 10 \
  --num-iters 30 \
  --output-json latency.json
```

This benchmark disables prefix caching by default. Random token prompts would
otherwise reuse cached blocks across iterations.

Use it to compare kernels or engine settings with one controlled batch. It does
not model queueing or arrival rate.

## Measure offline throughput

```bash
sonar bench throughput \
  --model Qwen/Qwen3-8B \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 1000 \
  --output-json throughput.json
```

Use a real dataset when prompt length, stop tokens, or chat templates affect
the result. Random prompts are useful for a shape-controlled comparison.

## Measure a running server

Start Sonar in one terminal:

```bash
sonar serve Qwen/Qwen3-8B \
  --served-model-name qwen3
```

Run the client benchmark from another terminal:

```bash
sonar bench serve \
  --backend openai \
  --base-url http://127.0.0.1:2242 \
  --model qwen3 \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --request-rate 8 \
  --num-prompts 1000 \
  --save-result
```

`request-rate` controls offered load. Test several rates. The service reaches
saturation when queueing increases latency faster than completed throughput.

Set a maximum concurrency when production has an admission limit. Use the same
API key and proxy path that production clients use when you need an end-to-end
measurement.

## Measure startup

```bash
sonar bench startup \
  --model Qwen/Qwen3-8B \
  --output-json startup.json
```

The startup benchmark separates cold and warm runs. Warm runs reuse model and
compilation caches. Run it on the same storage used by production.

## Inspect single-request performance

```bash
sonar bench perf --model Qwen/Qwen3-8B
```

`perf` reports prompt and generation behavior across sequence lengths. Use it
for a quick device check before a complete serving benchmark.

## Run parameter sweeps

```bash
sonar bench sweep --help
```

Sweep commands can run serving or startup experiments and plot results. Keep
the model revision and workload fixed while the sweep changes engine settings.

Useful sweep variables include `max_num_seqs`, `max_num_batched_tokens`,
`gpu_memory_utilization`, and speculative depth. Reject combinations that
change the accepted workload.

## Interpret results

Compare time to first token, inter-token latency, request latency, and completed
throughput. Report percentiles instead of only an average.

Warm up every configuration. Save JSON output and check that each run completed
the same number of requests. A faster result with request failures is invalid.
