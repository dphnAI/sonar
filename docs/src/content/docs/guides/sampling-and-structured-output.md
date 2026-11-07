---
title: Sampling and structured output
description: Control generation and constrain Sonar output.
---

Sampling parameters change token selection for each request. Server arguments
set model and scheduler behavior for the complete process.

## Deterministic generation

Use temperature zero for greedy token selection:

```python
params = SamplingParams(temperature=0.0, max_tokens=256)
```

A fixed seed helps reproduce stochastic sampling. GPU kernels and batch shapes
can still cause small numerical differences.

## Common sampling controls

`temperature` changes distribution sharpness. `top_p` keeps the smallest token
set that reaches the selected probability mass. `top_k` limits selection to a
fixed number of tokens. `min_p` removes tokens below a fraction of the most
likely token probability.

Start with one truncation method. Combining several methods makes behavior
harder to interpret.

Use `presence_penalty` and `frequency_penalty` for OpenAI-style penalties.
`repetition_penalty` applies a multiplicative penalty. DRY targets repeated
sequences and has separate controls in `SamplingParams`.

The OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints
accept the same extra sampler fields as `SamplingParams`, such as
`dry_multiplier`, `xtc_probability`, `top_a`, `tfs`, `nsigma`, `skew`,
`smoothing_factor`, `dynatemp_min`/`dynatemp_max` and `mirostat_mode`.

## Suppress phrases

`banned_strings` keeps whole phrases out of the output. When generated text
starts to spell a banned phrase, Sonar rewinds to the phrase start and samples
again without the token that began it. Matching is case-insensitive unless
`banned_strings_case_sensitive` is true. Pass both through `sonar_xargs`:

```json
{"prompt": "The capital of France is", "sonar_xargs": {"banned_strings": ["Paris"]}}
```

The server must run the phrase scheduler; otherwise the request is rejected
with HTTP 400:

```bash
sonar serve MODEL --scheduler-cls sonar.v1.phrase_guard.scheduler.PhraseScheduler
```

Phrase suppression requires plain-text generation with a fast Hugging Face
tokenizer. It cannot be combined with text `stop` strings (use
`stop_token_ids`), structured output, or Mirostat.

## Stop generation

Set `max_tokens` for every request. Add text stops or token ID stops when the
application has a protocol delimiter.

```python
params = SamplingParams(
    temperature=0.2,
    max_tokens=256,
    stop=["</answer>"],
)
```

The model end-of-sequence token also stops generation unless `ignore_eos` is
enabled.

## Generate JSON from Python

```python
from sonar import LLM, SamplingParams
from sonar.sampling_params import StructuredOutputsParams

params = SamplingParams(
    max_tokens=128,
    structured_outputs=StructuredOutputsParams(
        json={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        }
    ),
)
```

The schema constrains valid output tokens. Validate the returned JSON in the
application because a schema does not apply business rules.

Structured output also supports a regular expression, a choice list, a grammar,
and JSON-object mode. Set one primary constraint per request.

## Use the OpenAI response format

OpenAI-compatible clients can pass a JSON schema through `response_format`.
Use the current server schema at `/openapi.json` to confirm the accepted request
shape.

## Performance

Complex schemas increase preprocessing and token-mask work. Reuse schemas when
possible. Benchmark large choice sets and deeply nested schemas before
production use.
