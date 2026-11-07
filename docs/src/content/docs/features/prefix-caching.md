---
title: Prefix caching
description: Reuse KV cache blocks for repeated prompt prefixes.
---

Prefix caching is enabled by default. It avoids repeated prefill work when
requests begin with the same token sequence.

Common reusable prefixes include system prompts, tool definitions, shared
documents, and fixed few-shot examples.

## Get a cache hit

The cache key contains token IDs and request properties that affect model
execution. Two text strings must tokenize to the same prefix. Changing content
near the start prevents reuse of every block after that change.

Put stable content first. Put user-specific or time-varying content near the end
of the prompt.

## Measure the benefit

Send one request to populate the cache. Repeat the shared prefix with a different
suffix. Compare time to first token and prefix-cache metrics.

Prefix caching reduces prefill work. It does not make generated decode tokens
faster.

Use [`sonar bench`](/deployment/benchmarking/) with a prefix-repetition
dataset for a controlled comparison.

## Disable prefix caching

```bash
sonar serve MODEL --no-enable-prefix-caching
```

Disable it when you need a clean benchmark baseline or when a platform feature
requires it.

## Select a hash

The default hash is SHA-256. Reproducible CBOR variants are available for cache
systems that need canonical cross-language serialization. xxHash variants need
the optional `xxhash` package.

Use a cryptographic hash for an untrusted multi-tenant service unless you have
evaluated collision risk.

## Avoid ineffective layouts

Short shared prefixes save little work. Templates that insert a user ID, date,
or request identifier near the start produce poor reuse.

Large cached prefixes still occupy KV cache blocks. Watch cache pressure when
many unrelated prefixes compete for capacity.
