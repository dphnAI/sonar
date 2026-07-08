---
title: Optimization
---

## Faster Startup

Three mechanisms can reduce time-to-first-token on repeated boots of the same model, config, and hardware combination:

- **Reuse the compile cache.** Aphrodite persists `torch.compile` artifacts under `APHRODITE_CACHE_ROOT` (default `~/.cache/aphrodite`), and the cache directory can be copied between machines or baked into a container image. Set `APHRODITE_FORCE_AOT_LOAD=1` to fail loudly instead of silently recompiling when the cache misses. Any change to the model, config, relevant `APHRODITE_*` environment variables, torch build, or GPU model can invalidate the cache.
- **Skip memory profiling with `--kv-cache-memory-bytes`.** On startup, Aphrodite logs the exact `--kv-cache-memory-bytes` value that reproduces the current allocation. Passing it back on the next boot skips the memory-profiling measurement and the CUDA graph memory estimation pass. Note that this has performance implications: the KV cache is sized to exactly the given value instead of being measured, so a conservative value caps batch concurrency and throughput, while an optimistic one fails at allocation time. The value is only valid on the same GPU with the same initial free memory; if a boot runs out of memory after hardware or co-tenant changes, remove the flag to re-profile.
- **Serve without CUDA graphs using `--enforce-eager`.** This skips both compilation and CUDA graph capture for the fastest possible startup, at the cost of steady-state decode performance. It is useful for development loops and for measuring how much of a boot is compile or capture time.
