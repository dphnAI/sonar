---
title: Optimization
---

## Faster Startup

Three mechanisms can reduce time-to-first-token on repeated boots of the same model, config, and hardware combination:

- **Reuse the compile cache.** Aphrodite persists `torch.compile` artifacts under `APHRODITE_CACHE_ROOT` (default `~/.cache/aphrodite`), and the cache directory can be copied between machines or baked into a container image. Set `APHRODITE_FORCE_AOT_LOAD=1` to fail loudly instead of silently recompiling when the cache misses. Any change to the model, config, relevant `APHRODITE_*` environment variables, torch build, or GPU model can invalidate the cache.
- **Skip memory profiling with `--kv-cache-memory-bytes`.** On startup, Aphrodite logs the exact `--kv-cache-memory-bytes` value that reproduces the current allocation. Passing it back on the next boot skips the memory-profiling measurement and the CUDA graph memory estimation pass. Note that this has performance implications: the KV cache is sized to exactly the given value instead of being measured, so a conservative value caps batch concurrency and throughput, while an optimistic one fails at allocation time. The value is only valid on the same GPU with the same initial free memory; if a boot runs out of memory after hardware or co-tenant changes, remove the flag to re-profile.
- **Serve without CUDA graphs using `--enforce-eager`.** This skips both compilation and CUDA graph capture for the fastest possible startup, at the cost of steady-state decode performance. It is useful for development loops and for measuring how much of a boot is compile or capture time.

## KV Cache Offloading

Native KV cache offloading is enabled through `--kv-transfer-config` with the `OffloadingConnector` and an offloading spec such as `CPUOffloadingSpec` or `TieringOffloadingSpec`. The `kv_connector_extra_config` object controls the offloaded chunk size:

| Option | Default | Description |
| --- | --- | --- |
| `block_size` | GPU block size | Offloaded chunk size in tokens; must be a multiple of the GPU block size. Mutually exclusive with `blocks_per_chunk`. |
| `blocks_per_chunk` | `1` | Offloaded chunk size in GPU blocks; must be greater than `0`. Use this instead of `block_size` for models whose KV cache groups have different block sizes. |

Larger offloaded chunks reduce per-block bookkeeping overhead but increase the granularity of lookups. When self-describing KV events are enabled, `CPUOffloadingSpec` and `TieringOffloadingSpec` emit block-granular store/remove payloads for full-attention groups. With tiering, CPU promotions are self-describing when a local request observes a primary-tier hit before event translation; externally initiated promotions, pending-removal races, or re-promotions can still emit placeholders, and consumers should ignore removals for unknown hashes. In chunk mode (`block_size` greater than the GPU block size or `blocks_per_chunk` greater than `1`), overlapping chunks can re-announce shared per-block hashes, so external consumers should reference-count repeated store and remove announcements.
