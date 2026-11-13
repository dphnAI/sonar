# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local FlashStates snapshots on the native Mooncake Store API."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .layout import KeyMetadata, RankLocalStoreLayout

FRONTIER_LOOKUP_MSG = b"frontier_lookup_v1"


@dataclass
class FrontierTransfer:
    key: str
    locator: str
    num_tokens: int
    # (scheduler group, logical block index, physical block id).
    blocks: list[tuple[int, int, int]]


def locator_key(index: Any, request: Any, num_tokens: int) -> str:
    from sonar.v1.core.hybrid_checkpoint.keys import frontier_key

    # Probe one prefix per hash unit. The store appends the frontier's offset,
    # so several lengths share this probe without overwriting one another.
    start = (num_tokens - 1) // index.hash_block_size * index.hash_block_size
    return frontier_key(index.hash_fn, request, start + 1, index.hash_block_size).hex()


class FrontierStore:
    """Transfer whole manager blocks, including recurrent and ring state.

    Each rank commits its manifest only after all its payload puts succeed.
    A locator is merely a hint: lookup requires every rank's manifest and
    payloads. Eviction between lookup and load is handled as a failed load by
    the connector, never as a shorter successful checkpoint.
    """

    def __init__(self, store: Any, config: Any, kv_config: Any, cache_prefix: str, hash_block_size: int):
        self.store = store
        parallel = config.parallel_config
        # The external launcher includes DP in ParallelConfig.world_size.
        # Checkpoint manifests and layout collectives cover one engine only.
        self.world_size = (
            parallel.tensor_parallel_size * parallel.prefill_context_parallel_size * parallel.pipeline_parallel_size
        )
        self.rank = parallel.rank % self.world_size
        self.groups = kv_config.kv_cache_groups
        self.num_blocks = kv_config.num_blocks
        self.hash_block_size = hash_block_size
        # Capacity is excluded here. Actual tensor strides enter the layout
        # namespace during registration and can restrict compatible capacities.
        schema = (
            config.model_config.model,
            config.model_config.dtype,
            config.cache_config.cache_dtype,
            config.cache_config.mamba_cache_dtype,
            hash_block_size,
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.decode_context_parallel_size,
            parallel.prefill_context_parallel_size,
            parallel.cp_kv_cache_interleave_size,
            [(type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size) for g in self.groups],
        )
        digest = hashlib.sha256(repr(schema).encode()).hexdigest()
        self.prefix = f"{cache_prefix}@flashstates-v1:{digest}"
        self._base_prefix = self.prefix
        self.layouts: dict[int, RankLocalStoreLayout] = {}
        self.schema = ""

    def register(
        self,
        caches: dict[str, Any],
        registered: dict[int, int],
        borrowed: set[int] | None = None,
    ) -> None:
        schema: list[tuple[Any, ...]] = [(group.layer_names, group.kv_cache_spec) for group in self.groups]
        for group_id, group in enumerate(self.groups):
            tensors = []
            for name in group.layer_names:
                value = caches.get(name)
                if value is None:  # Pipeline stages own only their local layers.
                    continue
                views = value if isinstance(value, list) else [value]
                tensors.extend(views)
                for tensor in views:
                    schema.append((name, tuple(tensor.shape[1:]), tensor.stride(), str(tensor.dtype)))
                    storage = tensor.untyped_storage()
                    addr, size = storage.data_ptr(), storage.nbytes()
                    if addr not in registered and (borrowed is None or addr not in borrowed):
                        result = self.store.register_buffer(addr, size)
                        if result != 0:
                            raise RuntimeError(f"Mooncake frontier buffer registration failed: {result}")
                        registered[addr] = size
            layout = RankLocalStoreLayout(KeyMetadata("frontier", 0, 0, 0, 0), 1, 1)
            layout.register_kv_caches(tensors, self.num_blocks)
            self.layouts[group_id] = layout
            schema.append((group_id, layout.block_len, layout.block_stride))
        self.schema = hashlib.sha256(repr(schema).encode()).hexdigest()

    def set_layout_namespace(self, rank_schemas: list[str]) -> None:
        """Finalize a namespace shared by the engine's TP/PCP/PP ranks."""
        if len(rank_schemas) != self.world_size:
            raise ValueError("Frontier layout schemas must cover every worker rank")
        digest = hashlib.sha256(json.dumps(rank_schemas).encode()).hexdigest()
        self.prefix = f"{self._base_prefix}@layout:{digest}"

    def _manifest_key(self, key: str, rank: int) -> str:
        return f"{self.prefix}@{key}@rank:{rank}@manifest"

    def _locator_key(self, locator: str, offset: int) -> str:
        return f"{self.prefix}@locator:{locator}@offset:{offset}"

    def _locator_hint_key(self, locator: str) -> str:
        return f"{self.prefix}@locator:{locator}@present"

    def _payload_key(self, key: str, rank: int, group: int, index: int) -> str:
        return f"{self.prefix}@{key}@rank:{rank}@group:{group}@block:{index}"

    def _buffers(self, transfer: FrontierTransfer):
        keys, addresses, sizes = [], [], []
        for group, index, block in transfer.blocks:
            layout = self.layouts[group]
            # An empty PP group needs no payload on this rank.
            if not layout.block_len:
                continue
            keys.append(self._payload_key(transfer.key, self.rank, group, index))
            addresses.append(
                [
                    base + block * stride
                    for base, stride in zip(layout.kv_caches_base_addr, layout.block_stride, strict=True)
                ]
            )
            sizes.append(layout.block_len)
        return keys, addresses, sizes

    def save(self, transfer: FrontierTransfer, replicate_config: Any) -> None:
        keys, addresses, sizes = self._buffers(transfer)
        results = self.store.batch_put_from_multi_buffers(keys, addresses, sizes, replicate_config)
        if len(results) != len(keys) or any(result != 0 for result in results):
            raise RuntimeError("Mooncake frontier payload put failed")
        manifest = json.dumps(
            {
                "key": transfer.key,
                "num_tokens": transfer.num_tokens,
                "schema": self.schema,
                "payloads": keys,
            }
        ).encode()
        if self.store.put(self._manifest_key(transfer.key, self.rank), manifest) != 0:
            raise RuntimeError("Mooncake frontier manifest put failed")
        if self.rank == 0:
            locator = json.dumps(transfer.num_tokens).encode()
            offset = (transfer.num_tokens - 1) % self.hash_block_size + 1
            if self.store.put(self._locator_key(transfer.locator, offset), locator) != 0:
                raise RuntimeError("Mooncake frontier locator put failed")
            if self.store.put(self._locator_hint_key(transfer.locator), b"1") != 0:
                raise RuntimeError("Mooncake frontier locator hint put failed")

    def lookup_lengths(self, locators: list[str]) -> list[int]:
        found: list[int] = []
        hints = self.store.batch_is_exist([self._locator_hint_key(locator) for locator in locators])
        present_locators = [locator for locator, present in zip(locators, hints, strict=True) if present == 1]
        if not present_locators:
            return found
        locator_keys = [
            self._locator_key(locator, offset)
            for locator in present_locators
            for offset in range(1, self.hash_block_size + 1)
        ]
        exists = self.store.batch_is_exist(locator_keys)
        for locator, present in zip(locator_keys, exists, strict=True):
            if present != 1:
                continue
            value = self.store.get(locator)
            if not value:
                continue
            length = json.loads(value)
            if isinstance(length, int) and length > 0:
                found.append(length)
        return sorted(set(found), reverse=True)

    def lookup(self, candidates: list[tuple[int, str]]) -> list[tuple[int, str]]:
        found = []
        for length, key in candidates:
            payloads = []
            for rank in range(self.world_size):
                value = self.store.get(self._manifest_key(key, rank))
                if not value:
                    break
                manifest = json.loads(value)
                if manifest["key"] != key or manifest["num_tokens"] != length:
                    break
                if rank == self.rank and manifest["schema"] != self.schema:
                    break
                payloads.extend(manifest["payloads"])
            else:
                exists = self.store.batch_is_exist(payloads)
                if len(exists) == len(payloads) and all(result == 1 for result in exists):
                    found.append((length, key))
        return sorted(set(found), reverse=True)

    def load(self, transfer: FrontierTransfer) -> None:
        value = self.store.get(self._manifest_key(transfer.key, self.rank))
        if not value:
            raise RuntimeError("Mooncake frontier manifest was evicted")
        manifest = json.loads(value)
        if manifest["schema"] != self.schema or manifest["num_tokens"] != transfer.num_tokens:
            raise RuntimeError("Mooncake frontier layout mismatch")
        keys, addresses, sizes = self._buffers(transfer)
        if not set(keys).issubset(manifest["payloads"]):
            raise RuntimeError("Mooncake frontier manifest is incomplete")
        results = self.store.batch_get_into_multi_buffers(keys, addresses, sizes)
        if len(results) != len(keys) or any(
            result != sum(lengths) for result, lengths in zip(results, sizes, strict=True)
        ):
            raise RuntimeError("Mooncake frontier payload load failed")
