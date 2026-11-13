# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from sonar.v1.core.block_pool import BlockPool
from sonar.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from sonar.v1.core.single_type_kv_cache_manager import CircularBufferManager
from sonar.v1.kv_cache_interface import CircularBufferSpec

pytestmark = pytest.mark.cpu_test


def test_all_cached_duplicates_follow_removal_and_reuse():
    pool = BlockPool(6, enable_caching=True, hash_block_size=16)
    first, second = pool.get_new_blocks(2)
    key = make_block_hash_with_group_id(BlockHash(b"shared"), 0)
    alias = make_block_hash_with_group_id(BlockHash(b"frontier"), 0)
    pool.cache_block_under_key(first, key, 16)
    pool.cache_block_under_key(second, key, 16)
    pool.cache_block_under_key(first, alias, 19)
    cached = pool.cached_block_hash_to_block
    assert cached.get_all_blocks(key) == [first, second]
    # The returned list is not the map's mutable duplicate storage.
    cached.get_all_blocks(key).clear()
    assert cached.get_all_blocks(key) == [first, second]
    pool.uncache_key(first, alias)
    assert cached.get_all_blocks(alias) == []
    assert cached.get_all_blocks(key) == [first, second]
    pool._maybe_evict_cached_block(first)
    assert cached.get_all_blocks(key) == [second]
    pool._maybe_evict_cached_block(second)
    assert cached.get_all_blocks(key) == []
    pool.free_blocks([first, second])
    assert pool.get_num_free_blocks() == 5


def ring_manager(pool):
    manager = CircularBufferManager(
        CircularBufferSpec(
            block_size=8,
            compression_ratio=4,
            resume_catchup="decode",
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=64,
        needs_kv_cache_zeroing=True,
    )
    manager.takes_frontier_hits = True
    return manager


def test_ring_copy_endpoints_survive_request_retirement_until_execution():
    pool = BlockPool(5, enable_caching=True, hash_block_size=16)
    manager = ring_manager(pool)
    source = pool.get_new_blocks(1)[0]
    pool.free_blocks([source])
    manager.add_local_computed_blocks("resume", [source], 19, 0)
    assert source.ref_cnt == 1
    target = manager.allocate_new_blocks("resume", 20, 20)[0]
    assert manager.take_pending_cow_copies() == [(source, target)]
    assert target.ref_cnt == 2
    pool.free_blocks(manager.pop_blocks_for_free("resume"))
    assert target.ref_cnt == source.ref_cnt == 1
    # Neither endpoint can be reused while the copy is pending.
    pressure = pool.get_new_blocks(pool.get_num_free_blocks())
    assert source not in pressure and target not in pressure
    pool.free_blocks([source, target])
    pool.free_blocks(pressure)
    assert pool.get_num_free_blocks() == 4


def test_cancelled_ring_adoption_releases_its_source_reference():
    pool = BlockPool(4, enable_caching=True, hash_block_size=16)
    manager = ring_manager(pool)
    source = pool.get_new_blocks(1)[0]
    pool.free_blocks([source])
    manager.add_local_computed_blocks("cancelled", [source], 19, 0)
    pool.free_blocks(manager.pop_blocks_for_free("cancelled"))
    assert source.ref_cnt == 0
    assert not manager.take_pending_cow_copies()
    assert pool.get_num_free_blocks() == 3
