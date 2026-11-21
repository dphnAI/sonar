# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest
import torch
from transformers import OPTConfig

from sonar.config import CacheConfig, ModelConfig, SchedulerConfig, SonarConfig
from sonar.config.cache import PrefixCacheEviction
from sonar.sampling_params import SamplingParams
from sonar.utils.hashing import sha256
from sonar.v1.core.block_pool import BlockPool
from sonar.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from sonar.v1.core.queueguard import QueueGuard
from sonar.v1.core.sched.scheduler import Scheduler
from sonar.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from sonar.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowMLASpec
from sonar.v1.request import Request
from sonar.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test
BLOCK = 16


def request(name: str, *pages: int) -> Request:
    return Request(
        request_id=name,
        prompt_token_ids=[token for token in pages for _ in range(BLOCK)] + [0],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def test_shared_prefix_survives_tail_eviction_and_owner_departure() -> None:
    head = request("head", 1, 2)
    tail = request("tail", 1, 3, 4)
    idle = request("idle", 5)
    waiting = [head, tail]
    pool = BlockPool(6, enable_caching=True, hash_block_size=BLOCK)
    policy = QueueGuard(pool, BLOCK, lambda: waiting)
    pool.eviction_policy = policy
    shared, head_page, tail_page, tail_end, idle_page = pool.get_new_blocks(5)
    pool.cache_full_blocks(head, [shared, head_page], 0, 2, BLOCK, 0)
    pool.cache_full_blocks(tail, [shared, tail_page, tail_end], 1, 3, BLOCK, 0)
    pool.cache_full_blocks(idle, [idle_page], 0, 1, BLOCK, 0)
    pool.free_blocks([shared, head_page, tail_page, tail_end, idle_page])

    assert pool.get_new_blocks(2) == [idle_page, tail_end]
    assert pool.get_cached_block(head.block_hashes[0], [0]) == [shared]
    assert pool.get_cached_block(head.block_hashes[1], [0]) == [head_page]

    # The shared page now belongs to the remaining waiter; the other page
    # becomes evictable even after previously being parked for the head.
    waiting.pop(0)
    pool.touch([tail_page])
    pool.free_blocks([tail_page])
    policy.begin_step()
    assert pool.get_new_blocks(1) == [head_page]
    assert pool.get_cached_block(tail.block_hashes[0], [0]) == [shared]
    assert pool.get_cached_block(tail.block_hashes[1], [0]) == [tail_page]
    assert pool.get_new_blocks(2) == [tail_page, shared]
    assert pool.get_num_free_blocks() == 0


@pytest.mark.parametrize("eviction,keeps_waiter", [("lru", False), ("queueguard", True)])
def test_fcfs_prefill_preserves_next_prefix_with_swa_replay(
    tmp_path: Path, eviction: PrefixCacheEviction, keeps_waiter: bool
) -> None:
    OPTConfig(architectures=["OPTForCausalLM"], max_position_embeddings=128).save_pretrained(tmp_path)
    config = SonarConfig(
        model_config=ModelConfig(model=str(tmp_path), skip_tokenizer_init=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=64, max_model_len=128, watermark=0.0, is_encoder_decoder=False
        ),
        cache_config=CacheConfig(block_size=BLOCK, enable_prefix_caching=True, prefix_cache_eviction=eviction),
    )
    config.cache_config.num_gpu_blocks = 8
    register_all_kvcache_specs(config)
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["main"],
                FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=16, dtype=torch.bfloat16),
            ),
            KVCacheGroupSpec(
                ["window"],
                SlidingWindowMLASpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=16,
                    dtype=torch.bfloat16,
                    sliding_window=BLOCK,
                    bounded_replay=True,
                ),
            ),
        ],
    )
    scheduler = Scheduler(config, cache_config, StructuredOutputManager(config), block_size=BLOCK)
    pool = scheduler.kv_cache_manager.block_pool
    waiting = request("waiting", 1, 2, 3)
    idle = request("idle", 4, 5)
    cached, discard = pool.get_new_blocks(3), pool.get_new_blocks(2)
    pool.cache_full_blocks(waiting, cached, 0, 3, BLOCK, 0)
    pool.cache_full_blocks(idle, discard, 0, 2, BLOCK, 0)
    pool.free_blocks([*cached, *discard])

    first = request("first", 6)
    scheduler.add_request(first)
    scheduler.add_request(waiting)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == [first.request_id]
    hit_blocks = [pool.get_cached_block(h, [0]) for h in waiting.block_hashes]
    assert all(hit_blocks) is keeps_waiter
