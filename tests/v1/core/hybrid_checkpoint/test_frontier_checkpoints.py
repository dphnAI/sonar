# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba frontier checkpoints (`mamba_frontier_checkpoint`) on the real
scheduler with an attention group, two align-mode Mamba groups and an indexer
ring, driven step by step on CPU."""

import pytest

from sonar.utils.hashing import sha256
from sonar.v1.core.hybrid_checkpoint.keys import frontier_key
from sonar.v1.core.hybrid_checkpoint.manager import check_frontier_checkpoint_config
from sonar.v1.core.kv_cache_utils import (
    get_block_hash,
    init_none_hash,
    make_block_hash_with_group_id,
)
from sonar.v1.request import Request, RequestStatus

from .utils import (
    add_request,
    admitted_at,
    make_hybrid_scheduler,
    mock_connector,
    prompt,
    run,
    step,
)

pytestmark = pytest.mark.cpu_test

BLOCK = 64
RING = 4


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def make_scheduler(frontier: bool = True, **kwargs):
    return make_hybrid_scheduler(block_size=BLOCK, ring=RING, mamba_frontier_checkpoint=frontier, **kwargs)


@pytest.mark.parametrize("frontier", [False, True])
@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
def test_next_turn_resumes_at_the_previous_frontier(frontier: bool, dcp_world_size: int):
    scheduler = make_scheduler(frontier=frontier, dcp_world_size=dcp_world_size)
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    assert frontier_tokens % BLOCK

    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=4)
    # Without frontier checkpoints the turn resumes at its prompt's last block
    # boundary, where align mode cached the replay state.
    expected = frontier_tokens if frontier else 128
    assert admitted_at(scheduler, second) == expected


def test_frontier_hits_are_counted_in_prefix_cache_stats():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens

    tokens = list(first.all_token_ids) + prompt(2, 40)
    add_request(scheduler, "t1", tokens, max_tokens=4)
    scheduler.schedule()
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.hits) == (1, frontier_tokens)
    assert stats.frontier_hits == frontier_tokens - 2 * BLOCK


def test_resume_inside_an_indexer_group_takes_one_token_per_step():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 149), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    assert frontier_tokens % RING

    tokens = list(first.all_token_ids) + prompt(2, 300)
    second = add_request(scheduler, "t1", tokens, max_tokens=2)
    taken = [out.num_scheduled_tokens["t1"] for out in run(scheduler, second)]
    catchup = (-frontier_tokens) % RING
    assert taken[:catchup] == [1] * catchup
    assert taken[catchup] > 1


def test_decode_boundary_state_is_resumable():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 100), max_tokens=60)
    run(scheduler, first)
    assert first.num_computed_tokens > 2 * BLOCK

    # Shares the first request's tokens past its decode boundary at 128, but
    # not its frontier.
    tokens = list(first.all_token_ids[: 2 * BLOCK + 10]) + prompt(3, 50)
    sibling = add_request(scheduler, "s", tokens, max_tokens=2)
    assert admitted_at(scheduler, sibling) == 2 * BLOCK


def test_frontier_goes_whole_when_one_of_its_blocks_is_evicted():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    [entry] = checkpoints.frontiers.entries.values()
    page, *others = entry.blocks.values()

    scheduler.kv_cache_manager.block_pool.evict_blocks({page.block_id})

    assert not checkpoints.frontiers.entries
    assert all(block.block_hash is None for block in others)
    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=2)
    assert admitted_at(scheduler, second) == 128


def test_frontier_page_keeps_the_key_of_a_finer_hash_unit():
    """With a prefix-match unit finer than the block, the page holding the
    frontier is already cached under the request's own key at its prompt's
    last unit boundary: the frontier key joins that key, and releasing the
    frontier removes only its own key."""
    unit = 16
    scheduler = make_scheduler(prefix_match_unit=unit, hash_block_size=unit)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    pool = scheduler.kv_cache_manager.block_pool
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    assert frontier_tokens // BLOCK == 150 // BLOCK

    [entry] = checkpoints.frontiers.entries.values()
    page = entry.blocks[0]
    assert page.block_hash is not None
    assert get_block_hash(page.block_hash) == first.block_hashes[150 // unit - 1]
    frontier_page_key = make_block_hash_with_group_id(entry.key, 0)
    assert frontier_page_key in pool.cached_block_hashes_by_block[page.block_id]

    second = add_request(scheduler, "t1", list(first.all_token_ids) + prompt(2, 40), max_tokens=4)
    assert admitted_at(scheduler, second) == frontier_tokens
    run(scheduler, second)

    # The session moved on: its leaf replacement released the first frontier.
    assert entry.key not in checkpoints.frontiers.entries
    assert page.block_hash is not None
    assert get_block_hash(page.block_hash) == first.block_hashes[150 // unit - 1]
    assert page.block_id not in pool.cached_block_hashes_by_block
    # The page still serves the hit at the prompt's last unit boundary.
    tokens = list(first.all_token_ids[:150]) + prompt(3, 30)
    probe = add_request(scheduler, "p", tokens, max_tokens=1)
    assert admitted_at(scheduler, probe) == 150 // unit * unit


def test_aborted_request_publishes_no_frontier():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=50)
    while first.num_computed_tokens <= 150:
        step(scheduler)
    scheduler.finish_requests("t0", RequestStatus.FINISHED_ABORTED)
    assert scheduler.mamba_checkpoints is not None
    assert not scheduler.mamba_checkpoints.frontiers.entries


def test_session_leaf_is_replaced_unless_another_turn_needs_it():
    scheduler = make_scheduler()
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    entries = checkpoints.frontiers.entries

    def frontier_of(request: Request) -> bytes:
        return frontier_key(sha256, request, request.num_computed_tokens, BLOCK)

    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    tokens = list(first.all_token_ids)
    branch = add_request(scheduler, "b", tokens + prompt(5, 20), max_tokens=3)
    second = add_request(scheduler, "t1", tokens + prompt(2, 20), max_tokens=3)
    run(scheduler, branch, second)
    # Two continuations resumed at it: a branch point stays.
    assert frontier_of(first) in entries

    tokens = list(second.all_token_ids) + prompt(6, 20)
    third = add_request(scheduler, "t2", tokens, max_tokens=3)
    run(scheduler, third)
    # The session moved on from the second turn's frontier.
    assert frontier_of(second) not in entries
    assert frontier_of(third) in entries


def test_longer_connector_hit_supersedes_the_frontier():
    """The KV connector sees the block-aligned part of a frontier hit and loads
    past the frontier: the request resumes where the load ends, without the
    frontier's partial page, state or ring, and at the state the load caches
    there rather than at its last local block."""
    scheduler = make_scheduler(kv_transfer_config=mock_connector(2 * BLOCK))
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    first = add_request(scheduler, "t0", prompt(1, 149), max_tokens=10)
    assert admitted_at(scheduler, first) == 2 * BLOCK
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    # Inside a block and inside an indexer group: the frontier keeps a ring.
    assert frontier_tokens // BLOCK == 2 and frontier_tokens % RING

    tokens = list(first.all_token_ids) + prompt(2, 300)
    second = add_request(scheduler, "t1", tokens, max_tokens=3)
    assert admitted_at(scheduler, second) == 4 * BLOCK
    assert checkpoints.frontiers.num_hits == 0
    units = checkpoints.graph.units
    # Each load resumed its request at the state it ends with.
    assert units[second.block_hashes[1]].hits == 1
    assert units[second.block_hashes[3]].hits == 1


def test_a_load_resumes_at_the_state_of_its_hash_unit():
    """With a prefix-match unit finer than the block, the state a load ends
    with is cached under the request's hash at that unit (here the sixteenth
    and eighth), not at the block count's unit."""
    unit = 16
    scheduler = make_scheduler(
        kv_transfer_config=mock_connector(2 * BLOCK),
        prefix_match_unit=unit,
        hash_block_size=unit,
    )
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    first = add_request(scheduler, "t0", prompt(1, 149), max_tokens=10)
    assert admitted_at(scheduler, first) == 2 * BLOCK
    run(scheduler, first)
    tokens = list(first.all_token_ids) + prompt(2, 300)
    second = add_request(scheduler, "t1", tokens, max_tokens=3)
    assert admitted_at(scheduler, second) == 4 * BLOCK
    units = checkpoints.graph.units
    assert units[second.block_hashes[2 * BLOCK // unit - 1]].hits == 1
    assert units[second.block_hashes[4 * BLOCK // unit - 1]].hits == 1


@pytest.mark.parametrize(
    "kwargs,needs",
    [
        ({"prefix_cache_retention_interval": BLOCK}, "retention_interval 0"),
        ({"resume_catchup": None}, "resume_catchup"),
        (
            {"kv_transfer_config": mock_connector(0, kv_load_failure_policy="fail")},
            "kv_load_failure_policy 'recompute'",
        ),
        (
            {"kv_transfer_config": mock_connector(0, kv_role="kv_producer")},
            "kv_role 'kv_both'",
        ),
    ],
)
def test_unsupported_configurations_are_refused(kwargs, needs):
    with pytest.raises(ValueError, match=needs):
        make_scheduler(**kwargs)


def test_frontier_lookup_respects_request_and_explicit_bounds():
    scheduler = make_scheduler()
    first = add_request(scheduler, "finished", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier = first.num_computed_tokens
    continuation = add_request(scheduler, "bounded", list(first.all_token_ids) + prompt(2, 40), max_tokens=2)
    manager = scheduler.kv_cache_manager
    assert manager.get_computed_blocks(continuation)[1] == frontier
    assert manager.get_computed_blocks(continuation, frontier - 1)[1] < frontier

    continuation.prefix_cache_hit_limit = frontier - 1
    assert manager.get_computed_blocks(continuation, frontier + BLOCK)[1] < frontier
    assert manager.get_num_cached_tokens(continuation) < frontier
    continuation.prefix_cache_hit_limit = 0
    assert manager.get_computed_blocks(continuation)[1] == 0
    assert manager.get_num_cached_tokens(continuation) == 0


def test_read_only_frontier_probe_does_not_change_lifetime_or_admission():
    scheduler = make_scheduler()
    first = add_request(scheduler, "finished", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    continuation = add_request(scheduler, "probe", list(first.all_token_ids) + prompt(2, 40), max_tokens=2)
    manager = scheduler.kv_cache_manager
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    references = [block.ref_cnt for block in manager.block_pool.blocks]
    assert manager.get_num_cached_tokens(continuation) == first.num_computed_tokens
    assert continuation.request_id not in checkpoints.frontiers._lookups
    assert [block.ref_cnt for block in manager.block_pool.blocks] == references


def test_heterogeneous_blocks_remain_supported_without_frontiers():
    scheduler = make_scheduler(frontier=False, attention_block_size=32, hash_block_size=32)
    request = add_request(scheduler, "heterogeneous", prompt(1, 150), max_tokens=3)
    run(scheduler, request)
    assert request.is_finished()
    assert scheduler.kv_cache_manager.coordinator.single_type_managers[0].block_size == 32


def test_frontiers_reject_heterogeneous_physical_block_indices():
    with pytest.raises(ValueError, match="scheduler block size"):
        make_scheduler(attention_block_size=32, hash_block_size=32)


def test_frontiers_reject_hashes_that_cover_future_speculative_tokens():
    scheduler = make_scheduler(frontier=False)
    scheduler.sonar_config.num_hash_lookahead_tokens = 2
    with pytest.raises(ValueError, match="without speculative lookahead"):
        check_frontier_checkpoint_config(
            scheduler.sonar_config,
            block_size=BLOCK,
            use_v2_model_runner=True,
            enable_kv_cache_events=False,
        )


@pytest.mark.parametrize("dcp_world_size", [1, 2, 4])
@pytest.mark.parametrize("frontier_tokens", [63, 127, 191, 255, 319])
@pytest.mark.parametrize("hash_unit", [16, BLOCK])
def test_dcp_frontier_uses_global_positions_and_group_local_tables(
    dcp_world_size: int, frontier_tokens: int, hash_unit: int
):
    scheduler = make_scheduler(dcp_world_size=dcp_world_size, prefix_match_unit=hash_unit, hash_block_size=hash_unit)
    first = add_request(scheduler, "first", prompt(1, frontier_tokens - 9), max_tokens=10)
    run(scheduler, first)
    assert first.num_computed_tokens == frontier_tokens
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    assert checkpoints.block_size == BLOCK
    page_span = BLOCK * dcp_world_size
    assert scheduler.block_size == page_span
    [entry] = checkpoints.frontiers.entries.values()
    assert entry.num_tokens == frontier_tokens
    assert entry.key == frontier_key(sha256, first, frontier_tokens, hash_unit)

    second = add_request(scheduler, "second", list(first.all_token_ids) + prompt(2, 80), max_tokens=3)
    manager = scheduler.kv_cache_manager
    blocks, hit, _ = manager.get_computed_blocks(second)
    assert hit == frontier_tokens
    assert len(blocks.blocks[0]) == frontier_tokens // page_span + 1
    for i, page in enumerate(blocks.blocks[0][:-1]):
        grouped = make_block_hash_with_group_id(first.block_hashes[(i + 1) * page_span // hash_unit - 1], 0)
        assert manager.block_pool.cached_block_hash_to_block.contain(grouped, page.block_id)
    for group_id in checkpoints.frontiers.states:
        assert len(blocks.blocks[group_id]) == frontier_tokens // BLOCK + 1
        assert all(block.is_null for block in blocks.blocks[group_id][:-1])
        assert blocks.blocks[group_id][-1] is entry.blocks[group_id]
    assert blocks.blocks[0][-1] is entry.blocks[0]
    assert admitted_at(scheduler, second) == frontier_tokens
    # The resumed request owns a private tail on every rank. The scheduler
    # dispatches the same physical-page copy IDs to all workers.
    assert (
        manager.coordinator.single_type_managers[0].req_to_blocks["second"][frontier_tokens // page_span]
        is not entry.blocks[0]
    )
    run(scheduler, second)


@pytest.mark.parametrize("dcp_world_size", [2, 4])
def test_dcp_recurrent_boundary_inside_attention_page_is_resumable(dcp_world_size: int):
    scheduler = make_scheduler(dcp_world_size=dcp_world_size)
    first = add_request(scheduler, "first", prompt(1, 150), max_tokens=43)
    run(scheduler, first)
    assert first.num_computed_tokens == 3 * BLOCK
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    assert not checkpoints.frontiers.entries
    key = first.block_hashes[2]
    assert checkpoints.graph.units[key].num_tokens == 3 * BLOCK
    assert not checkpoints.graph.units[key].frontier
    page_key = make_block_hash_with_group_id(key, 0)
    assert scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block.get_one_block(page_key) is not None
    second = add_request(scheduler, "second", list(first.all_token_ids) + prompt(2, 40), max_tokens=2)
    assert admitted_at(scheduler, second) == 3 * BLOCK
    assert checkpoints.graph.units[key].hits == 1
    assert checkpoints.graph.resumed[second.request_id] == key


@pytest.mark.parametrize("eviction", ["lru", "gds"])
def test_dcp_frontier_eviction_and_reset(eviction: str):
    kwargs = {} if eviction == "lru" else {"prefill_cost_coefficients": (1.0, 0.0)}
    scheduler = make_scheduler(dcp_world_size=2, prefix_cache_eviction=eviction, **kwargs)
    first = add_request(scheduler, "first", prompt(1, 280), max_tokens=10)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    [entry] = checkpoints.frontiers.entries.values()
    pool = scheduler.kv_cache_manager.block_pool
    if checkpoints.eviction is not None:
        assert checkpoints.eviction.paths[entry.key] == (first.block_hashes[1], first.block_hashes[3])
    pool.evict_blocks({entry.blocks[0].block_id})
    assert not checkpoints.frontiers.entries
    second = add_request(scheduler, "second", list(first.all_token_ids) + prompt(2, 40), max_tokens=2)
    assert scheduler.kv_cache_manager.get_num_cached_tokens(second) < first.num_computed_tokens
    run(scheduler, second)
    assert scheduler.reset_prefix_cache()
    assert not checkpoints.frontiers.entries and not checkpoints.graph.units
    third = add_request(scheduler, "third", list(second.all_token_ids) + prompt(3, 40), max_tokens=1)
    assert scheduler.kv_cache_manager.get_num_cached_tokens(third) == 0


def test_dcp_eviction_tracks_partial_attention_dependency():
    scheduler = make_scheduler(dcp_world_size=4, prefix_cache_eviction="gds", prefill_cost_coefficients=(1.0, 0.0))
    first = add_request(scheduler, "first", prompt(1, 150), max_tokens=43)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    key = first.block_hashes[2]
    assert checkpoints.eviction.paths[key] == (key,)
    pool = scheduler.kv_cache_manager.block_pool
    page = pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(key, 0))
    assert page is not None
    pool.evict_blocks({page.block_id})
    assert checkpoints.graph.units[key].dead


def test_dcp_evicting_boundary_keeps_page_shared_with_frontier():
    scheduler = make_scheduler(dcp_world_size=4, prefix_cache_eviction="gds", prefill_cost_coefficients=(1.0, 0.0))
    first = add_request(scheduler, "first", prompt(1, 150), max_tokens=60)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    [entry] = checkpoints.frontiers.entries.values()
    boundary = checkpoints.graph.units[first.block_hashes[2]]
    page = entry.blocks[0]
    assert page not in checkpoints.eviction._exclusive(boundary)
    # The ordinary aligned checkpoint and decode scratch can both cache the
    # state. Count every retained copy, but not the shared attention page.
    assert checkpoints.eviction._size(boundary) == len(checkpoints.eviction._state_blocks(boundary))
    evicted = checkpoints.eviction._evict(boundary)
    assert page not in evicted
    assert entry.key in checkpoints.frontiers.entries
    second = add_request(scheduler, "second", list(first.all_token_ids) + prompt(2, 40), max_tokens=2)
    assert admitted_at(scheduler, second) == first.num_computed_tokens


@pytest.mark.parametrize("finish", [128, 256])
def test_dcp_attention_page_keeps_earlier_boundary_aliases(finish: int):
    scheduler = make_scheduler(
        dcp_world_size=4,
        prefix_cache_eviction="gds",
        prefill_cost_coefficients=(1.0, 0.0),
        prefix_cache_leaf_last=True,
    )
    first = add_request(scheduler, "first", prompt(1, 100), max_tokens=finish - 99)
    run(scheduler, first)
    assert first.num_computed_tokens == finish
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    pool = scheduler.kv_cache_manager.block_pool
    earlier = first.block_hashes[0]
    latest = first.block_hashes[finish // BLOCK - 1]
    earlier_page = pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(earlier, 0))
    latest_page = pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(latest, 0))
    assert earlier_page is not None and earlier_page is latest_page
    if finish == 256:
        assert latest_page.block_hash == make_block_hash_with_group_id(latest, 0)
        assert latest_page.block_hash_num_tokens == finish
    assert not checkpoints.graph.units[earlier].dead
    assert checkpoints.eviction._tier(checkpoints.graph.units[earlier]) == 1
    branch = add_request(scheduler, "branch", list(first.all_token_ids[:BLOCK]) + prompt(2, 40), max_tokens=2)
    assert admitted_at(scheduler, branch) == BLOCK


def test_dcp_mid_decode_boundaries_have_attention_aliases():
    scheduler = make_scheduler(dcp_world_size=4, prefix_cache_eviction="gds", prefill_cost_coefficients=(1.0, 0.0))
    first = add_request(scheduler, "first", prompt(1, 100), max_tokens=200)
    while first.num_computed_tokens < 196:
        step(scheduler)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    pool = scheduler.kv_cache_manager.block_pool
    for boundary in (128, 192):
        key = first.block_hashes[boundary // BLOCK - 1]
        assert pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(key, 0)) is not None
        assert not checkpoints.graph.units[key].dead
    branch = add_request(scheduler, "branch", list(first.all_token_ids[:128]) + prompt(2, 40), max_tokens=2)
    assert admitted_at(scheduler, branch) == 128


def test_dcp_partial_boundary_keys_do_not_make_a_false_fork():
    scheduler = make_scheduler(
        dcp_world_size=4,
        prefix_cache_eviction="gds",
        prefill_cost_coefficients=(1.0, 0.0),
        prefix_cache_leaf_last=True,
    )
    first = add_request(scheduler, "first", prompt(1, 280), max_tokens=105)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    unit = checkpoints.graph.units[first.block_hashes[3]]
    assert checkpoints.eviction._deeper(unit)
    assert not checkpoints.eviction._forks(unit)
    assert checkpoints.eviction._tier(unit) == 1
    partial = checkpoints.graph.units[first.block_hashes[4]]
    assert checkpoints.eviction._deeper(partial)
    assert checkpoints.eviction._tier(partial) == 1


def test_dcp_real_fork_inside_attention_page_stays_a_leaf():
    scheduler = make_scheduler(
        dcp_world_size=4,
        prefix_cache_eviction="gds",
        prefill_cost_coefficients=(1.0, 0.0),
        prefix_cache_leaf_last=True,
    )
    first = add_request(scheduler, "first", prompt(1, 100), max_tokens=29)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    parent = checkpoints.graph.units[first.block_hashes[1]]
    for seed in (2, 3):
        branch = add_request(
            scheduler, f"branch-{seed}", list(first.all_token_ids[:128]) + prompt(seed, 40), max_tokens=25
        )
        assert admitted_at(scheduler, branch) == 128
        run(scheduler, branch)
    assert parent.is_branch_point
    assert checkpoints.eviction._tier(parent) == 2
