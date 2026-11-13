# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-chunk Mamba state export (`mamba_prefill_export`) on CPU."""

import pytest

from sonar.utils.hashing import sha256
from sonar.v1.core.hybrid_checkpoint.export import plan_export, plan_filler
from sonar.v1.core.kv_cache_utils import init_none_hash

from .utils import add_request, make_hybrid_scheduler, prompt, run, step

pytestmark = pytest.mark.cpu_test

MAMBA_GROUPS = [1, 2]


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


@pytest.mark.parametrize(
    "start,end,stops,expected",
    [
        # One position inside: exported, and the chunk runs on.
        (192, 384, (256,), (384, 256)),
        # A second position ends the chunk.
        (192, 384, (256, 320), (320, 256)),
        # Off the checkpoint grid, or a chunk starting off the hash grid: stop.
        (192, 384, (264,), (264, None)),
        (208, 384, (256,), (256, None)),
        # Nothing strictly inside.
        (192, 256, (256,), (256, None)),
    ],
)
def test_plan_export(start, end, stops, expected):
    assert plan_export(start, end, stops, alignment=16, hash_block_size=64, block_size=64) == expected


@pytest.mark.parametrize(
    "num_new_tokens,prefill_end,stops,expected",
    [
        (1630, 6000, (), 1600),
        (1000, 6000, (), 0),
        # It would be the final chunk.
        (1630, 1500, (), 0),
        # It ends on 64 tokens before a required position, if that leaves 1024.
        (1630, 6000, (1100,), 1088),
        (1630, 6000, (1000,), 0),
    ],
)
def test_plan_filler(num_new_tokens, prefill_end, stops, expected):
    assert plan_filler(0, num_new_tokens, prefill_end=prefill_end, stops=stops) == expected


@pytest.mark.parametrize("unit", [None, 16])
@pytest.mark.parametrize("export", [False, True])
def test_chunk_exports_the_state_at_a_shared_prefix_junction(export: bool, unit: int | None):
    # 192-token steps: a lone request is exempt from the long-prefill cap.
    # A finer prefix-match unit leaves the junction on the block grid.
    scheduler = make_hybrid_scheduler(
        max_num_batched_tokens=192,
        ring=0,
        mamba_prefill_export=export,
        prefix_match_unit=unit,
        hash_block_size=unit,
    )
    shared = prompt(1, 640)
    run(scheduler, add_request(scheduler, "a", shared, max_tokens=1))

    # The first request leaves its Mamba state at 192 and attention pages up
    # to 640: the sibling resumes at 192 and its junction is at 256.
    sibling = add_request(scheduler, "b", shared[:300] + prompt(2, 400), max_tokens=1)
    outputs = run(scheduler, sibling)
    first = next(o for o in outputs if "b" in o.num_scheduled_tokens)
    [admitted] = [n for n in first.scheduled_new_reqs if n.req_id == "b"]
    assert admitted.num_computed_tokens == 192
    if export:
        assert first.num_scheduled_tokens["b"] == 192
        assert first.prefill_checkpoint_positions == {"b": 256}
    else:
        assert first.num_scheduled_tokens["b"] == 64
        assert all(o.prefill_checkpoint_positions is None for o in outputs)
    junction = scheduler.kv_cache_manager.block_pool.get_cached_block(
        sibling.block_hashes[256 // (unit or 64) - 1], MAMBA_GROUPS
    )
    assert junction is not None
    assert all(block.block_hash_num_tokens == 256 for block in junction)


@pytest.mark.parametrize("export", [False, True])
def test_leftover_smaller_than_a_block_gets_a_fill_chunk(export: bool):
    scheduler = make_hybrid_scheduler(
        block_size=2048,
        max_num_batched_tokens=4096,
        num_blocks=64,
        ring=0,
        mamba_prefill_export=export,
    )
    add_request(scheduler, "a", prompt(1, 2500), max_tokens=1)
    add_request(scheduler, "b", prompt(2, 6000), max_tokens=1)

    scheduled = step(scheduler).num_scheduled_tokens

    assert scheduled["a"] == 2500
    # 1,596 tokens are left: no whole block, so a fill chunk on 64 tokens.
    assert scheduled.get("b", 0) == (1536 if export else 0)


@pytest.mark.parametrize(
    "kwargs,needs",
    [
        ({"checkpoint_alignment": None}, "prefill checkpoints"),
    ],
)
def test_unsupported_configurations_are_refused(kwargs, needs):
    with pytest.raises(ValueError, match=needs):
        make_hybrid_scheduler(ring=0, mamba_prefill_export=True, **kwargs)
