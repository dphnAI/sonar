# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP-aware slot mapping for DFlash and DSpark draft inputs."""

from types import SimpleNamespace

import pytest
import torch

from aphrodite.platforms import current_platform
from aphrodite.v1.attention.backends.utils import PAD_SLOT_ID
from aphrodite.v1.worker.gpu.input_batch import InputBuffers
from aphrodite.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
)

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA required for DFlash kernel tests", allow_module_level=True)

DEVICE = "cuda"
BLOCK_SIZE = 16
NUM_SPECULATIVE_STEPS = 3
NUM_QUERY_PER_REQ = 1 + NUM_SPECULATIVE_STEPS
MAX_NUM_REQS = 8
MAX_NUM_TOKENS = 512
MAX_NUM_BLOCKS = 64
MAX_MODEL_LEN = 4096


def _ref_slots(
    positions: torch.Tensor,
    block_row: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    cp_interleave: int,
) -> torch.Tensor:
    virtual_block = BLOCK_SIZE * cp_size
    block_indices = (positions // virtual_block).clamp(max=block_row.shape[0] - 1)
    block_offsets = positions % virtual_block
    block_ids = block_row[block_indices].long()
    if cp_size == 1:
        slots = block_ids * BLOCK_SIZE + block_offsets
        return torch.where(block_ids != 0, slots, torch.full_like(slots, PAD_SLOT_ID))
    is_local = (block_offsets // cp_interleave) % cp_size == cp_rank
    local_offsets = block_offsets // (cp_interleave * cp_size) * cp_interleave + block_offsets % cp_interleave
    slots = block_ids * BLOCK_SIZE + local_offsets
    return torch.where(is_local & (block_ids != 0), slots, torch.full_like(slots, PAD_SLOT_ID))


def _run_prepare(
    cp_size: int,
    cp_rank: int,
    cp_interleave: int,
    *,
    use_global_cache_positions: bool = False,
    cached_tokens: list[int] | None = None,
    groups: list[tuple[torch.Tensor, int]] | None = None,
):
    context_positions = [
        torch.arange(100, 120, device=DEVICE),
        torch.arange(0, 1, device=DEVICE),
        torch.arange(37, 100, device=DEVICE),
    ]
    num_rejected = torch.tensor([2, 0, 0], dtype=torch.int32, device=DEVICE)
    num_sampled = torch.tensor([1, 1, 0], dtype=torch.int32, device=DEVICE)
    num_reqs = len(context_positions)

    positions = torch.cat(context_positions)
    num_context = torch.tensor([len(p) for p in context_positions], device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = num_context.cumsum(0)
    num_tokens = int(query_start_loc[-1])

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=num_context.cpu().numpy(),
        positions=positions,
        query_start_loc=query_start_loc,
        idx_mapping=torch.arange(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE),
    )
    input_buffers = InputBuffers(MAX_NUM_REQS, MAX_NUM_TOKENS, torch.device(DEVICE))
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    block_table = (
        torch.randperm(
            MAX_NUM_REQS * MAX_NUM_BLOCKS,
            generator=generator,
            device=DEVICE,
        )
        .view(MAX_NUM_REQS, MAX_NUM_BLOCKS)
        .to(torch.int32)
    )

    if groups is None:
        groups = [(block_table, BLOCK_SIZE)]
    query_slot_mappings = [torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE) for _ in groups]
    output_context_positions = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    context_slot_mappings = [torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE) for _ in groups]
    max_num_sampled = MAX_NUM_REQS * NUM_SPECULATIVE_STEPS
    sample_indices = torch.zeros(max_num_sampled, dtype=torch.int64, device=DEVICE)
    sample_pos = torch.zeros(max_num_sampled, dtype=torch.int64, device=DEVICE)
    sample_idx_mapping = torch.zeros(max_num_sampled, dtype=torch.int32, device=DEVICE)
    temperature = torch.zeros(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)
    seeds = torch.zeros(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
    last_sampled = torch.full((MAX_NUM_REQS,), 7, dtype=torch.int64, device=DEVICE)
    next_prefill_tokens = torch.full((MAX_NUM_REQS,), 11, dtype=torch.int64, device=DEVICE)
    num_cached_tokens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    if cached_tokens is not None:
        num_cached_tokens[:num_reqs] = torch.tensor(cached_tokens, dtype=torch.int32, device=DEVICE)

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mappings,
        output_context_positions,
        context_slot_mappings,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        [table for table, _ in groups],
        [block_size for _, block_size in groups],
        num_cached_tokens,
        parallel_drafting_token_id=1,
        num_query_per_req=NUM_QUERY_PER_REQ,
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
        use_global_cache_positions=use_global_cache_positions,
    )
    return SimpleNamespace(
        context_positions=context_positions,
        num_rejected=num_rejected,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        block_table=block_table,
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mappings[-1],
        output_context_positions=output_context_positions,
        context_slot_mapping=context_slot_mappings[-1],
        query_slot_mappings=query_slot_mappings,
        context_slot_mappings=context_slot_mappings,
        sample_indices=sample_indices,
        sample_pos=sample_pos,
        sample_idx_mapping=sample_idx_mapping,
        temperature=temperature,
        seeds=seeds,
    )


@pytest.mark.parametrize("cp_size,cp_interleave", [(1, 1), (2, 1), (4, 1), (2, 16), (4, 8)])
def test_dflash_slot_mapping_matches_dcp_layout(cp_size: int, cp_interleave: int):
    for cp_rank in range(cp_size):
        result = _run_prepare(cp_size, cp_rank, cp_interleave)
        for req in range(result.num_reqs):
            start = int(result.query_start_loc[req])
            end = int(result.query_start_loc[req + 1])
            context_positions = result.context_positions[req]
            expected_context_slots = _ref_slots(
                context_positions,
                result.block_table[req],
                cp_size,
                cp_rank,
                cp_interleave,
            )
            num_valid = len(context_positions) - int(result.num_rejected[req])
            expected_context_slots[num_valid:] = PAD_SLOT_ID
            expected_context_positions = context_positions.clone()
            expected_context_positions[num_valid:] = 0
            torch.testing.assert_close(
                result.context_slot_mapping[start:end],
                expected_context_slots,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                result.output_context_positions[start:end],
                expected_context_positions,
                rtol=0,
                atol=0,
            )

            query_start = req * NUM_QUERY_PER_REQ
            query_slots = result.query_slot_mapping[query_start : query_start + NUM_QUERY_PER_REQ]
            if cp_size == 1:
                last_valid_pos = int(context_positions[-1 - int(result.num_rejected[req])])
                query_positions = torch.arange(
                    last_valid_pos + 1,
                    last_valid_pos + 1 + NUM_QUERY_PER_REQ,
                    device=DEVICE,
                )
                expected_query_slots = _ref_slots(
                    query_positions,
                    result.block_table[req],
                    cp_size,
                    cp_rank,
                    cp_interleave,
                )
                torch.testing.assert_close(query_slots, expected_query_slots, rtol=0, atol=0)
            else:
                assert (query_slots == PAD_SLOT_ID).all()


@pytest.mark.parametrize("cp_size,cp_interleave", [(2, 1), (4, 1), (4, 8)])
def test_dflash_context_slots_partition_across_dcp(cp_size: int, cp_interleave: int):
    per_rank = [_run_prepare(cp_size, rank, cp_interleave) for rank in range(cp_size)]
    first = per_rank[0]
    for req, positions in enumerate(first.context_positions):
        start = int(first.query_start_loc[req])
        num_valid = len(positions) - int(first.num_rejected[req])
        for offset, position in enumerate(positions):
            token_idx = start + offset
            block_idx = int(position) // (BLOCK_SIZE * cp_size)
            resident = int(first.block_table[req, block_idx]) != 0
            owners = sum(int(result.context_slot_mapping[token_idx] != PAD_SLOT_ID) for result in per_rank)
            assert owners == int(offset < num_valid and resident)


@pytest.mark.parametrize("use_global_cache_positions", [False, True])
def test_restored_draft_prefix_keeps_cache_origin_consistent(use_global_cache_positions: bool):
    from aphrodite.v1.worker.gpu.spec_decode.dflash.speculator import shift_draft_block_tables

    cached_tokens = [64, 0, 32]
    result = _run_prepare(1, 0, 1, use_global_cache_positions=use_global_cache_positions, cached_tokens=cached_tokens)
    if not use_global_cache_positions:
        shift_draft_block_tables(
            result.block_table,
            torch.arange(result.num_reqs, dtype=torch.int32, device=DEVICE),
            torch.tensor(cached_tokens, dtype=torch.int32, device=DEVICE),
            result.input_buffers.seq_lens,
            BLOCK_SIZE,
            1,
        )
    for req, positions in enumerate(result.context_positions):
        last_valid = int(positions[-1 - int(result.num_rejected[req])])
        omitted = 0 if use_global_cache_positions else cached_tokens[req] // BLOCK_SIZE * BLOCK_SIZE
        seq_len = last_valid + 1 + NUM_QUERY_PER_REQ - omitted
        assert int(result.input_buffers.seq_lens[req]) == seq_len
        last_position = seq_len - 1
        block_id = int(result.block_table[req, last_position // BLOCK_SIZE])
        expected_slot = block_id * BLOCK_SIZE + last_position % BLOCK_SIZE if block_id else PAD_SLOT_ID
        assert int(result.query_slot_mapping[(req + 1) * NUM_QUERY_PER_REQ - 1]) == expected_slot


@pytest.mark.parametrize("use_global_cache_positions", [False, True])
def test_one_launch_prepares_every_draft_group_like_its_own_launch(use_global_cache_positions: bool):
    # Draft KV-cache groups differ in block table and block size; one launch
    # must give each group the slots it gets alone, and the shared outputs of
    # the last group (seq_lens depends on its block size via cached tokens).
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    groups = []
    for block_size in (16, 32, 8):
        blocks = MAX_MODEL_LEN // block_size
        table = torch.randperm(MAX_NUM_REQS * blocks, generator=generator, device=DEVICE).view(MAX_NUM_REQS, blocks)
        groups.append((table.to(torch.int32), block_size))
    cached = [64, 0, 40]
    together = _run_prepare(
        1, 0, 1, use_global_cache_positions=use_global_cache_positions, cached_tokens=cached, groups=groups
    )
    alone = [
        _run_prepare(
            1, 0, 1, use_global_cache_positions=use_global_cache_positions, cached_tokens=cached, groups=[group]
        )
        for group in groups
    ]
    for g, single in enumerate(alone):
        assert torch.equal(together.query_slot_mappings[g], single.query_slot_mappings[0])
        assert torch.equal(together.context_slot_mappings[g], single.context_slot_mappings[0])
    last = alone[-1]
    buffers = ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens")
    for name in buffers:
        assert torch.equal(getattr(together.input_buffers, name), getattr(last.input_buffers, name)), name
    for name in (
        "output_context_positions",
        "sample_indices",
        "sample_pos",
        "sample_idx_mapping",
        "temperature",
        "seeds",
    ):
        assert torch.equal(getattr(together, name), getattr(last, name)), name
