# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash and DSpark draft masking for cache-restored prefixes."""

import pytest
import torch

from sonar.platforms import current_platform
from sonar.v1.worker.gpu.spec_decode.dflash.speculator import (
    shift_draft_block_tables,
)

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

DEVICE = "cuda"
BLOCK_SIZE = 16
MAX_BLOCKS = 64
MAX_NUM_REQS = 8


def _make_block_table(num_reqs: int) -> torch.Tensor:
    return (
        torch.arange(MAX_NUM_REQS * MAX_BLOCKS, dtype=torch.int32, device=DEVICE)
        .view(MAX_NUM_REQS, MAX_BLOCKS)[:num_reqs]
        .contiguous()
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("cached_virtual_blocks", [0, 1, 3])
def test_shift_single_request(cp_size: int, cached_virtual_blocks: int):
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (MAX_NUM_REQS,),
        cached_virtual_blocks * BLOCK_SIZE * cp_size,
        dtype=torch.int32,
        device=DEVICE,
    )
    seq_lens = torch.full(
        (1,),
        (MAX_BLOCKS - cached_virtual_blocks) * BLOCK_SIZE * cp_size,
        dtype=torch.int32,
        device=DEVICE,
    )

    shift_draft_block_tables(
        block_table,
        idx_mapping,
        num_cached_tokens,
        seq_lens,
        BLOCK_SIZE,
        cp_size,
    )

    kept = MAX_BLOCKS - cached_virtual_blocks
    torch.testing.assert_close(
        block_table[0, :kept],
        original[0, cached_virtual_blocks:],
    )


def test_shift_ignores_partial_dcp_virtual_block():
    cp_size = 4
    block_table = _make_block_table(1)
    original = block_table.clone()
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    num_cached_tokens = torch.full(
        (MAX_NUM_REQS,),
        BLOCK_SIZE * cp_size - 1,
        dtype=torch.int32,
        device=DEVICE,
    )
    seq_lens = torch.full((1,), MAX_BLOCKS * BLOCK_SIZE * cp_size, dtype=torch.int32, device=DEVICE)

    shift_draft_block_tables(
        block_table,
        idx_mapping,
        num_cached_tokens,
        seq_lens,
        BLOCK_SIZE,
        cp_size,
    )

    torch.testing.assert_close(block_table, original)
