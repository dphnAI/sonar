# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused Q/KV norm must rotate KV exactly like the native KV rotation."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

b12x_rotary = pytest.importorskip("b12x.attention.compressed_sparse_mla.rotary")

from aphrodite.models.deepseek_v41.sm120.qkv_norm import (  # noqa: E402
    KV_WIDTH,
    Q_WIDTH,
    ROPE_DIM,
    ROW_STRIDE,
    fused_qkv_norm_quant,
    prepare_qkv_norm_quant,
)

MAX_POSITION = 8192


def _native_kv_rotation(kv, positions, table):
    query = b12x_rotary.Query(max_rows=64, heads=1, dim=KV_WIDTH, cos_sin_dtype="float32")
    ordinal = kv.device.index
    program = b12x_rotary.compile_rotation(b12x_rotary.TUNING.encode_query(query), ordinal)
    out = torch.empty_like(kv)
    return b12x_rotary._State(query, kv.device, program).run(kv, positions, table, out=out)


@pytest.mark.parametrize("rows, capacity", [(1, 8), (7, 8), (8, 8), (37, 64), (64, 64)])
def test_fused_kv_rotation_matches_native(rows, capacity):
    device = torch.device("cuda", torch.cuda.current_device())
    prepare_qkv_norm_quant(device)
    gen = torch.Generator(device="cuda").manual_seed(rows)
    source = (torch.randn(rows, ROW_STRIDE, generator=gen, device=device) * 4).to(torch.bfloat16)
    q_weight = (torch.rand(Q_WIDTH, generator=gen, device=device) + 0.5).to(torch.bfloat16)
    kv_weight = (torch.rand(KV_WIDTH, generator=gen, device=device) + 0.5).to(torch.bfloat16)
    angles = torch.rand(MAX_POSITION, ROPE_DIM // 2, generator=gen, device=device) * 6.3
    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    identity = torch.cat((torch.ones_like(angles), torch.zeros_like(angles)), dim=-1)
    positions = torch.randint(0, MAX_POSITION, (rows,), generator=gen, device=device, dtype=torch.int64)
    if rows > 1:
        # CUDA-graph padding rows carry negative positions.
        positions[-1] = -1

    _, _, normalized = fused_qkv_norm_quant(source, q_weight, kv_weight, 1e-6, capacity, positions, identity)
    _, _, kv = fused_qkv_norm_quant(source, q_weight, kv_weight, 1e-6, capacity, positions, table)
    expected = _native_kv_rotation(normalized, positions, table)

    assert torch.equal(kv.view(torch.int16), expected.view(torch.int16))
    if rows > 1:
        assert not kv[-1].any()
