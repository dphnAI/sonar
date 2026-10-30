# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused Q/KV norm must rotate KV exactly like the native KV rotation."""

import pytest
import torch

from aphrodite.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_device_capability(120), reason="requires SM120")

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
    query = b12x_rotary.Query(max_rows=kv.shape[0], heads=1, dim=KV_WIDTH, cos_sin_dtype="float32")
    ordinal = kv.device.index
    program = b12x_rotary.compile_rotation(b12x_rotary.TUNING.encode_query(query), ordinal)
    out = torch.empty_like(kv)
    return b12x_rotary._State(query, kv.device, program).run(kv, positions, table, out=out)


@pytest.mark.parametrize(
    "rows, capacity",
    [(1, 8), (7, 8), (8, 8), (37, 64), (64, 64), (65, 128), (128, 128), (129, 256), (257, 512), (4097, 8192)],
)
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


@pytest.mark.parametrize("rows, capacity", [(127, 128), (128, 128), (129, 256), (257, 512)])
def test_q_quantization_across_scale_row_tiles(rows, capacity):
    from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
    from b12x.norm import hyperconnection
    from b12x.norm.hyperconnection._impl import run_grouped_rmsnorm_impl
    from b12x.preparation import PreparedCall, prepare_default

    device = torch.device("cuda", torch.cuda.current_device())
    prepare_qkv_norm_quant(device)
    gen = torch.Generator(device=device).manual_seed(rows)
    source = torch.randn(rows, ROW_STRIDE, generator=gen, device=device).to(torch.bfloat16)
    q_weight = (torch.rand(Q_WIDTH, generator=gen, device=device) + 0.5).to(torch.bfloat16)
    kv_weight = torch.ones(KV_WIDTH, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)
    table = torch.cat((torch.ones(1, ROPE_DIM // 2, device=device), torch.zeros(1, ROPE_DIM // 2, device=device)), -1)
    values, scales, _ = fused_qkv_norm_quant(source, q_weight, kv_weight, 1e-6, capacity, positions, table)

    # Use the actual native RMS reduction and quantizer, not a different torch
    # reduction whose BF16 rounding can cross an FP8 quantization boundary.
    q_input = source[:, :Q_WIDTH].contiguous()
    normalized = torch.empty_like(q_input)
    plan = hyperconnection.plan(
        hyperconnection.Caps(device=device, max_tokens=capacity, hidden_size=Q_WIDTH, streams=1, lowrank=1),
        invocation={"operation": "grouped_rmsnorm", "zero_centered": False, "eps": 1e-6, "weight_dtype": "bfloat16"},
    )

    def prepare(state):
        return PreparedCall(
            run=lambda: run_grouped_rmsnorm_impl(
                q_input, q_weight, eps=1e-6, plan=state, out=normalized, zero_centered=False
            )
        )

    prepare_default(plan.request(name=f"qkv.reference.{rows}", prepare_call=prepare))
    run_grouped_rmsnorm_impl(q_input, q_weight, eps=1e-6, plan=plan.prepared.state, out=normalized, zero_centered=False)
    expected_values = torch.empty_like(values)
    scale_rows = torch.empty(rows, Q_WIDTH // 32, dtype=torch.uint8, device=device)
    physical = torch.empty((1, (rows + 127) // 128, 10, 32, 4, 4), dtype=torch.uint8, device=device)
    expected_scales = physical.permute(3, 4, 1, 5, 2, 0).view(torch.float8_e8m0fnu)
    quantize_mxfp8_rows_cute(
        normalized, expected_values, scale_rows, expected_scales, expected_m=capacity, min_amax=1e-4
    )

    assert torch.equal(values.view(torch.uint8), expected_values.view(torch.uint8))
    row = torch.arange(rows, device=device)[:, None]
    group = torch.arange(Q_WIDTH // 32, device=device)[None, :]
    live = (row % 32, (row // 32) % 4, row // 128, group % 4, group // 4, 0)
    assert torch.equal(scales.view(torch.uint8)[live], expected_scales.view(torch.uint8)[live])
