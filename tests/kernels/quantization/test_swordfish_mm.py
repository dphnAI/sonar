# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swordfish decode GEMM correctness: swordfish_mm vs a dequantized reference.

The oracle is A @ w_ref where w_ref is the fp reconstruction of the quantized
weight (from swordfish_quantize). Tolerances mirror test_machete_mm.
"""

import pytest
import torch

from aphrodite import _custom_ops as ops
from aphrodite.model_executor.layers.quantization.utils.swordfish_utils_test import (
    swordfish_quantize,
    swordfish_quantize_act_order,
    swordfish_quantize_awq,
)
from aphrodite.platforms import current_platform
from aphrodite.scalar_type import scalar_types
from tests.kernels.utils import opcheck

if not current_platform.is_cuda():
    pytest.skip(reason="swordfish requires CUDA", allow_module_level=True)
if not current_platform.has_device_capability(100):
    pytest.skip(reason="swordfish requires sm100 family", allow_module_level=True)

DEVICE = "cuda"
QT = scalar_types.uint4b8

# (M, K, N) decode sweep including regime boundaries, plus K/N variety
MNK = [
    (1, 256, 128),
    (52, 1024, 256),
    (60, 11008, 256),
    (2, 256, 128),
    (4, 512, 256),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (17, 512, 128),
    (32, 4096, 512),
    (33, 256, 256),
]
GROUPS = [-1, 32, 64, 128]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swordfish_mm_correct(mnk, group, dtype):
    m, k, n = mnk
    if group != -1 and k % group != 0:
        pytest.skip("k not divisible by group")

    torch.manual_seed(k * 13 + n + m)
    w = torch.randn((k, n), dtype=dtype, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, QT, group)

    a = torch.randn((m, k), dtype=dtype, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)

    out = ops.swordfish_mm(a, packed, scales, group, k, n)

    assert out.shape == (m, n)
    assert out.dtype == dtype
    # machete-style tolerance
    torch.testing.assert_close(
        out.to(torch.float32), ref, rtol=1e-1, atol=5e-2 if dtype == torch.float16 else 8e-2
    )


def test_swordfish_mm_opcheck():
    # M > 96 with fp16 exercises the deterministic decode epilogue. The
    # atomic window's summation order varies run to run, which opcheck's
    # trace comparison flags, and fp16 never routes to the prefill kernel.
    m, k, n = 128, 512, 128
    torch.manual_seed(1)
    w = torch.randn((k, n), dtype=torch.float16, device=DEVICE) / (k**0.5)
    _, packed, scales = swordfish_quantize(w, QT, 128)
    a = torch.randn((m, k), dtype=torch.float16, device=DEVICE)
    opcheck(torch.ops._C.swordfish_mm, (a, packed, scales, None, None, 4, 128, k, n))


@pytest.mark.parametrize("dtype", DTYPES)
def test_swordfish_mm_determinism(dtype):
    # M > 96 exercises the deterministic paths, fp16 through the
    # smem-reduction decode epilogue and bf16 through the tcgen05 prefill.
    # The window below uses the atomic epilogue, which is not run-stable by
    # design.
    m, k, n = 128, 1024, 256
    torch.manual_seed(9)
    w = torch.randn((k, n), dtype=dtype, device=DEVICE) / (k**0.5)
    _, packed, scales = swordfish_quantize(w, QT, 128)
    a = torch.randn((m, k), dtype=dtype, device=DEVICE)
    o0 = ops.swordfish_mm(a, packed, scales, 128, k, n)
    for _ in range(5):
        assert torch.equal(o0, ops.swordfish_mm(a, packed, scales, 128, k, n))


@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("group", [64, 128])
@pytest.mark.parametrize("dtype", DTYPES)
def test_swordfish_mm_awq_correct(mnk, group, dtype):
    m, k, n = mnk
    if k % group != 0:
        pytest.skip("k not divisible by group")

    torch.manual_seed(k * 7 + n + m)
    w = torch.randn((k, n), dtype=dtype, device=DEVICE) / (k**0.5)
    w_ref, packed, scales, zps_neg = swordfish_quantize_awq(w, group)

    a = torch.randn((m, k), dtype=dtype, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)

    out = ops.swordfish_mm(a, packed, scales, group, k, n, group_zps=zps_neg)

    assert out.shape == (m, n)
    assert out.dtype == dtype
    torch.testing.assert_close(
        out.to(torch.float32), ref, rtol=1e-1, atol=5e-2 if dtype == torch.float16 else 8e-2
    )


def test_swordfish_mm_awq_large_m():
    # M > 96 with bf16 routes through the prefill zero-point row.
    m, k, n = 256, 1024, 256
    torch.manual_seed(3)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales, zps_neg = swordfish_quantize_awq(w, 128)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_mm(a, packed, scales, 128, k, n, group_zps=zps_neg)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


@pytest.mark.parametrize("mnk", MNK)
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swordfish_mm_8bit_correct(mnk, group, dtype):
    m, k, n = mnk
    if group != -1 and k % group != 0:
        pytest.skip("k not divisible by group")

    torch.manual_seed(k * 3 + n + m)
    w = torch.randn((k, n), dtype=dtype, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, scalar_types.uint8b128, group)

    a = torch.randn((m, k), dtype=dtype, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)

    out = ops.swordfish_mm(a, packed, scales, group, k, n, num_bits=8)

    assert out.shape == (m, n)
    torch.testing.assert_close(
        out.to(torch.float32), ref, rtol=1e-1, atol=5e-2 if dtype == torch.float16 else 8e-2
    )


def test_swordfish_mm_8bit_large_m():
    # M > 96 with bf16 routes through the 8-bit prefill mainloop.
    m, k, n = 256, 1024, 256
    torch.manual_seed(5)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, scalar_types.uint8b128, 128)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_mm(a, packed, scales, 128, k, n, num_bits=8)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


@pytest.mark.parametrize("mnk", [(8, 512, 256), (33, 2048, 1024), (256, 2048, 512)])
@pytest.mark.parametrize("bits", [4, 8])
def test_swordfish_mm_act_order(mnk, bits):
    # The row sort realigns group boundaries, so the kernel runs the plain
    # grouped path; only the activation columns carry the permutation.
    m, k, n = mnk
    qt = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
    torch.manual_seed(k + n + m + bits)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales, sort_indices = swordfish_quantize_act_order(w, qt, 128)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_mm(a, packed, scales, 128, k, n, num_bits=bits,
                           perm=sort_indices)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", DTYPES)
def test_swordfish_mm_dense_tier(bits, dtype, monkeypatch):
    # Force the dequant + cuBLAS tier regardless of device size.
    monkeypatch.setenv("APHRODITE_SWORDFISH_DENSE_M", "64")
    m, k, n = 128, 1024, 512
    qt = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
    torch.manual_seed(bits + m)
    w = torch.randn((k, n), dtype=dtype, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, qt, 128)
    a = torch.randn((m, k), dtype=dtype, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_mm(a, packed, scales, 128, k, n, num_bits=bits)
    torch.testing.assert_close(
        out.to(torch.float32), ref, rtol=1e-1, atol=5e-2 if dtype == torch.float16 else 8e-2
    )


def test_swordfish_mm_dense_tier_awq(monkeypatch):
    monkeypatch.setenv("APHRODITE_SWORDFISH_DENSE_M", "64")
    m, k, n = 128, 1024, 512
    torch.manual_seed(2)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales, zps_neg = swordfish_quantize_awq(w, 128)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_mm(a, packed, scales, 128, k, n, group_zps=zps_neg)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


def test_swordfish_mm_dense_tier_act_order(monkeypatch):
    # The dense tier scatters weight rows through the sort instead of
    # permuting the activations.
    monkeypatch.setenv("APHRODITE_SWORDFISH_DENSE_M", "64")
    m, k, n = 128, 1024, 512
    torch.manual_seed(7)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales, sort_indices = swordfish_quantize_act_order(
        w, scalar_types.uint4b8, 128
    )
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_mm(a, packed, scales, 128, k, n, perm=sort_indices)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


def test_swordfish_mm_channelwise_replication():
    # The python layer replicates a channelwise scale row to group 128;
    # the two presentations must agree bit for bit.
    m, k, n = 64, 1024, 512
    torch.manual_seed(4)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, QT, -1)
    rep = scales.expand(k // 128, n).contiguous()
    o_cw = ops.swordfish_mm(a := torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE),
                            packed, scales, -1, k, n)
    o_g = ops.swordfish_mm(a, packed, rep, 128, k, n)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    torch.testing.assert_close(o_g.to(torch.float32), ref, rtol=1e-1, atol=8e-2)
    torch.testing.assert_close(o_cw.to(torch.float32), ref, rtol=1e-1, atol=8e-2)
