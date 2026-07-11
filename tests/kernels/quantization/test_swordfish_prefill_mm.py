# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swordfish prefill GEMM correctness: swordfish_prefill_mm (sm100 tcgen05
mixed-input mainloop fork) vs a dequantized reference.

Same oracle as test_swordfish_mm (A @ w_ref from swordfish_quantize,
machete-style tolerances) at prefill-sized M. v1 scope: bf16, group_size 128,
K % 128 == 0, N % 128 == 0.
"""

import pytest
import torch

from aphrodite import _custom_ops as ops
from aphrodite.model_executor.layers.quantization.utils.swordfish_utils_test import (
    swordfish_quantize,
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
GROUP = 128

# (M, K, N) with prefill-sized M plus non-multiple-of-tile M tails
MNK = [
    (256, 4096, 4096),
    (1024, 4096, 4096),
    (1024, 4096, 11008),
    (333, 2048, 4096),
    (128, 256, 128),
]


@pytest.mark.parametrize("mnk", MNK)
def test_swordfish_prefill_mm_correct(mnk):
    m, k, n = mnk

    torch.manual_seed(k * 13 + n + m)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, QT, GROUP)

    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)

    out = ops.swordfish_prefill_mm(a, packed, scales, GROUP, k, n)

    assert out.shape == (m, n)
    torch.testing.assert_close(
        out.to(torch.float32), ref, rtol=1e-1, atol=5e-2
    )


def test_swordfish_prefill_mm_opcheck():
    m, k, n = 256, 512, 256
    torch.manual_seed(7)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    _, packed, scales = swordfish_quantize(w, QT, GROUP)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    opcheck(
        torch.ops._C.swordfish_prefill_mm,
        (a, packed, scales, None, 4, GROUP, k, n),
    )


def test_swordfish_prefill_mm_rejects_bad_args():
    k, n = 512, 256
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE)
    _, packed, scales = swordfish_quantize(w, QT, GROUP)
    a = torch.randn((16, k), dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(Exception, match="dtype"):
        ops.swordfish_prefill_mm(a.to(torch.float16), packed, scales, GROUP, k, n)
    with pytest.raises(Exception, match="group"):
        ops.swordfish_prefill_mm(a, packed, scales, 48, k, n)


@pytest.mark.parametrize("mnk", [(256, 512, 256), (128, 2048, 1024), (512, 11008, 2048)])
def test_swordfish_prefill_mm_awq_correct(mnk):
    m, k, n = mnk
    torch.manual_seed(m + k + n)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales, zps_neg = swordfish_quantize_awq(w, GROUP)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_prefill_mm(a, packed, scales, GROUP, k, n, group_zps=zps_neg)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


@pytest.mark.parametrize("mnk", [(256, 512, 256), (128, 2048, 1024), (512, 11008, 2048)])
def test_swordfish_prefill_mm_8bit_correct(mnk):
    m, k, n = mnk
    torch.manual_seed(m + k + n + 1)
    w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, scalar_types.uint8b128, GROUP)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_prefill_mm(a, packed, scales, GROUP, k, n, num_bits=8)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


@pytest.mark.parametrize("mnk", [(256, 512, 256), (128, 2048, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_swordfish_prefill_mm_dtypes_groups(mnk, dtype, group):
    m, k, n = mnk
    torch.manual_seed(m + k + n + group)
    w = torch.randn((k, n), dtype=dtype, device=DEVICE) / (k**0.5)
    w_ref, packed, scales = swordfish_quantize(w, QT, group)
    a = torch.randn((m, k), dtype=dtype, device=DEVICE)
    ref = a.to(torch.float32) @ w_ref.to(torch.float32)
    out = ops.swordfish_prefill_mm(a, packed, scales, group, k, n)
    torch.testing.assert_close(
        out.to(torch.float32), ref, rtol=1e-1,
        atol=5e-2 if dtype == torch.float16 else 8e-2,
    )
