# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""swordfish_moe_mm correctness: per-expert dense references over a random
router assignment, exercised through the real moe_align_block_size buffers."""

import pytest
import torch

from aphrodite import _custom_ops as ops
from aphrodite.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from aphrodite.model_executor.layers.quantization.utils.swordfish_utils_test import (
    swordfish_quantize,
)
from aphrodite.platforms import current_platform
from aphrodite.scalar_type import scalar_types

if not current_platform.is_cuda():
    pytest.skip(reason="swordfish requires CUDA", allow_module_level=True)
if not current_platform.has_device_capability(100):
    pytest.skip(reason="swordfish requires sm100 family", allow_module_level=True)

DEVICE = "cuda"


def _moe_ref(a, w_refs, topk_ids, topk_weights, top_k, mul_topk_weights):
    m, k = a.shape
    n = w_refs[0].shape[1]
    out = torch.zeros((m * top_k, n), dtype=torch.float32, device=a.device)
    for t in range(m):
        for slot in range(top_k):
            e = int(topk_ids[t, slot])
            v = a[t].float() @ w_refs[e].float()
            if mul_topk_weights:
                v = v * topk_weights[t, slot]
            out[t * top_k + slot] = v
    return out


@pytest.mark.parametrize("m", [1, 7, 33, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("mul_topk", [False, True])
@pytest.mark.parametrize("block", [16, 32])
def test_swordfish_moe_mm(m, bits, mul_topk, block):
    e, top_k, k, n, group = 8, 2, 512, 256, 128
    qt = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
    torch.manual_seed(m * 31 + bits + mul_topk)

    w_refs, packs, scales = [], [], []
    for _ in range(e):
        w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
        w_ref, packed, s = swordfish_quantize(w, qt, group)
        w_refs.append(w_ref)
        packs.append(packed)
        scales.append(s)
    b_packed = torch.stack(packs).contiguous()
    group_scales = torch.stack(scales).contiguous()

    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    router = torch.randn((m, e), device=DEVICE)
    topk_weights, topk_ids = torch.topk(torch.softmax(router, -1), top_k)
    topk_weights = topk_weights.float().contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
        topk_ids, block, e
    )

    out = ops.swordfish_moe_mm(
        a, b_packed, group_scales, sorted_ids, expert_ids, num_post_padded,
        topk_weights if mul_topk else None, block, top_k, mul_topk, bits,
        group, k, n,
    )
    ref = _moe_ref(a, w_refs, topk_ids, topk_weights, top_k, mul_topk)

    assert out.shape == (m * top_k, n)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)


def test_swordfish_moe_mm_top1_w2_style():
    # The w2 GEMM runs with top_k=1 over the [M * topk, N] intermediate.
    e, k, n, group = 4, 256, 128, 128
    torch.manual_seed(11)
    w_refs, packs, scales = [], [], []
    for _ in range(e):
        w = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE) / (k**0.5)
        w_ref, packed, s = swordfish_quantize(w, scalar_types.uint4b8, group)
        w_refs.append(w_ref)
        packs.append(packed)
        scales.append(s)
    b_packed = torch.stack(packs).contiguous()
    group_scales = torch.stack(scales).contiguous()

    m = 48
    a = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    ids = torch.randint(0, e, (m, 1), dtype=torch.int32, device=DEVICE)
    weights = torch.rand((m, 1), dtype=torch.float32, device=DEVICE)
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(ids, 16, e)

    out = ops.swordfish_moe_mm(
        a, b_packed, group_scales, sorted_ids, expert_ids, num_post_padded,
        weights, 16, 1, True, 4, group, k, n,
    )
    ref = _moe_ref(a, w_refs, ids, weights, 1, True)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=1e-1, atol=8e-2)
