# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Folding indexer scales must preserve the unfused projection rounding."""

import pytest
import torch

from sonar.models.kimi_k3.amd.dsa import _index_weights_kernel
from sonar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm kernel")


@pytest.mark.parametrize("rows", [1, 96, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [32, 64])
def test_index_weights_preserve_projection_rounding(rows, dtype, heads):
    torch.manual_seed(7)
    values = torch.randn(rows, heads + 16, dtype=dtype, device="cuda")[:, :heads]
    scales = torch.exp2(torch.randint(-12, 5, (rows, heads), device="cuda").float())
    head_scale, softmax_scale = heads**-0.5, 128**-0.5
    expected = (values * head_scale).float() * scales * softmax_scale
    actual = torch.empty_like(expected)
    _index_weights_kernel[(rows,)](
        values,
        scales,
        actual,
        values.stride(0),
        head_scale,
        softmax_scale,
        NUM_HEADS=heads,
        BLOCK=heads,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_indexer_norm_reload_refreshes_cached_affine_in_place(dtype):
    from sonar.model_executor.model_loader.weight_utils import finalize_model_weights
    from sonar.models.kimi_k3.amd.dsa import KimiK3DSAIndexerKNorm

    norm = KimiK3DSAIndexerKNorm(128, eps=1e-6).to("cuda")
    x = torch.randn(4, 128, device="cuda", dtype=dtype)
    norm(x)
    weight, bias = norm.weight_bf16, norm.bias_bf16
    with torch.no_grad():
        norm.weight.fill_(2)
        norm.bias.fill_(0.5)
    finalize_model_weights(norm, torch.device("cuda"))
    assert norm.weight_bf16 is weight
    assert norm.bias_bf16 is bias
    expected = torch.nn.functional.layer_norm(
        x, norm.normalized_shape, norm.weight.to(dtype), norm.bias.to(dtype), norm.eps
    )
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
