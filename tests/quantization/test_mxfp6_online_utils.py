# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from sonar.model_executor.layers.quantization.utils.mxfp6_online_utils import (
    dequantize_mxfp6_reference,
    pack_mxfp6_codes,
    quantize_mxfp6_reference,
    unpack_mxfp6_codes,
)


def test_mxfp6_known_pack_order():
    codes = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
    assert pack_mxfp6_codes(codes).tolist() == [[64, 32, 12]]
    assert torch.equal(unpack_mxfp6_codes(pack_mxfp6_codes(codes)), codes)


@pytest.mark.parametrize("fmt,max_value", [("e2m3", 7.5), ("e3m2", 28.0)])
def test_mxfp6_all_codes_round_trip(fmt, max_value):
    codes = torch.arange(64, dtype=torch.uint8).view(1, 64)
    packed = pack_mxfp6_codes(codes)
    scales = torch.full((1, 2), 127, dtype=torch.uint8)
    values = dequantize_mxfp6_reference(packed, scales, fmt)
    requantized, requant_scales = quantize_mxfp6_reference(values, fmt)
    assert torch.equal(requant_scales, scales)
    assert torch.equal(requantized, packed)
    assert values.abs().max() == max_value


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
def test_mxfp6_block_scaling_and_signed_zero(fmt):
    x = torch.zeros((2, 32), dtype=torch.bfloat16)
    x[0, 0] = 1024
    x[0, 1] = -1024
    x[1, 0] = -0.0
    packed, scales = quantize_mxfp6_reference(x, fmt)
    restored = dequantize_mxfp6_reference(packed, scales, fmt, torch.bfloat16)
    assert restored[0, 0] == 1024
    assert restored[0, 1] == -1024
    assert torch.signbit(restored[1, 0])


def test_mxfp6_rejects_unaligned_k():
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp6_reference(torch.zeros((2, 31)))
