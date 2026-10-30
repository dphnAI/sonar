# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from aphrodite.model_executor.kernels.linear import (
    FlashInferCutedslMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from aphrodite.model_executor.layers.fusion.quant_activation import QuantizedActivation
from aphrodite.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
from aphrodite.platforms import current_platform

if not FlashInferCutedslMxfp8LinearKernel.is_supported()[0]:
    pytest.skip(reason="FlashInfer MXFP8 CuTe-DSL is unavailable", allow_module_level=True)


def _dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, k = values.shape
    exponents = scales.view(torch.uint8).reshape(rows, k // 32).float() - 127
    return (values.float().reshape(rows, k // 32, 32) * torch.exp2(exponents[..., None])).reshape(rows, k)


@pytest.mark.parametrize(
    "shape,dtype,prequantized,with_bias",
    [
        ((1, 1792, 5120), torch.bfloat16, False, False),
        ((5, 8192, 1280), torch.bfloat16, False, False),
        ((48, 5120, 2048), torch.bfloat16, True, True),
        ((128, 1792, 5120), torch.float16, False, True),
        ((512, 1152, 5120), torch.bfloat16, True, False),
        ((6, 5120, 576), torch.bfloat16, False, False),
        ((48, 5120, 576), torch.bfloat16, True, True),
        ((2, 128, 160), torch.float16, True, True),
        ((2, 128, 224), torch.bfloat16, True, False),
    ],
)
@torch.inference_mode()
def test_flashinfer_cutedsl_mxfp8_linear_graph(
    shape: tuple[int, int, int], dtype: torch.dtype, prequantized: bool, with_bias: bool
) -> None:
    torch.manual_seed(0)
    m, n, k = shape
    if k % 128 and not current_platform.is_device_capability_family(120):
        pytest.skip("SM120 padded-K specialization")
    weight = torch.randn((n, k // 32, 32), device="cuda", dtype=dtype)
    weight *= torch.exp2(torch.randint(-4, 4, (n, k // 32, 1), device="cuda").float())
    weight = weight.reshape(n, k)
    weight_q, weight_scales = mxfp8_e4m3_quantize(weight)
    reference_weight = _dequantize(weight_q, weight_scales)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_q, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scales, requires_grad=False)
    kernel = FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    input_shape = (m, k) if m % 2 else (2, m // 2, k)
    x = torch.randn(input_shape, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None
    input_q, input_scales = mxfp8_e4m3_quantize(x.reshape(m, k), is_sf_swizzled_layout=True)
    if prequantized and k % 128:
        input_q = torch.empty((m, 2 * k), device="cuda", dtype=input_q.dtype)[:, 1::2].copy_(input_q)
        input_scales = torch.empty(2 * input_scales.numel(), device="cuda", dtype=input_scales.dtype)[1::2].copy_(
            input_scales
        )
    activation = QuantizedActivation(input_q, input_scales, dtype, x.shape, kernel.input_quant_key())

    def poison_unused_scales() -> None:
        if prequantized and k % 128:
            input_scales.view(-1, (k + 127) // 128, 32, 4, 4)[:, -1, :, :, (k // 32) % 4 :].fill_(255)

    poison_unused_scales()

    def execute() -> torch.Tensor:
        return kernel.apply_weights(layer, activation if prequantized else x, bias)

    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()

    for source in (x.clone(), torch.randn_like(x) * 0.125, torch.zeros_like(x)):
        x.copy_(source)
        values, scales = mxfp8_e4m3_quantize(x.reshape(m, k))
        reference = _dequantize(values, scales) @ reference_weight.t()
        reference = reference.to(dtype)
        if bias is not None:
            reference = reference + bias
        aq, sf = mxfp8_e4m3_quantize(x.reshape(m, k), is_sf_swizzled_layout=True)
        input_q.copy_(aq)
        input_scales.copy_(sf)
        poison_unused_scales()
        saved_scales = input_scales.view(torch.uint8).clone()
        eager = execute()
        graph.replay()
        torch.cuda.synchronize()
        expected = reference.reshape(*input_shape[:-1], n)
        error = (captured.float() - expected.float()).norm()
        scale = expected.float().norm().clamp_min(1e-8)
        assert (error / scale).item() < 5e-3
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        assert torch.equal(input_scales.view(torch.uint8), saved_scales)
        assert torch.equal(input_q.view(torch.uint8), aq.view(torch.uint8))
