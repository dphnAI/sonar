# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical regressions against the real NF12 CUDA package, without model downloads."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sonar.model_executor.layers.fused_moe.activation import MoEActivation
from sonar.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from sonar.model_executor.layers.quantization.online.nf12 import Nf12OnlineLinearMethod
from sonar.model_executor.layers.quantization.online.nf12_moe import Nf12OnlineMoEMethod
from sonar.model_executor.model_loader.weight_utils import default_weight_loader
from sonar.model_executor.utils import replace_parameter
from sonar.platforms import current_platform


@pytest.fixture
def nf12_backend(default_sonar_config):
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("NF12 requires NVIDIA CUDA")
    if not current_platform.has_device_capability(80):
        pytest.skip("NF12 requires SM80 or newer")
    package = pytest.importorskip("nf12")
    default_sonar_config.model_config = SimpleNamespace(dtype=torch.bfloat16, is_moe=False)
    return package


@pytest.fixture
def rng(nf12_backend):
    return torch.Generator(device="cuda").manual_seed(1234)


def _randn(shape, rng, scale=0.1):
    return torch.randn(shape, generator=rng, device="cuda", dtype=torch.bfloat16) * scale


@pytest.fixture
def linear_layer(nf12_backend, dist_init):
    def create(weight):
        method = Nf12OnlineLinearMethod()
        layer = torch.nn.Module()
        layer.quant_method = method
        n, k = weight.shape
        method.create_weights(
            layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=weight.dtype,
            weight_loader=default_weight_loader,
        )
        replace_parameter(layer, "weight", weight.clone())
        method.process_weights_after_loading(layer)
        return method, layer

    return create


def test_nf12_recovers_bf16_escapes(nf12_backend, linear_layer, rng):
    weight = _randn((32, 128), rng)
    # Signed zeros, subnormals, infinities and distinct NaN payloads must survive.
    weight.view(torch.int16)[0, :8] = torch.tensor(
        [0, -32768, 1, -32767, 32640, -128, 32705, -63],
        dtype=torch.int16,
        device="cuda",
    )
    _, layer = linear_layer(weight)
    encoded = nf12_backend.NF12Weight(layer.weight, layer.weight_escapes, layer.weight_exp_base)
    assert encoded.num_escapes > 0
    torch.testing.assert_close(encoded.decode().view(torch.int16), weight.view(torch.int16), rtol=0, atol=0)


@pytest.mark.parametrize("shape,with_bias", [((128,), False), ((5, 128), True), ((2, 3, 128), True)])
def test_nf12_linear_matches_bf16(linear_layer, rng, shape, with_bias):
    weight = _randn((48, 128), rng)
    method, layer = linear_layer(weight)
    x = _randn(shape, rng, scale=0.5)
    if x.ndim == 3:
        x = x.transpose(0, 1)  # Preserve noncontiguous leading dimensions too.
    bias = _randn((48,), rng) if with_bias else None
    actual = method.apply(layer, x, bias)
    expected = F.linear(x, weight, bias)
    assert actual.shape == (*x.shape[:-1], 48)
    assert actual.dtype == torch.bfloat16
    # NF12 and cuBLAS may use different accumulation orders.
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)


@pytest.mark.parametrize("n,k", [(31, 128), (32, 96)])
def test_nf12_unsupported_local_shape_computes_unquantized(linear_layer, rng, n, k):
    weight = _randn((n, k), rng)
    method, layer = linear_layer(weight)
    x, bias = _randn((2, 3, k), rng), _randn((n,), rng)
    torch.testing.assert_close(method.apply(layer, x, bias), F.linear(x, weight, bias), rtol=0, atol=0)


@pytest.mark.parametrize("fp16_source", ["model", "weight"])
def test_nf12_rejects_fp16(nf12_backend, default_sonar_config, linear_layer, rng, fp16_source):
    if fp16_source == "model":
        default_sonar_config.model_config.dtype = torch.float16
    weight = _randn((32, 128), rng)
    if fp16_source == "weight":
        weight = weight.to(torch.float16)
    with pytest.raises(ValueError, match="BF16|bfloat16"):
        linear_layer(weight)


def test_nf12_moe_preserves_weighted_routes(nf12_backend, rng):
    moe = FusedMoEConfig(
        num_experts=3,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=3,
        num_logical_experts=3,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )
    layer = torch.nn.Module()
    layer.moe_config = moe
    layer.local_num_experts = 3
    layer.expert_map_manager = SimpleNamespace(num_fused_shared_experts=0)
    layer.apply_router_weight_on_input = False
    layer.expert_map = None
    method = Nf12OnlineMoEMethod(moe)
    layer.quant_method = method
    method.create_weights(
        layer,
        num_experts=3,
        hidden_size=128,
        intermediate_size_per_partition=128,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    w13, w2 = _randn((3, 256, 128), rng), _randn((3, 128, 128), rng)
    replace_parameter(layer, "w13_weight", w13.clone())
    replace_parameter(layer, "w2_weight", w2.clone())
    method.process_weights_after_loading(layer)
    for name, original in (("w13", w13), ("w2", w2)):
        encoded = nf12_backend.NF12MoEWeight(
            getattr(layer, f"{name}_weight"),
            getattr(layer, f"{name}_escapes"),
            getattr(layer, f"{name}_escape_offsets"),
            getattr(layer, f"{name}_exp_base"),
        )
        torch.testing.assert_close(encoded.decode().view(torch.int16), original.view(torch.int16), rtol=0, atol=0)

    x = _randn((4, 128), rng, scale=0.5)
    ids = [[2, 0], [1, 2], [0, 1], [2, 1]]
    topk_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    # Neither normalized nor uniform: catches accidental renormalization and
    # scaling the nonlinear MLP input rather than each expert's output.
    topk_weights = torch.tensor([[0.2, 0.6], [0.75, 0.1], [0.0, 0.5], [0.9, 0.3]], device="cuda")
    expected = torch.zeros_like(x, dtype=torch.float32)
    for token, experts in enumerate(ids):
        for route, expert in enumerate(experts):
            gate, up = F.linear(x[token], w13[expert]).float().chunk(2)
            hidden = (F.silu(gate) * up).to(torch.bfloat16)
            expert_output = F.linear(hidden, w2[expert]).float()
            expected[token] += expert_output * topk_weights[token, route]
    actual = method.apply(layer, x, topk_weights, topk_ids, None, None)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0.03, atol=0.006)
    # The same routes with no contribution must not retain a previous output.
    zero = method.apply(layer, x, torch.zeros_like(topk_weights), topk_ids, None, None)
    torch.testing.assert_close(zero, torch.zeros_like(x), rtol=0, atol=0)
