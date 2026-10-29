# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the b12x tensor-parallel MoE integration."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

import aphrodite.model_executor.layers.fused_moe.modular_kernel as mk
import aphrodite.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle
import aphrodite.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle
from aphrodite import _custom_ops as ops
from aphrodite.config import AphroditeConfig, ParallelConfig, set_current_aphrodite_config
from aphrodite.model_executor.layers.fused_moe import b12x_fused_decode, fused_topk
from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
from aphrodite.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from aphrodite.model_executor.layers.fused_moe.b12x import B12xExperts
from aphrodite.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    mxfp4_w4a16_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from aphrodite.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_deepseek_v4_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from aphrodite.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from aphrodite.model_executor.layers.quantization.utils.b12x_moe import (
    prepare_nvfp4_moe_layer_for_b12x,
)
from aphrodite.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from aphrodite.model_executor.warmup.b12x_warmup import warmup_b12x_model
from aphrodite.platforms import current_platform
from aphrodite.utils.torch_utils import set_random_seed
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.quantization.nvfp4_utils import (
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
    quant_nvfp4_tensor,
)
from tests.kernels.utils import torch_moe
from tests.quantization.reference_mxfp4 import dq_mxfp4_torch


def _quantize_nvfp4_linear(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights_q = []
    scales = []
    global_scales = []
    for expert_weight in weight:
        weight_q, scale, global_scale = quant_nvfp4_tensor(
            expert_weight,
            is_sf_swizzled_layout=False,
        )
        weights_q.append(weight_q)
        scales.append(scale)
        global_scales.append(global_scale)
    return torch.stack(weights_q), torch.stack(scales), torch.stack(global_scales)


def _dequantize_nvfp4_linear(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return dequantize_nvfp4_to_dtype(
        tensor_fp4,
        tensor_sf,
        global_scale,
        dtype=dtype,
        device=tensor_fp4.device,
        is_sf_linear_layout=True,
    )


def _nvfp4_activation_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    routed_input = hidden_states[:, None, :].expand(-1, topk, -1).reshape(tokens * topk, hidden_size)
    routed_output = torch.zeros(
        tokens * topk,
        hidden_size,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    flat_ids = topk_ids.reshape(-1)

    for expert in range(w1.shape[0]):
        mask = flat_ids == expert
        if not mask.any():
            continue
        a1_q, a1_block_scale = ops.scaled_fp4_quant(
            routed_input[mask],
            a1_scale[expert],
            is_sf_swizzled_layout=False,
        )
        a1 = _dequantize_nvfp4_linear(
            a1_q,
            a1_block_scale,
            a1_scale[expert],
            torch.float32,
        )
        fc1 = a1 @ w1[expert].float().t()
        gate, up = fc1.chunk(2, dim=-1)
        intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        a2_q, a2_block_scale = ops.scaled_fp4_quant(
            intermediate,
            a2_scale[expert],
            is_sf_swizzled_layout=False,
        )
        a2 = _dequantize_nvfp4_linear(
            a2_q,
            a2_block_scale,
            a2_scale[expert],
            torch.float32,
        )
        routed_output[mask] = a2 @ w2[expert].float().t()

    return routed_output.view(tokens, topk, hidden_size).mul(topk_weights[..., None]).sum(dim=1).to(hidden_states.dtype)


def _has_b12x_moe() -> bool:
    return (
        torch.cuda.is_available()
        and current_platform.is_device_capability_family(120)
        and B12xExperts._supports_current_device()
    )


def _count_fp4_negative_zeros(packed: torch.Tensor) -> int:
    low = (packed & 0x0F) == 0x08
    high = (packed & 0xF0) == 0x80
    return int(low.sum().item() + high.sum().item())


def _make_b12x_moe_kernel(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    activation: MoEActivation,
    quant_config: FusedMoEQuantConfig,
    *,
    layer: torch.nn.Module | None = None,
    swiglu_limit: float | None = None,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    max_num_tokens: int = 512,
) -> mk.FusedMoEKernel:
    num_experts = w1.shape[0]
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_states.shape[1],
        intermediate_size=w2.shape[2]
        * (
            32
            if quant_config.weight_quant_dtype == "q8_0"
            else 256
            if quant_config.weight_quant_dtype in ("iq2_xs", "iq2_xxs")
            else 2
        ),
        in_dtype=hidden_states.dtype,
        activation=activation,
        max_num_tokens=max_num_tokens,
    )
    moe_config.swiglu_limit = swiglu_limit
    moe_config.swiglu_alpha = swiglu_alpha
    moe_config.swiglu_beta = swiglu_beta
    experts = B12xExperts(moe_config, quant_config)
    if layer is None:
        layer = torch.nn.Module()
    layer.activation = activation
    layer.apply_router_weight_on_input = False
    layer.w13_weight = torch.nn.Parameter(w1, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    if quant_config.w1_scale is not None:
        layer.w13_weight_scale = torch.nn.Parameter(quant_config.w1_scale, requires_grad=False)
    if quant_config.w2_scale is not None:
        layer.w2_weight_scale = torch.nn.Parameter(quant_config.w2_scale, requires_grad=False)
    if quant_config.weight_quant_dtype == "nvfp4":
        layer.w13_weight_scale_2 = quant_config.g1_alphas
        layer.w2_weight_scale_2 = quant_config.g2_alphas
        if quant_config.quant_dtype is not None:
            assert quant_config.a1_gscale is not None
            assert quant_config.a2_gscale is not None
            layer.w13_input_scale = 1.0 / quant_config.a1_gscale
            layer.w2_input_scale = 1.0 / quant_config.a2_gscale
    experts.process_weights_after_loading(layer)
    return mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        experts,
    )


def _run_b12x_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    activation: MoEActivation,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    num_experts = w1.shape[0]
    kernel = _make_b12x_moe_kernel(
        hidden_states,
        w1,
        w2,
        topk,
        activation,
        quant_config,
    )
    topk_weights, topk_ids, _ = fused_topk(hidden_states, score, topk, renormalize=False)
    return kernel.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


def _quant_config(weight_dtype: str, activation_dtype: str | None):
    scale = torch.ones(1, dtype=torch.float32)
    return FusedMoEQuantConfig.make(
        quant_dtype=activation_dtype,
        weight_dtype=weight_dtype,
        w1_scale=scale,
        w2_scale=scale,
        g1_alphas=scale,
        g2_alphas=scale,
        a1_gscale=scale,
        a2_gscale=scale,
    )


@pytest.mark.parametrize(
    "hidden_size,expected",
    [
        (128, False),
        (256, False),
        (512, True),
        (768, False),
        (1024, True),
        (5120, True),
        (8192, True),
        (8448, False),
        (8704, False),
    ],
)
def test_b12x_fused_decode_hidden_size_eligibility(hidden_size: int, expected: bool) -> None:
    assert (
        b12x_fused_decode._supported(
            n=576,
            k=hidden_size,
            num_topk=6,
            max_tokens=8,
            ids_dtype=torch.int32,
            csf_inline=False,
            experts=384,
            swiglu_limit=10.0,
        )
        is expected
    )


def test_b12x_moe_supports_only_tensor_parallel() -> None:
    parallel = FusedMoEParallelConfig.make_no_parallel()

    assert B12xExperts._supports_parallel_config(parallel)
    assert not B12xExperts._supports_parallel_config(replace(parallel, use_ep=True, ep_size=2))
    all2all = replace(parallel, use_ep=True, dp_size=2)
    assert all2all.use_all2all_kernels
    assert not B12xExperts._supports_parallel_config(all2all)
    assert not B12xExperts._supports_parallel_config(replace(parallel, enable_eplb=True))


@pytest.mark.parametrize(
    "config_kwargs,overrides,weight_key,activation_key,expected_supported",
    [
        pytest.param(
            {"hidden_dim": 128, "activation": MoEActivation.SWIGLUOAI},
            {},
            kMxfp4Static,
            None,
            False,
            id="interleaved-swigluoai",
        ),
        pytest.param(
            {
                "hidden_dim": 128,
                "activation": MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            },
            {},
            kMxfp4Static,
            kMxfp8Dynamic,
            False,
            id="mxfp4-w4a8-uninterleaved-swigluoai",
        ),
        pytest.param(
            {
                "hidden_dim": 128,
                "activation": MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            },
            {},
            kMxfp4Static,
            None,
            True,
            id="w4a16-uninterleaved-swigluoai",
        ),
        pytest.param(
            {"activation": MoEActivation.RELU2_NO_MUL},
            {},
            kMxfp4Static,
            kMxfp8Dynamic,
            False,
            id="mxfp4-w4a8-relu2",
        ),
        pytest.param(
            {"hidden_dim": 128},
            {},
            kMxfp4Static,
            kMxfp8Dynamic,
            False,
            id="mxfp4-w4a8-alignment",
        ),
        pytest.param(
            {"hidden_dim": 128, "intermediate_size": 48},
            {"intermediate_size_per_partition": 64},
            kMxfp4Static,
            None,
            False,
            id="mxfp4-tp-scale-groups",
        ),
        pytest.param(
            {"activation": MoEActivation.SITU},
            {"activation_situ_beta": 3.0, "activation_situ_linear_beta": 25.0},
            kMxfp4Static,
            None,
            False,
            id="situ-beta",
        ),
        pytest.param(
            {"activation": MoEActivation.SITU},
            {"activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0},
            kMxfp4Static,
            None,
            True,
            id="situ-standard-parameters",
        ),
    ],
)
def test_b12x_moe_config_support(
    monkeypatch: pytest.MonkeyPatch,
    config_kwargs,
    overrides,
    weight_key,
    activation_key,
    expected_supported: bool,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(**{"hidden_dim": 256, "intermediate_size": 64, **config_kwargs})
    for name, value in overrides.items():
        setattr(config, name, value)

    supported, _ = B12xExperts.is_supported_config(
        B12xExperts,
        config,
        weight_key,
        activation_key,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert supported is expected_supported


@pytest.mark.parametrize(
    "activation_key,force_a16,expected_backend",
    [
        (kMxfp8Dynamic, False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
        (None, False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
    ],
)
def test_explicit_b12x_mxfp4_selection(
    monkeypatch: pytest.MonkeyPatch,
    activation_key,
    force_a16: bool,
    expected_backend: Mxfp4MoeBackend,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "APHRODITE_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(
        config,
        activation_key=activation_key,
    )

    assert backend == expected_backend
    assert experts_cls is B12xExperts


def test_explicit_b12x_mxfp4_force_a16_uses_a16_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "APHRODITE_B12X_MOE_FP4_FORCE_A16",
        True,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(
        config,
        activation_key=kMxfp8Dynamic,
    )

    assert backend == Mxfp4MoeBackend.B12X_MXFP4_BF16
    assert experts_cls is B12xExperts


@pytest.mark.parametrize(
    "force_a16,expected_backend",
    [
        (False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
        (True, Mxfp4MoeBackend.B12X_MXFP4_BF16),
    ],
)
def test_deepseek_v4_b12x_activation_selection(
    monkeypatch: pytest.MonkeyPatch,
    force_a16: bool,
    expected_backend: Mxfp4MoeBackend,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "APHRODITE_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)

    assert backend == expected_backend
    assert experts_cls is B12xExperts


def test_deepseek_v4_flashinfer_cutlass_falls_through_to_w4a8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit flashinfer_cutlass must try the W4A8 variant when the BF16
    variant is unsupported (the BF16 variant is gated to SM90)."""
    from aphrodite.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
        FlashInferExperts,
    )

    monkeypatch.setattr(FlashInferExperts, "_supports_current_device", lambda: True)

    def sm120_quant_gate(weight_key, activation_key):
        return (weight_key, activation_key) == (
            mxfp4_oracle.kMxfp4Static,
            mxfp4_oracle.kMxfp8Dynamic,
        )

    monkeypatch.setattr(FlashInferExperts, "_supports_quant_scheme", staticmethod(sm120_quant_gate))
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=64)
    config.moe_backend = "flashinfer_cutlass"

    backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)

    assert backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8
    assert experts_cls is FlashInferExperts


def test_b12x_mxfp4_falls_back_to_a16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle.envs,
        "APHRODITE_B12X_MOE_FP4_FORCE_A16",
        False,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64)
    config.moe_backend = "b12x"

    backend, experts_cls = select_mxfp4_moe_backend(config)

    assert backend == Mxfp4MoeBackend.B12X_MXFP4_BF16
    assert experts_cls is B12xExperts


@pytest.mark.parametrize(
    "activation_key,force_a16,activation,expected_supported",
    [
        (kNvfp4Dynamic, False, MoEActivation.SILU, True),
        (kMxfp8Dynamic, False, MoEActivation.SILU, True),
        (kNvfp4Dynamic, False, MoEActivation.SWIGLUOAI_UNINTERLEAVE, False),
        (kNvfp4Dynamic, True, MoEActivation.SWIGLUOAI_UNINTERLEAVE, True),
    ],
)
def test_explicit_b12x_nvfp4_selection(
    monkeypatch: pytest.MonkeyPatch,
    activation_key,
    force_a16: bool,
    activation: MoEActivation,
    expected_supported: bool,
) -> None:
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "APHRODITE_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=64, activation=activation)
    config.moe_backend = "b12x"

    if not expected_supported:
        with pytest.raises(ValueError, match="does not support swigluoai_uninterleave"):
            select_nvfp4_moe_backend(config, weight_key=kNvfp4Static, activation_key=activation_key)
        return

    backend, experts_cls = select_nvfp4_moe_backend(
        config,
        weight_key=kNvfp4Static,
        activation_key=activation_key,
    )

    assert backend == NvFp4MoeBackend.B12X
    assert experts_cls is B12xExperts


@pytest.mark.parametrize("force_a16,expected_quant_dtype", [(False, "nvfp4"), (True, None)])
def test_b12x_nvfp4_force_a16_updates_quant_config(
    monkeypatch: pytest.MonkeyPatch,
    force_a16: bool,
    expected_quant_dtype,
) -> None:
    monkeypatch.setattr(
        nvfp4_oracle.envs,
        "APHRODITE_B12X_MOE_FP4_FORCE_A16",
        force_a16,
    )
    scale = torch.ones(1)

    quant_config = nvfp4_oracle.make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.B12X,
        w13_scale=scale,
        w2_scale=scale,
        w13_scale_2=scale,
        w2_scale_2=scale,
        a13_scale=scale,
        a2_scale=scale,
    )

    assert quant_config.quant_dtype == expected_quant_dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_b12x_nvfp4_preparation_pads_each_gated_half() -> None:
    device = torch.device("cuda")
    num_experts, hidden_size, intermediate_size = 2, 64, 48
    w13 = torch.ones(
        num_experts,
        2 * intermediate_size,
        hidden_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.ones(
        num_experts,
        2 * intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w2 = torch.ones(
        num_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.ones(
        num_experts,
        hidden_size,
        intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w13[:, intermediate_size:].fill_(0x22)
    w13_scale[:, intermediate_size:].fill_(2.0)
    global_scale = torch.ones(num_experts, device=device)
    input_scale = torch.tensor([[1.0, 2.0], [3.0, 1.0]], device=device)

    prepared = prepare_nvfp4_moe_layer_for_b12x(
        w13,
        w13_scale,
        global_scale,
        input_scale,
        w2,
        w2_scale,
        global_scale,
        input_scale,
        is_act_and_mul=True,
    )

    prepared_w13, prepared_w13_scale, _, prepared_a13 = prepared[:4]
    prepared_w2, prepared_w2_scale, _, prepared_a2 = prepared[4:]
    assert prepared_w13.shape == (num_experts, 128, hidden_size // 2)
    assert prepared_w13_scale.shape == (num_experts, 128, hidden_size // 16)
    assert prepared_w2.shape == (num_experts, hidden_size, 32)
    assert prepared_w2_scale.shape == (num_experts, 128, 4)
    expected_w13 = torch.zeros_like(prepared_w13)
    expected_w13[:, :intermediate_size] = 0x01
    expected_w13[:, 64 : 64 + intermediate_size] = 0x22
    torch.testing.assert_close(prepared_w13, expected_w13, rtol=0, atol=0)
    torch.testing.assert_close(prepared_w2[:, :, : intermediate_size // 2], w2, rtol=0, atol=0)
    assert not prepared_w2[:, :, intermediate_size // 2 :].count_nonzero()
    for expert in range(num_experts):
        scale13 = convert_swizzled_to_linear(prepared_w13_scale[expert], 128, hidden_size, 16).float()
        expected_scale13 = torch.zeros_like(scale13)
        expected_scale13[:intermediate_size] = 1.0
        expected_scale13[64 : 64 + intermediate_size] = 2.0
        torch.testing.assert_close(scale13, expected_scale13, rtol=0, atol=0)
        scale2 = convert_swizzled_to_linear(prepared_w2_scale[expert], 128, 64, 16).float()
        expected_scale2 = torch.zeros_like(scale2)
        expected_scale2[:hidden_size, : intermediate_size // 16] = 1.0
        torch.testing.assert_close(scale2, expected_scale2, rtol=0, atol=0)
    torch.testing.assert_close(prepared_a13, torch.tensor([2.0, 3.0], device=device))
    torch.testing.assert_close(prepared_a2, torch.tensor([2.0, 3.0], device=device))


def test_b12x_moe_rejects_router_weight_on_input_for_w4a8() -> None:
    experts = B12xExperts(
        make_dummy_moe_config(hidden_dim=256, intermediate_size=64),
        _quant_config("mxfp4", "mxfp8"),
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=True,
    )

    with pytest.raises(
        ValueError,
        match="apply_router_weight_on_input only with W4A16",
    ):
        experts.process_weights_after_loading(layer)


@dataclass
class _B12xMoeCase:
    hidden_states: torch.Tensor
    score: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_ref: torch.Tensor
    w2_ref: torch.Tensor
    quant_config: FusedMoEQuantConfig
    activation: MoEActivation
    activation_dtype: str | None
    topk: int = 2


def _make_b12x_moe_case(
    weight_dtype: str,
    activation_dtype: str | None,
    *,
    activation: MoEActivation = MoEActivation.SILU,
    tokens: int = 16,
    seed: int = 19,
    num_experts: int = 4,
    hidden_size: int = 512,
    intermediate_size: int = 128,
    topk: int = 2,
) -> _B12xMoeCase:
    set_random_seed(seed)
    dtype = torch.bfloat16
    hidden_states = torch.randn((tokens, hidden_size), device="cuda", dtype=dtype) / 10
    w1_rows = 2 * intermediate_size if activation.is_gated else intermediate_size
    w1 = (
        torch.randn(
            (num_experts, w1_rows, hidden_size),
            device="cuda",
            dtype=dtype,
        )
        / 15
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device="cuda",
            dtype=dtype,
        )
        / 15
    )

    if weight_dtype == "mxfp4":
        w1_q, w1_scale = mxfp4_quantize(w1)
        w2_q, w2_scale = mxfp4_quantize(w2)
        w1_ref = torch.stack([dq_mxfp4_torch(w1_q[e], w1_scale[e], dtype) for e in range(num_experts)])
        w2_ref = torch.stack([dq_mxfp4_torch(w2_q[e], w2_scale[e], dtype) for e in range(num_experts)])
        if activation_dtype is None:
            quant_config = mxfp4_w4a16_moe_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
    elif weight_dtype == "nvfp4":
        w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
        w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)
        w1_ref = torch.stack(
            [_dequantize_nvfp4_linear(w1_q[e], w1_scale[e], w1_global_scale[e], dtype) for e in range(num_experts)]
        )
        w2_ref = torch.stack(
            [_dequantize_nvfp4_linear(w2_q[e], w2_scale[e], w2_global_scale[e], dtype) for e in range(num_experts)]
        )
        input_scale = torch.full(
            (num_experts,),
            1.0 if activation_dtype is None else 1.0 / 1024.0,
            device="cuda",
            dtype=torch.float32,
        )
        prepared = prepare_nvfp4_moe_layer_for_b12x(
            w1_q,
            w1_scale,
            1.0 / w1_global_scale,
            input_scale,
            w2_q,
            w2_scale,
            1.0 / w2_global_scale,
            input_scale,
            is_act_and_mul=activation.is_gated,
        )
        w1_q, w1_scale, w1_alpha, a1_scale = prepared[:4]
        w2_q, w2_scale, w2_alpha, a2_scale = prepared[4:]
        if activation_dtype is None:
            quant_config = nvfp4_w4a16_moe_quant_config(
                g1_alphas=w1_alpha,
                g2_alphas=w2_alpha,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
        else:
            quant_config = FusedMoEQuantConfig.make(
                quant_dtype=activation_dtype,
                weight_dtype=weight_dtype,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                g1_alphas=w1_alpha,
                g2_alphas=w2_alpha,
                a1_gscale=1.0 / a1_scale,
                a2_gscale=1.0 / a2_scale,
            )
    else:
        raise ValueError(f"unsupported test weight dtype {weight_dtype}")

    return _B12xMoeCase(
        hidden_states=hidden_states,
        score=torch.randn((tokens, num_experts), device="cuda", dtype=dtype),
        w1=w1_q,
        w2=w2_q,
        w1_ref=w1_ref,
        w2_ref=w2_ref,
        quant_config=quant_config,
        activation=activation,
        activation_dtype=activation_dtype,
        topk=topk,
    )


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype,activation",
    [
        pytest.param("mxfp4", "mxfp8", MoEActivation.SILU, id="mxfp4-mxfp8"),
        pytest.param("mxfp4", None, MoEActivation.SILU, id="mxfp4-bf16"),
        pytest.param("nvfp4", "nvfp4", MoEActivation.SILU, id="nvfp4-nvfp4"),
        pytest.param("nvfp4", "mxfp8", MoEActivation.SILU, id="nvfp4-mxfp8"),
        pytest.param("nvfp4", None, MoEActivation.SILU, id="nvfp4-bf16-silu"),
        pytest.param(
            "nvfp4",
            None,
            MoEActivation.RELU2_NO_MUL,
            id="nvfp4-bf16-relu2",
        ),
    ],
)
@torch.inference_mode()
def test_b12x_moe_matches_torch(
    weight_dtype: str,
    activation_dtype: str | None,
    activation: MoEActivation,
    workspace_init,
) -> None:
    with set_current_aphrodite_config(AphroditeConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            activation=activation,
        )
        reference = torch_moe(
            case.hidden_states,
            case.w1_ref,
            case.w2_ref,
            case.score,
            case.topk,
            activation=case.activation,
        )
        if activation_dtype == "nvfp4":
            topk_weights, topk_ids, _ = fused_topk(
                case.hidden_states,
                case.score,
                case.topk,
                renormalize=False,
            )
            reference = _nvfp4_activation_reference(
                case.hidden_states,
                case.w1_ref,
                case.w2_ref,
                topk_weights,
                topk_ids,
                case.quant_config.a1_gscale,
                case.quant_config.a2_gscale,
            )

        checks_zero_canonicalization = weight_dtype == "nvfp4" and activation_dtype is None
        if checks_zero_canonicalization:
            assert _count_fp4_negative_zeros(case.w1) > 0
            assert _count_fp4_negative_zeros(case.w2) > 0
        output = _run_b12x_moe(
            case.hidden_states,
            case.w1,
            case.w2,
            case.score,
            case.topk,
            case.activation,
            case.quant_config,
        )
        if checks_zero_canonicalization:
            assert _count_fp4_negative_zeros(case.w1) == 0
            assert _count_fp4_negative_zeros(case.w2) == 0

    torch.testing.assert_close(output, reference, atol=2e-1, rtol=2e-1)
    cosine = torch.nn.functional.cosine_similarity(
        output.flatten().float(),
        reference.flatten().float(),
        dim=0,
    )
    assert cosine > 0.99


def _assert_b12x_normwise_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    relative_l2: float,
    cosine: float,
) -> None:
    actual, expected = actual.float().flatten(), expected.float().flatten()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    assert torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected) <= relative_l2
    assert torch.nn.functional.cosine_similarity(actual, expected, dim=0) >= cosine


def _mxfp8_activation_roundtrip(tensor: torch.Tensor) -> torch.Tensor:
    blocks = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // 32, 32)
    scale_bits = (blocks.abs().amax(dim=-1, keepdim=True) * (1.0 / 448)).contiguous().view(torch.int32)
    # UE8M0 rounds the FP32 exponent upward whenever any mantissa bit is set.
    rounded_bits = torch.where(
        (scale_bits & 0x007FFFFF) != 0,
        (scale_bits + 0x00800000) & 0x7F800000,
        scale_bits,
    )
    scale = rounded_bits.view(torch.float32)
    exponent = (rounded_bits >> 23) & 0xFF
    inverse = ((254 - exponent).clamp(min=0) << 23).view(torch.float32)
    inverse = torch.where(exponent == 0, 0.0, inverse)
    payload = (blocks * inverse).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (payload * scale).reshape(tensor.shape)


def _clamped_swiglu_reference(
    case: _B12xMoeCase,
    *,
    swiglu_limit: float | None = 10.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
) -> torch.Tensor:
    weights, ids = case.score.float().softmax(dim=-1).topk(case.topk)
    output = torch.zeros_like(case.hidden_states, dtype=torch.float32)
    hidden_states = case.hidden_states.float()
    if case.activation_dtype == "mxfp8":
        hidden_states = _mxfp8_activation_roundtrip(hidden_states)
    for expert in range(case.w1_ref.shape[0]):
        token_ids, slots = torch.where(ids == expert)
        projections = (hidden_states[token_ids] @ case.w1_ref[expert].float().t()).to(case.hidden_states.dtype)
        gate, up = projections.float().chunk(2, dim=-1)
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        intermediate = (gate * torch.sigmoid(swiglu_alpha * gate) * (up + swiglu_beta)).to(case.hidden_states.dtype)
        intermediate = intermediate.float()
        if case.activation_dtype == "mxfp8":
            intermediate = _mxfp8_activation_roundtrip(intermediate)
        expert_output = intermediate @ case.w2_ref[expert].float().t()
        output.index_add_(0, token_ids, expert_output * weights[token_ids, slots, None])
    return output.to(case.hidden_states.dtype)


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize(
    "weight_dtype,activation_dtype,tokens,hidden_size,intermediate_size,num_experts,topk,activation",
    [
        pytest.param("mxfp4", "mxfp8", 4, 512, 128, 4, 2, MoEActivation.SILU, id="w4a8"),
        pytest.param("nvfp4", None, 4, 512, 128, 4, 2, MoEActivation.SILU, id="w4a16"),
        pytest.param("nvfp4", None, 4, 512, 128, 4, 2, MoEActivation.SWIGLUOAI_UNINTERLEAVE, id="w4a16-swigluoai"),
        pytest.param("mxfp4", "mxfp8", 1, 5120, 576, 8, 6, MoEActivation.SILU, id="compact-m1"),
        pytest.param("mxfp4", "mxfp8", 6, 5120, 576, 8, 6, MoEActivation.SILU, id="compact-m6"),
        pytest.param("mxfp4", "mxfp8", 48, 5120, 576, 8, 6, MoEActivation.SILU, id="compact-m48"),
    ],
)
@torch.inference_mode()
def test_b12x_moe_cuda_graph_replay(
    weight_dtype: str,
    activation_dtype: str | None,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: MoEActivation,
    workspace_init,
) -> None:
    from aphrodite.v1.worker.workspace import lock_workspace

    with set_current_aphrodite_config(AphroditeConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))):
        case = _make_b12x_moe_case(
            weight_dtype,
            activation_dtype,
            activation=activation,
            tokens=tokens,
            seed=23,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            topk=topk,
        )
        layer = torch.nn.Module()
        swiglu_params = {
            "swiglu_limit": 0.1 if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE else 10.0,
            "swiglu_alpha": 1.3 if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE else 1.0,
            "swiglu_beta": 0.75 if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE else 0.0,
        }
        kernel = _make_b12x_moe_kernel(
            case.hidden_states,
            case.w1,
            case.w2,
            case.topk,
            case.activation,
            case.quant_config,
            layer=layer,
            **swiglu_params,
            max_num_tokens=48,
        )
        topk_weights, topk_ids, _ = fused_topk(
            case.hidden_states,
            case.score,
            case.topk,
            renormalize=False,
        )
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()

        def apply() -> torch.Tensor:
            return kernel.apply(
                hidden_states=case.hidden_states,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=case.activation,
                global_num_experts=case.w1.shape[0],
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        warmup_counts = (1, 6, 48) if intermediate_size == 576 and tokens == 6 else (tokens,)
        warmup_b12x_model(layer, warmup_counts, case.hidden_states.dtype)
        expected = apply().clone()
        lock_workspace()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            actual = apply()

        for changed_input in (False, True):
            if changed_input:
                case.hidden_states.normal_(std=0.1)
                case.score.normal_()
                new_weights, new_ids, _ = fused_topk(case.hidden_states, case.score, case.topk, renormalize=False)
                topk_weights.copy_(new_weights)
                topk_ids.copy_(new_ids)
                expected = apply().clone()
            for _ in range(2):
                graph.replay()
            torch.accelerator.synchronize()
            # BF16 atomic reduction order varies in eager too; elementwise
            # relative error is unstable near cancellation.
            _assert_b12x_normwise_close(actual, expected, relative_l2=0.01, cosine=0.9999)
            reference = _clamped_swiglu_reference(case, **swiglu_params)
            _assert_b12x_normwise_close(actual, reference, relative_l2=0.05, cosine=0.999)

        if intermediate_size == 576 and tokens == 6:
            partial_case = replace(case, hidden_states=case.hidden_states[:5], score=case.score[:5])
            partial_weights, partial_ids = topk_weights[:5], topk_ids[:5]
            partial_graph = torch.cuda.CUDAGraph()
            # M5 is deliberately absent from warmup: capture cannot prepare a
            # new exact-M plan and must use the already-prepared capacity plan.
            with torch.cuda.graph(partial_graph, stream=stream):
                partial_output = kernel.apply(
                    hidden_states=partial_case.hidden_states,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=partial_weights,
                    topk_ids=partial_ids,
                    activation=case.activation,
                    global_num_experts=num_experts,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                )
            reference = _clamped_swiglu_reference(partial_case, **swiglu_params)
            for _ in range(2):
                partial_graph.replay()
            torch.accelerator.synchronize()
            _assert_b12x_normwise_close(partial_output, reference, relative_l2=0.05, cosine=0.999)
            partial_graph.reset()
            assert layer.w13_weight.numel() == layer.w2_weight.numel() == 0
            assert layer.w13_weight_scale.numel() == layer.w2_weight_scale.numel() == 0
            replacement = _make_b12x_moe_case(
                weight_dtype,
                activation_dtype,
                tokens=tokens,
                seed=29,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                topk=topk,
            )
            layer.w13_weight = torch.nn.Parameter(replacement.w1, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(replacement.w2, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(replacement.quant_config.w1_scale, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(replacement.quant_config.w2_scale, requires_grad=False)
            layer.b12x_warmup_provider.process_weights_after_loading(layer)
            case.w1_ref = replacement.w1_ref
            case.w2_ref = replacement.w2_ref
            expected = apply().clone()
            graph.replay()
            torch.accelerator.synchronize()
            _assert_b12x_normwise_close(actual, expected, relative_l2=0.01, cosine=0.9999)
            _assert_b12x_normwise_close(
                actual, _clamped_swiglu_reference(case, **swiglu_params), relative_l2=0.05, cosine=0.999
            )
        graph.reset()


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@torch.inference_mode()
def test_b12x_moe_warmup_prepares_each_owner_for_static_decode(workspace_init) -> None:
    with set_current_aphrodite_config(AphroditeConfig()):
        layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        cases = [
            _make_b12x_moe_case(
                "mxfp4",
                "mxfp8",
                tokens=tokens,
                seed=seed,
                hidden_size=512,
                intermediate_size=128,
                num_experts=4,
                topk=2,
            )
            for tokens, seed in ((7, 41), (8, 43))
        ]
        kernels = [
            _make_b12x_moe_kernel(
                case.hidden_states,
                case.w1,
                case.w2,
                case.topk,
                case.activation,
                case.quant_config,
                layer=layer,
                max_num_tokens=8,
            )
            for layer, case in zip(layers, cases)
        ]
        warmup_b12x_model(layers, (7, 8), torch.bfloat16)
        first = cases[0]
        _, scratch_shape, _ = kernels[0].fused_experts.workspace_shapes(
            8, 256, 512, first.topk, 4, 4, None, first.activation
        )
        scratch = torch.empty(scratch_shape, device="cuda", dtype=torch.bfloat16)
        outputs = [torch.empty_like(case.hidden_states) for case in cases]
        routes = [fused_topk(case.hidden_states, case.score, case.topk, renormalize=False)[:2] for case in cases]
        graph = torch.cuda.CUDAGraph()
        # Equal geometry must not deduplicate layer-owned plans. Neither owner
        # has had a serving invocation before this capture.
        with torch.cuda.graph(graph):
            for layer, kernel, case, output, (weights, ids) in zip(layers, kernels, cases, outputs, routes):
                kernel.fused_experts.apply(
                    output=output,
                    hidden_states=case.hidden_states,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=weights,
                    topk_ids=ids,
                    activation=case.activation,
                    global_num_experts=4,
                    expert_map=None,
                    a1q_scale=None,
                    a2_scale=None,
                    workspace13=None,
                    workspace2=scratch,
                    expert_tokens_meta=None,
                    apply_router_weight_on_input=False,
                )
        for _ in range(2):
            for case, (weights, ids) in zip(cases, routes):
                case.hidden_states.normal_(std=0.1)
                case.score.normal_()
                new_weights, new_ids, _ = fused_topk(case.hidden_states, case.score, case.topk, renormalize=False)
                weights.copy_(new_weights)
                ids.copy_(new_ids)
            allocated = torch.accelerator.memory_allocated()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.accelerator.memory_allocated() == allocated
            for case, output in zip(cases, outputs):
                # W4A8 rounds both activation boundaries to MXFP8; an A16
                # reference measures quantization error, not owner isolation.
                reference = _clamped_swiglu_reference(case, swiglu_limit=None)
                _assert_b12x_normwise_close(output, reference, relative_l2=0.05, cosine=0.999)
        graph.reset()


@pytest.mark.skipif(not _has_b12x_moe(), reason="requires b12x MoE on SM120")
@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("tokens", [1, 2, 4])
@torch.inference_mode()
def test_b12x_block_quant_relu2_moe_graph(codec, tokens, workspace_init):
    from b12x.testing.iq2_xs_reference import moe_reference_iq2_xs

    from tests.model_executor.kernels.test_b12x_linear import _block_weights

    with set_current_aphrodite_config(AphroditeConfig()):
        torch.manual_seed(71)
        e, h, i, topk = 4, 256, 256, 2
        raw1, raw2 = _block_weights(e * i, h, codec), _block_weights(e * h, i, codec)
        w1 = raw1.view(e, i, *raw1.shape[1:]).cuda()
        w2 = raw2.view(e, h, *raw2.shape[1:]).cuda()
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.125
        scores = torch.randn(tokens, e, device="cuda", dtype=torch.bfloat16)
        from aphrodite.model_executor.layers.fused_moe.config import FusedMoEQuantDesc

        quant = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype=codec),
            _w2=FusedMoEQuantDesc(dtype=codec),
        )
        activation = MoEActivation.RELU2_NO_MUL
        kernel = _make_b12x_moe_kernel(x, w1, w2, topk, activation, quant)
        weights, ids, _ = fused_topk(x, scores, topk, renormalize=False)

        def apply():
            return kernel.apply(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weights=weights,
                topk_ids=ids,
                activation=activation,
                global_num_experts=e,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        apply()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = apply()
        for _ in range(2):
            x.normal_(std=0.125)
            scores.normal_()
            new_weights, new_ids, _ = fused_topk(x, scores, topk, renormalize=False)
            weights.copy_(new_weights)
            ids.copy_(new_ids)
            allocated = torch.accelerator.memory_allocated()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.accelerator.memory_allocated() == allocated
            expected = moe_reference_iq2_xs(
                x,
                raw1.view(e, i, *raw1.shape[1:]),
                raw2.view(e, h, *raw2.shape[1:]),
                ids,
                weights,
                activation="relu2",
            )
            assert torch.isfinite(output).all()
            relative_l2 = (output.float() - expected.float()).norm() / expected.float().norm()
            cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), expected.float().flatten(), dim=0)
            assert cosine >= 0.999 and relative_l2 <= 0.01
        graph.reset()
