# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU shape-admission checks for MXFP8 linear kernels."""

from unittest.mock import patch

import pytest

from aphrodite.model_executor.kernels.linear import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    MarlinMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)

pytestmark = pytest.mark.cpu_test

# Kernels backed by FlashInfer mm_mxfp8, which requires N, K >= 128 and
# K % 32 == 0.
_MM_MXFP8_KERNELS = [
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
]

_SUPPORTED_SHAPE = (4096, 4096)
# (N, K): N < 128, K < 128, K % 32 != 0.
_UNSUPPORTED_SHAPES = [(64, 4096), (4096, 64), (4096, 4112)]

# (N, K): N <= 0, K <= 0, K % 32 != 0.
_MARLIN_UNSUPPORTED_SHAPES = [(0, 256), (256, 0), (256, 100)]


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
def test_mm_mxfp8_kernels_accept_supported_shape(kernel_cls):
    config = Mxfp8LinearLayerConfig(weight_shape=_SUPPORTED_SHAPE)
    can_implement, reason = kernel_cls.can_implement(config)
    assert can_implement, reason


@pytest.mark.parametrize("kernel_cls", _MM_MXFP8_KERNELS)
@pytest.mark.parametrize("weight_shape", _UNSUPPORTED_SHAPES)
def test_mm_mxfp8_kernels_reject_unsupported_shape(kernel_cls, weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = kernel_cls.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize("weight_shape", _MARLIN_UNSUPPORTED_SHAPES)
def test_marlin_mxfp8_rejects_unsupported_shape(weight_shape):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    can_implement, reason = MarlinMxfp8LinearKernel.can_implement(config)
    assert not can_implement
    assert reason


@pytest.mark.parametrize(
    ("capability", "expected"),
    [(100, True), (103, True), (120, False), (121, False)],
)
def test_cutlass_mxfp8_requires_aligned_n_only_on_sm12x(capability, expected):
    """(130, 256) passes the generic mm_mxfp8 limits. Only the SM12x CUTLASS
    backend needs N % 32 == 0, so SM100/103 must keep accepting it."""
    config = Mxfp8LinearLayerConfig(weight_shape=(130, 256))
    with patch(
        "aphrodite.model_executor.kernels.linear.mxfp8.flashinfer.current_platform.is_device_capability_family",
        side_effect=lambda family: family // 10 == capability // 10,
    ):
        can_implement, _ = FlashInferCutlassMxfp8LinearKernel.can_implement(config)
    assert can_implement == expected


@pytest.mark.parametrize(
    ("capability", "weight_shape", "expected"),
    [
        (120, (128, 128), True),
        (120, (128, 160), True),
        (120, (128, 576), True),
        (120, (160, 128), False),
        (120, (128, 144), False),
        (121, (160, 128), False),
        (100, (160, 128), True),
        (103, (160, 128), True),
    ],
)
def test_cutedsl_mxfp8_shape_admission(capability, weight_shape, expected):
    config = Mxfp8LinearLayerConfig(weight_shape=weight_shape)
    with patch(
        "aphrodite.model_executor.kernels.linear.mxfp8.flashinfer.current_platform.is_device_capability_family",
        side_effect=lambda family: family // 10 == capability // 10,
    ):
        supported, _ = FlashInferCutedslMxfp8LinearKernel.can_implement(config)
    assert supported == expected
