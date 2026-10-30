# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from aphrodite.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from aphrodite.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from aphrodite.platforms import current_platform
from aphrodite.utils import flashinfer as aphrodite_flashinfer
from aphrodite.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl

from ._padding import pad_mxfp8_columns
from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def _check_mm_mxfp8_shape(weight_shape: tuple[int, int]) -> tuple[bool, str | None]:
    N, K = weight_shape
    if K < 128:
        return False, f"mm_mxfp8 requires K >= 128, got K={K}."
    if K % MXFP8_BLOCK_SIZE != 0:
        return (
            False,
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}.",
        )
    if N < 128:
        return False, f"mm_mxfp8 requires N >= 128, got N={N}."
    return True, None


class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CUTLASS (SM100+)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda() and current_platform.has_device_capability(100)):
            return False, "requires >=sm_100 (Blackwell)"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        can_implement, reason = _check_mm_mxfp8_shape(c.weight_shape)
        if not can_implement:
            return False, reason
        N, _ = c.weight_shape
        if current_platform.is_device_capability_family(120) and N % 32 != 0:
            return (
                False,
                f"mm_mxfp8 on SM12x requires N to be divisible by 32, got N={N}.",
            )
        return True, None

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale_swizzled.contiguous(), requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        N, K = weight.shape

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = x.dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(x.view(-1, K), is_sf_swizzled_layout=True)

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = aphrodite_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CuTe-DSL (SM100/SM103/SM120)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and (current_platform.is_device_capability_family(100) or current_platform.is_device_capability_family(120))
        ):
            return False, "requires sm_100/sm_103/sm_120 (Blackwell)"
        if not has_flashinfer_cutedsl():
            return False, "requires FlashInfer CuTe-DSL module"
        if (
            current_platform.is_device_capability_family(120)
            and not aphrodite_flashinfer.has_flashinfer_b12x_mxfp8_gemm()
        ):
            return False, "requires FlashInfer's SM120 MXFP8 CuTe-DSL backend"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        supported, reason = _check_mm_mxfp8_shape(c.weight_shape)
        if not supported:
            return False, reason
        N, _ = c.weight_shape
        if current_platform.is_device_capability_family(120) and N % 128 != 0:
            return False, f"SM120 MXFP8 CuTe-DSL requires N divisible by 128, got N={N}."
        return True, None

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)
        if current_platform.is_device_capability_family(120):
            padded_k = (K + 127) // 128 * 128
            weight, weight_scale_swizzled = pad_mxfp8_columns(weight, weight_scale_swizzled, padded_k)

        # Store weight column-major [K, N] as mm_mxfp8 expects for operand B.
        layer.weight = Parameter(weight.contiguous().t(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale_swizzled.contiguous(), requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # [K, N], column-major
        weight_scale = layer.weight_scale
        K, N = weight.shape
        is_sm120 = current_platform.is_device_capability_family(120)

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = x.dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                x.view(-1, self.config.weight_shape[1]), is_sf_swizzled_layout=True
            )
        if is_sm120:
            input_mxfp8, input_scale = pad_mxfp8_columns(input_mxfp8, input_scale, K, sanitize_scales=qa is not None)

        output = aphrodite_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight,
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="b12x" if is_sm120 else "cute-dsl",
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not (current_platform.is_cuda() and current_platform.is_device_capability_family(100)):
            return False, "requires SM100-family GPU"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        if hasattr(layer, "_mxfp8_trtllm_output_size") and layer.weight_scale.ndim == 1:
            return

        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        if K % 256 != 0:
            raise ValueError(f"FlashInfer TRTLLM MXFP8 requires K to be divisible by 256, got K={K}.")

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        padded_n = ((N + 127) // 128) * 128
        if padded_n != N:
            padded_weight = weight.new_zeros((padded_n, K))
            padded_weight[:N] = weight
            weight = padded_weight

            padded_scale = weight_scale.new_zeros((padded_n, scale_k))
            padded_scale[:N] = weight_scale
            weight_scale = padded_scale
        else:
            weight = weight.contiguous()

        layer.weight = Parameter(
            shuffle_matrix_a(weight, 128).reshape(padded_n, K),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            shuffle_matrix_sf_a(
                weight_scale,
                128,
                num_elts_per_sf=MXFP8_BLOCK_SIZE,
            ).reshape(-1),
            requires_grad=False,
        )
        layer._mxfp8_trtllm_output_size = N

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, f"FlashInfer TRTLLM MXFP8 requires bfloat16 activations, got {x.dtype}."

        weight = layer.weight  # shuffled [padded N, K]
        weight_scale = layer.weight_scale
        _, K = weight.shape
        output_size = layer._mxfp8_trtllm_output_size
        input_shape = x.shape
        input_2d = x.view(-1, K)

        input_mxfp8, input_scale = aphrodite_flashinfer.flashinfer_mxfp8_quantize_8x4(input_2d)
        output = aphrodite_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=x.dtype,
            backend="trtllm",
            use_8x4_sf_layout=True,
        )
        if output.shape[-1] != output_size:
            output = output[:, :output_size].contiguous()

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_size)
        return output.view(output_shape)
