# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Blackwell MXFP8 x MXFP6 linear kernel."""

from functools import lru_cache
from typing import Literal

import torch
from torch.nn.parameter import Parameter

from aphrodite.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale
from aphrodite.platforms import current_platform
from aphrodite.utils.import_utils import has_cutedsl

from .base import Mxfp6LinearKernel


@lru_cache(maxsize=16)
def _compile_gemm(activation_format: str, weight_format: str, output_dtype: torch.dtype):
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass import utils

    from .cutedsl_kernel import (
        Sm100BlockScaledPersistentDenseGemmKernel,
        scaled_mm,
    )

    weight_dtype = cutlass.Float6E2M3FN if weight_format == "e2m3" else cutlass.Float6E3M2FN
    activation_dtype = {
        "mxfp8": cutlass.Float8E4M3FN,
        "mxfp6_e2m3": cutlass.Float6E2M3FN,
        "mxfp6_e3m2": cutlass.Float6E3M2FN,
    }[activation_format]
    out_dtype = cutlass.BFloat16 if output_dtype == torch.bfloat16 else cutlass.Float16
    cluster = (1, 1)
    gemm = Sm100BlockScaledPersistentDenseGemmKernel(32, (128, 128), cluster)
    max_clusters = utils.HardwareInfo().get_max_active_clusters(1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return scaled_mm(
        gemm,
        activation_dtype,
        weight_dtype,
        out_dtype,
        cutlass.Float8E8M0FNU,
        "k",
        "k",
        "n",
        max_clusters,
        stream,
    )


@torch.library.custom_op("aphrodite::cutedsl_mxfp6_gemm", mutates_args={"out"})
def _cutedsl_mxfp6_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    activation_format: str,
    weight_format: str,
) -> None:
    """Launch the CuTe DSL kernel behind an opaque Torch operator boundary."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from aphrodite.model_executor.layers.quantization.utils.mxfp6_online_utils import quantize_mxfp6_cuda
    from aphrodite.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    m, k = x.shape
    n = out.shape[1]
    if activation_format == "mxfp8":
        x_q, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        activation_dtype = cutlass.Float8E4M3FN
    else:
        activation_encoding: Literal["e2m3", "e3m2"] = "e2m3" if activation_format == "mxfp6_e2m3" else "e3m2"
        x_q, x_scale = quantize_mxfp6_cuda(x, activation_encoding)
        x_scale = swizzle_mxfp8_scale(x_scale, M=m, K=k)
        activation_dtype = cutlass.Float6E2M3FN if activation_encoding == "e2m3" else cutlass.Float6E3M2FN
    weight_dtype = cutlass.Float6E2M3FN if weight_format == "e2m3" else cutlass.Float6E3M2FN
    out_dtype = cutlass.BFloat16 if out.dtype == torch.bfloat16 else cutlass.Float16
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _compile_gemm(activation_format, weight_format, out.dtype)
    compiled(
        make_ptr(
            activation_dtype,
            x_q.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            weight_dtype,
            weight.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            x_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            cutlass.Float8E8M0FNU,
            weight_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        ),
        make_ptr(
            out_dtype,
            out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        (m, n, k, 1),
        stream,
    )


class CutedslMxfp6LinearKernel(Mxfp6LinearKernel):
    """Thor-native tcgen05 MXFP6 GEMM through NVIDIA CUTLASS DSL."""

    @classmethod
    def is_supported(cls) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "MXFP6 requires CUDA"
        capability = current_platform.get_device_capability()
        if capability is None or capability.to_int() != 110:
            return False, "the initial native MXFP6 kernel requires SM110"
        if not has_cutedsl():
            return False, "MXFP6 requires nvidia-cutlass-dsl"
        return True, None

    @classmethod
    def can_implement_shape(cls, n: int, k: int) -> tuple[bool, str | None]:
        if n < 128 or n % 128:
            return False, "output width must be a multiple of 128"
        if k < 128 or k % 128:
            return False, "input width must be a multiple of 128"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        n = layer.mxfp6_logical_n
        k = layer.mxfp6_logical_k
        supported, reason = self.can_implement_shape(n, k)
        if not supported:
            raise ValueError(reason)
        scales = swizzle_mxfp8_scale(layer.weight_scale.data, M=n, K=k)
        layer.weight = Parameter(layer.weight.data.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(scales.contiguous(), requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = layer.mxfp6_logical_n
        k = layer.mxfp6_logical_k
        input_shape = x.shape
        x_2d = x.reshape(-1, k)
        m = x_2d.shape[0]
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)

        _cutedsl_mxfp6_gemm(
            x_2d,
            layer.weight,
            layer.weight_scale,
            out,
            self.config.activation_format,
            self.config.weight_format,
        )
        if bias is not None:
            out.add_(bias)
        return out.view(*input_shape[:-1], n)
