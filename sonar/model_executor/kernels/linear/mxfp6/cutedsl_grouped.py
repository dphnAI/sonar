# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM110 grouped MXFP8 x MXFP6 matrix multiplication."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

from aphrodite.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)


def _grouped_swizzle_scales(
    scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
    k: int,
) -> torch.Tensor:
    """Swizzle row-major scales into one fixed-size slot per expert."""
    rows = scales.shape[0]
    num_k_tiles = (k + 127) // 128
    slot_size = ((rows + 127) // 128) * num_k_tiles * 512
    result = torch.zeros(
        (num_experts, slot_size),
        dtype=torch.uint8,
        device=scales.device,
    )
    if rows == 0:
        return result

    row = torch.arange(rows, device=scales.device)
    expert = torch.bucketize(row, expert_offsets[1:], right=True)
    local_row = row - expert_offsets[expert]
    scale_col = torch.arange(k // 32, device=scales.device)

    mt = local_row[:, None] // 128
    group4 = local_row[:, None] % 128 // 32
    row32 = local_row[:, None] % 32
    kt = scale_col[None, :] // 4
    col4 = scale_col[None, :] % 4
    index = (((mt * num_k_tiles + kt) * 32 + row32) * 4 + group4) * 4 + col4
    result[expert[:, None], index] = scales
    return result


def _grouped_swizzle_activation_scales(
    scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swizzle scales into compact, variable-size expert slots.

    The allocation has one 128-row alignment tile per expert plus enough tiles
    for all routed rows. It avoids reserving the worst-case row count for every
    expert while keeping all layout calculations on the GPU.
    """
    rows = scales.shape[0]
    num_k_tiles = (k + 127) // 128
    tile_size = num_k_tiles * 512
    counts = expert_offsets[1:] - expert_offsets[:-1]
    tiles = (counts + 127) // 128
    tile_bases = torch.cat((torch.zeros_like(tiles[:1]), tiles.cumsum(0)[:-1]))
    # sum(ceil(count / 128)) <= ceil(total / 128) + E - 1.
    max_tiles = (rows + 127) // 128 + max(num_experts - 1, 0)
    result = torch.zeros(max_tiles * tile_size, dtype=torch.uint8, device=scales.device)
    if rows == 0:
        return result, tile_bases * tile_size

    row = torch.arange(rows, device=scales.device)
    expert = torch.bucketize(row, expert_offsets[1:], right=True)
    local_row = row - expert_offsets[expert]
    scale_col = torch.arange(k // 32, device=scales.device)
    mt = local_row[:, None] // 128
    group4 = local_row[:, None] % 128 // 32
    row32 = local_row[:, None] % 32
    kt = scale_col[None, :] // 4
    col4 = scale_col[None, :] % 4
    local_index = (((mt * num_k_tiles + kt) * 32 + row32) * 4 + group4) * 4 + col4
    index = tile_bases[expert, None] * tile_size + local_index
    result[index] = scales
    return result, tile_bases * tile_size


@dataclass
class _CompiledGroupedGemm:
    fn: Callable[..., object]
    initial: tuple[object, object, object, object, object]
    tensormap: object
    backing_tensors: tuple[torch.Tensor, ...]
    max_active_clusters: int


@lru_cache(maxsize=16)
def _compile_grouped(
    num_experts: int,
    activation_format: str,
    weight_format: str,
    output_dtype: torch.dtype,
) -> _CompiledGroupedGemm:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass import utils

    from .cutedsl_grouped_kernel import (
        Sm100GroupedBlockScaledGemmKernel,
        create_tensor_and_stride,
    )

    a_dtype = {
        "mxfp8": cutlass.Float8E4M3FN,
        "mxfp6_e2m3": cutlass.Float6E2M3FN,
        "mxfp6_e3m2": cutlass.Float6E3M2FN,
    }[activation_format]
    b_dtype = cutlass.Float6E2M3FN if weight_format == "e2m3" else cutlass.Float6E3M2FN
    c_dtype = cutlass.BFloat16 if output_dtype == torch.bfloat16 else cutlass.Float16
    sf_dtype = cutlass.Float8E8M0FNU

    # The initial tensors carry only type and layout information.
    initial_a = create_tensor_and_stride(1, 128, 128, False, a_dtype)[1:3]
    initial_b = create_tensor_and_stride(1, 128, 128, False, b_dtype)[1:3]
    initial_c = create_tensor_and_stride(1, 128, 128, False, c_dtype)[1:3]
    initial_sfa = create_tensor_and_stride(1, 128, 128, False, sf_dtype)[1:3]
    initial_sfb = create_tensor_and_stride(1, 128, 128, False, sf_dtype)[1:3]
    initial_pairs = (initial_a, initial_b, initial_c, initial_sfa, initial_sfb)
    initial = tuple(pair[1] for pair in initial_pairs)

    dummy_shapes, shapes_backing = cutlass_torch.cute_tensor_like(
        torch.empty((num_experts, 4), dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    dummy_strides, strides_backing = cutlass_torch.cute_tensor_like(
        torch.empty((num_experts, 3, 2), dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    dummy_ptrs, ptrs_backing = cutlass_torch.cute_tensor_like(
        torch.empty((num_experts, 3), dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    dummy_sf_ptrs, sf_ptrs_backing = cutlass_torch.cute_tensor_like(
        torch.empty((num_experts, 2), dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    hardware = utils.HardwareInfo()
    sm_count = hardware.get_max_active_clusters(1)
    max_active_clusters = hardware.get_max_active_clusters(1)
    tensormap_shape = (
        sm_count,
        Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
        Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
    )
    tensormap, tensormap_backing = cutlass_torch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
    )
    kernel = Sm100GroupedBlockScaledGemmKernel(32, (128, 128), (1, 1))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(
        kernel,
        *initial,
        num_experts,
        dummy_shapes,
        dummy_strides,
        dummy_ptrs,
        dummy_sf_ptrs,
        max_active_clusters,
        tensormap,
        max_active_clusters,
        stream,
        options="--opt-level 2",
    )
    backing = tuple(pair[0] for pair in initial_pairs) + (
        shapes_backing,
        strides_backing,
        ptrs_backing,
        sf_ptrs_backing,
        tensormap_backing,
    )
    return _CompiledGroupedGemm(
        fn,
        initial,
        tensormap,
        backing,
        max_active_clusters,
    )


@torch.library.custom_op(
    "aphrodite::cutedsl_grouped_mxfp6_gemm",
    mutates_args={"out"},
)
def cutedsl_grouped_mxfp6_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    logical_n: int,
    logical_k: int,
    activation_format: str,
    weight_format: str,
) -> None:
    """Quantize grouped activations and execute one persistent grouped GEMM."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from aphrodite.model_executor.layers.quantization.utils.mxfp6_online_utils import (
        quantize_mxfp6_cuda,
    )

    num_experts = weight.shape[0]
    if activation_format == "mxfp8":
        x_q, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    else:
        encoding = "e2m3" if activation_format == "mxfp6_e2m3" else "e3m2"
        x_q, x_scale = quantize_mxfp6_cuda(x, encoding)  # type: ignore[arg-type]
    x_scale, x_scale_offsets = _grouped_swizzle_activation_scales(
        x_scale,
        expert_offsets,
        num_experts,
        logical_k,
    )

    counts = expert_offsets[1:] - expert_offsets[:-1]
    ones = torch.ones_like(counts)
    shapes = torch.stack(
        (counts, torch.full_like(counts, logical_n), torch.full_like(counts, logical_k), ones),
        dim=1,
    ).to(torch.int32)
    strides = torch.empty((num_experts, 3, 2), dtype=torch.int32, device=x.device)
    strides[:, 0, 0].fill_(logical_k)
    strides[:, 0, 1].fill_(1)
    strides[:, 1, 0].fill_(logical_k)
    strides[:, 1, 1].fill_(1)
    strides[:, 2, 0].fill_(logical_n)
    strides[:, 2, 1].fill_(1)

    expert = torch.arange(num_experts, dtype=torch.int64, device=x.device)
    ptrs = torch.stack(
        (
            x_q.data_ptr() + expert_offsets[:-1] * x_q.stride(0) * x_q.element_size(),
            weight.data_ptr() + expert * weight.stride(0) * weight.element_size(),
            out.data_ptr() + expert_offsets[:-1] * out.stride(0) * out.element_size(),
        ),
        dim=1,
    )
    sf_ptrs = torch.stack(
        (
            x_scale.data_ptr() + x_scale_offsets,
            weight_scale.data_ptr() + expert * weight_scale.stride(0) * weight_scale.element_size(),
        ),
        dim=1,
    )

    compiled = _compile_grouped(
        num_experts,
        activation_format,
        weight_format,
        out.dtype,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled.fn(
        *compiled.initial,
        from_dlpack(shapes, assumed_align=16),
        from_dlpack(strides, assumed_align=16),
        from_dlpack(ptrs, assumed_align=16),
        from_dlpack(sf_ptrs, assumed_align=16),
        compiled.tensormap,
        stream,
    )
