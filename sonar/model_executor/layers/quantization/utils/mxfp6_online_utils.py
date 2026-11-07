# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference utilities for online OCP MXFP6 quantization.

The production SM110 converter is implemented in CUDA. These routines define
the packed-weight ABI and provide a device-independent correctness oracle.
"""

from functools import lru_cache
from typing import Literal

import torch

from aphrodite.model_executor.layers.quantization.utils.mxfp8_utils import (
    swizzle_mxfp8_scale,
)

MXFP6_BLOCK_SIZE = 32
MXFP6_PACK_INPUT = 4
MXFP6_PACK_BYTES = 3
Mxfp6Format = Literal["e2m3", "e3m2"]


@lru_cache
def _mxfp6_quantize_triton_kernel():
    from aphrodite.triton_utils import tl, triton

    @triton.jit
    def kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        k: tl.constexpr,
        mantissa_bits: tl.constexpr,
        exponent_bias: tl.constexpr,
        max_value: tl.constexpr,
        min_normal: tl.constexpr,
        subnormal_step: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile_k = tl.program_id(1)
        offsets = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offsets < k
        values = tl.load(x_ptr + row * k + offsets, mask=mask, other=0.0).to(tl.float32)
        magnitude = tl.reshape(tl.abs(values), (BLOCK_K // 32, 32))
        amax = tl.max(magnitude, axis=1)
        scale_exp = tl.ceil(tl.log2(amax / max_value))
        scale_exp = tl.maximum(-127.0, tl.minimum(127.0, scale_exp))
        scale_exp = tl.where(amax == 0.0, -127.0, scale_exp)
        scaled = magnitude * tl.reshape(tl.exp2(-scale_exp), (BLOCK_K // 32, 1))
        scaled = tl.reshape(scaled, (BLOCK_K,))
        values = tl.reshape(values, (BLOCK_K,))

        mantissa_scale: tl.constexpr = 1 << mantissa_bits
        normal_exp = tl.floor(tl.log2(tl.maximum(scaled, min_normal)))
        exponent = normal_exp.to(tl.int32) + exponent_bias
        normal_mantissa = tl.extra.cuda.libdevice.rint((scaled * tl.exp2(-normal_exp) - 1.0) * mantissa_scale).to(
            tl.int32
        )
        carry = normal_mantissa == (1 << mantissa_bits)
        exponent += carry.to(tl.int32)
        normal_mantissa = tl.where(carry, 0, normal_mantissa)
        normal_code = (exponent << mantissa_bits) | normal_mantissa
        subnormal_code = tl.extra.cuda.libdevice.rint(scaled / subnormal_step).to(tl.int32)
        code = tl.where(scaled < min_normal, subnormal_code, normal_code)
        code = tl.maximum(0, tl.minimum(31, code))
        code |= tl.where(values < 0.0, 32, 0)

        grouped = tl.reshape(code, (BLOCK_K // 4, 4))
        shifts = tl.reshape(tl.arange(0, 4) * 6, (1, 4))
        words = tl.sum(
            grouped << shifts,
            axis=1,
        )
        byte_offsets = row * (k * 3 // 4) + tile_k * (BLOCK_K * 3 // 4) + tl.arange(0, BLOCK_K // 4) * 3
        byte_mask = byte_offsets < (row + 1) * (k * 3 // 4)
        tl.store(packed_ptr + byte_offsets, words & 0xFF, mask=byte_mask)
        tl.store(packed_ptr + byte_offsets + 1, (words >> 8) & 0xFF, mask=byte_mask)
        tl.store(packed_ptr + byte_offsets + 2, (words >> 16) & 0xFF, mask=byte_mask)
        scale_offsets = row * (k // 32) + tile_k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
        scale_mask = scale_offsets < (row + 1) * (k // 32)
        tl.store(scale_ptr + scale_offsets, scale_exp + 127.0, mask=scale_mask)

    return kernel


def quantize_mxfp6_cuda(
    x: torch.Tensor,
    fmt: Mxfp6Format = "e2m3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused CUDA conversion to packed OCP MXFP6 and row-major E8M0 scales."""
    if not x.is_cuda or x.ndim not in (2, 3):
        raise ValueError("MXFP6 CUDA quantization expects a 2D/3D CUDA tensor")
    if x.shape[-1] % MXFP6_BLOCK_SIZE:
        raise ValueError("MXFP6 requires K to be divisible by 32")
    from aphrodite.triton_utils import triton

    k = x.shape[-1]
    rows = x.numel() // k
    packed = torch.empty((*x.shape[:-1], k * 3 // 4), dtype=torch.uint8, device=x.device)
    scales = torch.empty((*x.shape[:-1], k // 32), dtype=torch.uint8, device=x.device)
    if fmt == "e2m3":
        mantissa_bits, exponent_bias, max_value = 3, 1, 7.5
        min_normal, subnormal_step = 1.0, 0.125
    elif fmt == "e3m2":
        mantissa_bits, exponent_bias, max_value = 2, 3, 28.0
        min_normal, subnormal_step = 0.25, 0.0625
    else:
        raise ValueError(f"unsupported MXFP6 format: {fmt}")
    kernel = _mxfp6_quantize_triton_kernel()
    block_k = 256
    kernel[(rows, triton.cdiv(k, block_k))](
        x,
        packed,
        scales,
        k=k,
        mantissa_bits=mantissa_bits,
        exponent_bias=exponent_bias,
        max_value=max_value,
        min_normal=min_normal,
        subnormal_step=subnormal_step,
        BLOCK_K=block_k,
    )
    return packed, scales


def _positive_codebook(fmt: Mxfp6Format, device: torch.device) -> torch.Tensor:
    if fmt == "e2m3":
        exponent_bits, mantissa_bits, bias = 2, 3, 1
    elif fmt == "e3m2":
        exponent_bits, mantissa_bits, bias = 3, 2, 3
    else:
        raise ValueError(f"unsupported MXFP6 format: {fmt}")

    codes = torch.arange(1 << (exponent_bits + mantissa_bits), device=device)
    exponent = codes >> mantissa_bits
    mantissa = codes & ((1 << mantissa_bits) - 1)
    normal = (1.0 + mantissa.float() / (1 << mantissa_bits)) * torch.exp2(exponent.float() - bias)
    subnormal = mantissa.float() * 2.0 ** (1 - bias - mantissa_bits)
    return torch.where(exponent == 0, subnormal, normal)


def _quantize_codes(x: torch.Tensor, fmt: Mxfp6Format) -> torch.Tensor:
    """Quantize values already divided by their block scale to raw FP6 codes."""
    values = _positive_codebook(fmt, x.device)
    magnitude = x.abs().float().clamp(max=values[-1])
    upper = torch.searchsorted(values, magnitude).clamp(max=values.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    lower_dist = magnitude - values[lower]
    upper_dist = values[upper] - magnitude
    choose_upper = upper_dist < lower_dist
    tied = upper_dist == lower_dist
    # IEEE round-to-nearest-even selects the candidate with an even LSB.
    choose_upper |= tied & ((upper & 1) == 0)
    positive = torch.where(choose_upper, upper, lower).to(torch.uint8)
    sign = torch.signbit(x).to(torch.uint8) << 5
    # OCP finite FP6 preserves signed zero.
    return positive | sign


def pack_mxfp6_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack four consecutive six-bit codes into three little-endian bytes."""
    if codes.shape[-1] % MXFP6_PACK_INPUT != 0:
        raise ValueError("MXFP6 packing requires the last dimension to be divisible by 4")
    grouped = codes.to(torch.int32).view(*codes.shape[:-1], -1, 4)
    words = grouped[..., 0] | grouped[..., 1] << 6 | grouped[..., 2] << 12 | grouped[..., 3] << 18
    return (
        torch.stack(
            (
                words & 0xFF,
                words >> 8 & 0xFF,
                words >> 16 & 0xFF,
            ),
            dim=-1,
        )
        .flatten(-2)
        .to(torch.uint8)
    )


def unpack_mxfp6_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack the canonical three-byte/four-value MXFP6 representation."""
    if packed.shape[-1] % MXFP6_PACK_BYTES != 0:
        raise ValueError("packed MXFP6 data must contain complete three-byte groups")
    grouped = packed.to(torch.int32).view(*packed.shape[:-1], -1, 3)
    words = grouped[..., 0] | grouped[..., 1] << 8 | grouped[..., 2] << 16
    return (
        torch.stack(
            tuple(words >> shift & 0x3F for shift in (0, 6, 12, 18)),
            dim=-1,
        )
        .flatten(-2)
        .to(torch.uint8)
    )


def dequantize_mxfp6_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    fmt: Mxfp6Format,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    codes = unpack_mxfp6_codes(packed)
    values = _positive_codebook(fmt, packed.device)[(codes & 0x1F).long()]
    values = torch.where((codes & 0x20) != 0, -values, values)
    blocked = values.view(*values.shape[:-1], -1, MXFP6_BLOCK_SIZE)
    descale = torch.exp2(scales.float() - 127.0).unsqueeze(-1)
    return (blocked * descale).flatten(-2).to(dtype)


def quantize_mxfp6_reference(
    x: torch.Tensor,
    fmt: Mxfp6Format = "e2m3",
    *,
    swizzle_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D/3D tensor using OCP block-32 MXFP6 semantics."""
    if x.ndim not in (2, 3):
        raise ValueError(f"MXFP6 reference quantization expects a 2D or 3D tensor, got {x.ndim}D")
    if x.shape[-1] % MXFP6_BLOCK_SIZE != 0:
        raise ValueError("MXFP6 requires K to be divisible by 32")

    codebook = _positive_codebook(fmt, x.device)
    blocked = x.float().view(*x.shape[:-1], -1, MXFP6_BLOCK_SIZE)
    amax = blocked.abs().amax(dim=-1)
    safe_amax = amax.clamp_min(torch.finfo(torch.float32).tiny)
    scale_exp = torch.ceil(torch.log2(safe_amax / codebook[-1])).clamp(-127, 127)
    scale_exp = torch.where(amax == 0, torch.full_like(scale_exp, -127), scale_exp)
    scales = (scale_exp + 127).to(torch.uint8)
    scaled = blocked / torch.exp2(scale_exp).unsqueeze(-1)
    packed = pack_mxfp6_codes(_quantize_codes(scaled.flatten(-2), fmt))

    if swizzle_scales:
        if x.ndim == 2:
            scales = swizzle_mxfp8_scale(scales, x.shape[0], x.shape[1])
        else:
            scales = torch.cat([swizzle_mxfp8_scale(s, x.shape[1], x.shape[2]) for s in scales])
    return packed, scales
