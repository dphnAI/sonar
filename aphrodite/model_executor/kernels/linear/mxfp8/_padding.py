# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

_PAD_KERNEL = None


def _make_pad_kernel():
    from aphrodite.triton_utils import tl, triton

    @triton.jit
    def _pad_mxfp8_columns_kernel(
        src,
        dst,
        scales,
        padded_scales,
        ROWS: tl.constexpr,
        K: tl.constexpr,
        PADDED_K: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        SF_SIZE: tl.constexpr,
    ):
        offset = tl.program_id(0) * 256 + tl.arange(0, 256)
        row, col = offset // PADDED_K, offset % PADDED_K
        value = tl.load(src + row * S0 + col * S1, (row < ROWS) & (col < K), other=0)
        tl.store(dst + offset, value, offset < ROWS * PADDED_K)
        if SF_SIZE:
            scale_col = ((offset // 512) % (PADDED_K // 128)) * 4 + offset % 4
            value = tl.load(scales + offset, (offset < SF_SIZE) & (scale_col < K // 32), other=0)
            tl.store(padded_scales + offset, value, offset < SF_SIZE)

    return _pad_mxfp8_columns_kernel


def pad_mxfp8_columns(
    data: torch.Tensor,
    scales: torch.Tensor,
    columns: int,
    *,
    sanitize_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round K up within its existing F8_128x4 scale tile without requantizing.

    Locally quantized scales already have zero tails. For prequantized input,
    ``sanitize_scales`` clears newly consumed columns in a copy: an arbitrary
    E8M0 tail can contain NaNs even though the padded payload is zero.
    """
    k = data.shape[1]
    if columns == k:
        return data, scales
    global _PAD_KERNEL
    if _PAD_KERNEL is None:
        _PAD_KERNEL = _make_pad_kernel()
    output = torch.empty((data.shape[0], columns), dtype=data.dtype, device=data.device)
    if sanitize_scales:
        scales = scales.contiguous()
        output_scales = torch.empty_like(scales)
    else:
        output_scales = scales
    sf_size = scales.numel() if sanitize_scales else 0
    _PAD_KERNEL[(max(data.shape[0] * columns, sf_size) + 255) // 256,](
        data.view(torch.uint8),
        output.view(torch.uint8),
        scales.view(torch.uint8),
        output_scales.view(torch.uint8),
        data.shape[0],
        k,
        columns,
        *data.stride(),
        sf_size,
        num_warps=4,
    )
    return output, output_scales
