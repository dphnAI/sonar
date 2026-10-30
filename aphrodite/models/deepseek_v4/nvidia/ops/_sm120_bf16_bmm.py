# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from aphrodite.triton_utils import tl, triton


@triton.jit
def _sm120_wo_a_bf16_partial(
    A,
    W,
    P,
    M: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ASG: tl.int64,
    ASM: tl.int64,
    ASK: tl.int64,
    CHUNK: tl.constexpr,
):
    group = tl.program_id(1) // tl.cdiv(M, 16)
    row = tl.program_id(1) % tl.cdiv(M, 16) * 16 + tl.arange(0, 16)
    col = tl.program_id(0) * 64 + tl.arange(0, 64)
    split = tl.program_id(2)
    acc = tl.zeros((16, 64), tl.float32)
    for step in range(CHUNK // 128):
        k = split * CHUNK + step * 128 + tl.arange(0, 128)
        a = tl.load(A + group * ASG + row[:, None] * ASM + k[None, :] * ASK, row[:, None] < M, other=0)
        w = tl.load(W + group * N * K + col[None, :] * K + k[:, None], col[None, :] < N, other=0)
        acc = tl.dot(a, w, acc)
    tl.store(
        P + ((split * G + group) * M + row[:, None]) * N + col[None, :], acc, (row[:, None] < M) & (col[None, :] < N)
    )


@triton.jit
def _sm120_wo_a_bf16_reduce(P, OUT, M: tl.constexpr, G: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr):
    pos = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, SPLITS)
    total = G * M * N
    partial = tl.load(P + split[:, None] * total + pos[None, :], pos[None, :] < total, other=0)
    value = tl.sum(partial, axis=0)
    group, row, col = pos // (M * N), pos // N % M, pos % N
    tl.store(OUT + (row * G + group) * N + col, value, pos < total)


def bf16_grouped_bmm(a: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Project [tokens, groups, K] using the existing contiguous BF16 weights.

    The caller restricts this to SM120, 1 < tokens <= 64, N divisible by 64,
    and K divisible by 128 * splits. Input strides are preserved; output is
    contiguous [tokens, groups, N].
    """
    m, groups, k = a.shape
    n = weight.shape[1]
    splits = 8 if m <= 32 else 16
    partial = torch.empty((splits, groups, m, n), device=a.device, dtype=torch.float32)
    _sm120_wo_a_bf16_partial[(triton.cdiv(n, 64), groups * triton.cdiv(m, 16), splits)](
        a,
        weight,
        partial,
        m,
        groups,
        n,
        k,
        a.stride(1),
        a.stride(0),
        a.stride(2),
        k // splits,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    _sm120_wo_a_bf16_reduce[(triton.cdiv(groups * m * n, 256),)](partial, out, m, groups, n, splits, num_warps=4)
