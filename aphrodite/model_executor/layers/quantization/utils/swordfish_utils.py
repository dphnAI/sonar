# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swordfish (Blackwell sm100/sm110 w4a16) support queries.

ABI v1 (frozen): int4 weights x fp16/bf16 activations, group scales
{-1, 64, 128}, no act_order. Zero-point checkpoints (AWQ) fold
(8 - zp) * scale into a second scale-shaped tensor at load time.
"""

import torch

from aphrodite.scalar_type import ScalarType, scalar_types

SWORDFISH_BLOCK_N = 64
SWORDFISH_BLOCK_K = 64


def query_swordfish_supported_quant_types(zero_points: bool) -> list[ScalarType]:
    if zero_points:
        return [scalar_types.uint4]
    return [scalar_types.uint4b8]


def query_swordfish_supported_group_sizes(act_type: torch.dtype) -> list[int]:
    if act_type not in (torch.float16, torch.bfloat16):
        return []
    return [-1, 64, 128]


def check_swordfish_supports_shape(
    in_features: int, out_features: int
) -> tuple[bool, str | None]:
    if in_features % SWORDFISH_BLOCK_K != 0:
        return (
            False,
            f"in_features ({in_features}) not divisible by {SWORDFISH_BLOCK_K}",
        )
    if out_features % SWORDFISH_BLOCK_N != 0:
        return (
            False,
            f"out_features ({out_features}) not divisible by {SWORDFISH_BLOCK_N}",
        )
    return True, None
