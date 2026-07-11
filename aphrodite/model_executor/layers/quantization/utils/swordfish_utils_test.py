# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch reference implementation of the Swordfish packed-weight ABI v1.

Test oracle for `ops.swordfish_prepack_B` (bit-exact contract) and for the
mm kernels' dequant reference.

ABI v1 is Marlin's in-tile permutation (16x64 tiles, pack_idx nibble
interleave, reused verbatim from marlin_utils_test) re-tiled into
(NB, KB, 512) int32 block-linear order:
  block (nb, kb) <- marlin rows [4*kb, 4*kb+4) x int32 cols [128*nb, 128*(nb+1))
"""

import torch

from aphrodite.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
    marlin_weights,
)
from aphrodite.scalar_type import ScalarType, scalar_types

SWORDFISH_BLOCK_N = 64
SWORDFISH_BLOCK_K = 64
SWORDFISH_BLOCK_INT32 = 512
SWORDFISH_SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]
SWORDFISH_SUPPORTED_GROUP_SIZES = [-1, 64, 128]


def swordfish_shape_ok(size_k: int, size_n: int) -> bool:
    return (
        size_k > 0
        and size_n > 0
        and size_k % SWORDFISH_BLOCK_K == 0
        and size_n % SWORDFISH_BLOCK_N == 0
    )


def swordfish_pack_weights_ref(q_w: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    """Pack unpacked int4 codes q_w [K, N] (values 0..15, u4b8 domain) into the
    Swordfish ABI v1 tensor: int32 [NB, KB, 512]."""
    assert swordfish_shape_ok(size_k, size_n), (size_k, size_n)
    assert q_w.shape == (size_k, size_n)

    # Stage 1, Marlin permutation and nibble pack, int32 [K/16, N*2].
    perm = get_weight_perm(num_bits=4)
    marlin_flat = marlin_weights(q_w, size_k, size_n, num_bits=4, perm=perm)
    assert marlin_flat.shape == (size_k // 16, size_n * 2)

    # Stage 2, block re-tile to (NB, KB, 512).
    kb = size_k // SWORDFISH_BLOCK_K
    nb = size_n // SWORDFISH_BLOCK_N
    # rows split as (KB, 4) k16-slices, cols as (NB, 128) int32 runs
    x = marlin_flat.reshape(kb, 4, nb, 128)
    x = x.permute(2, 0, 1, 3)  # (NB, KB, 4, 128)
    return x.reshape(nb, kb, SWORDFISH_BLOCK_INT32).contiguous()


def swordfish_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
):
    """Quantize fp weight [K, N] GPTQ-style (no act_order) and pack to the
    Swordfish ABI. Returns (w_ref, packed, scales)."""
    from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
        gptq_quantize_weights,
    )

    assert quant_type in SWORDFISH_SUPPORTED_QUANT_TYPES
    size_k, size_n = w.shape

    w_ref, q_w, s, _, _ = gptq_quantize_weights(
        w, quant_type, group_size, act_order=False
    )
    packed = swordfish_pack_weights_ref(q_w, size_k, size_n)
    return w_ref, packed, s


def swordfish_quantize_awq(
    w: torch.Tensor,
    group_size: int,
):
    """Quantize fp weight [K, N] AWQ-style (uint4 + group zero points) and
    pack to the Swordfish ABI. Returns (w_ref, packed, scales, zps_neg)
    where zps_neg holds the prescaled (8 - zp) * scale rows the kernel
    consumes."""
    from aphrodite.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )

    size_k, size_n = w.shape
    # The kernel applies the zero point after scaling, (w - 8) * s plus
    # (8 - zp) * s, so the matching reference keeps tolerances tight.
    w_ref, q_w, s, zp = quantize_weights(
        w,
        scalar_types.uint4,
        group_size,
        zero_points=True,
        ref_zero_points_after_scales=True,
    )
    packed = swordfish_pack_weights_ref(q_w, size_k, size_n)
    zps_neg = ((8.0 - zp.to(s.dtype)) * s).contiguous()
    return w_ref, packed, s, zps_neg
