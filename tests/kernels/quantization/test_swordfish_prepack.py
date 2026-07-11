# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swordfish ABI v1 prepack: bit-exact contract tests.

The CUDA op `swordfish_prepack_B` must reproduce the pure-torch reference
(`swordfish_pack_weights_ref`) bit-exactly — the ABI is frozen against these
tests (csrc/libtorch_stable/quantization/swordfish/docs/abi-design.md).
"""

import pytest
import torch

from aphrodite import _custom_ops as ops
from aphrodite.model_executor.layers.quantization.utils.marlin_utils_test import (
    get_weight_perm,
    marlin_permute_weights,
)
from aphrodite.model_executor.layers.quantization.utils.quant_utils import pack_rows
from aphrodite.model_executor.layers.quantization.utils.swordfish_utils_test import (
    SWORDFISH_BLOCK_INT32,
    swordfish_pack_weights_ref,
    swordfish_shape_ok,
)
from aphrodite.platforms import current_platform
from tests.kernels.utils import opcheck

if not current_platform.is_cuda():
    pytest.skip(reason="swordfish requires CUDA", allow_module_level=True)

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="swordfish requires Blackwell (sm100 family)",
        allow_module_level=True,
    )

# (K, N) coverage includes the minimum, odd tile multiples, and typical
# model shapes
SHAPES = [
    (64, 64),
    (192, 320),
    (256, 128),
    (2048, 4096),
    (4096, 4096),
]

DEVICE = "cuda"


def _random_q4(size_k: int, size_n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 16, (size_k, size_n), generator=g, dtype=torch.int32)


def _gptq_pack(q_w: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    return pack_rows(q_w, 4, size_k, size_n)


@pytest.mark.parametrize("shape", SHAPES)
def test_prepack_bit_exact(shape):
    size_k, size_n = shape
    q_w = _random_q4(size_k, size_n, seed=size_k * 31 + size_n)

    ref = swordfish_pack_weights_ref(q_w, size_k, size_n)

    gptq = _gptq_pack(q_w, size_k, size_n).to(DEVICE)
    got = ops.swordfish_prepack_B(gptq, size_k, size_n)

    assert got.dtype == torch.int32
    assert tuple(got.shape) == (size_n // 64, size_k // 64, SWORDFISH_BLOCK_INT32)
    assert torch.equal(got.cpu(), ref.cpu()), (
        f"prepack mismatch at shape {shape}: "
        f"first diff word {torch.nonzero(got.cpu() != ref.cpu())[0].tolist()}"
    )


@pytest.mark.parametrize("shape", [(256, 128), (2048, 4096)])
def test_prepack_roundtrip(shape):
    """Unpack the packed ABI back to the original int4 codes via the
    index-tracked inverse of the reference permutation."""
    size_k, size_n = shape
    q_w = _random_q4(size_k, size_n, seed=7)

    packed = swordfish_pack_weights_ref(q_w, size_k, size_n)

    # Rebuild the flat Marlin layout from blocks (inverse of stage 2).
    nb, kb = size_n // 64, size_k // 64
    marlin_flat = (
        packed.reshape(nb, kb, 4, 128)
        .permute(1, 2, 0, 3)
        .reshape(size_k // 16, size_n * 2)
    )

    # Index-tracked inverse of stage 1, the permutation run on an index grid.
    perm = get_weight_perm(num_bits=4)
    idx = torch.arange(size_k * size_n, dtype=torch.int64).reshape(size_k, size_n)
    idx_perm = marlin_permute_weights(idx, size_k, size_n, perm)  # (K/16, N*16)

    # Nibble-unpack marlin_flat in pack order and scatter to original slots.
    words = marlin_flat.reshape(-1).to(torch.int64)
    nibbles = torch.stack([(words >> (4 * i)) & 0xF for i in range(8)], dim=1)
    recovered = torch.empty(size_k * size_n, dtype=torch.int32)
    recovered[idx_perm.reshape(-1)] = nibbles.reshape(-1).to(torch.int32)
    assert torch.equal(recovered.reshape(size_k, size_n), q_w)


@pytest.mark.parametrize("shape", [(96, 64), (64, 96), (100, 100)])
def test_prepack_rejects_bad_shapes(shape):
    size_k, size_n = shape
    assert not swordfish_shape_ok(size_k, size_n)
    # build a plausibly-shaped gptq tensor; op must reject on k/n args
    gptq = torch.zeros(
        (max(size_k // 8, 1), size_n), dtype=torch.int32, device=DEVICE
    )
    with pytest.raises(Exception):
        ops.swordfish_prepack_B(gptq, size_k, size_n)


def test_prepack_opcheck():
    size_k, size_n = 256, 128
    q_w = _random_q4(size_k, size_n, seed=3)
    gptq = _gptq_pack(q_w, size_k, size_n).to(DEVICE)
    opcheck(
        torch.ops._C.swordfish_prepack_B,
        (gptq, size_k, size_n, 4),
    )


@pytest.mark.parametrize("shape", [(64, 64), (256, 128), (2048, 4096)])
def test_prepack_bit_exact_8bit(shape):
    size_k, size_n = shape
    g = torch.Generator(device="cpu").manual_seed(size_k + size_n)
    q_w = torch.randint(0, 256, (size_k, size_n), generator=g, dtype=torch.int32)
    ref = swordfish_pack_weights_ref(q_w, size_k, size_n, num_bits=8)
    gptq = pack_rows(q_w, 8, size_k, size_n).to(DEVICE)
    got = ops.swordfish_prepack_B(gptq, size_k, size_n, num_bits=8)
    assert got.shape == (size_n // 64, size_k // 64, 1024)
    assert torch.equal(got.cpu(), ref)
