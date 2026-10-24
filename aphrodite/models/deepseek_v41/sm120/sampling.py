# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the rounded Markov-logit addition with full-vocabulary draft sampling."""

import torch

from aphrodite.triton_utils import tl, triton
from aphrodite.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed


@triton.jit
def _rounded_sum_gumbel_kernel(
    Base,
    Bias,
    base_rs,
    bias_rs,
    Mapping,
    Temperature,
    Seeds,
    Positions,
    Cache,
    cache_rs,
    cache_cs,
    Column,
    LocalIDs,
    LocalScores,
    partial_rs,
    V: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    keys = tile * BLOCK + tl.arange(0, BLOCK)
    mask = keys < V
    base = tl.load(Base + row * base_rs + keys, mask, other=0)
    bias = tl.load(Bias + row * bias_rs + keys, mask, other=0)
    # Native torch.add writes the same-dtype tensor before the sampler reads it.
    # Preserve that exact seam: BF16/FP16 round-trip is NOT optional.
    rounded = (base.to(tl.float32) + bias.to(tl.float32)).to(Base.dtype.element_ty)
    logits = tl.where(mask, rounded.to(tl.float32), -float("inf"))
    req = tl.load(Mapping + row).to(tl.int64)
    valid_req = req >= 0
    temperature = tl.load(Temperature + req, valid_req, other=0).to(tl.float32)
    col = tl.load(Column)
    tl.store(Cache + req * cache_rs + col * cache_cs + keys, rounded, mask & valid_req)
    seed = tl.load(Seeds + req, valid_req, other=0)
    seed = spec_step_seed(seed, col)
    pos = tl.load(Positions + row)
    # Reuse production noise, salt, temperature division, FP64 choice and ties.
    score, index = gumbel_noised_argmax(
        logits,
        keys,
        mask,
        seed,
        pos,
        temperature,
        IS_DRAFTING=True,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=True,
    )
    tl.store(LocalIDs + row * partial_rs + tile, tile * BLOCK + index)
    tl.store(LocalScores + row * partial_rs + tile, score)


def sample_rounded_sum(base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, *, use_fp64=False):
    """Sample one full-vocab position; same key/cache API as native gumbel_sample.

    `column` is a scalar tensor, `positions_minus_one` is already P-1.
    Caller owns graph/capture and all production buffers. No global RNG is used.
    Unsupported layouts/dtypes raise rather than silently changing arithmetic.
    """
    if (
        base.ndim != 2
        or base.shape != bias.shape
        or base.dtype != bias.dtype
        or base.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not base.is_cuda
        or base.device != bias.device
        or base.stride(-1) != 1
        or bias.stride(-1) != 1
    ):
        raise ValueError("requires matching native CUDA full-vocabulary row tensors")
    batch, vocab = base.shape
    if (
        cache is None
        or cache.ndim != 3
        or cache.dtype != base.dtype
        or cache.device != base.device
        or cache.shape[-1] < vocab
        or cache.stride(-1) != 1
        or column.ndim != 0
        or idx_map.shape != (batch,)
        or positions_minus_one.shape != (batch,)
    ):
        raise ValueError("requires native same-dtype proposal cache and scalar step column")
    if any(value.device != base.device for value in (idx_map, temperature, seeds, positions_minus_one, column)):
        raise ValueError("all sampling metadata must be on the native device")
    idx_map = idx_map.contiguous()
    positions_minus_one = positions_minus_one.contiguous()
    column = column.contiguous()
    blocks = triton.cdiv(vocab, 1024)
    local_ids = base.new_empty((batch, blocks), dtype=torch.int64)
    local_scores = base.new_empty((batch, blocks), dtype=torch.float64 if use_fp64 else torch.float32)
    _rounded_sum_gumbel_kernel[(batch, blocks)](
        base,
        bias,
        base.stride(0),
        bias.stride(0),
        idx_map,
        temperature,
        seeds,
        positions_minus_one,
        cache,
        cache.stride(0),
        cache.stride(1),
        column,
        local_ids,
        local_scores,
        local_ids.stride(0),
        V=vocab,
        USE_FP64=use_fp64,
        BLOCK=1024,
    )
    max_block = local_scores.argmax(dim=-1, keepdim=True)
    return local_ids.gather(dim=-1, index=max_block).view(-1)
