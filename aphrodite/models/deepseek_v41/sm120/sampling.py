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
    key_offset,
    V: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    local = tile * BLOCK + tl.arange(0, BLOCK)
    mask = local < V
    # Noise is keyed by the global token id; `key_offset` is the shard start.
    keys = key_offset + local
    base = tl.load(Base + row * base_rs + local, mask, other=0)
    bias = tl.load(Bias + row * bias_rs + local, mask, other=0)
    # Native torch.add writes the same-dtype tensor before the sampler reads it.
    # Preserve that exact seam: BF16/FP16 round-trip is NOT optional.
    rounded = (base.to(tl.float32) + bias.to(tl.float32)).to(Base.dtype.element_ty)
    logits = tl.where(mask, rounded.to(tl.float32), -float("inf"))
    req = tl.load(Mapping + row).to(tl.int64)
    valid_req = req >= 0
    temperature = tl.load(Temperature + req, valid_req, other=0).to(tl.float32)
    col = tl.load(Column)
    tl.store(Cache + req * cache_rs + col * cache_cs + local, rounded, mask & valid_req)
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
    tl.store(LocalIDs + row * partial_rs + tile, key_offset + tile * BLOCK + index)
    tl.store(LocalScores + row * partial_rs + tile, score)


@triton.jit
def _first_max(best, best_index, score, index):
    # torch.argmax order: a later value wins only if strictly greater, and the
    # first NaN beats every number.
    take = (score > best) | ((score != score) & (best == best))
    return tl.where(take, score, best), tl.where(take, index, best_index)


@triton.jit
def _block_winner_kernel(Scores, IDs, partial_rs, Out, BLOCKS: tl.constexpr, PACKET: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    best = tl.load(Scores + row * partial_rs)
    best_block = tl.full([], 0, tl.int32)
    for block in tl.static_range(1, BLOCKS):
        best, best_block = _first_max(best, best_block, tl.load(Scores + row * partial_rs + block), block)
    winner = tl.load(IDs + row * partial_rs + best_block)
    if PACKET:
        # FP32 scores and ids below 2^53 are exact in float64.
        tl.store(Out + row * 2, best.to(tl.float64))
        tl.store(Out + row * 2 + 1, winner.to(tl.float64))
    else:
        tl.store(Out + row, winner)


@triton.jit(do_not_specialize=["rows"])
def _rank_winner_kernel(Packets, IDs, rows, WORLD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    best = tl.load(Packets + row * 2)
    best_id = tl.load(Packets + row * 2 + 1)
    for rank in tl.static_range(1, WORLD):
        packet = (rank * rows + row) * 2
        best, best_id = _first_max(best, best_id, tl.load(Packets + packet), tl.load(Packets + packet + 1))
    tl.store(IDs + row, best_id.to(tl.int64))


def _rounded_sum_local(
    base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, key_offset, *, use_fp64, packet
):
    """Run the rounded-sum Gumbel kernel over `base`'s columns.

    Column `i` is token `key_offset + i`. Writes cache columns [0, V) and
    returns each row's winner: a [B, 2] float64 (score, global id) packet, or
    int64 global ids [B] when `packet` is false.
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
    out = base.new_empty((batch, 2) if packet else (batch,), dtype=torch.float64 if packet else torch.int64)
    if batch == 0:
        return out
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
        key_offset,
        V=vocab,
        USE_FP64=use_fp64,
        BLOCK=1024,
    )
    # First max -> lowest block -> lowest token id, as within each block.
    _block_winner_kernel[(batch,)](local_scores, local_ids, local_ids.stride(0), out, BLOCKS=blocks, PACKET=packet)
    return out


def sample_rounded_sum(base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, *, use_fp64=False):
    """Sample one full-vocab position; same key/cache API as native gumbel_sample.

    `column` is a scalar tensor, `positions_minus_one` is already P-1.
    Caller owns graph/capture and all production buffers. No global RNG is used.
    Unsupported layouts/dtypes raise rather than silently changing arithmetic.
    """
    return _rounded_sum_local(
        base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, 0, use_fp64=use_fp64, packet=False
    )


def rounded_sum_shard_partial(
    base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, shard, *, use_fp64=False
):
    """Rank-local phase of `sample_rounded_sum_vocab_parallel`.

    `base`/`bias`/`cache` hold only `shard`'s columns. Writes the cache shard
    and returns the packed [B, 2] float64 (score, global id) row winner. FP32
    scores and ids below 2^53 are exact in float64.
    """
    if base.shape[-1] != shard.size:
        raise ValueError("vocab-parallel sampling requires shard-width logits")
    return _rounded_sum_local(
        base,
        bias,
        idx_map,
        temperature,
        seeds,
        positions_minus_one,
        cache,
        column,
        shard.start,
        use_fp64=use_fp64,
        packet=True,
    )


def combine_rounded_sum_partials(packets: torch.Tensor) -> torch.Tensor:
    """Pick each row's winner from [world, B, 2] packets in rank order.

    First max wins, so exact ties resolve to the lowest rank and therefore the
    lowest global token id.
    """
    if packets.ndim != 3 or packets.shape[-1] != 2 or packets.dtype != torch.float64:
        raise ValueError("requires [world, B, 2] float64 winner packets")
    packets = packets.contiguous()
    world, rows, _ = packets.shape
    ids = packets.new_empty((rows,), dtype=torch.int64)
    if rows:
        _rank_winner_kernel[(rows,)](packets, ids, rows, WORLD=world)
    return ids


def sample_rounded_sum_vocab_parallel(
    base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, shard, *, use_fp64=False
):
    """`sample_rounded_sum` over this rank's vocab shard; returns global ids [B].

    Same noise, cache and tie semantics as the full-vocab sampler; `cache` is
    the [.., .., shard.size] local shard. One all-gather of [B, 2] float64.
    """
    packet = rounded_sum_shard_partial(
        base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, shard, use_fp64=use_fp64
    )
    return combine_rounded_sum_partials(shard.all_gather(packet))
