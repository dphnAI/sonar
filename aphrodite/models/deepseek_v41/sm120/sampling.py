# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the rounded Markov-logit addition with full-vocabulary draft sampling.

On a vocab shard the Markov bias projection runs inside the sampling kernel:
each program projects the step's Markov embedding through its own block of
`markov_w2` rows, so the bias row is never written and no separate GEMV runs.
"""

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


@triton.jit(do_not_specialize=["rows"])
def _markov_rounded_sum_gumbel_kernel(
    Base,
    base_rs,
    Embed,
    W2,
    rows,
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
    RANK: tl.constexpr,
    ROWS: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    local = tile * BLOCK + tl.arange(0, BLOCK)
    mask = local < V
    keys = key_offset + local
    k = tl.arange(0, RANK)
    r = tl.arange(0, ROWS)
    # Every row's bias for this tile from one weight tile, FP32 accumulation.
    w2 = tl.load(W2 + local[:, None] * RANK + k[None, :], mask=mask[:, None], other=0)
    embeds = tl.load(Embed + r[:, None] * RANK + k[None, :], mask=r[:, None] < rows, other=0)
    biases = tl.dot(embeds, tl.trans(w2))
    col = tl.load(Column)
    for row in range(rows):
        # The unfused path materializes the bias in the logits dtype first.
        bias = tl.sum(tl.where(r[:, None] == row, biases, 0.0), axis=0).to(Base.dtype.element_ty)
        base = tl.load(Base + row * base_rs + local, mask, other=0)
        rounded = (base.to(tl.float32) + bias.to(tl.float32)).to(Base.dtype.element_ty)
        logits = tl.where(mask, rounded.to(tl.float32), -float("inf"))
        req = tl.load(Mapping + row).to(tl.int64)
        valid_req = req >= 0
        temperature = tl.load(Temperature + req, valid_req, other=0).to(tl.float32)
        tl.store(Cache + req * cache_rs + col * cache_cs + local, rounded, mask & valid_req)
        seed = spec_step_seed(tl.load(Seeds + req, valid_req, other=0), col)
        pos = tl.load(Positions + row)
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
def _first_max_combine(score, index, other, other_index):
    # The scan order of _first_max as an associative reduce: NaN beats every
    # number, then the larger score, and ties go to the lower index.
    nan, other_nan = score != score, other != other
    take = (other_nan & (~nan | (other_index < index))) | (
        ~nan & ~other_nan & ((other > score) | ((other == score) & (other_index < index)))
    )
    return tl.where(take, other, score), tl.where(take, other_index, index)


@triton.jit
def _block_winner_kernel(Scores, IDs, partial_rs, Out, BLOCKS: tl.constexpr, WIDTH: tl.constexpr, PACKET: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, WIDTH)
    live = block < BLOCKS
    scores = tl.load(Scores + row * partial_rs + block, live, other=-float("inf"))
    best, best_block = tl.reduce((scores, block), 0, _first_max_combine)
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


def _check_rows(base, idx_map, temperature, seeds, positions_minus_one, cache, column):
    if (
        base.ndim != 2
        or base.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not base.is_cuda
        or base.stride(-1) != 1
    ):
        raise ValueError("requires native CUDA full-vocabulary row tensors")
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
    return idx_map.contiguous(), positions_minus_one.contiguous(), column.contiguous()


def sample_rounded_sum(base, bias, idx_map, temperature, seeds, positions_minus_one, cache, column, *, use_fp64=False):
    """Sample one full-vocab position; same key/cache API as native gumbel_sample.

    `column` is a scalar tensor, `positions_minus_one` is already P-1.
    Caller owns graph/capture and all production buffers. No global RNG is used.
    Unsupported layouts/dtypes raise rather than silently changing arithmetic.
    """
    if base.shape != bias.shape or base.dtype != bias.dtype or base.device != bias.device or bias.stride(-1) != 1:
        raise ValueError("requires a bias row tensor matching the logits")
    idx_map, positions_minus_one, column = _check_rows(
        base, idx_map, temperature, seeds, positions_minus_one, cache, column
    )
    batch, vocab = base.shape
    blocks = triton.cdiv(vocab, 1024)
    out = base.new_empty((batch,), dtype=torch.int64)
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
        0,
        V=vocab,
        USE_FP64=use_fp64,
        BLOCK=1024,
    )
    # First max -> lowest block -> lowest token id, as within each block.
    _block_winner_kernel[(batch,)](
        local_scores,
        local_ids,
        local_ids.stride(0),
        out,
        BLOCKS=blocks,
        WIDTH=triton.next_power_of_2(blocks),
        PACKET=False,
    )
    return out


def markov_shard_partial(
    base,
    markov_embed,
    markov_w2,
    idx_map,
    temperature,
    seeds,
    positions_minus_one,
    cache,
    column,
    shard,
    *,
    use_fp64=False,
):
    """Rank-local phase of `sample_markov_vocab_parallel`.

    `base`/`cache` hold only `shard`'s columns and `markov_w2` the shard's
    `[shard.size, rank]` rows. The bias `markov_embed @ markov_w2.T` is formed
    in FP32 inside the kernel, cast to the logits dtype as a materialized bias
    would be, then added exactly like `sample_rounded_sum`. Writes the cache
    shard and returns the packed [B, 2] float64 (score, global id) row winner.
    FP32 scores and ids below 2^53 are exact in float64.
    """
    idx_map, positions_minus_one, column = _check_rows(
        base, idx_map, temperature, seeds, positions_minus_one, cache, column
    )
    batch, vocab = base.shape
    rank = markov_embed.shape[-1]
    if (
        vocab != shard.size
        or markov_embed.shape != (batch, rank)
        or not markov_embed.is_contiguous()
        or markov_w2.shape != (vocab, rank)
        or not markov_w2.is_contiguous()
        or markov_embed.dtype != markov_w2.dtype
        or markov_w2.dtype not in (torch.bfloat16, torch.float16)
        or rank < 16
        or rank & (rank - 1)
        or markov_embed.device != base.device
        or markov_w2.device != base.device
    ):
        raise ValueError("requires shard-width logits and contiguous half-precision power-of-two-rank Markov factors")
    block = 64
    blocks = triton.cdiv(vocab, block)
    out = base.new_empty((batch, 2), dtype=torch.float64)
    if batch == 0:
        return out
    local_ids = base.new_empty((batch, blocks), dtype=torch.int64)
    local_scores = base.new_empty((batch, blocks), dtype=torch.float64 if use_fp64 else torch.float32)
    # One tensor-core tile covers 16 rows; larger batches take one launch per slab.
    slab = 16
    for start in range(0, batch, slab):
        rows = slice(start, start + slab)
        _markov_rounded_sum_gumbel_kernel[(blocks,)](
            base[rows],
            base.stride(0),
            markov_embed[rows],
            markov_w2,
            min(slab, batch - start),
            idx_map[rows],
            temperature,
            seeds,
            positions_minus_one[rows],
            cache,
            cache.stride(0),
            cache.stride(1),
            column,
            local_ids[rows],
            local_scores[rows],
            local_ids.stride(0),
            shard.start,
            V=vocab,
            RANK=rank,
            ROWS=slab,
            USE_FP64=use_fp64,
            BLOCK=block,
        )
    _block_winner_kernel[(batch,)](
        local_scores,
        local_ids,
        local_ids.stride(0),
        out,
        BLOCKS=blocks,
        WIDTH=triton.next_power_of_2(blocks),
        PACKET=True,
    )
    return out


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


def sample_markov_vocab_parallel(
    base,
    markov_embed,
    markov_w2,
    idx_map,
    temperature,
    seeds,
    positions_minus_one,
    cache,
    column,
    shard,
    *,
    use_fp64=False,
):
    """Draft sampling with the Markov bias over this rank's vocab shard; global ids [B].

    Same noise, cache and tie semantics as the full-vocab sampler; `cache` is
    the [.., .., shard.size] local shard. One all-gather of [B, 2] float64.
    """
    packet = markov_shard_partial(
        base,
        markov_embed,
        markov_w2,
        idx_map,
        temperature,
        seeds,
        positions_minus_one,
        cache,
        column,
        shard,
        use_fp64=use_fp64,
    )
    return combine_rounded_sum_partials(shard.all_gather(packet))
