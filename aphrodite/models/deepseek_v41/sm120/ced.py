# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED row packing and batch metadata; no model computation lives here."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from aphrodite.triton_utils import tl, triton

if TYPE_CHECKING:
    from .sparse_mla import DeepseekV41B12xMetadata

CED_WINDOW = 128


def ced_decoder_start(hf_config: Any) -> int | None:
    """Find the full-resolution KV source following the compressed encoder."""
    ratios = getattr(hf_config, "compress_ratios", ())
    layers = getattr(hf_config, "num_hidden_layers", len(ratios))
    for source in sorted(getattr(hf_config, "kv_source_layer_ids", ())):
        if (
            0 < source < min(layers, len(ratios))
            and ratios[source] == 1
            and any(ratio > 1 for ratio in ratios[:source])
        ):
            return source
    return None


@triton.jit(do_not_specialize=["source_rows", "output_rows"])
def _copy_rows(
    Source,
    Indices,
    Output,
    source_rows,
    output_rows,
    DIMS: tl.constexpr,
    STRIDES: tl.constexpr,
    WIDTH: tl.constexpr,
    SCATTER: tl.constexpr,
    B: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    row = offset // WIDTH
    col = offset % WIDTH
    limit = source_rows if SCATTER else output_rows
    index = tl.load(Indices + row, row < limit, other=-1).to(tl.int64)
    valid = (row < limit) & (index >= 0)
    valid &= index < (output_rows if SCATTER else source_rows)
    source_row = row if SCATTER else index
    source_offset = source_row * STRIDES[0]
    remainder = col
    for axis in tl.static_range(len(DIMS) - 1, -1, -1):
        source_offset += (remainder % DIMS[axis]) * STRIDES[axis + 1]
        remainder = remainder // DIMS[axis]
    value = tl.load(Source + source_offset, valid, other=0)
    if SCATTER:
        tl.store(Output + index * WIDTH + col, value, valid)
    else:
        tl.store(Output + offset, value, row < output_rows)


@torch.library.custom_op("aphrodite::dsv41_gather_rows", mutates_args=())
def gather_rows(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather first-axis rows, zeroing invalid indices (including graph padding)."""
    output = source.new_empty((indices.numel(), *source.shape[1:]))
    width = source.numel() // source.shape[0] if source.shape[0] else int(np.prod(source.shape[1:]))
    if output.numel():
        _copy_rows[(triton.cdiv(output.numel(), 512),)](
            source,
            indices,
            output,
            source.shape[0],
            indices.numel(),
            tuple(source.shape[1:]),
            source.stride(),
            width,
            False,
            512,
        )
    return output


@gather_rows.register_fake
def _gather_rows_fake(source, indices):
    return source.new_empty((indices.numel(), *source.shape[1:]))


@torch.library.custom_op("aphrodite::dsv41_scatter_rows", mutates_args=())
def scatter_rows(source: torch.Tensor, indices: torch.Tensor, full_rows: int) -> torch.Tensor:
    """Restore the full-row ABI; discarded rows are always zero."""
    output = source.new_zeros((full_rows, *source.shape[1:]))
    width = source.numel() // source.shape[0] if source.shape[0] else int(np.prod(source.shape[1:]))
    if source.numel():
        _copy_rows[(triton.cdiv(source.numel(), 512),)](
            source,
            indices,
            output,
            source.shape[0],
            full_rows,
            tuple(source.shape[1:]),
            source.stride(),
            width,
            True,
            512,
        )
    return output


@scatter_rows.register_fake
def _scatter_rows_fake(source, indices, full_rows):
    return source.new_empty((full_rows, *source.shape[1:]))


@dataclass(frozen=True)
class CEDPlan:
    """CPU scheduling bounds; GPU starts remain authoritative for actual rows."""

    capacity: int
    num_tokens: int
    max_query_len: int


def plan_ced(query_lengths, keep_full, capacities: tuple[int, ...]) -> CEDPlan | None:
    kept = tuple(int(q) if full else min(int(q), CED_WINDOW) for q, full in zip(query_lengths, keep_full))
    if all(k == int(q) for k, q in zip(kept, query_lengths)):
        return None
    count = sum(kept)
    return CEDPlan(
        capacities[bisect_left(capacities, count)],
        count,
        max(kept, default=0),
    )


@triton.jit(do_not_specialize=["nr"])
def _decoder_requests(
    Starts,
    Seq,
    KeepFull,
    PrefixStart,
    OutStarts,
    RequestPositions,
    ReplayStart,
    Counts,
    nr,
    R: tl.constexpr,
    B: tl.constexpr,
    WINDOW: tl.constexpr,
):
    r = tl.arange(0, B)
    valid = r < nr
    start = tl.load(Starts + r, valid, other=0)
    end = tl.load(Starts + r + 1, valid, other=0)
    seq = tl.load(Seq + r, valid, other=0).to(tl.int64)
    full = tl.load(KeepFull + r, valid, other=0)
    q = end - start
    kept = tl.where(full, q, tl.minimum(q, WINDOW))
    cumulative = tl.cumsum(kept)
    count = tl.sum(kept)
    tl.store(OutStarts + r, tl.where(valid, cumulative - kept, count), r <= R)
    tl.store(OutStarts + nr, count)
    tl.store(RequestPositions + r, tl.where(valid, seq - kept, -1), r < R)
    prefix = tl.load(PrefixStart + r, valid, other=0)
    replay = tl.maximum(prefix, tl.where(q > kept, seq - kept, 0))
    tl.store(ReplayStart + r, replay, r < R)
    tl.store(Counts, count)
    tl.store(Counts + 1, nr)


@triton.jit(do_not_specialize=["nr", "nt", "capacity"])
def _decoder_indices(Starts, CompactStarts, Indices, nr, nt, capacity, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    lo = tl.full((B,), 0, tl.int32)
    hi = tl.full((B,), nr, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(CompactStarts + mid + 1, mid < nr, other=nt)
        right = end <= row
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    valid = lo < nr
    compact_end = tl.load(CompactStarts + lo + 1, valid, other=0)
    original_end = tl.load(Starts + lo + 1, valid, other=0).to(tl.int64)
    tl.store(
        Indices + row,
        tl.where(valid, original_end - compact_end + row, -1),
        row < capacity,
    )


class CEDState:
    """Stable per-model staging buffers shared by dummy, capture, and runtime."""

    def __init__(self, max_requests: int, max_tokens: int, capacities: tuple[int, ...], device):
        self.capacities = capacities
        self.max_requests = max_requests
        self.indices = torch.empty(max_tokens, dtype=torch.int64, device=device)
        self.starts = torch.empty(max_requests + 1, dtype=torch.int32, device=device)
        self.request_positions = torch.empty(max_requests, dtype=torch.int64, device=device)
        self.replay_start = torch.empty_like(self.request_positions)
        self.counts = torch.empty(2, dtype=torch.int32, device=device)
        self.keep_full = torch.empty(max_requests, dtype=torch.bool, device=device)
        self.prefix_start = torch.empty_like(self.request_positions)
        self.plan: CEDPlan | None = None

    def warmup(self, hidden_size: int, hc_mult: int) -> None:
        """Resolve both packing directions and row mapping before graph capture."""
        device = self.indices.device
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CED packing must be warmed before graph capture")
        if self.indices.numel() <= CED_WINDOW:
            return
        count = CED_WINDOW + 1
        starts = torch.tensor([0, count], dtype=torch.int32, device=device)
        seq = torch.tensor([count], dtype=torch.int32, device=device)
        self.stage(starts, seq, [count], [False], [0])
        selected = torch.tensor([0, 1, -1, 3], dtype=torch.int64, device=device)
        for dtype, tail in (
            (torch.int32, ()),
            (torch.int64, ()),
            (torch.bfloat16, (hidden_size,)),
            (torch.bfloat16, (hc_mult, hidden_size)),
            (torch.float32, (hc_mult,)),
            (torch.float32, (hc_mult, hc_mult)),
        ):
            source = torch.zeros((4, *tail), dtype=dtype, device=device)
            packed = gather_rows(source, selected)
            scatter_rows(packed, selected, 4)
        self.plan = None
        self.replay_start.zero_()

    def stage(
        self,
        starts: torch.Tensor,
        seq: torch.Tensor,
        query_lengths,
        keep_full,
        prefix_start,
    ) -> None:
        nr = len(query_lengths)
        self.plan = plan_ced(query_lengths, keep_full, self.capacities)
        # Fresh pinned sources remain owned by the async-copy allocator until
        # transfer completion; reusing a CPU staging buffer could race overlap.
        keep_host = torch.tensor(keep_full, dtype=torch.bool, device="cpu", pin_memory=True)
        prefix_host = torch.tensor(prefix_start, dtype=torch.int64, device="cpu", pin_memory=True)
        self.keep_full[:nr].copy_(keep_host, non_blocking=True)
        self.prefix_start[:nr].copy_(prefix_host, non_blocking=True)
        _decoder_requests[(1,)](
            starts,
            seq,
            self.keep_full,
            self.prefix_start,
            self.starts,
            self.request_positions,
            self.replay_start,
            self.counts,
            nr,
            self.max_requests,
            triton.next_power_of_2(self.max_requests + 1),
            CED_WINDOW,
        )
        if self.plan is not None:
            _decoder_indices[(triton.cdiv(self.plan.capacity, 128),)](
                starts,
                self.starts,
                self.indices,
                nr,
                self.plan.num_tokens,
                self.plan.capacity,
                128,
            )

    def get_indices(self) -> torch.Tensor | None:
        return None if self.plan is None else self.indices[: self.plan.num_tokens]

    def decoder_metadata(self, metadata: DeepseekV41B12xMetadata) -> DeepseekV41B12xMetadata:
        plan = self.plan
        assert plan is not None
        indices = self.indices[: plan.num_tokens]
        # Generic packing zeros invalid rows; native metadata uses -1 sentinels.
        valid = indices >= 0

        def gather(field, sentinel=0):
            result = gather_rows(field, indices)
            return result if sentinel == 0 else result.masked_fill(~valid, sentinel)

        return replace(
            metadata,
            num_actual_tokens=plan.num_tokens,
            max_query_len=plan.max_query_len,
            query_start_loc=self.starts,
            request_positions=self.request_positions,
            live_counts=self.counts,
            positions=gather(metadata.positions, -1),
            req_id_per_token=gather(metadata.req_id_per_token, -1),
            slot_mapping=gather(metadata.slot_mapping, -1),
            cache_lengths=gather(metadata.cache_lengths),
            is_decode=plan.max_query_len <= 1,
            decoder=None,
            swa_replay_start=self.replay_start,
        )
