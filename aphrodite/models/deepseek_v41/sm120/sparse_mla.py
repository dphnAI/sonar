# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-exact V4.1 cache metadata. No foreign attention scheduler is involved."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from aphrodite.triton_utils import tl, triton
from aphrodite.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from aphrodite.v1.kv_cache_interface import CircularBufferSpec, KVCacheLayout

MAIN_BLOCK_SIZE = 256
SWA_BLOCK_SIZE = 128


@triton.jit(do_not_specialize=["nr", "nt", "stride", "table_width"])
def _tokens(
    Starts,
    Seq,
    Table,
    InputSlots,
    Positions,
    Reqs,
    Slots,
    Visible,
    OutStarts,
    RequestPositions,
    Counts,
    nr,
    nt,
    stride,
    table_width,
    R: tl.constexpr,
    T: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    B: tl.constexpr,
    CIRCULAR: tl.constexpr = False,
):
    if tl.program_id(0) == 0:
        # Per-request metadata rides in the first program instead of a
        # separate launch: request starts, prefix positions and live counts.
        for first in range(0, R + 1, B):
            r = first + tl.arange(0, B)
            start = tl.load(Starts + r, r <= nr, other=nt)
            end = tl.load(Starts + r + 1, r < nr, other=nt)
            seq = tl.load(Seq + r, r < nr, other=0)
            tl.store(OutStarts + r, start, r <= R)
            tl.store(RequestPositions + r, tl.where(r < nr, seq - (end - start), -1), r < R)
        tl.store(Counts, nt)
        tl.store(Counts + 1, nr)
    t = tl.program_id(0) * B + tl.arange(0, B)
    lo = tl.full((B,), 0, tl.int32)
    hi = tl.full((B,), nr, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(Starts + mid + 1, mid < nr, other=nt)
        right = end <= t
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    valid = (t < nt) & (lo < nr)
    start = tl.load(Starts + lo, valid, other=0)
    end = tl.load(Starts + lo + 1, valid, other=0)
    seq = tl.load(Seq + lo, valid, other=0)
    pos = seq.to(tl.int64) - (end - start) + t - start
    valid = valid & (pos >= 0)
    logical = pos // RATIO
    page_col = tl.full((B,), 0, tl.int64) if CIRCULAR else logical // PAGE
    valid = valid & (page_col < table_width)
    block = tl.load(Table + lo.to(tl.int64) * stride + page_col, valid, other=0).to(tl.int64)
    valid = valid & (block > 0)
    slot = block * PAGE + logical % PAGE
    emit = valid & ((pos + 1) % RATIO == 0)
    if not CIRCULAR:
        # Prefix replay suppresses writes, not reads of the cached prefix.
        input_slot = tl.load(InputSlots + t, t < nt, other=-1)
        emit = emit & (input_slot >= 0)
    if CIRCULAR:
        # Only the final capacity rows may write: earlier rows would race
        # later writers to the same ring slots during a long prefill.
        emit = emit & (t >= end - PAGE)
    tl.store(Positions + t, tl.where(valid, pos, -1), t < T)
    tl.store(Reqs + t, tl.where(valid, lo, -1), t < T)
    tl.store(Slots + t, tl.where(emit, slot, -1), t < T)
    tl.store(Visible + t, tl.where(valid, (pos + 1) // RATIO, 0), t < T)


@triton.jit(do_not_specialize=["offset", "table_stride"])
def _chunk(
    Positions,
    Reqs,
    Table,
    Swa,
    Lengths,
    TopLengths,
    Visible,
    Starts,
    RequestPositions,
    offset,
    table_stride,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    DRAFT: tl.constexpr,
    BLOCK: tl.constexpr,
    swa_replay_start=None,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    req = tl.load(Reqs + offset + row)
    pos = tl.load(Positions + offset + row).to(tl.int64)
    valid = (req >= 0) & (pos >= 0)
    if DRAFT:
        first = tl.load(Starts + req, valid, other=0).to(tl.int64)
        end = tl.load(Starts + req + 1, valid, other=0).to(tl.int64)
        prefix = tl.load(RequestPositions + req, valid, other=0).to(tl.int64)
        start = tl.maximum(prefix - WINDOW, 0)
        if swa_replay_start is not None:
            replay = tl.load(swa_replay_start + req, valid, other=0).to(tl.int64)
            start = tl.maximum(start, replay)
        length = prefix + end - first - start
    else:
        start = tl.maximum(pos + 1 - WINDOW, 0)
        if swa_replay_start is not None:
            replay = tl.load(swa_replay_start + req, valid, other=0).to(tl.int64)
            start = tl.maximum(start, replay)
        length = pos + 1 - start
    length = tl.where(valid, tl.minimum(tl.maximum(length, 0), WIDTH), 0)
    logical = start + col
    page = tl.load(
        Table + req.to(tl.int64) * table_stride + logical // PAGE,
        valid & (logical >= 0) & (col < length),
        other=-1,
    ).to(tl.int64)
    physical = page * PAGE + logical % PAGE
    tl.store(
        Swa + row * WIDTH + col,
        tl.where((page > 0) & (logical >= 0) & (col < length), physical, -1),
        col < WIDTH,
    )
    tl.store(Lengths + row, tl.maximum(length, 0))
    visible = tl.load(Visible + offset + row)
    tl.store(TopLengths + row, tl.where(valid, tl.minimum(tl.maximum(visible, 0), 512), 0))


@dataclass
class DeepseekV41B12xMetadata(AttentionMetadata):
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    block_size: int
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    request_positions: torch.Tensor
    live_counts: torch.Tensor
    positions: torch.Tensor
    req_id_per_token: torch.Tensor
    slot_mapping: torch.Tensor
    cache_lengths: torch.Tensor
    is_decode: bool
    max_seq_len: int
    decoder: "DeepseekV41B12xMetadata | None" = None
    swa_replay_start: torch.Tensor | None = None
    swa_prefix_replay_start: torch.Tensor | None = None


class DeepseekV41B12xMetadataBuilder(AttentionMetadataBuilder[DeepseekV41B12xMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, aphrodite_config, device):
        super().__init__(kv_cache_spec, layer_names, aphrodite_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.tokens = aphrodite_config.scheduler_config.max_num_batched_tokens
        self.requests = aphrodite_config.scheduler_config.max_num_seqs
        self.ratio = int(kv_cache_spec.tokens_per_state)
        self.page = int(kv_cache_spec.num_states)
        self.circular = isinstance(kv_cache_spec, CircularBufferSpec)
        if self.ratio not in (1, 2) or self.page <= 0:
            raise ValueError("V4.1 metadata requires ratio1/2 and nonempty pages")

        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        self.starts = alloc((self.requests + 1,), torch.int32)
        self.request_positions = alloc((self.requests,), torch.int64)
        self.counts = alloc((2,), torch.int32)
        self.positions = alloc((self.tokens,), torch.int64)
        self.reqs = alloc((self.tokens,), torch.int32)
        self.slots = alloc((self.tokens,), torch.int64)
        self.lengths = alloc((self.tokens,), torch.int32)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build=False,
    ):
        cm = common_attn_metadata
        table = cm.block_table_tensor
        # num_actual_tokens already includes the selected graph's padding.
        # Retain capacity-sized storage, but refresh only that readable domain.
        _tokens[(max(1, triton.cdiv(cm.num_actual_tokens, 128)),)](
            cm.query_start_loc,
            cm.seq_lens,
            table,
            cm.slot_mapping,
            self.positions,
            self.reqs,
            self.slots,
            self.lengths,
            self.starts,
            self.request_positions,
            self.counts,
            cm.num_reqs,
            cm.num_actual_tokens,
            table.stride(0),
            table.shape[1],
            self.requests,
            self.tokens,
            self.page,
            self.ratio,
            128,
            self.circular,
        )
        return DeepseekV41B12xMetadata(
            cm.num_actual_tokens,
            cm.num_reqs,
            cm.max_query_len,
            self.page,
            table,
            self.starts,
            self.request_positions,
            self.counts,
            self.positions,
            self.reqs,
            self.slots,
            self.lengths,
            self.reorder_batch_threshold is not None and cm.max_query_len <= self.reorder_batch_threshold,
            cm.max_seq_len,
        )


class DeepseekV41B12xBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "fp8", "fp8_ds_mla"]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # SWA, main KV, index keys, and partial states have different byte pages.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_name():
        return "B12X_DSV41"

    @staticmethod
    def get_builder_cls():
        return DeepseekV41B12xMetadataBuilder

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("V4.1 native attention runs through DeepseekV41B12xAttention")

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(32)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size):
        return MAIN_BLOCK_SIZE

    @classmethod
    def get_supported_head_sizes(cls):
        return [68, 512, 1024]

    @classmethod
    def is_mla(cls):
        return True

    @classmethod
    def is_sparse(cls):
        return True

    @classmethod
    def supports_sink(cls):
        return True

    @classmethod
    def supports_sliding_window(cls):
        return True

    @classmethod
    def supports_compute_capability(cls, capability):
        return capability.major == 12
