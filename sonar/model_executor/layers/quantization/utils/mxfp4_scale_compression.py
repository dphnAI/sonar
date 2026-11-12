# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless storage compression for native-layout MXFP4 expert scale bytes.

The encoder reads each expert's w13/w2 byte rows as one sequence without copying
them. Two-bit codes represent values 120 through 122 or mark a literal byte.
Each group of 128 values has an int32 literal offset for independent decoding.
Encoding synchronizes once to size the literal allocation. Decoding uses only
caller-owned storage and is safe to capture in a CUDA graph after warmup.
"""

from dataclasses import dataclass
from math import prod

import torch

from sonar.triton_utils import tl, triton

MAX_SELECTED_EXPERTS = 128


@triton.jit
def _pack_codes(
    W13,
    W2,
    Packed,
    Counts,
    N13: tl.constexpr,
    N2: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUPS: tl.constexpr,
):
    expert = tl.program_id(1)
    group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    index = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    a = tl.load(W13 + expert * N13 + index, index < N13, other=120)
    b = tl.load(W2 + expert * N2 + index - N13, (index >= N13) & (index < N13 + N2), other=120)
    value = tl.where(index < N13, a, b).to(tl.int32)
    escaped = (value < 120) | (value > 122)
    code = tl.where(escaped, 3, value - 120)
    codes = tl.reshape(code, (GROUPS, 32, 4))
    shifts = tl.arange(0, 4)[None, None, :] * 2
    packed = tl.sum(codes << shifts, axis=2)
    word = (expert * GROUP_COUNT + group[:, None]) * 32 + tl.arange(0, 32)[None, :]
    tl.store(Packed + word, packed, group[:, None] < GROUP_COUNT)
    count = tl.sum(escaped.to(tl.int32), axis=1)
    tl.store(Counts + expert * GROUP_COUNT + group, count, group < GROUP_COUNT)


@triton.jit
def _pack_literals(
    W13,
    W2,
    Offsets,
    Literals,
    N13: tl.constexpr,
    N2: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUPS: tl.constexpr,
):
    expert = tl.program_id(1)
    group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    index = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    a = tl.load(W13 + expert * N13 + index, index < N13, other=120)
    b = tl.load(W2 + expert * N2 + index - N13, (index >= N13) & (index < N13 + N2), other=120)
    value = tl.where(index < N13, a, b)
    escaped = ((value < 120) | (value > 122)) & (index < N13 + N2)
    rank = tl.cumsum(escaped.to(tl.int32), axis=1) - 1
    start = tl.load(Offsets + expert * GROUP_COUNT + group, group < GROUP_COUNT, other=0)
    tl.store(Literals + start[:, None] + rank, value, escaped)


@triton.jit
def _decode_scales(
    Packed,
    Offsets,
    Literals,
    Destination,
    ExpertIds,
    ExpertIdsOut,
    N13: tl.constexpr,
    N2: tl.constexpr,
    EXPERTS: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    SELECTED: tl.constexpr,
    WRITE_IDS: tl.constexpr,
    EXPERT_OFFSET: tl.constexpr,
    ID_BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    slot = tl.program_id(1)
    if SELECTED:
        global_expert = tl.load(ExpertIds + slot).to(tl.int32)
        if WRITE_IDS and tl.program_id(0) == 0:
            tl.store(ExpertIdsOut + slot, global_expert)
        expert = global_expert - EXPERT_OFFSET
        earlier = tl.arange(0, ID_BLOCK)
        ids = tl.load(ExpertIds + earlier, earlier < slot, other=-1).to(tl.int32)
        first = tl.sum(((earlier < slot) & (ids == global_expert)).to(tl.int32), axis=0) == 0
        active = first & (expert >= 0) & (expert < EXPERTS)
    else:
        expert = slot
        active = True
    if active:
        first_group = tl.program_id(0) * GROUPS
        group = first_group + tl.arange(0, GROUPS)
        word = group[:, None] * 32 + tl.arange(0, 32)[None, :]
        valid = word * 4 < N13 + N2
        p = tl.load(Packed + expert * GROUP_COUNT * 32 + word, valid, other=0).to(tl.uint32)
        tile_start = tl.load(Offsets + expert * GROUP_COUNT + first_group)
        tile_end = tl.load(Offsets + expert * GROUP_COUNT + tl.minimum(first_group + GROUPS, GROUP_COUNT))
        # Tiles without literals expand directly, avoiding dependent gathers
        # and the warp prefix scan.
        result = ((p & 3) | ((p & 12) << 6) | ((p & 48) << 12) | ((p & 192) << 18)) + 0x78787878
        if tile_start != tile_end:
            escaped_bits = p & (p >> 1) & 0x55
            count = tl.inline_asm_elementwise(
                "popc.b32 $0, $1;",
                constraints="=r,r",
                args=[escaped_bits],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            prefix = tl.cumsum(count, axis=1) - count
            start = tl.load(Offsets + expert * GROUP_COUNT + group, group < GROUP_COUNT, other=0)
            rank = start[:, None] + prefix
            result = tl.full((GROUPS, 32), 0, tl.uint32)
            for k in tl.static_range(4):
                code = (p >> (k * 2)) & 3
                escaped = code == 3
                raw = tl.load(Literals + rank, valid & escaped, other=0).to(tl.uint32)
                value = tl.where(escaped, raw, 120 + code)
                rank += escaped.to(tl.int32)
                result |= value << (k * 8)
        if N13 % 4 == 0 and N2 % 4 == 0:
            destination = tl.where(
                word < N13 // 4, expert * (N13 // 4) + word, EXPERTS * (N13 // 4) + expert * (N2 // 4) + word - N13 // 4
            )
            output = tl.cast(Destination, tl.pointer_type(tl.uint32))
            tl.store(output + destination, result, valid)
        else:
            for k in tl.static_range(4):
                index = word * 4 + k
                destination = tl.where(index < N13, expert * N13 + index, EXPERTS * N13 + expert * N2 + index - N13)
                tl.store(Destination + destination, (result >> (k * 8)) & 255, index < N13 + N2)


@dataclass(frozen=True)
class CompressedMXFP4Scales:
    """Compressed buffers without references to the original scale tensors."""

    packed: torch.Tensor
    offsets: torch.Tensor
    literals: torch.Tensor
    w13_shape: tuple[int, ...]
    w2_shape: tuple[int, ...]
    raw_nbytes: int
    encoded_nbytes: int

    def _check_destination(self, scratch: torch.Tensor) -> None:
        if (
            scratch.dtype != torch.uint8
            or not scratch.is_contiguous()
            or scratch.device != self.packed.device
            or scratch.numel() < self.raw_nbytes
        ):
            raise ValueError("scale scratch must be contiguous uint8 on the codec device with raw_nbytes capacity")

    def destination_views(self, scratch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """View decoded storage in the original native layouts."""
        self._check_destination(scratch)
        flat = scratch.view(-1)
        split = prod(self.w13_shape)
        return (flat[:split].view(self.w13_shape), flat[split : self.raw_nbytes].view(self.w2_shape))

    def decode_into(
        self,
        scratch: torch.Tensor,
        expert_ids: torch.Tensor | None = None,
        expert_offset: int = 0,
        expert_ids_out: torch.Tensor | None = None,
    ) -> None:
        """Decode all experts or at most 128 IDs into their original expert slots.

        Subtract expert_offset from each global ID to obtain its local slot.
        Optional expert_ids_out receives every input ID cast to int32, including
        duplicates and invalid IDs, in the same launch as scale decoding.
        Duplicate IDs have a single writer. Unselected slots are untouched,
        and IDs outside the local expert range are ignored. Decoding requires
        no GPU allocation or host synchronization. Keep scratch alive until
        its consumer finishes.
        """
        self._check_destination(scratch)
        experts = self.w13_shape[0]
        n13, n2 = prod(self.w13_shape[1:]), prod(self.w2_shape[1:])
        selected = expert_ids is not None
        if expert_ids is not None:
            if (
                expert_ids.dtype not in (torch.int16, torch.int32)
                or expert_ids.device != self.packed.device
                or expert_ids.ndim not in (1, 2)
                or not expert_ids.is_contiguous()
                or expert_ids.numel() > MAX_SELECTED_EXPERTS
            ):
                raise ValueError(
                    "expert IDs must be contiguous 1D/2D int16/int32 on the codec device, with at most 128 entries"
                )
            count = expert_ids.numel()
        else:
            count = experts
        if expert_ids_out is not None and (
            expert_ids is None
            or expert_ids_out.dtype != torch.int32
            or expert_ids_out.device != self.packed.device
            or expert_ids_out.shape != expert_ids.shape
            or not expert_ids_out.is_contiguous()
        ):
            raise ValueError("expert_ids_out must be contiguous int32 with the same shape/device as expert_ids")
        if count == 0:
            return
        group_count = triton.cdiv(n13 + n2, 128)
        groups = 8 if selected and count <= 16 else 16
        _decode_scales[(triton.cdiv(group_count, groups), count)](
            self.packed,
            self.offsets,
            self.literals,
            scratch,
            expert_ids if selected else scratch,
            expert_ids_out if expert_ids_out is not None else scratch,
            n13,
            n2,
            experts,
            group_count,
            selected,
            expert_ids_out is not None,
            expert_offset,
            triton.next_power_of_2(count) if selected else 1,
            groups,
            num_warps=4,
        )


def encode_mxfp4_scales(w13: torch.Tensor, w2: torch.Tensor) -> CompressedMXFP4Scales:
    """Pack native-layout scale storage bytes on GPU during model loading.

    Input dtypes are viewed as uint8, never numerically converted. Temporary
    counts and a prefix sum are discarded after packing. One loading-time host
    synchronization obtains the exact literal buffer size.
    """
    if (
        not w13.is_cuda
        or w13.device != w2.device
        or w13.ndim < 2
        or w2.ndim < 2
        or not w13.is_contiguous()
        or not w2.is_contiguous()
        or w13.shape[0] != w2.shape[0]
        or w13.numel() == 0
        or w2.numel() == 0
    ):
        raise ValueError("scales must be nonempty contiguous CUDA tensors with a matching expert dimension")
    a, b = w13.view(torch.uint8), w2.view(torch.uint8)
    shape13, shape2 = tuple(a.shape), tuple(b.shape)
    experts = shape13[0]
    n13, n2 = prod(shape13[1:]), prod(shape2[1:])
    raw_nbytes = a.numel() + b.numel()
    if raw_nbytes > torch.iinfo(torch.int32).max:
        raise ValueError("a scale pair must fit int32 literal offsets")
    groups = triton.cdiv(n13 + n2, 128)
    packed = torch.empty((experts, groups, 32), dtype=torch.uint8, device=a.device)
    counts = torch.empty(experts * groups, dtype=torch.int32, device=a.device)
    offsets = torch.empty(experts * groups + 1, dtype=torch.int32, device=a.device)
    grid = (triton.cdiv(groups, 8), experts)
    _pack_codes[grid](a, b, packed, counts, n13, n2, groups, 8, num_warps=4)
    offsets[0].zero_()
    torch.cumsum(counts, dim=0, dtype=torch.int32, out=offsets[1:])
    literal_count = int(offsets[-1].item())
    literals = torch.empty(max(1, literal_count), dtype=torch.uint8, device=a.device)
    _pack_literals[grid](a, b, offsets, literals, n13, n2, groups, 8, num_warps=4)
    return CompressedMXFP4Scales(
        packed=packed,
        offsets=offsets,
        literals=literals,
        w13_shape=shape13,
        w2_shape=shape2,
        raw_nbytes=raw_nbytes,
        encoded_nbytes=packed.numel() + offsets.numel() * offsets.element_size() + literals.numel(),
    )
