# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors
# Adapted from FlashInfer 0.7.2 bundled b12x/norm/mhc/_kernels.py
# (MHCPostPrePartialKernel and MHCFinalizeGramKernel, prepared lagged decode).
# Modified to split the critical-path output from the next coefficients.
"""Lagged mHC post_pre with the next coefficients off the critical path.

V4.1 collapses each sublayer input with the previous boundary's pre-mix, so a
boundary's mix projection and Sinkhorn only produce coefficients consumed at
the next boundary. The critical path keeps post, the lagged collapse and its
RMSNorm. The coefficient pass runs on a side stream that overlaps the next
sublayer; every later mHC operation joins it before reading the coefficients.

Each kernel keeps b12x's prepared lagged decode expressions, roundings and
reduction orders, so all five outputs match its fused post_pre bit for bit.
"""

import hashlib
import inspect

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from b12x._lib import compiler, utils
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x.norm.hyperconnection import _cute as provider
from cutlass import BFloat16, Float32, Int32, Int64

from aphrodite.utils.b12x import PreparationResourceUnavailableError

MULT = 4
MIXES = 24
PARTIALS = 1 + MIXES
SOURCE_TILE = 128
SOURCE_WARPS = SOURCE_TILE // 32
PARTIALS_PER_CTA = 4
PARTIAL_GROUPS = (PARTIALS + PARTIALS_PER_CTA - 1) // PARTIALS_PER_CTA
NORM_THREADS = 1024
COEFFICIENT_THREADS = 64
MAX_TOKENS = 64


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _contract_four(
    p0: Float32,
    p1: Float32,
    p2: Float32,
    p3: Float32,
    r0: Float32,
    r1: Float32,
    r2: Float32,
    r3: Float32,
) -> Float32:
    return p0 * r0 + p1 * r1 + p2 * r2 + p3 * r3


def _rows(pointer, rows, *tail):
    shape = (rows, *tail)
    return cute.make_tensor(pointer, cute.make_ordered_layout(shape, order=tuple(reversed(range(len(shape))))))


class _PostCollapse:
    """Post-mix the four streams and collapse them with the carried pre-mix."""

    def __init__(self, hidden: int):
        self.hidden = hidden
        self.tiles = hidden // SOURCE_TILE

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        residual: cute.Pointer,
        prev_post: cute.Pointer,
        prev_comb: cute.Pointer,
        pre_mix: cute.Pointer,
        out: cute.Pointer,
        y: cute.Pointer,
        squares: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _rows(x, rows, self.hidden),
            _rows(residual, rows, MULT, self.hidden),
            _rows(prev_post, rows, MULT),
            _rows(prev_comb, rows, MULT, MULT),
            _rows(pre_mix, rows, MULT),
            _rows(out, rows, MULT, self.hidden),
            _rows(y, rows, self.hidden),
            _rows(squares, rows, self.tiles),
        ).launch(grid=(self.tiles, rows, 1), block=(SOURCE_TILE, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        pre_mix: cute.Tensor,
        out: cute.Tensor,
        y: cute.Tensor,
        squares: cute.Tensor,
    ) -> None:
        tile, token, _ = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        token = Int64(token)
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        allocator = cutlass.memory.SmemAllocator()
        warp_sums = allocator.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((SOURCE_WARPS,)), byte_alignment=16
        )
        h = Int64(tile) * Int64(SOURCE_TILE) + Int64(tidx)
        r0 = Float32(residual[token, Int32(0), h])
        r1 = Float32(residual[token, Int32(1), h])
        r2 = Float32(residual[token, Int32(2), h])
        r3 = Float32(residual[token, Int32(3), h])
        xh = Float32(x[token, h])
        o0 = (
            Float32(prev_post[token, Int32(0)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(0)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(0)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(0)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(0)]) * r3
        ).to(BFloat16)
        o1 = (
            Float32(prev_post[token, Int32(1)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(1)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(1)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(1)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(1)]) * r3
        ).to(BFloat16)
        o2 = (
            Float32(prev_post[token, Int32(2)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(2)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(2)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(2)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(2)]) * r3
        ).to(BFloat16)
        o3 = (
            Float32(prev_post[token, Int32(3)]) * xh
            + Float32(prev_comb[token, Int32(0), Int32(3)]) * r0
            + Float32(prev_comb[token, Int32(1), Int32(3)]) * r1
            + Float32(prev_comb[token, Int32(2), Int32(3)]) * r2
            + Float32(prev_comb[token, Int32(3), Int32(3)]) * r3
        ).to(BFloat16)
        out[token, Int32(0), h] = o0
        out[token, Int32(1), h] = o1
        out[token, Int32(2), h] = o2
        out[token, Int32(3), h] = o3
        collapsed = _contract_four(
            Float32(pre_mix[token, Int32(0)]),
            Float32(pre_mix[token, Int32(1)]),
            Float32(pre_mix[token, Int32(2)]),
            Float32(pre_mix[token, Int32(3)]),
            Float32(o0),
            Float32(o1),
            Float32(o2),
            Float32(o3),
        ).to(BFloat16)
        y[token, h] = collapsed
        square = _warp_sum(Float32(collapsed) * Float32(collapsed))
        if lane == Int32(0):
            warp_sums[warp] = square
        cute.arch.sync_threads()
        if tidx == Int32(0):
            total = Float32(0.0)
            for source_warp in cutlass.range_constexpr(SOURCE_WARPS):
                total += Float32(warp_sums[source_warp])
            squares[token, Int32(tile)] = total


class _Normalize:
    """RMS-normalize the collapsed input in place from its per-tile squares."""

    def __init__(self, hidden: int, eps: float):
        self.hidden = hidden
        self.tiles = hidden // SOURCE_TILE
        self.tile_warps = (self.tiles + 31) // 32
        self.eps = eps

    @cute.jit
    def __call__(
        self,
        squares: cute.Pointer,
        weight: cute.Pointer,
        y: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _rows(squares, rows, self.tiles),
            cute.make_tensor(weight, cute.make_layout((self.hidden,))),
            _rows(y, rows, self.hidden),
        ).launch(grid=(self.hidden // NORM_THREADS, rows, 1), block=(NORM_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, squares: cute.Tensor, weight: cute.Tensor, y: cute.Tensor) -> None:
        group, token, _ = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        token = Int64(token)
        allocator = cutlass.memory.SmemAllocator()
        tile_sums = allocator.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((self.tile_warps,)), byte_alignment=16
        )
        scale = allocator.allocate_tensor(element_type=Float32, layout=cute.make_layout((1,)), byte_alignment=4)
        if tidx < Int32(self.tile_warps * 32):
            value = Float32(0.0)
            if tidx < Int32(self.tiles):
                value = Float32(squares[token, tidx])
            value = _warp_sum(value)
            if tidx % Int32(32) == Int32(0):
                tile_sums[tidx // Int32(32)] = value
        cute.arch.sync_threads()
        if tidx == Int32(0):
            total = Float32(0.0)
            for tile_warp in cutlass.range_constexpr(self.tile_warps):
                total += Float32(tile_sums[tile_warp])
            scale[0] = cute.math.rsqrt(total / Float32(self.hidden) + Float32(self.eps), fastmath=True)
        cute.arch.sync_threads()
        h = Int64(group) * Int64(NORM_THREADS) + Int64(tidx)
        value = Float32(y[token, h])
        value = value * Float32(scale[0]) * Float32(weight[h])
        y[token, h] = value.to(BFloat16)


class _MixPartials:
    """Per-tile squared norm and mix projections of the post-mixed streams."""

    def __init__(self, hidden: int):
        self.hidden = hidden
        self.tiles = hidden // SOURCE_TILE

    @cute.jit
    def __call__(
        self,
        residual: cute.Pointer,
        fn: cute.Pointer,
        partials: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _rows(residual, rows, MULT, self.hidden),
            cute.make_tensor(fn, cute.make_ordered_layout((MIXES, MULT * self.hidden), order=(1, 0))),
            _rows(partials, rows, self.tiles, PARTIALS),
        ).launch(grid=(self.tiles, PARTIAL_GROUPS, rows), block=(SOURCE_TILE, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, residual: cute.Tensor, fn: cute.Tensor, partials: cute.Tensor) -> None:
        tile, group, token = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        token = Int64(token)
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        allocator = cutlass.memory.SmemAllocator()
        warp_sums = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((PARTIALS_PER_CTA, SOURCE_WARPS), stride=(SOURCE_WARPS, 1)),
            byte_alignment=16,
        )
        first = group * Int32(PARTIALS_PER_CTA)
        h = Int64(tile) * Int64(SOURCE_TILE) + Int64(tidx)
        r0 = Float32(residual[token, Int32(0), h])
        r1 = Float32(residual[token, Int32(1), h])
        r2 = Float32(residual[token, Int32(2), h])
        r3 = Float32(residual[token, Int32(3), h])
        values = cute.make_rmem_tensor(cute.make_layout((PARTIALS_PER_CTA,), stride=(1,)), Float32)
        for slot in cutlass.range_constexpr(PARTIALS_PER_CTA):
            partial = first + Int32(slot)
            value = Float32(0.0)
            if partial == Int32(0):
                value = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
            elif partial < Int32(PARTIALS):
                mix = partial - Int32(1)
                value = (
                    Float32(fn[mix, h]) * r0
                    + Float32(fn[mix, Int64(self.hidden) + h]) * r1
                    + Float32(fn[mix, Int64(2 * self.hidden) + h]) * r2
                    + Float32(fn[mix, Int64(3 * self.hidden) + h]) * r3
                )
            values[slot] = _warp_sum(value)
        if lane == Int32(0):
            for slot in cutlass.range_constexpr(PARTIALS_PER_CTA):
                warp_sums[slot, warp] = values[slot]
        cute.arch.sync_threads()
        if tidx == Int32(0):
            for slot in cutlass.range_constexpr(PARTIALS_PER_CTA):
                total = Float32(0.0)
                for source_warp in cutlass.range_constexpr(SOURCE_WARPS):
                    total += Float32(warp_sums[slot, source_warp])
                partial = first + Int32(slot)
                if partial < Int32(PARTIALS):
                    partials[token, Int32(tile), partial] = total


class _Coefficients:
    """Turn the mix partials into post, comb and the next pre-mix."""

    def __init__(self, hidden: int, rms_eps: float, hc_eps: float, iterations: int):
        self.hidden = hidden
        self.tiles = hidden // SOURCE_TILE
        self.rms_eps = rms_eps
        self.hc_eps = hc_eps
        self.iterations = iterations

    @cute.jit
    def __call__(
        self,
        partials: cute.Pointer,
        scale: cute.Pointer,
        bias: cute.Pointer,
        post: cute.Pointer,
        comb: cute.Pointer,
        pre_out: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            _rows(partials, rows, self.tiles, PARTIALS),
            cute.make_tensor(scale, cute.make_layout((3,))),
            cute.make_tensor(bias, cute.make_layout((MIXES,))),
            _rows(post, rows, MULT),
            _rows(comb, rows, MULT, MULT),
            _rows(pre_out, rows, MULT),
        ).launch(grid=(rows, 1, 1), block=(COEFFICIENT_THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        pre_out: cute.Tensor,
    ) -> None:
        token, _, _ = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        token = Int64(token)
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        allocator = cutlass.memory.SmemAllocator()
        source_sums = allocator.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((2 * PARTIALS,)), byte_alignment=16
        )
        s_pre = allocator.allocate_tensor(element_type=Float32, layout=cute.make_layout((MULT,)), byte_alignment=16)
        sums = cute.make_rmem_tensor(cute.make_layout((PARTIALS,), stride=(1,)), Float32)
        source_tile = warp * Int32(32) + lane
        for column in cutlass.range_constexpr(PARTIALS):
            value = Float32(0.0)
            if source_tile < Int32(self.tiles):
                value = Float32(partials[token, source_tile, column])
            value = _warp_sum(value)
            if lane == Int32(0):
                source_sums[Int32(column * 2) + warp] = value
        cute.arch.sync_threads()
        if tidx == Int32(0):
            for column in cutlass.range_constexpr(PARTIALS):
                sums[column] = Float32(source_sums[Int32(column * 2)]) + Float32(source_sums[Int32(column * 2 + 1)])
            mixes = cute.make_rmem_tensor(cute.make_layout((MIXES,), stride=(1,)), Float32)
            for mix in cutlass.range_constexpr(MIXES):
                mixes[mix] = Float32(sums[mix + 1])
            inv_rms = cute.math.rsqrt(
                Float32(sums[0]) / Float32(MULT * self.hidden) + Float32(self.rms_eps),
                fastmath=True,
            )
            for mix in cutlass.range_constexpr(MIXES):
                mixes[mix] = mixes[mix] * inv_rms

            s0 = Float32(scale[0])
            s1 = Float32(scale[1])
            s2 = Float32(scale[2])
            one = Float32(1.0)
            two = Float32(2.0)
            eps = Float32(self.hc_eps)
            for index in cutlass.range_constexpr(MULT):
                s_pre[index] = (
                    one / (one + cute.math.exp(-(mixes[index] * s0 + Float32(bias[index])), fastmath=True)) + eps
                )
            for index in cutlass.range_constexpr(MULT):
                post[token, Int32(index)] = two / (
                    one
                    + cute.math.exp(
                        -(mixes[MULT + index] * s1 + Float32(bias[MULT + index])),
                        fastmath=True,
                    )
                )

            c = cute.make_rmem_tensor(cute.make_layout((MULT, MULT), stride=(MULT, 1)), Float32)
            for row in cutlass.range_constexpr(MULT):
                for col in cutlass.range_constexpr(MULT):
                    index = 2 * MULT + row * MULT + col
                    c[row, col] = mixes[index] * s2 + Float32(bias[index])
            for row in cutlass.range_constexpr(MULT):
                peak = c[row, 0]
                for col in cutlass.range_constexpr(1, MULT):
                    if c[row, col] > peak:
                        peak = c[row, col]
                for col in cutlass.range_constexpr(MULT):
                    c[row, col] = cute.math.exp(c[row, col] - peak, fastmath=True)
            for row in cutlass.range_constexpr(MULT):
                inv_row = cute.arch.rcp_approx(c[row, 0] + c[row, 1] + c[row, 2] + c[row, 3])
                for col in cutlass.range_constexpr(MULT):
                    c[row, col] = c[row, col] * inv_row + eps
            for col in cutlass.range_constexpr(MULT):
                inv_col = cute.arch.rcp_approx(c[0, col] + c[1, col] + c[2, col] + c[3, col] + eps)
                for row in cutlass.range_constexpr(MULT):
                    c[row, col] = c[row, col] * inv_col
            for _ in cutlass.range_constexpr(self.iterations - 1):
                for row in cutlass.range_constexpr(MULT):
                    inv_row = cute.arch.rcp_approx(c[row, 0] + c[row, 1] + c[row, 2] + c[row, 3] + eps)
                    for col in cutlass.range_constexpr(MULT):
                        c[row, col] = c[row, col] * inv_row
                for col in cutlass.range_constexpr(MULT):
                    inv_col = cute.arch.rcp_approx(c[0, col] + c[1, col] + c[2, col] + c[3, col] + eps)
                    for row in cutlass.range_constexpr(MULT):
                        c[row, col] = c[row, col] * inv_col
            for row in cutlass.range_constexpr(MULT):
                for col in cutlass.range_constexpr(MULT):
                    comb[token, Int32(row), Int32(col)] = c[row, col]
        cute.arch.sync_threads()
        if tidx < Int32(MULT):
            pre_out[token, tidx] = Float32(s_pre[tidx])


_KERNEL_SHA = hashlib.sha256(
    "".join(
        inspect.getsource(item)
        for item in (_warp_sum, _contract_four, _rows, _PostCollapse, _Normalize, _MixPartials, _Coefficients)
    ).encode()
).hexdigest()


class _Split:
    def __init__(self, device, hidden, rms_eps, hc_eps, iterations, launchers):
        self.recipe = (hidden, rms_eps, hc_eps, iterations)
        self.launchers = launchers
        self.stream = torch.cuda.Stream(device)
        self.ready = torch.cuda.Event()
        self.pending = False
        tiles = hidden // SOURCE_TILE
        self.squares = torch.empty((MAX_TOKENS, tiles), dtype=torch.float32, device=device)
        self.partials = torch.empty((MAX_TOKENS, tiles, PARTIALS), dtype=torch.float32, device=device)


_SPLITS: dict[int, _Split] = {}


def supported_hidden(hidden: int) -> bool:
    """Shapes whose b12x prepared lagged decode reductions this port matches."""
    tiles = hidden // SOURCE_TILE
    return hidden % NORM_THREADS == 0 and 32 < tiles <= COEFFICIENT_THREADS


def _compile(kernel, name, key, pointers):
    return compiler.compile(
        kernel,
        *(provider._fake_pointer(dtype) for dtype in pointers),
        Int32(1),
        utils.current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(f"aphrodite.dsv41.mhc_split.{name}", 1, (*key, _KERNEL_SHA)),
    )


def prepare_mhc_split(device: torch.device, hidden: int, rms_eps: float, hc_eps: float, iterations: int) -> None:
    """Compile and real-warm the split before freeze/capture; one recipe per device."""
    ordinal = device.index
    assert ordinal is not None
    recipe = (hidden, float(rms_eps), float(hc_eps), int(iterations))
    existing = _SPLITS.get(ordinal)
    if existing is not None:
        if existing.recipe != recipe:
            raise ValueError("V4.1 mHC split supports one numerical recipe per device")
        return
    if not supported_hidden(hidden):
        raise ValueError(f"V4.1 mHC split does not support hidden size {hidden}")
    with torch.accelerator.device_index(ordinal):
        if torch.cuda.is_current_stream_capturing():
            raise PreparationResourceUnavailableError("V4.1 mHC split must be prepared before capture")
        raise_if_kernel_resolution_frozen("mhc_split.prepare")
        bf16, f32 = torch.bfloat16, torch.float32
        launchers = {
            "post_collapse": _compile(
                _PostCollapse(hidden), "post_collapse", (hidden,), (bf16, bf16, f32, f32, f32, bf16, bf16, f32)
            ),
            "normalize": _compile(_Normalize(hidden, recipe[1]), "normalize", recipe[:2], (f32, bf16, bf16)),
            "mix_partials": _compile(
                _MixPartials(hidden), "mix_partials", (hidden, PARTIALS_PER_CTA), (bf16, f32, f32)
            ),
            "coefficients": _compile(_Coefficients(*recipe), "coefficients", recipe, (f32,) * 6),
        }
        split = _Split(device, hidden, *recipe[1:], launchers)
        rows = MAX_TOKENS
        x = torch.zeros((rows, hidden), dtype=bf16, device=device)
        residual = torch.zeros((rows, MULT, hidden), dtype=bf16, device=device)
        mix = torch.zeros((rows, MULT), dtype=f32, device=device)
        comb = torch.zeros((rows, MULT, MULT), dtype=f32, device=device)
        fn = torch.zeros((MIXES, MULT * hidden), dtype=f32, device=device)
        scale = torch.zeros(3, dtype=f32, device=device)
        bias = torch.zeros(MIXES, dtype=f32, device=device)
        weight = torch.ones(hidden, dtype=bf16, device=device)
        _critical(split, x, residual, mix, comb, mix, residual.clone(), x.clone(), weight)
        _coefficients(split, residual, fn, scale, bias, mix.clone(), comb.clone(), mix.clone())
        torch.accelerator.current_stream().synchronize()
        _SPLITS[ordinal] = split


def _critical(split, x, residual, prev_post, prev_comb, pre_mix, out, y, weight):
    rows = residual.shape[0]
    stream = utils.current_cuda_stream()
    pointer = provider._pointer
    run_compiled(
        split.launchers["post_collapse"],
        (
            pointer(x),
            pointer(residual),
            pointer(prev_post),
            pointer(prev_comb),
            pointer(pre_mix),
            pointer(out),
            pointer(y),
            pointer(split.squares),
            rows,
            stream,
        ),
    )
    run_compiled(split.launchers["normalize"], (pointer(split.squares), pointer(weight), pointer(y), rows, stream))


def _coefficients(split, residual, fn, scale, bias, post, comb, pre_out):
    rows = residual.shape[0]
    stream = utils.current_cuda_stream()
    pointer = provider._pointer
    run_compiled(
        split.launchers["mix_partials"], (pointer(residual), pointer(fn), pointer(split.partials), rows, stream)
    )
    run_compiled(
        split.launchers["coefficients"],
        (
            pointer(split.partials),
            pointer(scale),
            pointer(bias),
            pointer(post),
            pointer(comb),
            pointer(pre_out),
            rows,
            stream,
        ),
    )


def split_available(device: torch.device, rows: int, hidden: int) -> bool:
    split = _SPLITS.get(device.index)
    return split is not None and split.recipe[0] == hidden and 1 <= rows <= MAX_TOKENS


def join_mhc_split(device: torch.device) -> None:
    """Order the current stream after the pending coefficient pass."""
    split = _SPLITS.get(device.index)
    if split is not None and split.pending:
        torch.cuda.current_stream(device).wait_event(split.ready)
        split.pending = False


def lagged_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    pre_mix: torch.Tensor,
    residual_out: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre_out: torch.Tensor,
) -> None:
    """Write residual_out and normalized y now; post, comb and pre_out after a join.

    Inputs and outputs are contiguous: BF16 x/y[M,H], residual/residual_out
    [M,4,H] and weight[H]; FP32 prev_post/pre_mix/post/pre_out[M,4],
    prev_comb/comb[M,4,4], fn[24,4H], scale[3] and bias[24]. Callers must call
    ``join_mhc_split`` before reading post, comb or pre_out, or reusing them.
    """
    split = _SPLITS.get(residual.device.index)
    if split is None:
        raise PreparationResourceUnavailableError("V4.1 mHC split was not prepared")
    if split.pending:
        raise RuntimeError("V4.1 mHC split requires a join before the next boundary")
    rows, hidden = x.shape
    if not 1 <= rows <= MAX_TOKENS or hidden != split.recipe[0]:
        raise ValueError(f"V4.1 mHC split requires M1..{MAX_TOKENS} and its prepared hidden size")
    _critical(split, x, residual, prev_post, prev_comb, pre_mix, residual_out, y, weight)
    split.stream.wait_stream(torch.cuda.current_stream(residual.device))
    with torch.cuda.stream(split.stream):
        _coefficients(split, residual_out, fn, scale, bias, post, comb, pre_out)
        split.ready.record(split.stream)
    split.pending = True
