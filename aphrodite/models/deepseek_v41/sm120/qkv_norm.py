# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors
# Adapted from FlashInfer 0.7.2 bundled b12x/norm/hyperconnection/_cute.py,
# b12x/_lib/quant/mxfp8_rows.py and
# b12x/attention/compressed_sparse_mla/rotary.py.
# Modified to fuse strided Q/KV normalization, native MXFP8 Q quantization and
# KV RoPE.
"""Native SM120 Q/KV normalization with a shared BF16 Q quantization seam."""

import hashlib
import inspect

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from b12x._lib import compiler, intrinsics, utils
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x.norm.hyperconnection import _cute as provider
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from aphrodite.utils.b12x import PreparationResourceUnavailableError

Q_WIDTH = 1280
KV_WIDTH = 512
ROW_STRIDE = Q_WIDTH + KV_WIDTH
ROPE_DIM = 64
ROPE_START = KV_WIDTH - ROPE_DIM
THREADS = 256
# kv_norm gives each thread two KV columns and pairs RoPE columns across lanes.
assert 2 * THREADS == KV_WIDTH and ROPE_START % 32 == 0
_COMPILED: dict[int, tuple[object, object, object]] = {}


class _QKVNormQuant:
    def __init__(self, subgroup_width: int, wide_rows: bool = False):
        self.subgroup_width = subgroup_width
        self.wide_rows = wide_rows

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        q_weight: cute.Pointer,
        kv_weight: cute.Pointer,
        kv_output: cute.Pointer,
        q_values: cute.Pointer,
        q_scale_mma: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        table_stride: Int64,
        eps: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        values_u32 = cute.make_tensor(q_values, cute.make_ordered_layout((rows, Q_WIDTH // 4), order=(1, 0)))
        scale_mma = cute.make_tensor(q_scale_mma, cute.make_layout((10 * 512,)))
        self.kernel(
            source, q_weight, kv_weight, kv_output, values_u32, scale_mma, positions, cos_sin, table_stride, eps
        ).launch(
            grid=(rows, 2, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Pointer,
        q_weight: cute.Pointer,
        kv_weight: cute.Pointer,
        kv_output: cute.Pointer,
        values_u32: cute.Tensor,
        scale_mma: cute.Tensor,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        table_stride: Int64,
        eps: Float32,
    ) -> None:
        row, segment, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        if segment == 0:
            self.q_norm_quant(source, q_weight, values_u32, scale_mma, eps, Int32(row), Int32(thread))
        else:
            self.kv_norm(source, kv_weight, kv_output, positions, cos_sin, table_stride, eps, Int64(row), Int32(thread))

    @cute.jit
    def q_norm_quant(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        values_u32: cute.Tensor,
        scale_mma: cute.Tensor,
        eps: Float32,
        row: Int32,
        thread: Int32,
    ) -> None:
        allocator = cutlass.utils.SmemAllocator()
        rounded = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=cute.make_layout((Q_WIDTH,)),
            byte_alignment=16,
        )
        reduction = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1, THREADS // 32)),
            byte_alignment=16,
        )
        inverse = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        # Preserve the native per-thread items, reduction and FP32 operation order.
        values = cute.make_rmem_tensor((5,), Float32)
        square_sum = Float32(0.0)
        base = row.to(Int64) * Int64(ROW_STRIDE)
        for item in cutlass.range_constexpr(5):
            column = thread + Int32(item * THREADS)
            value = Float32(0.0)
            if column < Int32(Q_WIDTH):
                value = Float32(source[base + column.to(Int64)])
            values[item] = value
            square_sum += value * value
        total = provider._block_sum(square_sum, reduction)
        if thread == Int32(0):
            inverse[0] = cute.math.rsqrt(total / Float32(Q_WIDTH) + eps, fastmath=True)
        cute.arch.sync_threads()
        scale = Float32(inverse[0])
        for item in cutlass.range_constexpr(5):
            column = thread + Int32(item * THREADS)
            if column < Int32(Q_WIDTH):
                learned = Float32(weight[column.to(Int64)])
                rounded[column] = BFloat16(values[item] * scale * (Float32(0.0) + learned))
        # Quantization must read the actual BF16-rounded normalized values.
        cute.arch.sync_threads()
        lane = thread % Int32(self.subgroup_width)
        groups_per_round = THREADS // self.subgroup_width
        for tile in cutlass.range_constexpr((40 + groups_per_round - 1) // groups_per_round):
            group = thread // Int32(self.subgroup_width) + Int32(tile * groups_per_round)
            if group < Int32(40):
                if cutlass.const_expr(self.subgroup_width == 8):
                    values4 = cute.make_rmem_tensor((4,), Float32)
                    k0 = group * Int32(32) + lane * Int32(4)
                    for elem in cutlass.range_constexpr(4):
                        values4[elem] = Float32(rounded[k0 + Int32(elem)])
                    max_abs = intrinsics.fabs_f32(values4[0])
                    for elem in cutlass.range_constexpr(1, 4):
                        max_abs = intrinsics.fmax_f32(max_abs, intrinsics.fabs_f32(values4[elem]))
                    for shift in cutlass.range_constexpr(3):
                        max_abs = intrinsics.fmax_f32(
                            max_abs,
                            cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
                        )
                    max_abs = intrinsics.fmax_f32(max_abs, Float32(1e-4))
                    _, scale_byte = intrinsics.pow2_ceil_ue8m0(max_abs * Float32(1.0 / intrinsics.FLOAT8_E4M3_MAX))
                    inv_scale = intrinsics.ue8m0_to_output_scale(scale_byte)
                    values_u32[row, group * Int32(8) + lane] = intrinsics.cvt_f32x4_to_e4m3x4(
                        values4[0] * inv_scale,
                        values4[1] * inv_scale,
                        values4[2] * inv_scale,
                        values4[3] * inv_scale,
                    )
                else:
                    values8 = cute.make_rmem_tensor((8,), Float32)
                    k0 = group * Int32(32) + lane * Int32(8)
                    for elem in cutlass.range_constexpr(8):
                        values8[elem] = Float32(rounded[k0 + Int32(elem)])
                    max_abs = intrinsics.fabs_f32(values8[0])
                    for elem in cutlass.range_constexpr(1, 8):
                        max_abs = intrinsics.fmax_f32(max_abs, intrinsics.fabs_f32(values8[elem]))
                    for shift in cutlass.range_constexpr(2):
                        max_abs = intrinsics.fmax_f32(
                            max_abs,
                            cute.arch.shuffle_sync_bfly(max_abs, offset=1 << shift),
                        )
                    max_abs = intrinsics.fmax_f32(max_abs, Float32(1e-4))
                    _, scale_byte = intrinsics.pow2_ceil_ue8m0(max_abs * Float32(1.0 / intrinsics.FLOAT8_E4M3_MAX))
                    if max_abs == Float32(0.0):
                        scale_byte = Uint32(127)
                    inv_scale = intrinsics.ue8m0_to_output_scale(scale_byte)
                    word0 = group * Int32(8) + lane * Int32(2)
                    values_u32[row, word0] = intrinsics.cvt_f32x4_to_e4m3x4(
                        values8[0] * inv_scale,
                        values8[1] * inv_scale,
                        values8[2] * inv_scale,
                        values8[3] * inv_scale,
                    )
                    values_u32[row, word0 + Int32(1)] = intrinsics.cvt_f32x4_to_e4m3x4(
                        values8[4] * inv_scale,
                        values8[5] * inv_scale,
                        values8[6] * inv_scale,
                        values8[7] * inv_scale,
                    )
                if lane == Int32(0):
                    # Native F8_128x4 physical offset; decode stays in tile_m=0.
                    row32 = row % Int32(32)
                    row4 = (row // Int32(32)) % Int32(4)
                    k4 = group % Int32(4)
                    tile_k = group // Int32(4)
                    offset = row32 * Int32(16) + row4 * Int32(4) + k4 + tile_k * Int32(512)
                    if cutlass.const_expr(self.wide_rows):
                        offset += (row // Int32(128)) * Int32(10 * 512)
                    scale_mma[offset] = Uint8(scale_byte)

    @cute.jit
    def kv_norm(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        table_stride: Int64,
        eps: Float32,
        row: Int64,
        thread: Int32,
    ) -> None:
        base = row * Int64(ROW_STRIDE) + Int64(Q_WIDTH)
        output_base = row * Int64(KV_WIDTH)
        values = cute.make_rmem_tensor((2,), Float32)
        square_sum = Float32(0.0)
        for item in cutlass.range_constexpr(2):
            column = thread + Int32(item * THREADS)
            value = Float32(0.0)
            if column < Int32(KV_WIDTH):
                value = Float32(source[base + column.to(Int64)])
            values[item] = value
            square_sum += value * value
        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1, THREADS // 32)),
            byte_alignment=16,
        )
        inverse = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        total = provider._block_sum(square_sum, reduction)
        if thread == Int32(0):
            inverse[0] = cute.math.rsqrt(total / Float32(KV_WIDTH) + eps, fastmath=True)
        cute.arch.sync_threads()
        scale = Float32(inverse[0])
        position = Int64(positions[row])
        for item in cutlass.range_constexpr(2):
            column = thread + Int32(item * THREADS)
            learned = Float32(weight[column.to(Int64)])
            # The native rotation reads the BF16-rounded normalized KV.
            value = Float32(BFloat16(values[item] * scale * (Float32(0.0) + learned)))
            if cutlass.const_expr(item * THREADS >= ROPE_START - THREADS):
                # RoPE columns live in the last warps; each pair is adjacent lanes.
                partner = cute.arch.shuffle_sync_bfly(value, offset=1)
                # Nested: CuTe DSL does not trace Python `and` on runtime values.
                if position >= Int64(0):  # noqa: SIM102
                    if column >= Int32(ROPE_START):
                        local = column - Int32(ROPE_START)
                        cosine = Float32(cos_sin[position * table_stride + Int64(local // 2)])
                        sine = Float32(cos_sin[position * table_stride + Int64(ROPE_DIM // 2 + local // 2)])
                        sign = Float32(-1.0)
                        if local % 2 == 1:
                            sign = Float32(1.0)
                        value = value * cosine + sign * partner * sine
            if position < Int64(0):
                value = Float32(0.0)
            output[output_base + column.to(Int64)] = BFloat16(value)


_KERNEL_SHA = hashlib.sha256(inspect.getsource(_QKVNormQuant).encode()).hexdigest()


def _allocate_outputs(rows, device):
    values = torch.empty((rows, Q_WIDTH), dtype=torch.float8_e4m3fn, device=device)
    physical = torch.empty((1, (rows + 127) // 128, 10, 32, 4, 4), dtype=torch.uint8, device=device)
    scale_mma = physical.permute(3, 4, 1, 5, 2, 0).view(torch.float8_e8m0fnu)
    kv = torch.empty((rows, KV_WIDTH), dtype=torch.bfloat16, device=device)
    return values, scale_mma, kv


def _run(compiled, source, q_weight, kv_weight, values, scale_mma, kv, positions, cos_sin_cache, eps):
    run_compiled(
        compiled,
        (
            provider._pointer(source),
            provider._pointer(q_weight),
            provider._pointer(kv_weight),
            provider._pointer(kv),
            utils.make_ptr(Uint32, values.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
            utils.make_ptr(Uint8, scale_mma.data_ptr(), cute.AddressSpace.gmem, assumed_align=1),
            utils.make_ptr(Int64, positions.data_ptr(), cute.AddressSpace.gmem, assumed_align=8),
            utils.make_ptr(Float32, cos_sin_cache.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
            Int64(cos_sin_cache.stride(0)),
            float(eps),
            source.shape[0],
            utils.current_cuda_stream(),
        ),
    )


def prepare_qkv_norm_quant(device: torch.device) -> None:
    """Compile and real-warm decode and multi-tile prefill before freeze/capture."""
    ordinal = device.index
    assert ordinal is not None
    if ordinal in _COMPILED:
        return
    with torch.accelerator.device_index(ordinal):
        if torch.cuda.is_current_stream_capturing():
            raise PreparationResourceUnavailableError(
                "V4.1 Q/KV normalization and quantization must be prepared before capture"
            )
        raise_if_kernel_resolution_frozen("qkv_norm_quant.prepare")
        source = torch.ones((256, ROW_STRIDE), dtype=torch.bfloat16, device=device)
        q_weight = torch.ones(Q_WIDTH, dtype=torch.bfloat16, device=device)
        kv_weight = torch.ones(KV_WIDTH, dtype=torch.bfloat16, device=device)
        positions = torch.zeros(256, dtype=torch.int64, device=device)
        cos_sin_cache = torch.ones((1, ROPE_DIM), dtype=torch.float32, device=device)
        values, scale_mma, kv = _allocate_outputs(256, device)
        launchers = []
        for subgroup, rows, wide_rows in ((8, 1, False), (4, 64, False), (4, 256, True)):
            compiled = compiler.compile(
                _QKVNormQuant(subgroup, wide_rows),
                *(provider._fake_pointer() for _ in range(4)),
                utils.make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4),
                utils.make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=1),
                utils.make_ptr(Int64, 16, cute.AddressSpace.gmem, assumed_align=8),
                utils.make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
                Int64(1),
                Float32(1.0e-6),
                Int32(1),
                utils.current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "aphrodite.dsv41.qkv_norm_quant",
                    1,
                    (Q_WIDTH, KV_WIDTH, ROW_STRIDE, ROPE_DIM, THREADS, subgroup, wide_rows, _KERNEL_SHA),
                ),
            )
            _run(
                compiled,
                source[:rows],
                q_weight,
                kv_weight,
                values[:rows],
                scale_mma,
                kv[:rows],
                positions[:rows],
                cos_sin_cache,
                1.0e-6,
            )
            launchers.append(compiled)
        torch.accelerator.current_stream().synchronize()
        _COMPILED[ordinal] = (launchers[0], launchers[1], launchers[2])


def fused_qkv_norm_quant(
    source: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    capacity: int,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contiguous MXFP8 Q, canonical F8_128x4 scales and rotated BF16 KV.

    Source is contiguous BF16[M,1792], weights are separate BF16[1280]/BF16[512],
    and capacity is the actual selected original Q-plan capacity, not live M.
    KV is normalized, rounded to BF16, then GPT-J rotated over its last 64
    columns with the FP32 cos/sin table, exactly as the native KV rotation.
    KV rows with a negative position are zero.
    """
    rows = source.shape[0]
    if not 1 <= rows <= capacity:
        raise ValueError("V4.1 fused Q/KV norm/quant requires positive M within capacity")
    if positions.dtype != torch.int64 or positions.shape != (rows,) or not positions.is_contiguous():
        raise ValueError("V4.1 fused KV rotation requires contiguous int64 positions[M]")
    if (
        cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] < ROPE_DIM
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("V4.1 fused KV rotation requires an FP32 [positions, 64] cos/sin table")
    launchers = _COMPILED.get(source.device.index)
    if launchers is None:
        raise PreparationResourceUnavailableError("V4.1 Q/KV normalization and quantization was not prepared")
    values, scale_mma, kv = _allocate_outputs(rows, source.device)
    _run(
        launchers[2 if rows > 128 else 0 if capacity <= 8 else 1],
        source,
        q_weight,
        kv_weight,
        values,
        scale_mma,
        kv,
        positions,
        cos_sin_cache,
        eps,
    )
    return values, scale_mma, kv
