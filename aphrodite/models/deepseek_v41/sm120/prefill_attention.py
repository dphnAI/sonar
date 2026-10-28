# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 by the b12x authors
# Adapted from FlashInfer 0.7.2 bundled b12x/attention/_shared/mla.
"""Native-order H16 BF16 mixed-cache prefill for SM120.

Native cache bytes, 64-candidate tile order, BF16 MMA operands and the provider's
online-softmax/epilogue are retained. Each gathered K=V pair is decoded once into
shared memory, rather than independently in QK and PV. Indexed page mapping is
performed at gather time. This route is deliberately separate from decode.
"""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from b12x._lib.intrinsics import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_shared_u32,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    mxfp8_pair_to_bf16x2_sm120,
    nvfp4_pair_to_bf16x2_sm120,
    shared_ptr_to_u32,
    st_shared_u32,
    st_shared_v4_u32,
)
from b12x.attention._shared.cute.ops import LOG2_E
from b12x.attention._shared.mla.decode_math import (
    EPILOGUE_FINAL_BF16,
    ld_shared_f32,
    s7_epilogue,
    st_shared_f32,
)
from b12x.attention._shared.mla.prefill_mg import (
    _cache_base_tensor,
    _to_cute,
    s0_load_q_bf16_to_smem_mg,
    s4_finalize_row_sum_mg2,
    s4_online_softmax_mg2,
    s5_fill_sm_p_full_mg2_nvfp4,
)
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@cute.struct
class _Storage:
    q: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 16 * 520], 128]
    kv: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 64 * 520], 128]
    reduce: cute.struct.MemRange[Float32, 256]
    p: cute.struct.MemRange[cutlass.BFloat16, 16 * 72]
    valid: cute.struct.MemRange[Int32, 64]


@cute.jit
def _gather(
    kv,
    indices,
    lengths,
    table,
    dst,
    valid,
    row,
    start,
    tid,
    *,
    page: cutlass.Constexpr,
    stride: cutlass.Constexpr,
    width: cutlass.Constexpr,
    pages: cutlass.Constexpr,
    indexed: cutlass.Constexpr,
):
    # Eight coordinates per task: one vector load and one scale feed four
    # native packed conversions. The resulting BF16 bits serve both QK and PV.
    for batch in cutlass.range_constexpr(16):
        task = tid + Int32(batch * 256)
        candidate = task // Int32(64)
        dim = (task % Int32(64)) * Int32(8)
        pos = start + candidate
        idx = Int32(-1)
        if pos < Int32(width) and pos < lengths[row]:
            idx = indices[row, pos]
        # Keep compile-time specialization separate from the runtime predicate.
        if cutlass.const_expr(indexed):  # noqa: SIM102
            if idx >= Int32(0):
                logical_page = idx // Int32(page)
                physical_page = Int32(-1)
                if logical_page < cute.size(table.shape[1]):
                    physical_page = table[row, logical_page]
                idx = physical_page * Int32(page) + idx % Int32(page)
                if physical_page < Int32(0) or physical_page >= Int32(pages):
                    idx = Int32(-1)
        if dim == Int32(0):
            st_shared_u32(valid + candidate * Int32(4), idx.to(Uint32))
        raw0, raw1, scale = Uint32(0), Uint32(0), Uint32(0)
        if idx >= Int32(0):
            block = idx // Int32(page)
            local = idx % Int32(page)
            base = get_ptr_as_int64(kv, block.to(Int64) * Int64(stride))
            if cutlass.const_expr(indexed):
                raw0 = ld_global_nc_u32(base + (local * Int32(288) + dim // Int32(2)).to(Int64))
                sc_off = local * Int32(288) + Int32(256) + dim // Int32(16)
            else:
                raw0, raw1 = ld_global_nc_v2_u32(base + (local * Int32(528) + dim).to(Int64))
                sc_off = local * Int32(528) + Int32(512) + dim // Int32(32)
            scale = (
                ld_global_nc_u32(base + (sc_off & ~Int32(3)).to(Int64)) >> ((sc_off & Int32(3)) * Int32(8))
            ) & Uint32(255)
        pairs = cute.make_rmem_tensor(4, Uint32)
        for j in cutlass.range_constexpr(4):
            pair = Uint32(0)
            if idx >= Int32(0):
                if cutlass.const_expr(indexed):
                    packed = (raw0 >> Uint32(j * 8)) & Uint32(255)
                    pair = nvfp4_pair_to_bf16x2_sm120(packed, scale)
                else:
                    if cutlass.const_expr(j < 2):
                        packed = (raw0 >> Uint32(j * 16)) & Uint32(65535)
                    else:
                        packed = (raw1 >> Uint32((j - 2) * 16)) & Uint32(65535)
                    pair = mxfp8_pair_to_bf16x2_sm120(packed, scale)
            pairs[j] = pair
        # Each thread owns one aligned eight-BF16 segment.
        st_shared_v4_u32(
            dst + (candidate * Int32(520) + dim) * Int32(2),
            pairs[0],
            pairs[1],
            pairs[2],
            pairs[3],
        )
    cute.arch.barrier(barrier_id=3, number_of_threads=256)


@cute.jit
def _mask(qk, valid, warp, lane, scale):
    first = warp * Int32(8) + (lane & Int32(3)) * Int32(2)
    for j in cutlass.range_constexpr(2):
        idx = ld_shared_u32(valid + (first + Int32(j)) * Int32(4)).to(Int32)
        if idx < Int32(0):
            qk[j] = Float32(-1e30)
            qk[j + 2] = Float32(-1e30)
    for j in cutlass.range_constexpr(4):
        qk[j] = qk[j] * scale
    return qk


@cute.jit
def _qk(qaddr, kvaddr, warp, lane):
    gid = lane >> Int32(2)
    sub = lane & Int32(3)
    arow = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    acol = (lane >> Int32(4)) * Int32(8)
    result = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    for k in cutlass.range_constexpr(32):
        ko = Int32(k * 16)
        b0 = ld_shared_u32(kvaddr + ((warp * Int32(8) + gid) * Int32(520) + ko + sub * Int32(2)) * Int32(2))
        b1 = ld_shared_u32(kvaddr + ((warp * Int32(8) + gid) * Int32(520) + ko + sub * Int32(2) + Int32(8)) * Int32(2))
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(qaddr + (arow * Int32(520) + ko + acol) * Int32(2))
        result[0], result[1], result[2], result[3] = mma_m16n8k16_f32_bf16(
            result[0], result[1], result[2], result[3], a0, a1, a2, a3, b0, b1
        )
    return result


@dsl_user_op
def _ldmatrix_trans_b16(address: Int32, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [address.ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@cute.jit
def _pv(acc, paddr, kvaddr, warp, lane):
    arow = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    acol = (lane >> Int32(4)) * Int32(8)
    # All output tiles consume the same P fragment. Hoist its ldmatrix without
    # folding the tile-local sum into the persistent attention accumulator.
    xv = [[Float32(0.0) for _ in range(4)] for _ in range(8)]
    for ks in cutlass.range_constexpr(4):
        k = Int32(ks * 16)
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(paddr + (arow * Int32(72) + k + acol) * Int32(2))
        for vc in cutlass.range_constexpr(8):
            col = Int32(vc * 64) + warp * Int32(8)
            addr = kvaddr + ((k + (lane & Int32(15))) * Int32(520) + col) * Int32(2)
            b0, b1 = _ldmatrix_trans_b16(addr)
            xv[vc][0], xv[vc][1], xv[vc][2], xv[vc][3] = mma_m16n8k16_f32_bf16(
                xv[vc][0], xv[vc][1], xv[vc][2], xv[vc][3], a0, a1, a2, a3, b0, b1
            )
    for vc in cutlass.range_constexpr(8):
        for j in cutlass.range_constexpr(4):
            acc[vc][j] = acc[vc][j] + xv[vc][j]
    return acc


class _Prefill:
    def __init__(self, q, swa, swa_page, indices, indexed, indexed_page, extra_indices):
        self.heads = int(q.shape[1])
        self.swa_page = int(swa_page)
        self.swa_stride = int(swa.stride(0))
        self.swa_pages = int(swa.shape[0])
        self.swa_width = int(indices.shape[1])
        self.indexed = indexed is not None
        self.extra_page = int(indexed_page)
        self.extra_stride = 0 if indexed is None else int(indexed.stride(0))
        self.extra_pages = 0 if indexed is None else int(indexed.shape[0])
        self.extra_width = 0 if extra_indices is None else int(extra_indices.shape[1])

    @cute.jit
    def __call__(
        self, q, swa, indices, lengths, extra, extra_indices, extra_lengths, table, sink, out, lse, scale, rows, stream
    ):
        self.kernel(q, swa, indices, lengths, extra, extra_indices, extra_lengths, table, sink, out, lse, scale).launch(
            grid=(rows * (self.heads // 16), 1, 1), block=(256, 1, 1), smem=_Storage.size_in_bytes(), stream=stream
        )

    @cute.kernel
    def kernel(self, q, swa, indices, lengths, extra, extra_indices, extra_lengths, table, sink, out, lse, scale):
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid >> Int32(5)
        lane = cute.arch.lane_idx()
        block = Int32(cute.arch.block_idx()[0])
        row = block // Int32(self.heads // 16)
        head = (block % Int32(self.heads // 16)) * Int32(16)
        st = utils.SmemAllocator().allocate(_Storage)
        qaddr = shared_ptr_to_u32(st.q.data_ptr())
        kvaddr = shared_ptr_to_u32(st.kv.data_ptr())
        reduce = shared_ptr_to_u32(st.reduce.data_ptr())
        paddr = shared_ptr_to_u32(st.p.data_ptr())
        valid = shared_ptr_to_u32(st.valid.data_ptr())
        qt = cute.make_tensor(
            q.iterator + row.to(Int64) * q.stride[0],
            cute.make_layout((self.heads, 512), stride=(q.stride[1], q.stride[2])),
        )
        s0_load_q_bf16_to_smem_mg(
            qt,
            qaddr,
            qaddr,
            paddr,
            paddr,
            head,
            tid,
            d_nope=512,
            d_rope=0,
            hpb=16,
            q_nope_bf16_stride=520,
            q_rope_stride=8,
            num_threads=256,
            barrier_id=2,
            n_hg=1,
            valid_hpb=16,
        )
        if tid < Int32(16):
            st_shared_f32(reduce + Int32(512) + tid * Int32(4), Float32(-1e30))
        cute.arch.barrier(barrier_id=3, number_of_threads=256)
        af = cute.make_rmem_tensor(32, Float32)
        rf = cute.make_rmem_tensor(4, Float32)
        sf = cute.make_rmem_tensor(2, Float32)
        for j in cutlass.range_constexpr(32):
            af[j] = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            rf[j] = Float32(0.0)
        sf[0], sf[1] = Float32(0.0), Float32(0.0)
        length = min(max(lengths[row], Int32(0)), Int32(self.swa_width))
        extra_length = Int32(0)
        tiles = (length + Int32(63)) // Int32(64)
        if cutlass.const_expr(self.indexed):
            extra_length = min(max(extra_lengths[row], Int32(0)), Int32(self.extra_width))
            tiles = Int32(self.swa_width // 64) + (extra_length + Int32(63)) // Int32(64)
        for tile in cutlass.range(tiles, unroll=1):
            start = tile * Int32(64)
            if cutlass.const_expr(self.indexed):
                if tile >= Int32(self.swa_width // 64):
                    start = (tile - Int32(self.swa_width // 64)) * Int32(64)
                    _gather(
                        extra,
                        extra_indices,
                        extra_lengths,
                        table,
                        kvaddr,
                        valid,
                        row,
                        start,
                        tid,
                        page=self.extra_page,
                        stride=self.extra_stride,
                        width=self.extra_width,
                        pages=self.extra_pages,
                        indexed=True,
                    )
                else:
                    _gather(
                        swa,
                        indices,
                        lengths,
                        table,
                        kvaddr,
                        valid,
                        row,
                        start,
                        tid,
                        page=self.swa_page,
                        stride=self.swa_stride,
                        width=self.swa_width,
                        pages=self.swa_pages,
                        indexed=False,
                    )
            else:
                _gather(
                    swa,
                    indices,
                    lengths,
                    table,
                    kvaddr,
                    valid,
                    row,
                    start,
                    tid,
                    page=self.swa_page,
                    stride=self.swa_stride,
                    width=self.swa_width,
                    pages=self.swa_pages,
                    indexed=False,
                )
            qk = _qk(qaddr, kvaddr, warp, lane)
            qk = _mask(qk, valid, warp, lane, scale)
            acc = [[af[j * 4 + k] for k in range(4)] for j in range(8)]
            rope = [rf[j] for j in range(4)]
            sums = [sf[0], sf[1]]
            zeros = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
            p, _, _, _, _, _ = s4_online_softmax_mg2(
                qk,
                zeros,
                zeros,
                zeros,
                acc,
                acc,
                rope,
                rope,
                sums,
                sums,
                af,
                af,
                rf,
                rf,
                sf,
                sf,
                reduce,
                warp,
                lane,
                tid,
                n_v_chunks=8,
                hpb=16,
                n_warps=8,
                num_threads=256,
                barrier_id=3,
                n_acc_tiles=8,
                reduce_group_bytes=1024,
                reduce_sum_off=512,
                n_hg=1,
                valid_hpb=16,
            )
            s5_fill_sm_p_full_mg2_nvfp4(
                p, p, paddr, warp, lane, bi=64, hpb=16, sm_p_stride=72, num_threads=256, barrier_id=3, n_hg=1
            )
            acc = _pv(acc, paddr, kvaddr, warp, lane)
            for j in cutlass.range_constexpr(8):
                for k in cutlass.range_constexpr(4):
                    af[j * 4 + k] = acc[j][k]
            sf[0], sf[1] = sums[0], sums[1]
            cute.arch.barrier(barrier_id=3, number_of_threads=256)
        gid = lane >> Int32(2)
        final_max = [
            ld_shared_f32(reduce + Int32(512) + gid * Int32(4)),
            ld_shared_f32(reduce + Int32(512) + (gid + Int32(8)) * Int32(4)),
        ]
        sums = [sf[0], sf[1]]
        if length == Int32(0) and extra_length == Int32(0):
            sums = [Float32(0.0), Float32(0.0)]
        sums, _ = s4_finalize_row_sum_mg2(
            sums,
            sums,
            reduce + Int32(512),
            reduce + Int32(512),
            warp,
            lane,
            tid,
            hpb=16,
            n_warps=8,
            num_threads=256,
            barrier_id=3,
            n_hg=1,
            valid_hpb=16,
        )
        acc = [[af[j * 4 + k] for k in range(4)] for j in range(8)]
        output_tile = cute.make_tensor(
            out.iterator + row.to(Int64) * out.stride[0] + head.to(Int64) * out.stride[1],
            cute.make_layout((16, 512), stride=(out.stride[1], out.stride[2])),
        )
        lse_tile = cute.make_tensor(lse.iterator + row.to(Int64) * lse.stride[0] + head.to(Int64), cute.make_layout(16))
        s7_epilogue(
            acc,
            [rf[j] for j in range(4)],
            final_max,
            sums,
            output_tile,
            lse_tile,
            warp,
            lane,
            n_v_chunks=8,
            v_chunk=64,
            d_nope=512,
            d_rope=0,
            n_warps=8,
            valid_hpb=16,
            nt_per_warp_xv=1,
            v_has_rope=False,
            epilogue_mode=EPILOGUE_FINAL_BF16,
            has_attn_sink=True,
            attn_sink=sink,
            head_base=head,
            fast_rcp=True,
            staging_base_addr=kvaddr,
            d_v=512,
            num_threads=256,
            barrier_id=3,
            coalesced_output=True,
        )


@dataclass(frozen=True)
class PreparedPrefill:
    """Row-dynamic program; no retained query, cache, output or scratch tensors."""

    compiled: object
    kernel: _Prefill

    def run(
        self,
        *,
        q,
        out,
        lse,
        swa_cache,
        swa_indices,
        swa_lengths,
        attn_sink,
        sm_scale,
        indexed_cache=None,
        indexed_indices=None,
        indexed_lengths=None,
        indexed_page_table=None,
    ):
        self.compiled(
            *_arguments(
                q,
                out,
                lse,
                swa_cache,
                swa_indices,
                swa_lengths,
                attn_sink,
                sm_scale,
                indexed_cache,
                indexed_indices,
                indexed_lengths,
                indexed_page_table,
            )
        )


def _arguments(
    q,
    out,
    lse,
    swa_cache,
    swa_indices,
    swa_lengths,
    attn_sink,
    sm_scale,
    indexed_cache,
    indexed_indices,
    indexed_lengths,
    indexed_page_table,
):
    indexed = indexed_cache is not None
    extra = indexed_cache if indexed else swa_cache
    extra_indices = indexed_indices if indexed else swa_indices
    extra_lengths = indexed_lengths if indexed else swa_lengths
    table = indexed_page_table if indexed else swa_indices
    return (
        _to_cute(q, cutlass.BFloat16, dynamic_layout=True),
        _to_cute(_cache_base_tensor(swa_cache), cutlass.Uint8),
        _to_cute(swa_indices, Int32, align=4, dynamic_layout=True),
        _to_cute(swa_lengths, Int32, align=4, dynamic_layout=True),
        _to_cute(_cache_base_tensor(extra), cutlass.Uint8),
        _to_cute(extra_indices, Int32, align=4, dynamic_layout=True),
        _to_cute(extra_lengths, Int32, align=4, dynamic_layout=True),
        _to_cute(table, Int32, align=4, dynamic_layout=True),
        _to_cute(attn_sink, Float32, align=4),
        _to_cute(out, cutlass.BFloat16, dynamic_layout=True),
        _to_cute(lse, Float32, align=4, dynamic_layout=True),
        Float32(sm_scale * LOG2_E),
        Int32(q.shape[0]),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def prepare_prefill(
    *,
    q,
    out,
    lse,
    swa_cache,
    swa_page_size,
    swa_indices,
    swa_lengths,
    attn_sink,
    sm_scale,
    indexed_cache=None,
    indexed_page_size=1,
    indexed_indices=None,
    indexed_lengths=None,
    indexed_page_table=None,
):
    """Compile before freeze using real capacity descriptors.

    Candidate widths, cache page sizes/counts and page pitches specialize the
    H16 kernel. Rows and tensor row strides are dynamic, so one handle serves
    the warmed capacity and its live slices without recompiling. The caller
    owns contiguous output/LSE storage and supplies it on every invocation.
    """
    if q.shape[1] != 16 or swa_indices.shape[1] % 64:
        raise ValueError("single-stage prefill requires H16 and K64 tiles")
    if attn_sink is None:
        raise ValueError("single-stage prefill requires an attention sink")
    kernel = _Prefill(q, swa_cache, swa_page_size, swa_indices, indexed_cache, indexed_page_size, indexed_indices)
    arguments = _arguments(
        q,
        out,
        lse,
        swa_cache,
        swa_indices,
        swa_lengths,
        attn_sink,
        sm_scale,
        indexed_cache,
        indexed_indices,
        indexed_lengths,
        indexed_page_table,
    )
    return PreparedPrefill(cute.compile(kernel, *arguments), kernel)
