# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Static-M SIMT skinny GEMM with a shared-add multicast epilogue."""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int64, Uint16, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

from sonar.models.common.ops.moe_tail_fusion.primitives import (
    CUDAGraphCompatibleWrapper,
    bf16x2_to_u32,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    fma_f32_bf16,
    sanitize_negative_zero_u32,
    sanitize_negative_zero_u32x2,
    sanitize_negative_zero_u32x4,
    store_global_u32,
    store_global_u32x2,
    store_global_u32x4,
)


@dataclass(frozen=True)
class SkinnyConfig:
    block_size: int = 224
    outputs_per_block: int = 2
    k_unroll: int = 2
    vector_width: int = 8
    prefetch_b_before_pdl: bool = True


def config_for_m(num_rows: int, shard_dim: int = 896, *, nf12: bool = False) -> SkinnyConfig:
    if nf12:
        if not 1 <= num_rows <= 5 or shard_dim not in (448, 896):
            raise ValueError("NF12 skinny requires M in [1, 5] and N=448 or 896")
        return SkinnyConfig(block_size=224, outputs_per_block=4, k_unroll=1, vector_width=16)
    if num_rows <= 5 and shard_dim in (448, 896):
        return SkinnyConfig(
            block_size=224,
            outputs_per_block=2,
            k_unroll=1,
            vector_width=16,
        )
    if shard_dim == 448:
        if num_rows >= 6:
            return SkinnyConfig(
                block_size=224,
                outputs_per_block=2,
                k_unroll=1,
            )
        outputs_per_block = 2 if num_rows <= 3 else 4
        return SkinnyConfig(
            block_size=448,
            outputs_per_block=outputs_per_block,
            k_unroll=1,
        )
    if num_rows == 1:
        return SkinnyConfig(outputs_per_block=8, k_unroll=1)
    return SkinnyConfig(outputs_per_block=4)


def _as_cute(tensor: torch.Tensor):
    return from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()),
        assumed_align=32,
    )


@dsl_user_op
def _bf16_from_bits(bits: Uint16, *, loc=None, ip=None) -> BFloat16:
    return BFloat16(llvm.bitcast(BFloat16.mlir_type, bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _prmt(a: Uint32, b: Uint32, selector: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Uint32(value).ir_value(loc=loc, ip=ip) for value in (a, b, selector)],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _copy_nf12_tile(destination: cute.Pointer, source: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [cutlass.Int32(destination.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip), source.ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class FusedAddMulticastSkinnyGemm:
    """SIMT GEMM adapted from the existing Skinny GEMM."""

    def __init__(
        self,
        *,
        num_rows: int,
        hidden_dim: int,
        config: SkinnyConfig,
        nf12: bool = False,
    ) -> None:
        if config.block_size % 32:
            raise ValueError("skinny block_size must be a multiple of 32")
        if nf12 and (config.outputs_per_block != 4 or config.vector_width != 16):
            raise ValueError("NF12 skinny requires four paired-row outputs and vector_width=16")
        self.num_rows = num_rows
        self.hidden_dim = hidden_dim
        self.block_size = config.block_size
        self.outputs_per_block = config.outputs_per_block
        self.k_unroll = config.k_unroll
        self.vector_width = config.vector_width
        self.prefetch_b_before_pdl = config.prefetch_b_before_pdl
        self.num_warps = config.block_size // 32
        self.nf12 = nf12
        self.shared_bytes = num_rows * self.outputs_per_block * self.num_warps * 4
        if nf12:
            self.shared_bytes = ((self.shared_bytes + 127) // 128) * 128 + (self.block_size // 4) * 384

    @cute.jit
    def load_nf12(
        self,
        gB: cute.Tensor,
        b_regs: cute.Tensor,
        n_base: cutlass.Int32,
        tidx: cutlass.Int32,
        k_extent: cutlass.Int32,
        escapes_ptr: Int64,
        exp_base: cutlass.Int32,
    ) -> None:
        # Four neighboring threads copy the three 128-byte planes for their
        # paired rows. Keep staging separate from the epilogue's partials.
        partial_words = self.num_rows * self.outputs_per_block * self.num_warps
        stage_offset = ((partial_words + 31) // 32) * 32
        staged = cute.make_tensor(
            cute.arch.get_dyn_smem(Uint32, alignment=128) + stage_offset,
            cute.make_layout((self.block_size // 4) * 96),
        )
        super_atom = tidx // 4
        atom = tidx % 4
        source_base = ((n_base // 16) * (k_extent // 64) + super_atom) * 1536 + (n_base % 8) * 64
        for plane in cutlass.range_constexpr(3):
            for half in cutlass.range_constexpr(2):
                source_offset = source_base + plane * 512 + atom * 16 + half * 64
                destination_offset = super_atom * 96 + plane * 32 + atom * 4 + half * 16
                _copy_nf12_tile(
                    staged.iterator + destination_offset,
                    gB.iterator.toint() + source_offset.to(Int64),
                )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        escapes = cute.make_tensor(
            cute.make_ptr(BFloat16, escapes_ptr, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout(self.hidden_dim * k_extent),
        )
        base_x4 = exp_base.to(Uint32) * Uint32(0x01010101)
        for ni in cutlass.range_constexpr(2):
            for pair in cutlass.range_constexpr(4):
                lane = ni * 4 + pair
                meta = staged[super_atom * 96 + lane * 4 + atom]
                low_offset = super_atom * 96 + 32 + (atom // 2) * 32 + lane * 4 + (atom % 2) * 2
                lo0 = staged[low_offset]
                if meta == 0:
                    # Sliced packed rows retain indices into the full escape table.
                    for row_half in cutlass.range_constexpr(2):
                        for half in cutlass.range_constexpr(2):
                            slot = row_half * 2 + half * 4
                            for odd in cutlass.range_constexpr(2):
                                b_regs[ni + row_half * 2, half * 8 + pair * 2 + odd] = escapes[
                                    lo0.to(Int64) * 8 + slot + odd
                                ]
                else:
                    lo1 = staged[low_offset + 1]
                    even_high = (meta & Uint32(0x87878787)) + base_x4
                    odd_high = (((meta << 1) & Uint32(0x80808080)) | ((meta >> 3) & Uint32(0x07070707))) + base_x4
                    high0 = _prmt(even_high, odd_high, 0x5140)
                    high1 = _prmt(even_high, odd_high, 0x7362)
                    decoded = (
                        _prmt(lo0, high0, 0x5140),
                        _prmt(lo0, high0, 0x7362),
                        _prmt(lo1, high1, 0x5140),
                        _prmt(lo1, high1, 0x7362),
                    )
                    for row_half in cutlass.range_constexpr(2):
                        for half in cutlass.range_constexpr(2):
                            for odd in cutlass.range_constexpr(2):
                                bits = decoded[half * 2 + row_half] >> (odd * 16)
                                b_regs[ni + row_half * 2, half * 8 + pair * 2 + odd] = _bf16_from_bits(bits.to(Uint16))

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gShared: cute.Tensor,
        output_multicast_ptr: Int64,
        escapes_ptr: Int64,
        exp_base: cutlass.Int32,
        stream: cuda.CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        k = cute.size(gA, mode=[1])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=self.vector_width * BFloat16.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=self.vector_width * BFloat16.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(
            gA,
            gB,
            gShared,
            output_multicast_ptr,
            escapes_ptr,
            exp_base,
            k,
            copy_a,
            copy_b,
        ).launch(
            grid=[cute.ceil_div(n, self.outputs_per_block), 1, 1],
            block=[self.block_size, 1, 1],
            smem=self.shared_bytes,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gShared: cute.Tensor,
        output_multicast_ptr: Int64,
        escapes_ptr: Int64,
        exp_base: cutlass.Int32,
        k_extent: cutlass.Int32,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        num_rows: cutlass.Constexpr = self.num_rows

        acc = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, outputs_per_block),
                stride=(outputs_per_block, 1),
            ),
            Float32,
        )
        acc.fill(0.0)

        n_base = block_idx * outputs_per_block
        if const_expr(self.nf12):
            n_base = (block_idx // 4) * 16 + (block_idx % 4) * 2
        k_tile_size: cutlass.Constexpr = block_size * vector_width
        num_k_tiles = k_extent // k_tile_size
        gA_vec = cute.logical_divide(gA, (None, vector_width))
        if const_expr(not self.nf12):
            gB_vec = cute.logical_divide(gB, (None, vector_width))
            tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        a_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, vector_width),
                stride=(vector_width, 1),
            ),
            BFloat16,
        )
        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, vector_width),
                stride=(vector_width, 1),
            ),
            BFloat16,
        )

        if const_expr(self.prefetch_b_before_pdl):
            if const_expr(self.nf12):
                self.load_nf12(gB, b_regs, n_base, tidx, k_extent, escapes_ptr, exp_base)
            else:
                for ni in cutlass.range_constexpr(outputs_per_block):
                    tB = tB_all[n_base + ni, (None, (tidx, None))]
                    cute.copy(copy_b, tB[None, 0], b_regs[ni, None])

        cute.arch.griddepcontrol_wait()

        for mi in cutlass.range_constexpr(num_rows):
            cute.copy(copy_a, tA[mi, None, 0], a_regs[mi, None])
        if const_expr(not self.prefetch_b_before_pdl):
            if const_expr(self.nf12):
                self.load_nf12(gB, b_regs, n_base, tidx, k_extent, escapes_ptr, exp_base)
            else:
                for ni in cutlass.range_constexpr(outputs_per_block):
                    tB = tB_all[n_base + ni, (None, (tidx, None))]
                    cute.copy(copy_b, tB[None, 0], b_regs[ni, None])
        for vi in cutlass.range_constexpr(vector_width):
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    acc[mi, ni] = fma_f32_bf16(
                        a_regs[mi, vi],
                        b_regs[ni, vi],
                        acc[mi, ni],
                    )

        for k_tile in cutlass.range(1, num_k_tiles, unroll=self.k_unroll):
            for mi in cutlass.range_constexpr(num_rows):
                cute.copy(
                    copy_a,
                    tA[mi, None, k_tile],
                    a_regs[mi, None],
                )
            if const_expr(self.nf12):
                self.load_nf12(gB, b_regs, n_base, tidx + k_tile * block_size, k_extent, escapes_ptr, exp_base)
            else:
                for ni in cutlass.range_constexpr(outputs_per_block):
                    tB = tB_all[n_base + ni, (None, (tidx, None))]
                    cute.copy(copy_b, tB[None, k_tile], b_regs[ni, None])
            for vi in cutlass.range_constexpr(vector_width):
                for mi in cutlass.range_constexpr(num_rows):
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        acc[mi, ni] = fma_f32_bf16(
                            a_regs[mi, vi],
                            b_regs[ni, vi],
                            acc[mi, ni],
                        )

        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem_layout = cute.make_layout(
            (num_rows, outputs_per_block, num_warps),
            stride=(outputs_per_block * num_warps, num_warps, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            Float32,
            smem_layout,
            byte_alignment=16,
        )
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx == 0:
            fused = cute.make_rmem_tensor(
                cute.make_layout((outputs_per_block,)),
                BFloat16,
            )
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    total = (
                        partials[mi, ni, None]
                        .load()
                        .reduce(
                            cute.ReductionOp.ADD,
                            init_val=Float32(0.0),
                            reduction_profile=0,
                        )
                    )
                    gemm_value = Float32(total).to(BFloat16)
                    output_row = n_base + ni
                    if const_expr(self.nf12):
                        output_row = n_base + (ni % 2) + (ni // 2) * 8
                    fused[ni] = (gemm_value.to(Float32) + gShared[mi, output_row].to(Float32)).to(BFloat16)
                output_offset = Int64((mi * self.hidden_dim + n_base) * 2)
                if const_expr(self.nf12):
                    packed = sanitize_negative_zero_u32x2(bf16x4_to_packed_u32x2(fused.load()))
                    store_global_u32(output_multicast_ptr + output_offset, packed[0])
                    store_global_u32(output_multicast_ptr + output_offset + 16, packed[1])
                elif const_expr(outputs_per_block == 2):
                    packed = sanitize_negative_zero_u32(bf16x2_to_u32(fused.load()))
                    store_global_u32(
                        output_multicast_ptr + output_offset,
                        packed,
                    )
                elif const_expr(outputs_per_block == 4):
                    packed = sanitize_negative_zero_u32x2(bf16x4_to_packed_u32x2(fused.load()))
                    store_global_u32x2(
                        output_multicast_ptr + output_offset,
                        packed,
                    )
                else:
                    packed = sanitize_negative_zero_u32x4(bf16x8_to_packed_u32x4(fused.load()))
                    store_global_u32x4(
                        output_multicast_ptr + output_offset,
                        packed,
                    )

        cute.arch.griddepcontrol_launch_dependents()


_COMPILED: dict[tuple[object, ...], object] = {}


def compile_kernel(
    *,
    num_rows: int,
    latent_dim: int,
    hidden_dim: int,
    shard_dim: int,
    config: SkinnyConfig,
    nf12: bool = False,
):
    key = (
        torch.accelerator.current_device_index(),
        num_rows,
        latent_dim,
        hidden_dim,
        shard_dim,
        config,
        nf12,
    )
    if key in _COMPILED:
        return _COMPILED[key]

    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    if shard_dim % config.outputs_per_block:
        raise ValueError("shard_dim must be divisible by outputs_per_block")
    if latent_dim % (config.block_size * config.vector_width):
        raise ValueError("latent_dim must be divisible by block_size * vector_width")
    if nf12 and (
        not 1 <= num_rows <= 5
        or latent_dim != 3584
        or shard_dim not in (448, 896)
        or config.vector_width != 16
        or config.outputs_per_block != 4
    ):
        raise ValueError("NF12 skinny requires M in [1, 5], K=3584, N=448 or 896, and four paired-row outputs")

    a = make_fake_tensor(
        BFloat16,
        (num_rows, latent_dim),
        stride=(latent_dim, 1),
        assumed_align=32,
    )
    b_columns = latent_dim * 3 // 8 if nf12 else latent_dim
    b = make_fake_tensor(
        Uint32 if nf12 else BFloat16,
        (shard_dim, b_columns),
        stride=(b_columns, 1),
        assumed_align=32,
    )
    shared = make_fake_tensor(
        BFloat16,
        (num_rows, shard_dim),
        stride=(hidden_dim, 1),
        assumed_align=32,
    )
    compiled = cute.compile(
        FusedAddMulticastSkinnyGemm(
            num_rows=num_rows,
            hidden_dim=hidden_dim,
            config=config,
            nf12=nf12,
        ),
        a,
        b,
        shared,
        Int64(0),
        Int64(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--ptxas-options -maxrregcount=128",
    )
    _COMPILED[key] = compiled
    return compiled


class FusedAddMulticastSkinnyGemmKernel:
    """Buffer-free compiled launcher for one static M."""

    def __init__(
        self,
        *,
        rank: int,
        tp_size: int,
        latent_dim: int,
        hidden_dim: int,
        num_rows: int,
        nf12: bool = False,
    ) -> None:
        if not 1 <= num_rows <= 8:
            raise ValueError("skinny backend requires static M in [1, 8]")
        if hidden_dim % tp_size:
            raise ValueError("hidden_dim must be divisible by TP size")
        self.rank = rank
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.shard_dim = hidden_dim // tp_size
        self.num_rows = num_rows
        self.nf12 = nf12
        self._skinny = compile_kernel(
            num_rows=num_rows,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            shard_dim=self.shard_dim,
            config=config_for_m(num_rows, self.shard_dim, nf12=nf12),
            nf12=nf12,
        )

    def __call__(
        self,
        latent: torch.Tensor,
        weight: torch.Tensor,
        shared_shard: torch.Tensor,
        mailbox: torch.Tensor,
        mailbox_multicast_ptr: int,
        *,
        weight_escapes: torch.Tensor | None = None,
        weight_exp_base: int | None = None,
    ) -> torch.Tensor:
        device = mailbox.device
        expected = (
            (
                latent,
                (self.num_rows, self.latent_dim),
                torch.bfloat16,
                "latent",
            ),
            (
                weight,
                (self.shard_dim, self.latent_dim * 3 // 2 if self.nf12 else self.latent_dim),
                torch.uint8 if self.nf12 else torch.bfloat16,
                "weight",
            ),
            (
                shared_shard,
                (mailbox.shape[1], self.shard_dim),
                torch.bfloat16,
                "shared_shard",
            ),
            (
                mailbox,
                (1, mailbox.shape[1], self.hidden_dim),
                torch.bfloat16,
                "mailbox",
            ),
        )
        for tensor, shape, dtype, name in expected:
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
                raise ValueError(f"{name} must be CUDA {dtype} {list(shape)}")
        if (
            not latent.is_contiguous()
            or not weight.is_contiguous()
            or shared_shard.stride() != (self.hidden_dim, 1)
            or not mailbox.is_contiguous()
        ):
            raise ValueError("skinny up-projection inputs have unsupported strides")
        if any(tensor.data_ptr() % 32 for tensor in (latent, weight, shared_shard)):
            raise ValueError("skinny up-projection inputs must be 32-byte aligned")
        if mailbox.shape[1] < self.num_rows:
            raise ValueError("mailbox capacity is smaller than runtime M")
        if self.nf12:
            if weight_escapes is None or weight_exp_base is None:
                raise ValueError("NF12 weights require the original escape table and exponent base")
            if (
                weight_escapes.dtype != torch.int16
                or weight_escapes.ndim != 2
                or weight_escapes.shape[1] != 8
                or not isinstance(weight_exp_base, int)
                or not 0 <= weight_exp_base <= 120
                or weight_escapes.device != device
                or not weight_escapes.is_contiguous()
                or weight_escapes.data_ptr() % 16
            ):
                raise ValueError("NF12 metadata requires contiguous CUDA int16 [E, 8] and an integer base in [0, 120]")
        elif weight_escapes is not None or weight_exp_base is not None:
            raise ValueError("BF16 weights must not supply NF12 metadata")

        with torch.accelerator.device_index(device.index):
            self._skinny(
                _as_cute(latent),
                _as_cute(weight.view(torch.uint32) if self.nf12 else weight),
                _as_cute(shared_shard[: self.num_rows]),
                Int64(mailbox_multicast_ptr + self.rank * self.shard_dim * 2),
                Int64(weight_escapes.data_ptr() if weight_escapes is not None else 0),
                cutlass.Int32(weight_exp_base if self.nf12 else 0),
                cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
            )
        return mailbox
