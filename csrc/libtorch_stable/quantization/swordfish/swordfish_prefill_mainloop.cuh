// Swordfish prefill mainloop, an sm100 tcgen05 mixed-input collective reading the
// Swordfish packed-weight ABI v1 directly.
//
// This is a fork of CUTLASS 4.4.2's
//   include/cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp
// (BSD-3-Clause, Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES),
// following the Machete precedent of forking a stock collective. Line
// references below are against that file. The pipeline structure (Load2Transform
// / Load2Mma / Transform2Mma / Mma2Accum), the B (activation) path, the scale
// TMA path, and the MMA stage are stock. Two things change.
//
// 1. B-operand TMA over the packed ABI (stock lines 310-313, 460-467,
//    529-535, 735-814, 883-973). The quantized operand (riding the swapped
//    A slot) is
//    presented to TMA as a dense byte tensor (256, 8, KB, NB); each (nb, kb)
//    ABI block is a linear 2048 B run (invariant I3), and the Marlin in-tile
//    permutation is invisible to a byte copy (I4). The input smem staging is a
//    plain byte buffer; the canonical-layout input atom machinery is deleted.
//
// 2. The Transform stage (stock lines 975-1144). Instead of
//    MixedInputUtils::dequantize_A_kblock_for_transform over canonical int4,
//    each transform thread consumes packed words in Marlin tile order (lane T
//    <-> word 4T of each 512 B sub-tile, the contract proven by the decode
//    path, swordfish_decode.cuh) and dequantizes with the marlin u4b8 LOP3
//    idiom, applies group scales, and writes K-major bf16 into the
//    tcgen05-descriptor-legal compute smem buffer (invariant I5, four 32-bit
//    lane-local stores per packed word). Because the stores are free-form smem
//    writes, this fork uses the SS (A-from-SMEM) UMMA atom, i.e. the TiledMma
//    of KernelTmaWarpSpecialized1SmMixedInputSmemSm100. (The stock Smem
//    schedule does not compile in CUTLASS 4.4.2, since its transform_init calls
//    make_tmem_copy on an SS descriptor fragment; this fork replaces that code
//    entirely.)
//
// Scope: 1-SM or 2-SM (cta_group::2) MMA, u4b8 weights, bf16
// activations/scales, ConvertAndScale without zero points, group_size 64 or
// 128, L = 1, N and K multiples of 128 (the ABI tail policy rejects rests).
#pragma once

#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/detail/collective/mixed_input_utils.hpp"
#include "cutlass/detail/sm100_mixed_dtype_blockwise_layout.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm100.hpp"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

namespace swordfish_detail {

// u4b8 -> bf16x2 pair dequant, the marlin LOP3 idiom
// (csrc/libtorch_stable/quantization/marlin/dequant.h,
// dequant<nv_bfloat162, kU4B8>; re-stated here so this header only depends on
// CUTLASS/CUDA). Consumes nibbles {0,4} into frag[0] and {1,5} into frag[1];
// feed `q >> 8` for nibbles {2,6}/{3,7}. Under the Marlin pack_idx interleave
// {0,2,4,6,1,3,5,7} this yields, for a word held by lane T (column c = T/4,
// k-quad t = T%4):
//   dequant(q):      frag[0] = (k=2t, 2t+1),  frag[1] = (k=2t+8, 2t+9)   @ col c
//   dequant(q >> 8): same k positions                                    @ col c+8
__device__ __forceinline__ void dequant_u4b8_bf16x2(uint32_t q,
                                                    __nv_bfloat162 frag[2]) {
  static constexpr uint32_t kMask = 0x000f000f;
  static constexpr uint32_t kEx = 0x43004300;
  static constexpr uint32_t kSub = 0x43084308;  // {136, 136}: (v | 0x4300) - 136 = v - 8
  static constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  uint32_t lo, hi;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(lo)
               : "r"(q), "n"(kMask), "n"(kEx), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(hi)
               : "r"(q >> 4), "n"(kMask), "n"(kEx), "n"(kImmLut));
  frag[0] = __hsub2(reinterpret_cast<__nv_bfloat162 const&>(lo),
                    reinterpret_cast<__nv_bfloat162 const&>(kSub));
  frag[1] = __hsub2(reinterpret_cast<__nv_bfloat162 const&>(hi),
                    reinterpret_cast<__nv_bfloat162 const&>(kSub));
}

// u4b8 -> f16x2 pair dequant (marlin dequant<half2, kU4B8>). Same fragment
// semantics as the bf16 helper; the high nibble plane rides an embedded
// x16 exponent folded out by the fma.
__device__ __forceinline__ void dequant_u4b8_f16x2(uint32_t q,
                                                   __half2 frag[2]) {
  static constexpr uint32_t kLo = 0x000f000f;
  static constexpr uint32_t kHi = 0x00f000f0;
  static constexpr uint32_t kEx = 0x64006400;
  static constexpr uint32_t kSub = 0x64086408;
  static constexpr uint32_t kMul = 0x2c002c00;
  static constexpr uint32_t kAdd = 0xd480d480;
  static constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  uint32_t lo, hi;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(lo)
               : "r"(q), "n"(kLo), "n"(kEx), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(hi)
               : "r"(q), "n"(kHi), "n"(kEx), "n"(kImmLut));
  frag[0] = __hsub2(reinterpret_cast<__half2 const&>(lo),
                    reinterpret_cast<__half2 const&>(kSub));
  frag[1] = __hfma2(reinterpret_cast<__half2 const&>(hi),
                    reinterpret_cast<__half2 const&>(kMul),
                    reinterpret_cast<__half2 const&>(kAdd));
}

// u8b128 -> f16x2 pair dequant (marlin dequant<half2, kU8B128>).
__device__ __forceinline__ void dequant_u8b128_f16x2(uint32_t q,
                                                     __half2 frag[2]) {
  static constexpr uint32_t kMask01 = 0x5250;
  static constexpr uint32_t kMask23 = 0x5351;
  static constexpr uint32_t kBase = 0x64646464;
  static constexpr uint32_t kSub = 0x64806480;
  const uint32_t lo = __byte_perm(q, kBase, kMask01);
  const uint32_t hi = __byte_perm(q, kBase, kMask23);
  frag[0] = __hsub2(reinterpret_cast<__half2 const&>(lo),
                    reinterpret_cast<__half2 const&>(kSub));
  frag[1] = __hsub2(reinterpret_cast<__half2 const&>(hi),
                    reinterpret_cast<__half2 const&>(kSub));
}

// u8b128 -> bf16x2 pair dequant (marlin dequant<nv_bfloat162, kU8B128>, the
// FasterTransformer fp32-bias idiom). Under the 8-bit pack interleave
// {0,2,1,3} one word yields the same fragment pair as the 4-bit dequant of
// one nibble plane: frag[0] = perm values (0,1), frag[1] = values (2,3).
__device__ __forceinline__ void dequant_u8b128_bf16x2(uint32_t q,
                                                      __nv_bfloat162 frag[2]) {
  float f[4];
  uint32_t* fc = reinterpret_cast<uint32_t*>(f);
  static constexpr uint32_t kBase = 0x4B000000;
  fc[0] = __byte_perm(q, kBase, 0x7650);
  fc[1] = __byte_perm(q, kBase, 0x7652);
  fc[2] = __byte_perm(q, kBase, 0x7651);
  fc[3] = __byte_perm(q, kBase, 0x7653);
  f[0] -= 8388736.f;
  f[1] -= 8388736.f;
  f[2] -= 8388736.f;
  f[3] -= 8388736.f;
  uint32_t* out = reinterpret_cast<uint32_t*>(frag);
  out[0] = __byte_perm(fc[0], fc[1], 0x7632);
  out[1] = __byte_perm(fc[2], fc[3], 0x7632);
}

}  // namespace swordfish_detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// Same template signature as the stock CollectiveMma specialization (stock
// lines 62-103) so the instantiation site can mirror the stock builder; a
// standalone struct (the kernel layer dispatches on DispatchPolicy::Schedule
// only).
template <
  int Load2TransformPipelineStageCount_,
  int Transform2MmaPipelineStageCount_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  class ClusterShape,
  class TileShape_,
  class ElementAOptionalTuple_,
  class StridePairA_,
  class ElementBOptionalTuple_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomsA_,
  class CopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomsB_,
  class CopyAtomsB_,
  class TransformB_,
  int WBits = 4>
struct SwordfishMainloopSm100MixedInput {
public:
  //
  // Type Aliases (stock lines 105-263, SwapAB plumbing removed: the quantized
  // operand is ALWAYS the A slot here)
  //
  using ConversionMode = cutlass::detail::ConversionMode;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;
  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedMixedInput<
                            Load2TransformPipelineStageCount_,
                            Transform2MmaPipelineStageCount_,
                            SchedulerPipelineStageCount_,
                            AccumulatorPipelineStageCount_,
                            ClusterShape>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  using KernelSchedule = typename DispatchPolicy::Schedule;
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static_assert(!IsDynamicCluster, "swordfish prefill v1 requires a static cluster");
  static_assert(cute::is_same_v<ClusterShape, Shape<_1,_1,_1>> ||
                    cute::is_same_v<ClusterShape, Shape<_2,_1,_1>>,
                "swordfish prefill supports 1-SM (cluster 1x1x1) or 2-SM "
                "(cluster 2x1x1) only");
  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));
  // Number of CTAs the 2-SM MMA atom spans in the tile-M (weight-N) axis:
  // 1 for the 1-SM atom, 2 for the 2-SM atom. Drives the per-CTA A split.
  static constexpr int kAtomCtasM = size<0>(AtomThrShapeMNK{});

  using ElementAOptionalTuple = ElementAOptionalTuple_;
  using ElementBOptionalTuple = ElementBOptionalTuple_;
  static_assert(cute::is_tuple<ElementAOptionalTuple>::value &&
                !cute::is_tuple<ElementBOptionalTuple>::value,
                "swordfish prefill: quantized operand must ride the A slot "
                "(pass {ElementA, ElementScale} for A)");

  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = true;
  using ElementScale = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ElementZero = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  // The packed operand is staged as raw bytes.
  static_assert(cute::is_same_v<ElementA, uint8_t>,
                "swordfish prefill: pass uint8_t as the (packed) A element");
  // Zero-point checkpoints (AWQ/HQQ) pass a third element in the A tuple.
  // The zero tensor holds prescaled (8 - zp) * scale rows, scale-shaped and
  // scale-typed, so it rides the scale TMA machinery verbatim and the
  // transform's scaling multiply becomes an fma.
  static constexpr bool HasZp = !cute::is_void_v<ElementZero>;
  static_assert(!HasZp || cute::is_same_v<ElementZero, ElementScale>,
                "swordfish prefill zero points are scale-typed (8 - zp) * s");

  using StrideA = cute::remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
  using LayoutScale = cute::remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  using CtaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using CtaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;
  static constexpr bool kActF16 = cute::is_same_v<ElementAMma, cutlass::half_t>;
  static_assert(kActF16 || cute::is_same_v<ElementAMma, cutlass::bfloat16_t>,
                "swordfish prefill dequantizes to fp16 or bf16");
  static_assert(cute::is_same_v<NonVoidElementScale, ElementAMma>,
                "swordfish prefill group scales match the activation dtype");
  // Register pair type matching ElementAMma for the transform.
  using Elem2 = cute::conditional_t<kActF16, __half2, __nv_bfloat162>;

  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = GmemTiledCopyA_;

  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;
  using CopyAtomsA = CopyAtomsA_;
  using CopyAtomsB = CopyAtomsB_;

  using SmemLayoutAtomACompute = typename SmemLayoutAtomsA::ComputeLayoutAtom;
  using SmemLayoutAtomB = typename SmemLayoutAtomsB::InputLayoutAtom;

  using TmaElementA = uint8_t;

  using ArchTag = typename DispatchPolicy::ArchTag;

  using Load2TransformPipeline = cutlass::PipelineTmaTransformAsync<
                             DispatchPolicy::Load2TransformPipelineStageCount,
                             AtomThrShapeMNK>;
  using Load2TransformPipelineState = typename Load2TransformPipeline::PipelineState;

  using Load2MmaPipeline = cutlass::PipelineTmaUmmaAsync<
                             DispatchPolicy::Load2TransformPipelineStageCount,
                             ClusterShape,
                             AtomThrShapeMNK>;
  using Load2MmaPipelineState = typename Load2MmaPipeline::PipelineState;

  using Transform2MmaPipeline = cutlass::PipelineUmmaConsumerAsync<
                              DispatchPolicy::Transform2MmaPipelineStageCount,
                              AtomThrShapeMNK>;
  using Transform2MmaPipelineState = typename Transform2MmaPipeline::PipelineState;

  using Mma2AccumPipeline = cutlass::PipelineUmmaAsync<
                              DispatchPolicy::Schedule::AccumulatorPipelineStageCount,
                              AtomThrShapeMNK>;
  using Mma2AccumPipelineState = typename Mma2AccumPipeline::PipelineState;

  // ---- scale layout machinery (stock lines 265-278, unchanged) --------------
  static constexpr int ScaleGranularityMN = size<0,0>(LayoutScale{});
  static constexpr int ScaleGranularityK = size<1,0>(LayoutScale{});
  static_assert(ScaleGranularityMN == 1, "swordfish scales are per-column");
  static_assert(ScaleGranularityK == 32 || ScaleGranularityK == 64 ||
                    ScaleGranularityK == 128,
                "swordfish prefill supports group sizes 32, 64 and 128");
  // At granularity 32 a k64 sub-block spans two scale groups, so the
  // transform keeps one register set per 32-row group of its share.
  static constexpr int kScaleGroupsPerWarp =
      ScaleGranularityK == 32 ? 2 : 1;
  using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<
      ScaleGranularityMN,
      ScaleGranularityK>;

  using ScaleTileShape = decltype(make_shape(size<0>(TileShape{}), size<2>(TileShape{})));
  using SmemLayoutAtomScaleFull = decltype(ScaleConfig::smem_atom_layout_scale(ScaleTileShape{}));
  using SmemLayoutAtomScale = decltype(slice(make_coord(make_coord(_,0),make_coord(_,0)), SmemLayoutAtomScaleFull{}));

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Thread counts (stock lines 292-294)
  // Parametric transform width. The forked kernel layer derives its warp
  // layout from this. 128 means one warp per packed block; 256 (one per half
  // block) measured slightly worse on B200, so the transform is not the
  // bottleneck there.
  static constexpr uint32_t NumTransformationThreads = 128;
  static constexpr uint32_t NumAccumThreads = 128;

  constexpr static int AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  constexpr static int StagesPerTile = size<2>(CtaShapeA_MK{});

  // ---- Swordfish geometry ----------------------------------------------------
  // ABI v1 constants (swordfish_types.cuh; restated to keep this header
  // dependent only on CUTLASS/CUDA).
  static constexpr int kBlockN = 64;         // columns per packed block
  static constexpr int kBlockK = 64;         // K rows per packed block
  static_assert(WBits == 4 || WBits == 8, "swordfish weights are 4 or 8 bit");
  static constexpr int kSubTileBytes = WBits == 8 ? 1024 : 512;  // 16x64 tile
  static constexpr int kBlockBytes = 4 * kSubTileBytes;
  static constexpr int kBlockRows = kBlockBytes / 256;  // TMA inner rows

  // Full MMA-tile weight-N drives the packed-A gmem tiler so the tile
  // coordinate matches the scheduler. Per-CTA weight-N drives the smem
  // buffers, since each CTA of a 2-SM pair stages only its own half.
  static constexpr int TileN_Weights_Full = size<0>(TileShape{});
  static constexpr int CtaTileN_Weights = size<0>(CtaShape_MNK{});  // per-CTA
  static constexpr int CtaTileK = size<2>(CtaShape_MNK{});
  static_assert(CtaTileN_Weights == 128 && CtaTileK == 128,
                "swordfish prefill is tuned/asserted for a 128x128x128 CTA tile");
  static constexpr int kBlocksPerTileN = CtaTileN_Weights / kBlockN;  // 2 (per CTA)
  static constexpr int kBlocksPerTileK = CtaTileK / kBlockK;          // 2
  static constexpr int kStageBytes = kBlocksPerTileN * kBlocksPerTileK * kBlockBytes;  // 8192

  // CHANGE (1): input staging is the packed byte stream, laid out
  // (byte-lo, byte-hi, kb, nb, PIPE). Modes 0-1 pre-split the 2048 B block run
  // so every TMA box extent is <= 256. Sized/tiled PER-CTA (128 weight-cols);
  // in 2-SM the load loop maps the cluster's weight-N tile coord to this CTA's
  // 128-col half via kAtomCtasM*coord + atom_half.
  using SmemLayoutA = Layout<
      Shape< _256, Int<kBlockRows>, Int<kBlocksPerTileK>, Int<kBlocksPerTileN>, Int<DispatchPolicy::Load2TransformPipelineStageCount>>,
      Stride<_1, _256, Int<kBlockBytes>, Int<kBlocksPerTileK * kBlockBytes>, Int<kStageBytes>>>;
  using SwordfishTilerA = Shape<_256, Int<kBlockRows>, Int<kBlocksPerTileK>, Int<kBlocksPerTileN>>;

  // B (activations) staging: stock (lines 320-323).
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(CtaShapeB_NK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  // CHANGE (2): the compute buffer keeps the stock tcgen05-descriptor-legal
  // core-matrix layout, but is built flat-first so the transform can address
  // it by logical (n, k, stage) with a memory function IDENTICAL (by
  // construction) to the MMA-facing grouped view.
  // Per-CTA shape. Each CTA dequants only its own weight-N half.
  using SmemLayoutAComputeMK = decltype(tile_to_shape(
      SmemLayoutAtomACompute{},
      make_shape(size<0>(CtaShape_MNK{}), size<2>(CtaShape_MNK{}),
                 Int<DispatchPolicy::Transform2MmaPipelineStageCount>{}),
      Step<_1,_2,_3>{}));
  // Same regroup tile_to_mma_shape applies (cute/atom/mma_traits_sm100.hpp:116).
  using SmemLayoutACompute = decltype(tiled_divide(
      SmemLayoutAComputeMK{}, product_each(shape<0>(CtaShapeA_MK{}))));

  // Closed-form of the compute buffer's memory function (element offset for
  // logical (n, k, stage)), used by the transform's store path so each 32-bit
  // store is one ADD instead of a hierarchical cute crd2idx evaluation. For
  // the Sw<3,4,3> (8, 64):(64, 1) 16-bit-flagged atom tiled M-first
  // (Step<_1,_2,_3>):
  //   pre = (n%8)*64 + (k%64) + (n/8)*512 + (k/64)*8192 + stage*16384
  //   off = pre XOR ((n%8) << 3)
  // The swizzle of the smem_ptr_flag_bits<16> atom acts in the BYTE domain
  // (element bits [3..6) ^= element bits [6..9) = n%8), matching the UMMA
  // SWIZZLE_128B descriptor; evaluating the flagged ComposedLayout directly
  // applies the swizzle in the wrong unit, so the static_asserts below pin
  // the formula against the position-independent view (what the numerics
  // verified end-to-end).
  CUTLASS_HOST_DEVICE
  static constexpr int compute_elem_offset(int n, int k, int stage) {
    int const pre = (n & 7) * 64 + (k & 63) + (n >> 3) * 512 + (k >> 6) * 8192 +
                    stage * 16384;
    return pre ^ ((n & 7) << 3);
  }
  using SmemLayoutAComputePosInd = decltype(as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(static_cast<ElementAMma*>(nullptr)),
                  SmemLayoutAComputeMK{})).layout());
  static_assert(SmemLayoutAComputePosInd{}(0, 0, 0) == compute_elem_offset(0, 0, 0));
  static_assert(SmemLayoutAComputePosInd{}(1, 0, 0) == compute_elem_offset(1, 0, 0));
  static_assert(SmemLayoutAComputePosInd{}(9, 8, 0) == compute_elem_offset(9, 8, 0));
  static_assert(SmemLayoutAComputePosInd{}(3, 17, 1) == compute_elem_offset(3, 17, 1));
  static_assert(SmemLayoutAComputePosInd{}(77, 101, 1) == compute_elem_offset(77, 101, 1));
  static_assert(SmemLayoutAComputePosInd{}(127, 127, 0) == compute_elem_offset(127, 127, 0));
  static_assert(SmemLayoutAComputePosInd{}(64, 64, 0) == compute_elem_offset(64, 64, 0));
  static_assert(SmemLayoutAComputePosInd{}(15, 40, 0) == compute_elem_offset(15, 40, 0));

  // Scale staging: stock (lines 325-328).
  using SmemLayoutScale = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomScale{},
      append(CtaShapeA_MK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
      Step<_1,_2,_3>{}));
  static constexpr int kScalesPerStage = cosize(take<0,3>(SmemLayoutScale{}));
  static constexpr int kScaleKGroupsPerStage = CtaTileK / ScaleGranularityK;
  static_assert(kScalesPerStage == CtaTileN_Weights * kScaleKGroupsPerStage,
                "unexpected scale smem layout");

  static_assert(DispatchPolicy::Load2TransformPipelineStageCount >= 2 &&
                DispatchPolicy::Transform2MmaPipelineStageCount >= 2,
                "Specialization requires Stages set to value 2 or more.");
  // SS MMA only: A operand consumed from SMEM through a descriptor.
  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "swordfish prefill requires an SS UMMA atom (A and B from SMEM)");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>,
                "swordfish prefill v1 uses plain (non-multicast) TMA for the packed operand");

  static constexpr ConversionMode KernelConversionMode = ConversionMode::ConvertAndScale;
  static constexpr bool ModeHasScales = true;

  static constexpr size_t SmemAlignmentA = 1024;
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});

  struct PipelineStorage {
    using Load2TransformPipelineStorage = typename Load2TransformPipeline::SharedStorage;
    alignas(16) Load2TransformPipelineStorage load2transform_pipeline;
    using Load2MmaPipelineStorage = typename Load2MmaPipeline::SharedStorage;
    alignas(16) Load2MmaPipelineStorage load2mma_pipeline;
    using Transform2MmaPipelineStorage = typename Transform2MmaPipeline::SharedStorage;
    alignas(16) Transform2MmaPipelineStorage transform2mma_pipeline;
    using Mma2AccumPipelineStorage = typename Mma2AccumPipeline::SharedStorage;
    alignas(16) Mma2AccumPipelineStorage mma2accum_pipeline;
  };

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      struct TensorStorageUntransformed {
        alignas(1024) cute::ArrayEngine<uint8_t, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
        cute::ArrayEngine<NonVoidElementScale, cute::cosize_v<SmemLayoutScale>> smem_scale;
        cute::ArrayEngine<NonVoidElementScale, HasZp ? cute::cosize_v<SmemLayoutScale> : 1> smem_zero;
      };
      struct TensorStorageTransformed {
        alignas(1024) cute::ArrayEngine<ElementAMma, cute::cosize_v<SmemLayoutACompute>> smem_ACompute;
      };
      TensorStorageUntransformed input;
      TensorStorageTransformed compute;
    } tensors;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  // Per-stage mbarrier transaction bytes: packed bytes + scales. Only TMA
  // bytes count here (the transform's smem stores arrive through the
  // transform2mma pipeline and not this barrier, per the stock collective's mbarrier
  // caution).
  static constexpr uint32_t kScaleTxBytes =
      cutlass::bits_to_bytes(kScalesPerStage * cute::sizeof_bits_v<NonVoidElementScale>);
  static constexpr uint32_t TmaTransactionBytes_A =
      kStageBytes + kScaleTxBytes * (HasZp ? 2 : 1);
  // AtomThrShape-scaled as in stock. The cta_group::2 TMA loads both CTAs'
  // B halves with one instruction and its arrival, covering both halves'
  // bytes, lands on the MMA leader's barrier. In 1-SM this reduces to one
  // tile.
  static constexpr uint32_t TmaTransactionBytes_B =
      cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytes_A + TmaTransactionBytes_B;

  // Host side kernel arguments (stock lines 424-432; ptr_A is the packed ABI
  // tensor base; dA is carried for interface parity but the packed layout is
  // derived from the problem shape).
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementScale const* ptr_S{nullptr};
    LayoutScale layout_S{};
    ElementZero const* ptr_Z{nullptr};  // scale-shaped (8 - zp) * scale rows
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(ClusterShape{}),
                                        make_tile(typename TiledMma::AtomThrID{})));

    // CHANGE (1): TMA over the packed byte tensor (256, 8, KB, NB).
    using GmemLayoutAPacked = Layout<Shape<_256, Int<kBlockRows>, int32_t, int32_t>,
                                     Stride<_1, _256, Int<kBlockBytes>, int64_t>>;
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(make_gmem_ptr(static_cast<uint8_t const*>(nullptr)), GmemLayoutAPacked{}),
        SmemLayoutA{}(_,_,_,_,cute::Int<0>{})));

    using TMA_B = decltype(make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
    );

    using TMA_Scale = decltype(make_tma_atom_A_sm100(
        GmemTiledCopyScale{},
        make_tensor(static_cast<NonVoidElementScale const*>(nullptr), LayoutScale{}),
        SmemLayoutScale{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
    );

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_Scale tma_load_scale;
    TMA_Scale tma_load_zero;  // constructed over ptr_Z (layout shared with S)
    uint32_t tma_transaction_bytes{TmaTransactionBytes};
    int32_t blocks_k{0};  // KB = K / 64
    int32_t blocks_n{0};  // NB = N_weights / 64
  };

  CUTLASS_DEVICE
  SwordfishMainloopSm100MixedInput(Params const& params, ClusterShape, uint32_t block_rank_in_cluster)
    : observed_tma_load_a_(&params.tma_load_a)
    , observed_tma_load_b_(&params.tma_load_b)
    , block_rank_in_cluster_(block_rank_in_cluster) {}

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
    ProblemShape const& problem_shape,
    Arguments const& args,
    void* workspace,
    cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    (void) workspace;
    (void) hw_info;

    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;  // M = weight N (swapped operands)

    int32_t const blocks_k = int32_t(K / kBlockK);
    int32_t const blocks_n = int32_t(M / kBlockN);

    // Packed operand as a dense byte tensor (ABI invariant I3).
    auto gA_layout = make_layout(
        make_shape(_256{}, Int<kBlockRows>{}, blocks_k, blocks_n),
        make_stride(_1{}, _256{}, Int<kBlockBytes>{}, int64_t(kBlockBytes) * blocks_k));
    Tensor tensor_a = make_tensor(
        make_gmem_ptr(reinterpret_cast<uint8_t const*>(args.ptr_A)), gA_layout);
    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_,_,_,_,cute::Int<0>{}));

    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));
    auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    Tensor tensor_scale = make_tensor(detail::get_logical_ptr(args.ptr_S), args.layout_S);
    typename Params::TMA_Scale tma_load_scale = make_tma_atom_A_sm100(
        GmemTiledCopyScale{},
        tensor_scale,
        SmemLayoutScale{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    Tensor tensor_zero = make_tensor(
        detail::get_logical_ptr(reinterpret_cast<NonVoidElementScale const*>(args.ptr_Z)),
        args.layout_S);
    typename Params::TMA_Scale tma_load_zero = make_tma_atom_A_sm100(
        GmemTiledCopyScale{},
        tensor_zero,
        SmemLayoutScale{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    return {
        tma_load_a,
        tma_load_b,
        tma_load_scale,
        tma_load_zero,
        TmaTransactionBytes,
        blocks_k,
        blocks_n };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    // ABI v1 tail policy: reject non-multiples (M here = weight N).
    implementable &= (M % CtaTileN_Weights == 0);
    implementable &= (K % CtaTileK == 0);
    implementable &= (L == 1);
    implementable &= (args.ptr_S != nullptr);
    implementable &= ((args.ptr_Z != nullptr) == HasZp);

    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB>();
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: swordfish prefill shape/argument requirements not met.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE static void
  prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_scale.get_tma_descriptor());
    if constexpr (HasZp) {
      cute::prefetch_tma_descriptor(params.tma_load_zero.get_tma_descriptor());
    }
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE auto
  partition_accumulator_shape() {
    return partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));
  }

  /// Produce the inputs to the transform threads by loading inputs from gmem -> smem.
  /// (stock lines 735-814; the A copy now walks the packed block tensor)
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TileCoordMNKL,
    class KTileIterator,
    class... Ts
  >
  CUTLASS_DEVICE auto
  load_A(
      Params const& params,
      Load2TransformPipeline load2xform_pipeline,
      Load2TransformPipelineState load2xform_pipeline_state,
      cute::tuple<GTensorA, GTensorB,
                  GTensorPartitionedA, GTensorPartitionedB,
                  STensorA, STensorB,
                  uint16_t, uint16_t,
                  cute::tuple<Ts...>> const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_gA, unused_gB,
          tAgA_nk, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b, extra_input_partitions] = load_inputs;

    // tAgA_nk : ((TMA), 1, 1, K_TILES, N_TILES) over the packed byte tensor,
    // tiled PER-CTA (128 weight-cols). Unlike the scales (multicast, routed
    // through cta_mma.partition_A which applies the atom split), the packed A
    // is DISJOINT across the 2-SM pair: each CTA loads its own 128-col half.
    // So index the CTA's own 128-tile: AtomThrID*(base MMA tile) + atom half.
    // Reduces to get<0>(cta_coord_mnkl) in the 1-SM case (AtomThrID size 1).
    constexpr int kAtom = size(typename TiledMma::AtomThrID{});
    const int a_ntile = kAtom * (get<0>(cta_coord_mnkl) / kAtom) +
                        int(block_rank_in_cluster_);
    Tensor tAgA = tAgA_nk(_, _0{}, _0{}, _, a_ntile);

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2xform_pipeline_flag = load2xform_pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

    using BarrierType = typename Load2TransformPipeline::ProducerBarrierType;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      load2xform_pipeline.producer_acquire(load2xform_pipeline_state, load2xform_pipeline_flag);

      int tile_A_write_stage = load2xform_pipeline_state.index();
      BarrierType* load2xform_tma_barrier = load2xform_pipeline.producer_get_barrier(load2xform_pipeline_state);

      ++load2xform_pipeline_state;
      skip_wait = (k_tile_count <= 1);
      load2xform_pipeline_flag = load2xform_pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

      // TMA load: 4 packed blocks (2 nb x 2 kb) = one k tile of this CTA's
      // weight stripe, plus the k tile's group scales.
      copy(observed_tma_load_a_->with(*load2xform_tma_barrier, mcast_mask_a),
           tAgA(_,*k_tile_iter), tAsA(_,tile_A_write_stage));

      auto tSgS_mkl = get<0>(extra_input_partitions);
      auto tSgS = tSgS_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
      auto tSsS = get<1>(extra_input_partitions);
      copy(params.tma_load_scale.with(*load2xform_tma_barrier, mcast_mask_a),
           tSgS(_,*k_tile_iter), tSsS(_,tile_A_write_stage));

      if constexpr (HasZp) {
        auto tZgZ_mkl = get<2>(extra_input_partitions);
        auto tZgZ = tZgZ_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
        auto tZsZ = get<3>(extra_input_partitions);
        copy(params.tma_load_zero.with(*load2xform_tma_barrier, mcast_mask_a),
             tZgZ(_,*k_tile_iter), tZsZ(_,tile_A_write_stage));
      }

      ++k_tile_iter;
    }

    return cute::make_tuple(load2xform_pipeline_state, k_tile_iter);
  }

  /// (stock lines 816-876, unchanged: activations)
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TileCoordMNKL,
    class KTileIterator,
    class... Ts
  >
  CUTLASS_DEVICE auto
  load_B(
      Params const& params,
      Load2MmaPipeline load2mma_pipeline,
      Load2MmaPipelineState load2mma_pipeline_state,
      cute::tuple<GTensorA, GTensorB,
                  GTensorPartitionedA, GTensorPartitionedB,
                  STensorA, STensorB,
                  uint16_t, uint16_t,
                  cute::tuple<Ts...>> const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_gA, unused_gB,
          tAgA_nk, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b, extra_input_partitions] = load_inputs;

    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2mma_pipeline_flag = load2mma_pipeline.producer_try_acquire(load2mma_pipeline_state, skip_wait);

    using BarrierType = typename Load2TransformPipeline::ProducerBarrierType;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      load2mma_pipeline.producer_acquire(load2mma_pipeline_state, load2mma_pipeline_flag);

      int tile_B_write_stage = load2mma_pipeline_state.index();
      BarrierType* load2mma_tma_barrier = load2mma_pipeline.producer_get_barrier(load2mma_pipeline_state);

      ++load2mma_pipeline_state;
      skip_wait = (k_tile_count <= 1);
      load2mma_pipeline_flag = load2mma_pipeline.producer_try_acquire(load2mma_pipeline_state, skip_wait);

      copy(observed_tma_load_b_->with(*load2mma_tma_barrier, mcast_mask_b),
           tBgB(_,*k_tile_iter), tBsB(_,tile_B_write_stage));

      ++k_tile_iter;
    }

    return cute::make_tuple(load2mma_pipeline_state, k_tile_iter);
  }

  /// Set up the data needed by this collective for load.
  /// (stock lines 883-973; A partitioning swapped for the packed byte tensor)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_storage) const {
    auto [M,N,K,L] = problem_shape_MNKL;

    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    // ---- A: packed byte tensor, tiled by (256, 8, kb-per-tile, nb-per-tile).
    Tensor mA_packed = observed_tma_load_a_->get_tma_tensor(
        make_shape(_256{}, _8{}, params.blocks_k, params.blocks_n));
    Tensor gA_packed = flat_divide(mA_packed, SwordfishTilerA{});
    // (256, 8, KBt, NBt, 1, 1, K_TILES, N_TILES)

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});

    Layout cta_layout_mnk  = make_layout(ClusterShape{});
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    auto [tAgA_nk, tAsA] = tma_partition(*observed_tma_load_a_,
                                      Int<0>{}, Layout<_1>{},
                                      group_modes<0,4>(sA), group_modes<0,4>(gA_packed));

    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    // ---- scales: stock path (lines 925-947).
    Tensor mS_mkl = params.tma_load_scale.get_tma_tensor(shape(LayoutScale{}));
    Tensor gS_mkl = local_tile(mS_mkl, TileShape{}, make_coord(_,_,_), Step<_1, cute::Underscore,_1>{});
    Tensor sS = make_tensor(make_smem_ptr(shared_storage.input.smem_scale.begin()), SmemLayoutScale{});
    Tensor tCgS_mkl = cta_mma.partition_A(gS_mkl);

    auto [tSgS_mkl, tSsS] = tma_partition(params.tma_load_scale,
                                    get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                    group_modes<0,3>(sS), group_modes<0,3>(tCgS_mkl));

    // Zero rows share the scale layout; the partitions are layout-only and
    // never copied from when HasZp is false.
    Tensor mZ_mkl = params.tma_load_zero.get_tma_tensor(shape(LayoutScale{}));
    Tensor gZ_mkl = local_tile(mZ_mkl, TileShape{}, make_coord(_,_,_), Step<_1, cute::Underscore,_1>{});
    Tensor sZ = make_tensor(make_smem_ptr(shared_storage.input.smem_zero.begin()), SmemLayoutScale{});
    Tensor tCgZ_mkl = cta_mma.partition_A(gZ_mkl);

    auto [tZgZ_mkl, tZsZ] = tma_partition(params.tma_load_zero,
                                    get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                    group_modes<0,3>(sZ), group_modes<0,3>(tCgZ_mkl));

    return cute::make_tuple(
        gA_mkl, gB_nkl,                        // for scheduler (shapes only)
        tAgA_nk, tBgB_nkl, tAsA, tBsB,         // for input tensor values
        mcast_mask_a, mcast_mask_b,            // multicast masks
        cute::make_tuple(tSgS_mkl, tSsS, tZgZ_mkl, tZsZ));
  }

  /// CHANGE (2): the Transform stage (stock lines 975-1059). Consumes the
  /// packed bytes in Marlin tile order, dequantizes via the marlin u4b8 LOP3
  /// sequences, applies group scales, and writes K-major bf16 into the
  /// tcgen05 compute buffer.
  ///
  /// Thread assignment (128 threads, 4 warps; ABI read contract):
  ///   warp w      <-> packed block (nb = w>>1, kb = w&1) of the k tile
  ///   lane T      <-> words [4T, 4T+4) of each of the block's 4 sub-tiles
  ///                   (one 16 B vector load per sub-tile)
  ///   word 4T+j   ->  dequants to columns {16j+c, 16j+8+c} (c = T/4) at
  ///                   k rows 16s + 2t + {0,1,8,9} (t = T%4) of the sub-tile.
  template<
    class KTileIterator, class Accumulator,
    class GTensorA, class SrcTensorA, class DstTensorA, class ScaleTensor,
    class ZeroTensor
  >
  CUTLASS_DEVICE auto
  transform(
      Load2TransformPipeline load2transform_pipeline,
      Load2TransformPipelineState load2transform_pipeline_consumer_state,
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_producer_state,
      Accumulator accumulators,
      cute::tuple<GTensorA, SrcTensorA, DstTensorA, ScaleTensor, ZeroTensor> input_operands,
      KTileIterator k_tile_iter, int k_tile_count) {

    cutlass::arch::NamedBarrier transform_bar(NumTransformationThreads, cutlass::arch::ReservedNamedBarriers::TransformBarrier);

    auto [unused_gA, sA, sACompute, sS, sZ] = input_operands;
    uint8_t const* smem_a_base = reinterpret_cast<uint8_t const*>(raw_pointer_cast(sA.data()));
    NonVoidElementScale const* smem_s_base = raw_pointer_cast(sS.data());
    NonVoidElementScale const* smem_z_base = raw_pointer_cast(sZ.data());
    ElementAMma* smem_c_base = raw_pointer_cast(sACompute.data());

    const int tid  = threadIdx.x % NumTransformationThreads;
    const int lane = tid % 32;
    const int warp = tid / 32;
    // Parametric width: kWarpsPerBlock warps share a packed block, each
    // covering kSubTilesPerWarp of its 4 sub-tiles (4 warps -> whole block,
    // 8 warps -> half each).
    constexpr int kWarpsPerBlock = int(NumTransformationThreads) / 128;
    constexpr int kSubTilesPerWarp = 4 / kWarpsPerBlock;
    const int blk_id = warp / kWarpsPerBlock;
    const int nbi  = blk_id >> 1;  // n64 block within the 128-col stripe
    const int kbi  = blk_id & 1;   // k64 block within the k tile
    const int sh   = (warp % kWarpsPerBlock) * kSubTilesPerWarp;
    const int c    = lane >> 2;   // column octet within each n16 group
    const int t    = lane & 3;    // k-quad

    // This warp's k rows fall in [kbi*64, kbi*64+64), one scale k-group for
    // group sizes 64 and 128 and two consecutive groups at 32.
    const int scale_kg = (kbi * kBlockK) / ScaleGranularityK;

    // Per-thread compute-buffer addressing (see compute_elem_offset). The
    // word (s, j) covers columns n = nbi*64 + 16j + 8b + c and rows
    // k = kbi*64 + 16s + 2t + 8p, and n%8 == c for all of them, so the
    // swizzle XOR mask (c << 3) is a per-thread constant confined to the
    // (k%64) bits:
    //   off = A(j,b) + (((16s + 8p) ^ (c<<3)) + 2t)
    //   A   = c*64 + (nbi*8 + 2j + b)*512 + kbi*8192 + stage*16384
    // Everything except the stage term is loop-invariant.
    int klow[4][2];
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < 4; s++) {
      klow[s][0] = ((16 * s) ^ (c << 3)) + 2 * t;
      klow[s][1] = ((16 * s + 8) ^ (c << 3)) + 2 * t;
    }
    ElementAMma* const thread_c_base = smem_c_base + c * 64 + kbi * 8192;

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
    auto transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      load2transform_pipeline.consumer_wait(load2transform_pipeline_consumer_state, load2transform_flag);
      transform2mma_pipeline.producer_acquire(transform2mma_pipeline_producer_state, transform2mma_flag);

      int load2transform_consumer_index = load2transform_pipeline_consumer_state.index();
      int transform2mma_producer_index = transform2mma_pipeline_producer_state.index();

      auto curr_load2transform_pipeline_consumer_state = load2transform_pipeline_consumer_state;
      auto curr_transform2mma_pipeline_producer_state = transform2mma_pipeline_producer_state;

      // ---- pull this thread's packed words and scales into registers
      // (this warp's share of the block: sub-tiles [sh, sh+kSubTilesPerWarp))
      uint32_t w[kSubTilesPerWarp][WBits == 8 ? 8 : 4];
      uint8_t const* blk = smem_a_base
          + load2transform_consumer_index * kStageBytes
          + (nbi * kBlocksPerTileK + kbi) * kBlockBytes
          + lane * (WBits == 8 ? 32 : 16);
      CUTLASS_PRAGMA_UNROLL
      for (int si = 0; si < kSubTilesPerWarp; si++) {
        uint4 v = *reinterpret_cast<uint4 const*>(blk + (sh + si) * kSubTileBytes);
        w[si][0] = v.x; w[si][1] = v.y; w[si][2] = v.z; w[si][3] = v.w;
        if constexpr (WBits == 8) {
          uint4 v1 = *reinterpret_cast<uint4 const*>(
              blk + (sh + si) * kSubTileBytes + 16);
          w[si][4] = v1.x; w[si][5] = v1.y; w[si][6] = v1.z; w[si][7] = v1.w;
        }
      }

      auto bcast2 = [](NonVoidElementScale v) {
        if constexpr (kActF16) {
          return __half2half2(reinterpret_cast<__half const&>(v));
        } else {
          return __bfloat162bfloat162(
              reinterpret_cast<__nv_bfloat16 const&>(v));
        }
      };
      // [group of the warp's share][j][column octet 0/1] broadcast pairs
      Elem2 sreg[kScaleGroupsPerWarp][4][2];
      Elem2 zreg[kScaleGroupsPerWarp][4][2];
      CUTLASS_PRAGMA_UNROLL
      for (int kg = 0; kg < kScaleGroupsPerWarp; kg++) {
        NonVoidElementScale const* srow = smem_s_base
            + load2transform_consumer_index * kScalesPerStage
            + (scale_kg + kg) * CtaTileN_Weights + nbi * kBlockN;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; j++) {
          sreg[kg][j][0] = bcast2(srow[16 * j + c]);
          sreg[kg][j][1] = bcast2(srow[16 * j + 8 + c]);
        }
        if constexpr (HasZp) {
          NonVoidElementScale const* zrow = smem_z_base
              + load2transform_consumer_index * kScalesPerStage
              + (scale_kg + kg) * CtaTileN_Weights + nbi * kBlockN;
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 4; j++) {
            zreg[kg][j][0] = bcast2(zrow[16 * j + c]);
            zreg[kg][j][1] = bcast2(zrow[16 * j + 8 + c]);
          }
        }
      }

      // Loads from SMEM are done. Signal the mainloop load as early as possible
      transform_bar.sync();
      load2transform_pipeline.consumer_release(curr_load2transform_pipeline_consumer_state);

      // ---- dequant + scale + I5 stores (4x 32-bit per packed word)
      ElementAMma* const stage_base =
          thread_c_base + transform2mma_producer_index * 16384;
      CUTLASS_PRAGMA_UNROLL
      for (int si = 0; si < kSubTilesPerWarp; si++) {
        const int s = sh + si;
        // Sub-tile s covers k rows [16s, 16s+16); group index within the
        // warp's register set (always 0 at granularity 64 and 128).
        const int kg = kScaleGroupsPerWarp == 1 ? 0 : (s / 2) % 2;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; j++) {
          ElementAMma* const row0 = stage_base + (nbi * 8 + 2 * j) * 512;
          ElementAMma* const row1 = row0 + 512;
          Elem2 f0[2], f1[2];
          if constexpr (WBits == 8 && kActF16) {
            swordfish_detail::dequant_u8b128_f16x2(w[si][2 * j], f0);
            swordfish_detail::dequant_u8b128_f16x2(w[si][2 * j + 1], f1);
          } else if constexpr (WBits == 8) {
            swordfish_detail::dequant_u8b128_bf16x2(w[si][2 * j], f0);
            swordfish_detail::dequant_u8b128_bf16x2(w[si][2 * j + 1], f1);
          } else if constexpr (kActF16) {
            swordfish_detail::dequant_u4b8_f16x2(w[si][j], f0);
            swordfish_detail::dequant_u4b8_f16x2(w[si][j] >> 8, f1);
          } else {
            swordfish_detail::dequant_u4b8_bf16x2(w[si][j], f0);
            swordfish_detail::dequant_u4b8_bf16x2(w[si][j] >> 8, f1);
          }
          if constexpr (HasZp) {
            f0[0] = __hfma2(f0[0], sreg[kg][j][0], zreg[kg][j][0]);
            f0[1] = __hfma2(f0[1], sreg[kg][j][0], zreg[kg][j][0]);
            f1[0] = __hfma2(f1[0], sreg[kg][j][1], zreg[kg][j][1]);
            f1[1] = __hfma2(f1[1], sreg[kg][j][1], zreg[kg][j][1]);
          } else {
            f0[0] = __hmul2(f0[0], sreg[kg][j][0]);
            f0[1] = __hmul2(f0[1], sreg[kg][j][0]);
            f1[0] = __hmul2(f1[0], sreg[kg][j][1]);
            f1[1] = __hmul2(f1[1], sreg[kg][j][1]);
          }
          *reinterpret_cast<uint32_t*>(row0 + klow[s][0]) = reinterpret_cast<uint32_t const&>(f0[0]);
          *reinterpret_cast<uint32_t*>(row0 + klow[s][1]) = reinterpret_cast<uint32_t const&>(f0[1]);
          *reinterpret_cast<uint32_t*>(row1 + klow[s][0]) = reinterpret_cast<uint32_t const&>(f1[0]);
          *reinterpret_cast<uint32_t*>(row1 + klow[s][1]) = reinterpret_cast<uint32_t const&>(f1[1]);
        }
      }

      // Publish the generic STS to the async proxy the tcgen05 MMA reads
      // through, then let the MMA know this stage is transformed.
      cutlass::arch::fence_view_async_shared();
      transform2mma_pipeline.producer_commit(curr_transform2mma_pipeline_producer_state);

      ++load2transform_pipeline_consumer_state;
      ++transform2mma_pipeline_producer_state;

      skip_wait = (k_tile_count <= 1);
      load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
      transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);
    }
    return cute::make_tuple(load2transform_pipeline_consumer_state, transform2mma_pipeline_producer_state);
  }

  /// (stock lines 1061-1144, replaced. No tiled-copy plumbing, the transform
  /// addresses the byte staging and the compute buffer directly)
  template<class ProblemShape_MNKL, class Accumulator>
  CUTLASS_DEVICE auto
  transform_init(
      Params const& params,
      ProblemShape_MNKL const& problem_shape_MNKL,
      Accumulator accumulators,
      TensorStorage& shared_storage) {
    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});
    Tensor sACompute = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutAComputeMK{}));
    Tensor sS = make_tensor(make_smem_ptr(shared_storage.input.smem_scale.begin()), SmemLayoutScale{});
    Tensor sZ = make_tensor(make_smem_ptr(shared_storage.input.smem_zero.begin()), SmemLayoutScale{});

    return cute::make_tuple(gA_mkl, sA, sACompute, sS, sZ);
  }

  /// Perform a collective-scoped matrix multiply-accumulate: stock (lines
  /// 1146-1232), SS descriptor path.
  template <
    class FrgEngine, class FrgLayout,
    class TensorA, class TensorB
  >
  CUTLASS_DEVICE auto
  mma(
      Load2MmaPipeline load2mma_pipeline,
      Load2MmaPipelineState load2mma_pipeline_consumer_state,
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_consumer_state,
      Mma2AccumPipeline mma2accum_pipeline,
      Mma2AccumPipelineState mma2accum_pipeline_producer_state,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      cute::tuple<TensorA, TensorB> const& input_operands,
      int k_tile_count
  ) {
    TiledMma tiled_mma;

    auto curr_load2mma_pipeline_consumer_state = load2mma_pipeline_consumer_state;
    auto next_load2mma_pipeline_consumer_state = load2mma_pipeline_consumer_state;

    auto curr_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;
    auto next_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;

    uint32_t skip_wait = (k_tile_count <= 0);
    auto transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);
    auto load2mma_flag = load2mma_pipeline.consumer_try_wait(next_load2mma_pipeline_consumer_state, skip_wait);
    ++next_transform2mma_pipeline_consumer_state;
    ++next_load2mma_pipeline_consumer_state;

    auto const [tCrA, tCrB] = input_operands;

    mma2accum_pipeline.producer_acquire(mma2accum_pipeline_producer_state);

    int mma2accum_pipeline_producer_state_index = mma2accum_pipeline_producer_state.index();
    auto tCtC = accumulators(_,_,_,mma2accum_pipeline_producer_state_index);
    auto curr_mma2accum_pipeline_producer_state = mma2accum_pipeline_producer_state;
    ++mma2accum_pipeline_producer_state;

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      load2mma_pipeline.consumer_wait(curr_load2mma_pipeline_consumer_state, load2mma_flag);
      transform2mma_pipeline.consumer_wait(curr_transform2mma_pipeline_consumer_state, transform2mma_flag);

      int load2mma_pipeline_consumer_state_index = curr_load2mma_pipeline_consumer_state.index();
      int transform2mma_pipeline_consumer_state_index = curr_transform2mma_pipeline_consumer_state.index();

      auto tCrA0 = tCrA(_,_,_,transform2mma_pipeline_consumer_state_index);
      auto tCrB0 = tCrB(_,_,_,load2mma_pipeline_consumer_state_index);

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); k_block ++) {
        cute::gemm(tiled_mma, tCrA0(_,_,k_block), tCrB0(_,_,k_block), tCtC);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      load2mma_pipeline.consumer_release(curr_load2mma_pipeline_consumer_state);
      transform2mma_pipeline.consumer_release(curr_transform2mma_pipeline_consumer_state);

      skip_wait = (k_tile_count <= 1);
      load2mma_flag = load2mma_pipeline.consumer_try_wait(next_load2mma_pipeline_consumer_state, skip_wait);
      transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);

      curr_load2mma_pipeline_consumer_state = next_load2mma_pipeline_consumer_state;
      curr_transform2mma_pipeline_consumer_state = next_transform2mma_pipeline_consumer_state;

      ++next_load2mma_pipeline_consumer_state;
      ++next_transform2mma_pipeline_consumer_state;
    }

    mma2accum_pipeline.producer_commit(curr_mma2accum_pipeline_producer_state);

    return cute::make_tuple(curr_load2mma_pipeline_consumer_state, curr_transform2mma_pipeline_consumer_state, mma2accum_pipeline_producer_state);
  }

  /// (stock lines 1234-1255; SS branch only)
  template<class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  mma_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TensorStorage& shared_storage) const {
    TiledMma tiled_mma;

    Tensor sACompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutACompute{});
    Tensor tCrA = tiled_mma.make_fragment_A(sACompute);

    Tensor sB = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});
    Tensor tCrB = tiled_mma.make_fragment_B(sB);
    return cute::make_tuple(tCrA, tCrB);
  }

  template<class FrgEngine, class FrgLayout, class TmemCopyAtom, class EpilogueTile>
  CUTLASS_DEVICE auto
  accum_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TmemCopyAtom tmem_cp_atom, EpilogueTile epilogue_tile) {
    return accumulators;
  }

private:
  /// gA_mkl slot 0 is only consumed for its SHAPE (the kernel derives the
  /// k-tile count from shape<3>); return an identity tensor with the stock
  /// tiled shape. gB_nkl is the stock activation tiling.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  constexpr auto
  tile_input_tensors(Params const& params, ProblemShape_MNKL const& problem_shape_MNKL) const {
    using X = cute::Underscore;
    auto [M,N,K,L] = problem_shape_MNKL;

    auto gA_mkl = make_identity_tensor(make_shape(
        size<0>(TileShape{}), size<2>(TileShape{}),
        ceil_div(M, size<0>(TileShape{})), ceil_div(K, size<2>(TileShape{})), L));

    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  typename Params::TMA_A const* observed_tma_load_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_b_ = nullptr;
  uint32_t block_rank_in_cluster_ = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
