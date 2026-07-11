// w4a16 prefill GEMM over the Swordfish ABI v1 packed weight, using the
// forked sm100 tcgen05 mixed-input collective
// (swordfish_prefill_mainloop.cuh) that reads the packed layout directly.
// v1 accepts bf16 activations/scales, group_size 128, and K and N multiples
// of 128, with a 2-SM (cta_group::2) MMA.

#include <algorithm>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"
#include "swordfish_types.cuh"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "swordfish_prefill_mainloop.cuh"
#include "swordfish_prefill_kernel.cuh"

namespace swordfish {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace prefill {

using namespace cute;

using MmaType = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using ClusterShape = Shape<_2, _1, _1>;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<MmaType>::value;

constexpr int kScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<
    /*GranN=*/1, kScaleGranularityK>;
using LayoutScale = decltype(ScaleConfig::deduce_layout_scale());

using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;  // packed slot (unused)
using StrideB = cutlass::gemm::TagToStrideB_t<
    typename cutlass::layout::LayoutTranspose<cutlass::layout::RowMajor>::type>;  // activations
using StridePairA = decltype(cute::make_tuple(StrideA{}, LayoutScale{}));
using ElementPairA = cute::tuple<uint8_t, MmaType>;

static constexpr cute::UMMA::Major UmmaMajorA = cute::UMMA::Major::K;
static constexpr cute::UMMA::Major UmmaMajorB = cute::UMMA::Major::K;

// 2-SM (cta_group::2) MMA config, parameterized on the instruction N width.
// Tile-M (the weight N dimension) spans the two CTAs of an SM pair, 128
// columns each. N=256 wins compute-bound shapes (per-instruction issue
// overhead caps a 256x128 UMMA at about two thirds of the 256x256 rate);
// N=128 wins K-heavy shapes, where the wide tile starves the K pipeline.
template <int kTileN>
struct PrefillCfg {
  using MmaTileShape = Shape<_256, Int<kTileN>, _128>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      MmaType, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
      MmaType, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
      EpilogueSchedule
    >::CollectiveOp;

  // The CUTLASS convenience builder has no branch for 2-SM atoms with
  // smem-sourced A, so the SS atom is constructed directly. M is the full
  // cluster tile-M spanning both CTAs.
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_2x1SM_SS<MmaType, MmaType, ElementAccumulator,
          256, kTileN, UmmaMajorA, UmmaMajorB>{}));

  // partition_shape_A/B take CTA-local shapes for a 2-SM atom.
  using CtaShape = decltype(shape_div(MmaTileShape{}, ClusterShape{}));
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(CtaShape{}),
                                                                         cute::size<2>(CtaShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(CtaShape{}),
                                                                         cute::size<2>(CtaShape{}))));
  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));

  using SmemLayoutAtomACompute = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, MmaType, BlockTileA_M, BlockTileA_K>());
  using SmemLayoutAtomPairA = cutlass::gemm::collective::detail::CollectiveMmaEmulatedLayoutAtomType<
      SmemLayoutAtomACompute, SmemLayoutAtomACompute>;
  using CopyAtomPairA = cutlass::gemm::collective::detail::CollectiveMmaEmulatedCopyType<
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, uint8_t>,
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, MmaType>>;

  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, MmaType, BlockTileB_N, BlockTileB_K>());
  using SmemLayoutAtomPairB = cutlass::gemm::collective::detail::CollectiveMmaEmulatedLayoutAtomType<
      SmemLayoutAtomB, SmemLayoutAtomB>;
  using CopyAtomPairB = cutlass::gemm::collective::detail::CollectiveMmaEmulatedCopyType<
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, MmaType>,
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, MmaType>>;

  // Pipeline stage counts, derived from the smem budget.
  static constexpr int kSmemCapacity = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes;
  static constexpr int kKernelCarveout = 2048;
  static constexpr int kEpilogueBytes = int(sizeof(typename CollectiveEpilogue::SharedStorage));
  // TMEM is 512 columns and an accumulator stage needs kTileN of them.
  static constexpr int kAccumStages = 512 / kTileN;
  // B stage is the CTA's half of the N-split activation tile, kTileN/2 rows.
  static constexpr int kInputStageBytes = 8192 + kTileN * 128 + 256 + 64;
  static constexpr int kComputeStageBytes = 32768 + 32;
  static constexpr int kT2MStages = 2;
  static constexpr int kAvail = kSmemCapacity - kKernelCarveout - kEpilogueBytes
                              - kAccumStages * 32 - kT2MStages * kComputeStageBytes;
  static constexpr int kL2TStages = kAvail / kInputStageBytes < 4 ? kAvail / kInputStageBytes : 4;
  static_assert(kL2TStages >= 2, "not enough SMEM for two input stages");

  using CollectiveMainloop = cutlass::gemm::collective::SwordfishMainloopSm100MixedInput<
      kL2TStages, kT2MStages, /*SchedulerStages=*/3, kAccumStages, ClusterShape,
      MmaTileShape,
      ElementPairA, StridePairA,
      MmaType, StrideB,
      TiledMma,
      cute::SM90_TMA_LOAD, SmemLayoutAtomPairA, CopyAtomPairA, cute::identity,
      // Activations must use the cta_group::2 TMA. The 2x1SM atom N-splits B
      // across the CTA pair and the leader's MMA reads the peer's half, so
      // the copy must deliver both halves with one arrival on the leader's
      // barrier. A per-CTA TMA arrives at local barriers and races.
      cute::SM100_TMA_2SM_LOAD, SmemLayoutAtomPairB, CopyAtomPairB, cute::identity>;

  // Forked kernel layer whose warp layout derives from the collective's
  // NumTransformationThreads.
  using GemmKernel = cutlass::gemm::kernel::SwordfishPrefillKernel<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Cfg>
void run(torch::stable::Tensor& a, torch::stable::Tensor& b_packed,
         torch::stable::Tensor& group_scales, torch::stable::Tensor& c,
         int M, int N, int K, cudaStream_t stream) {
  using Gemm = typename Cfg::Gemm;
  using GemmKernel = typename Cfg::GemmKernel;

  // L2-aware M chunking. The activation stream loses all L2 reuse once a
  // chunk outgrows Thor's 32 MB L2, a two-thirds throughput loss, so the
  // per-launch activation footprint is capped near 12 MB.
  constexpr int64_t kAChunkBytes = int64_t(12) << 20;
  int m_chunk = int(kAChunkBytes / (int64_t(K) * 2));
  m_chunk = std::max(256, (m_chunk / 128) * 128);

  LayoutScale layout_S = ScaleConfig::tile_atom_to_shape_scale(cute::make_shape(N, K, 1));
  auto* c_base = reinterpret_cast<MmaType*>(c.mutable_data_ptr());
  const auto* a_base = reinterpret_cast<MmaType const*>(a.const_data_ptr());

  torch::stable::Tensor workspace;
  size_t ws_alloc = 0;

  for (int m0 = 0; m0 < M; m0 += m_chunk) {
    const int mc = std::min(m_chunk, M - m0);

    auto stride_act = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(mc, K, 1));
    auto stride_c = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, cute::make_shape(N, mc, 1));
    auto stride_d = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(N, mc, 1));

    MmaType* c_ptr = c_base + int64_t(m0) * N;

    // Swapped problem shape (N, mc, K). The packed weight rides the A slot.
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, mc, K, 1},
        {reinterpret_cast<uint8_t const*>(b_packed.const_data_ptr()),
         StrideA{}, a_base + int64_t(m0) * K, stride_act,
         reinterpret_cast<MmaType const*>(group_scales.const_data_ptr()),
         layout_S, nullptr},
        {{1.0f, 0.0f}, c_ptr, stride_c, c_ptr, stride_d}};

    Gemm gemm;
    STD_TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
                    "swordfish_prefill_mm: unsupported problem");

    const size_t ws_bytes = Gemm::get_workspace_size(arguments);
    if (ws_bytes > ws_alloc) {
      workspace = torch::stable::empty({int64_t(ws_bytes)},
                                       torch::headeronly::ScalarType::Byte,
                                       std::nullopt, a.device());
      ws_alloc = ws_bytes;
    }

    STD_TORCH_CHECK(
        gemm.initialize(arguments,
                        ws_alloc ? workspace.mutable_data_ptr() : nullptr,
                        stream) == cutlass::Status::kSuccess,
        "swordfish_prefill_mm: initialize failed");
    STD_TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
                    "swordfish_prefill_mm: launch failed");
  }
}

}  // namespace prefill

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

torch::stable::Tensor swordfish_prefill_mm(torch::stable::Tensor& a,
                                           torch::stable::Tensor& b_packed,
                                           torch::stable::Tensor& group_scales,
                                           int64_t group_size, int64_t size_k,
                                           int64_t size_n) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  STD_TORCH_CHECK(shape_ok(size_k, size_n) && size_k % 128 == 0 &&
                      size_n % 128 == 0,
                  "swordfish prefill v1 requires K % 128 == 0 and "
                  "N % 128 == 0; got K=", size_k, " N=", size_n);
  STD_TORCH_CHECK(a.dim() == 2 && a.size(1) == size_k,
                  "a must be [M, K] with K=", size_k);
  STD_TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == size_k,
                  "a must be contiguous");
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "swordfish prefill v1 requires bf16 activations");
  STD_TORCH_CHECK(
      group_scales.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "swordfish prefill v1 requires bf16 group scales");
  STD_TORCH_CHECK(group_size == prefill::kScaleGranularityK,
                  "swordfish prefill v1 supports group_size ",
                  prefill::kScaleGranularityK, " only; got ", group_size);

  const int64_t nb = num_blocks_n(size_n);
  const int64_t kb = num_blocks_k(size_k);
  STD_TORCH_CHECK(
      b_packed.scalar_type() == torch::headeronly::ScalarType::Int &&
          b_packed.dim() == 3 && b_packed.size(0) == nb &&
          b_packed.size(1) == kb && b_packed.size(2) == kBlockInt32,
      "b_packed must be int32 [", nb, ", ", kb, ", ", kBlockInt32, "]");

  const int64_t num_groups = size_k / group_size;
  STD_TORCH_CHECK(group_scales.dim() == 2 &&
                      group_scales.size(0) == num_groups &&
                      group_scales.size(1) == size_n,
                  "group_scales must be [", num_groups, ", ", size_n, "]");

  const int64_t size_m = a.size(0);

  const int32_t device_index = a.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  torch::stable::Tensor c = torch::stable::empty(
      {size_m, size_n}, a.scalar_type(), std::nullopt, a.device());
  if (size_m == 0) return c;

  const int M = int(size_m), N = int(size_n), K = int(size_k);

  // Tile-N dispatch. The 256-wide tile wins compute-bound shapes; K-heavy
  // shapes starve its K pipeline and prefer 256x128 (measured on both
  // Thor and B200 at K=14336).
  if (K >= 2 * N && K >= 8192) {
    prefill::run<prefill::PrefillCfg<128>>(a, b_packed, group_scales, c, M, N,
                                           K, stream);
  } else {
    prefill::run<prefill::PrefillCfg<256>>(a, b_packed, group_scales, c, M, N,
                                           K, stream);
  }
  return c;
#else
  STD_TORCH_CHECK(false,
                  "swordfish_prefill_mm requires a CUDA >= 12.8 build with "
                  "sm100-family support");
#endif
}

}  // namespace swordfish

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("swordfish_prefill_mm", TORCH_BOX(&swordfish::swordfish_prefill_mm));
}
