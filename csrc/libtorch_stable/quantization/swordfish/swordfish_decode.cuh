// Swordfish decode GEMM. w4a16 mma.sync mainloop over the ABI v1
// block-linear layout.
//
// One CTA (4 warps) covers M_TILES m16 tiles of one n64 block-column against
// a single weight stream. Warps split K into contiguous pair ranges (pair =
// 2 consecutive k16 slices = 1024 B of packed weights), which keeps each
// warp's group scales register-resident. Weights and the tiles' activation
// rows ride one cp.async pipeline, each lane copying exactly the bytes it
// later consumes into its own smem slot, so bytes in flight cost no
// registers and no warp synchronization. Each staged word is dequantized
// once and fans out to M_TILES mmas.
//
// The in-tile read contract is Marlin's, proven bit-exact by the prepack
// tests. Lane T consumes I4[T] at word 4T of each 512 B sub-tile; word j
// dequants into the fragments of n8-tiles 2j and 2j+1, where lane T owns
// column T/4 and rows 2*(T%4) + {0,1,8,9} of the k16xn8 fragment.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <climits>
#include <type_traits>

#include "libtorch_stable/quantization/marlin/marlin.cuh"
#include "libtorch_stable/quantization/marlin/marlin_dtypes.cuh"
#include "libtorch_stable/quantization/marlin/dequant.h"
#include "libtorch_stable/quantization/marlin/marlin_mma.h"

#include "swordfish_abi.cuh"

namespace swordfish {

inline constexpr int kDecodeWarps = 4;
inline constexpr int kDecodeThreads = kDecodeWarps * 32;
// One k16-slice pair = two consecutive 512 B sub-tiles.
inline constexpr int kPairInt32 = 2 * (kSubTileBytes / 4);
// cp.async pipeline depth in pairs (stages in flight per warp).
inline constexpr int kStages = 5;

// One ldmatrix.x4 loads a full m16k16 activation fragment from smem in
// tensor-core register order.
template <aphrodite::ScalarTypeId type_id>
__device__ __forceinline__ void ldsm4(
    typename marlin::MarlinScalarType<type_id>::FragA& frag_a,
    const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// cp.async.cg with an evict_first L2 hint. Packed weights are read once per
// launch and must not evict the re-read activation rows and scales.
__device__ __forceinline__ void cp_async4_evict_first(void* smem_ptr,
                                                      const void* glob_ptr,
                                                      uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(
          smem),
      "l"(glob_ptr), "l"(pol));
#else
  marlin::cp_async4(smem_ptr, glob_ptr);
#endif
}

__device__ __forceinline__ uint64_t l2_evict_first_policy() {
  uint64_t pol = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(pol));
#endif
  return pol;
}

// Packed-pair global atomic add. The atomicAdd(half2*) intrinsic compiles to
// a CAS loop on this toolchain, so emit red.global directly.
template <typename scalar_t2>
__device__ __forceinline__ void red_add2(scalar_t2* p, scalar_t2 v) {
  if constexpr (sizeof(scalar_t2) == 4) {
    if constexpr (std::is_same_v<scalar_t2, nv_bfloat162>) {
      asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" ::"l"(p),
                   "r"(*reinterpret_cast<uint32_t*>(&v))
                   : "memory");
    } else {
      asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(p),
                   "r"(*reinterpret_cast<uint32_t*>(&v))
                   : "memory");
    }
  }
}

// ATOMIC_EPI replaces the cross-warp smem reduction with red.global adds
// into a zeroed C, freeing 16 KB smem per CTA and both __syncthreads, and
// making cross-CTA split-K free. Summation order is nondeterministic. The
// launcher uses it for the decode window (M <= 47).
template <aphrodite::ScalarTypeId type_id, bool ATOMIC_EPI, int M_TILES = 1>
__global__ void swordfish_decode_kernel(
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ A,
    const int32_t* __restrict__ B,
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ S,
    typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ C,
    int M, int K, int N, int group_size) {
  static_assert(M_TILES >= 1 && M_TILES <= 3, "supported m-tile fusion: 1-3");
  static_assert(M_TILES == 1 || ATOMIC_EPI,
                "multi-m-tile is an atomic-epilogue configuration");
  // Pipeline depth scales inversely with M_TILES. Compute per staged pair
  // grows with the tile count, so fewer stages hide the same copy latency,
  // and the astage footprint stays inside the 48 KB static smem budget.
  constexpr int kStagesT = M_TILES == 1 ? kStages : (M_TILES == 2 ? 4 : 3);
  using Dtype = marlin::MarlinScalarType<type_id>;
  using scalar_t = typename Dtype::scalar_t;
  using scalar_t2 = typename Dtype::scalar_t2;
  using FragA = typename Dtype::FragA;
  using FragB = typename Dtype::FragB;
  using FragC = typename Dtype::FragC;

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group_id = lane / 4;
  const int tig = lane % 4;  // thread-in-group

  const int nb = blockIdx.y;                        // n64 block-column
  const int m_base = blockIdx.x * (16 * M_TILES);   // first m16 tile
  const int col_base = nb * kBlockN;
  const int num_kb = K / kBlockK;

  // Output rows owned by this lane's C fragments, per m16 tile.
  int row0[M_TILES];
  bool r0ok[M_TILES], r1ok[M_TILES];
#pragma unroll
  for (int t = 0; t < M_TILES; t++) {
    row0[t] = m_base + 16 * t + group_id;
    r0ok[t] = row0[t] < M;
    r1ok[t] = row0[t] + 8 < M;
  }

  // acc[t][j][b]: tile t's n8-tile at columns col_base + 16*j + 8*b + [0, 8).
  FragC acc[M_TILES][4][2];
#pragma unroll
  for (int t = 0; t < M_TILES; t++)
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int b = 0; b < 2; b++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[t][j][b][i] = 0.0f;

  const scalar_t2 zero2 = Dtype::num2num2(Dtype::float2num(0.0f));

  // At split-K 1 this CTA owns its C tile exclusively and zeroes it here,
  // sparing the launcher a flat ~2 us cudaMemsetAsync. At split > 1 tiles
  // are shared and the launcher memsets instead.
  if constexpr (ATOMIC_EPI) {
    if (gridDim.z == 1) {
      const int rows = min(16 * M_TILES, M - m_base);
      auto* c2 = reinterpret_cast<scalar_t2*>(C);
      for (int i = threadIdx.x; i < rows * 32; i += kDecodeThreads) {
        c2[int64_t(m_base + (i >> 5)) * (N >> 1) + (col_base >> 1) +
           (i & 31)] = zero2;
      }
      __syncthreads();  // tile zeroed before any warp's atomic epilogue
    }
  }

  // Global just-in-time A fragment load, used only by the deterministic
  // path. Register order per mma contract, reg0/reg1 at k0, reg2/reg3 at
  // k0+8.
  auto load_a_global = [&](int k0, FragA& fa) {
    const int ca = k0 + 2 * tig;
    const int cb = ca + 8;
    const auto* a_row0 = reinterpret_cast<const scalar_t2*>(
        A + int64_t(r0ok[0] ? row0[0] : 0) * K);
    const auto* a_row1 = reinterpret_cast<const scalar_t2*>(
        A + int64_t(r1ok[0] ? row0[0] + 8 : 0) * K);
    fa[0] = r0ok[0] ? a_row0[ca / 2] : zero2;
    fa[1] = r1ok[0] ? a_row1[ca / 2] : zero2;
    fa[2] = r0ok[0] ? a_row0[cb / 2] : zero2;
    fa[3] = r1ok[0] ? a_row1[cb / 2] : zero2;
  };
  // ldmatrix per-lane addressing. Lanes 0-15 cover rows 0-15 cols 0-7 and
  // lanes 16-31 cols 8-15, yielding FragA's register order. Rows >= M hold
  // unstaged garbage whose products only reach acc components the epilogue
  // guards never store.
  const int ldsm_row = lane & 15;
  const int ldsm_col = (lane >> 4) << 3;
  auto load_a = [&](const scalar_t (*sa)[32], int ks, FragA& fa) {
    ldsm4<type_id>(fa, &sa[ldsm_row][ldsm_col + ks]);
  };

  // Group scales, hoisted. The contiguous k-split makes each warp see each
  // group exactly once. One 4-byte load per lane covers the 64-wide row and
  // shfl broadcasts the pairs. Fetch and expand are split so the next
  // group's row prefetches a full group early, hiding its latency under
  // compute (on-demand loads were the dominant stall on K-heavy shapes).
  scalar_t2 s_reg[4][2];  // [j][b] broadcast pairs for this lane's column
  auto fetch_scale_row = [&](int g) -> scalar_t2 {
    return reinterpret_cast<const scalar_t2*>(S + int64_t(g) * N +
                                              col_base)[lane];
  };
  auto expand_scales = [&](scalar_t2 mine) {
    const uint32_t sel = group_id & 1;  // half index within the shfl'd pair
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int b = 0; b < 2; b++) {
        // half index i = 16j + 8b + group_id -> lane i/2, element i%2
        const scalar_t2 v = __shfl_sync(
            0xffffffffu, mine, 8 * j + 4 * b + (group_id >> 1));
        const uint32_t bits = reinterpret_cast<const uint32_t&>(v);
        const uint16_t h16 = sel ? uint16_t(bits >> 16) : uint16_t(bits);
        s_reg[j][b] = Dtype::num2num2(reinterpret_cast<const scalar_t&>(h16));
      }
    }
  };

  // One k16 slice. Dequant the lane's word once, scale once, then fan the
  // mma out across the CTA's m16 tiles.
  auto process_slice = [&](const FragA (&fa)[M_TILES], const marlin::I4& bq) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int b_quant_0 = bq.elems[j];
      const int b_quant_1 = b_quant_0 >> 8;

      FragB frag_b0, frag_b1;
      marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
          b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
      marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
          b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));

      frag_b0[0] = __hmul2(frag_b0[0], s_reg[j][0]);
      frag_b0[1] = __hmul2(frag_b0[1], s_reg[j][0]);
      frag_b1[0] = __hmul2(frag_b1[0], s_reg[j][1]);
      frag_b1[1] = __hmul2(frag_b1[1], s_reg[j][1]);

#pragma unroll
      for (int t = 0; t < M_TILES; t++) {
        marlin::mma<type_id, /*use_fp16_accum=*/false>(fa[t], frag_b0,
                                                       acc[t][j][0]);
        marlin::mma<type_id, /*use_fp16_accum=*/false>(fa[t], frag_b1,
                                                       acc[t][j][1]);
      }
    }
  };

  // One pair = two consecutive k16 slices from one staged buffer.
  auto process_pair = [&](int p, const scalar_t (*sa)[16][32],
                          const int4* buf) {
    FragA fa0[M_TILES], fa1[M_TILES];
    if constexpr (ATOMIC_EPI) {
#pragma unroll
      for (int t = 0; t < M_TILES; t++) {
        load_a(sa[t], 0, fa0[t]);
        load_a(sa[t], 16, fa1[t]);
      }
    } else {
      load_a_global(32 * p, fa0[0]);
      load_a_global(32 * p + 16, fa1[0]);
    }
    const marlin::I4 bq0 = *reinterpret_cast<const marlin::I4*>(&buf[lane]);
    const marlin::I4 bq1 =
        *reinterpret_cast<const marlin::I4*>(&buf[32 + lane]);
    process_slice(fa0, bq0);
    process_slice(fa1, bq1);
  };

  // Weight staging, one self-slot buffer per warp and stage.
  __shared__ int4 bstage[kDecodeWarps][kStagesT][2 * 32];
  // Activation slices for the in-flight pairs, 1 KB per tile and stage,
  // copied in the same commit group as the weights so one wait covers both.
  __shared__ scalar_t
      astage[kDecodeWarps][ATOMIC_EPI ? kStagesT : 1][M_TILES][16][32];
  const int32_t* b_col = B + int64_t(nb) * num_kb * kBlockInt32;
  const int num_pairs = K / (2 * kMarlinTileK);
  // Cross-CTA split-K. blockIdx.z slices the pair space and the atomic
  // epilogue merges slices in the zeroed C. gridDim.z > 1 only on the
  // ATOMIC_EPI path.
  const int slice_pairs = (num_pairs + gridDim.z - 1) / gridDim.z;
  const int s_beg = blockIdx.z * slice_pairs;
  const int s_end = min(s_beg + slice_pairs, num_pairs);
  const int pairs_per_warp =
      (s_end - s_beg + kDecodeWarps - 1) / kDecodeWarps;
  const int p_beg = s_beg + warp * pairs_per_warp;
  const int p_end = min(p_beg + pairs_per_warp, s_end);

  // Scale-group bookkeeping: `left` = pairs left in the current group
  // (host guarantees group_size % 32 == 0, so pairs never straddle groups).
  const int ppg = group_size > 0 ? group_size / 32 : INT_MAX;
  int g = 0;
  int left = INT_MAX;

  // One commit group per stage. Fences are unconditional so the wait<N>
  // accounting stays uniform through the tail (empty groups are legal).
  const int g_last =
      group_size > 0 && p_beg < p_end ? (32 * (p_end - 1)) / group_size : 0;
  scalar_t2 s_next = zero2;  // prefetched next-group row (lane's slice)
  if (p_beg < p_end) {
    if (group_size > 0) {
      g = p_beg / ppg;
      left = ppg - p_beg % ppg;
    }
    expand_scales(fetch_scale_row(g));
    if (g + 1 <= g_last) s_next = fetch_scale_row(g + 1);
  }

  // Incremental int32 issue cursors. A modulo-indexed ring compiles to
  // magic-number division and per-pair int64 address math costs multiplies,
  // both measurable in this loop. Offsets fit int32 for every shape the
  // host validates.
  const uint64_t bpol = l2_evict_first_policy();
  const int32_t* ipair_ptr = b_col + p_beg * kPairInt32 + 4 * lane;
  const int ia_row = lane >> 1;
  const int ia_c0 = 16 * (lane & 1);  // 0 or 16
  // Per-tile activation cursors. Only valid rows are staged, since clamping
  // out-of-range rows to row 0 multiplies activation traffic at small M.
  bool ia_okt[M_TILES];
  const scalar_t* ia_ptr[M_TILES];
#pragma unroll
  for (int t = 0; t < M_TILES; t++) {
    const int r = m_base + 16 * t + ia_row;
    ia_okt[t] = r < M;
    ia_ptr[t] = A + (ia_okt[t] ? r : 0) * K + 32 * p_beg + ia_c0;
  }
  int ipend = p_end - p_beg;  // pairs left to issue

  auto issue_pair = [&](int slot) {
    if (ipend > 0) {
      ipend--;
      cp_async4_evict_first(&bstage[warp][slot][lane], ipair_ptr, bpol);
      cp_async4_evict_first(&bstage[warp][slot][32 + lane],
                            ipair_ptr + kPairInt32 / 2, bpol);
      ipair_ptr += kPairInt32;
      if constexpr (ATOMIC_EPI) {
#pragma unroll
        for (int t = 0; t < M_TILES; t++) {
          if (ia_okt[t]) {
            marlin::cp_async4(&astage[warp][slot][t][ia_row][ia_c0],
                              ia_ptr[t]);
            marlin::cp_async4(&astage[warp][slot][t][ia_row][ia_c0 + 8],
                              ia_ptr[t] + 8);
          }
          ia_ptr[t] += 32;
        }
      }
    }
    marlin::cp_async_fence();
  };

#pragma unroll
  for (int s = 0; s < kStagesT - 1; s++) issue_pair(s);

  int slot = 0;
  int islot = kStagesT - 1;
  for (int p = p_beg; p < p_end; p++) {
    issue_pair(islot);
    if (++islot == kStagesT) islot = 0;
    marlin::cp_async_wait<kStagesT - 2>();  // oldest stage (slot) complete
    if (left == 0) {
      ++g;
      expand_scales(s_next);  // value already in flight since last boundary
      if (g + 1 <= g_last) s_next = fetch_scale_row(g + 1);
      left = ppg;
    }
    process_pair(p, astage[warp][ATOMIC_EPI ? slot : 0], bstage[warp][slot]);
    left--;
    if (++slot == kStagesT) slot = 0;
  }

  if constexpr (ATOMIC_EPI) {
    // Direct atomic epilogue: every warp adds its partial fragments to C.
#pragma unroll
    for (int t = 0; t < M_TILES; t++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          const int col = col_base + 8 * (2 * j + b) + 2 * tig;
          const float4 v = *reinterpret_cast<float4*>(&acc[t][j][b]);
          if (r0ok[t])
            red_add2(
                reinterpret_cast<scalar_t2*>(C + int64_t(row0[t]) * N + col),
                Dtype::nums2num2(Dtype::float2num(v.x), Dtype::float2num(v.y)));
          if (r1ok[t])
            red_add2(
                reinterpret_cast<scalar_t2*>(C + int64_t(row0[t] + 8) * N + col),
                Dtype::nums2num2(Dtype::float2num(v.z), Dtype::float2num(v.w)));
        }
      }
    }
    return;
  }

  // Deterministic epilogue. Partials meet in smem and warp w reduces and
  // writes n8-tiles {2w, 2w+1}. Tile index is 2*j + b.
  __shared__ float4 red[kDecodeWarps][8][32];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int b = 0; b < 2; b++)
      red[warp][2 * j + b][lane] = *reinterpret_cast<float4*>(&acc[0][j][b]);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int tile = 2 * warp + i;
    float4 sum = red[0][tile][lane];
#pragma unroll
    for (int w = 1; w < kDecodeWarps; w++) {
      const float4 v = red[w][tile][lane];
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    const int col = col_base + 8 * tile + 2 * tig;
    if (r0ok[0]) {
      *reinterpret_cast<scalar_t2*>(C + int64_t(row0[0]) * N + col) =
          Dtype::nums2num2(Dtype::float2num(sum.x), Dtype::float2num(sum.y));
    }
    if (r1ok[0]) {
      *reinterpret_cast<scalar_t2*>(C + int64_t(row0[0] + 8) * N + col) =
          Dtype::nums2num2(Dtype::float2num(sum.z), Dtype::float2num(sum.w));
    }
  }
}

// Stream-K decode for the fused window (M 17 to 96). Persistent CTAs; one
// unit of flat work is one k16-slice pair of one (m-group, n64-column)
// tile, and each warp owns a contiguous range of units. The m-group index
// varies fastest after the pair so consecutive segments in a warp's range
// revisit the same weight column while it is L2-resident. Accumulators
// flush through red.global at segment boundaries into a launcher-zeroed C,
// so a tile's K slices may come from any number of warps with no locks and
// no fixup pass. This removes split-K heuristics and wave quantization for
// the window entirely.
template <aphrodite::ScalarTypeId type_id, int M_TILES>
__global__ void swordfish_decode_streamk_kernel(
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ A,
    const int32_t* __restrict__ B,
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ S,
    typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ C,
    int M, int K, int N, int group_size, int m_groups) {
  constexpr int kStagesT = M_TILES == 1 ? kStages : (M_TILES == 2 ? 4 : 3);
  using Dtype = marlin::MarlinScalarType<type_id>;
  using scalar_t = typename Dtype::scalar_t;
  using scalar_t2 = typename Dtype::scalar_t2;
  using FragA = typename Dtype::FragA;
  using FragB = typename Dtype::FragB;
  using FragC = typename Dtype::FragC;

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group_id = lane / 4;
  const int tig = lane % 4;
  const int ldsm_row = lane & 15;
  const int ldsm_col = (lane >> 4) << 3;
  const int ia_row = lane >> 1;
  const int ia_c0 = 16 * (lane & 1);

  const int num_pairs = K / (2 * kMarlinTileK);
  const int nb_cnt = N / kBlockN;
  const int num_kb = K / kBlockK;
  const int ppg = group_size > 0 ? group_size / 32 : INT_MAX;
  const scalar_t2 zero2 = Dtype::num2num2(Dtype::float2num(0.0f));
  const uint64_t bpol = l2_evict_first_policy();

  const int64_t total = int64_t(m_groups) * nb_cnt * num_pairs;
  const int total_warps = gridDim.x * kDecodeWarps;
  const int64_t per = (total + total_warps - 1) / total_warps;
  int64_t w = int64_t(blockIdx.x * kDecodeWarps + warp) * per;
  const int64_t w_end = w + per < total ? w + per : total;

  __shared__ int4 bstage[kDecodeWarps][kStagesT][2 * 32];
  __shared__ scalar_t astage[kDecodeWarps][kStagesT][M_TILES][16][32];

  FragC acc[M_TILES][4][2];
  scalar_t2 s_reg[4][2];

  while (w < w_end) {
    const int64_t cg = w / num_pairs;
    const int p_beg = int(w - cg * num_pairs);
    const int col = int(cg / m_groups);
    const int g_idx = int(cg - int64_t(col) * m_groups);
    const int p_end =
        int(int64_t(num_pairs) < p_beg + (w_end - w) ? int64_t(num_pairs)
                                                     : p_beg + (w_end - w));
    w += p_end - p_beg;

    const int m_base = g_idx * (16 * M_TILES);
    const int col_base = col * kBlockN;

    int row0[M_TILES];
    bool r0ok[M_TILES], r1ok[M_TILES];
#pragma unroll
    for (int t = 0; t < M_TILES; t++) {
      row0[t] = m_base + 16 * t + group_id;
      r0ok[t] = row0[t] < M;
      r1ok[t] = row0[t] + 8 < M;
    }
#pragma unroll
    for (int t = 0; t < M_TILES; t++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int b = 0; b < 2; b++)
#pragma unroll
          for (int i = 0; i < 4; i++) acc[t][j][b][i] = 0.0f;

    auto fetch_scale_row = [&](int g) -> scalar_t2 {
      return reinterpret_cast<const scalar_t2*>(S + int64_t(g) * N +
                                                col_base)[lane];
    };
    auto expand_scales = [&](scalar_t2 mine) {
      const uint32_t sel = group_id & 1;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          const scalar_t2 v = __shfl_sync(
              0xffffffffu, mine, 8 * j + 4 * b + (group_id >> 1));
          const uint32_t bits = reinterpret_cast<const uint32_t&>(v);
          const uint16_t h16 = sel ? uint16_t(bits >> 16) : uint16_t(bits);
          s_reg[j][b] =
              Dtype::num2num2(reinterpret_cast<const scalar_t&>(h16));
        }
      }
    };
    auto load_a = [&](const scalar_t (*sa)[32], int ks, FragA& fa) {
      ldsm4<type_id>(fa, &sa[ldsm_row][ldsm_col + ks]);
    };
    auto process_slice = [&](const FragA (&fa)[M_TILES],
                             const marlin::I4& bq) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int b_quant_0 = bq.elems[j];
        const int b_quant_1 = b_quant_0 >> 8;
        FragB frag_b0, frag_b1;
        marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
            b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
        marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
            b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));
        frag_b0[0] = __hmul2(frag_b0[0], s_reg[j][0]);
        frag_b0[1] = __hmul2(frag_b0[1], s_reg[j][0]);
        frag_b1[0] = __hmul2(frag_b1[0], s_reg[j][1]);
        frag_b1[1] = __hmul2(frag_b1[1], s_reg[j][1]);
#pragma unroll
        for (int t = 0; t < M_TILES; t++) {
          marlin::mma<type_id, false>(fa[t], frag_b0, acc[t][j][0]);
          marlin::mma<type_id, false>(fa[t], frag_b1, acc[t][j][1]);
        }
      }
    };
    auto process_pair = [&](int aslot, const int4* buf) {
      FragA fa0[M_TILES], fa1[M_TILES];
#pragma unroll
      for (int t = 0; t < M_TILES; t++) {
        load_a(astage[warp][aslot][t], 0, fa0[t]);
        load_a(astage[warp][aslot][t], 16, fa1[t]);
      }
      const marlin::I4 bq0 = *reinterpret_cast<const marlin::I4*>(&buf[lane]);
      const marlin::I4 bq1 =
          *reinterpret_cast<const marlin::I4*>(&buf[32 + lane]);
      process_slice(fa0, bq0);
      process_slice(fa1, bq1);
    };

    const int32_t* ipair_ptr =
        B + int64_t(col) * num_kb * kBlockInt32 + p_beg * kPairInt32 + 4 * lane;
    bool ia_okt[M_TILES];
    const scalar_t* ia_ptr[M_TILES];
#pragma unroll
    for (int t = 0; t < M_TILES; t++) {
      const int r = m_base + 16 * t + ia_row;
      ia_okt[t] = r < M;
      ia_ptr[t] = A + (ia_okt[t] ? r : 0) * K + 32 * p_beg + ia_c0;
    }
    int ipend = p_end - p_beg;

    auto issue_pair = [&](int slot) {
      if (ipend > 0) {
        ipend--;
        cp_async4_evict_first(&bstage[warp][slot][lane], ipair_ptr, bpol);
        cp_async4_evict_first(&bstage[warp][slot][32 + lane],
                              ipair_ptr + kPairInt32 / 2, bpol);
        ipair_ptr += kPairInt32;
#pragma unroll
        for (int t = 0; t < M_TILES; t++) {
          if (ia_okt[t]) {
            marlin::cp_async4(&astage[warp][slot][t][ia_row][ia_c0],
                              ia_ptr[t]);
            marlin::cp_async4(&astage[warp][slot][t][ia_row][ia_c0 + 8],
                              ia_ptr[t] + 8);
          }
          ia_ptr[t] += 32;
        }
      }
      marlin::cp_async_fence();
    };

    int g = 0;
    int left = INT_MAX;
    const int g_last =
        group_size > 0 && p_beg < p_end ? (32 * (p_end - 1)) / group_size : 0;
    scalar_t2 s_next = zero2;
    if (p_beg < p_end) {
      if (group_size > 0) {
        g = p_beg / ppg;
        left = ppg - p_beg % ppg;
      }
      expand_scales(fetch_scale_row(g));
      if (g + 1 <= g_last) s_next = fetch_scale_row(g + 1);
    }

#pragma unroll
    for (int st = 0; st < kStagesT - 1; st++) issue_pair(st);

    int slot = 0;
    int islot = kStagesT - 1;
    for (int p = p_beg; p < p_end; p++) {
      issue_pair(islot);
      if (++islot == kStagesT) islot = 0;
      marlin::cp_async_wait<kStagesT - 2>();
      if (left == 0) {
        ++g;
        expand_scales(s_next);
        if (g + 1 <= g_last) s_next = fetch_scale_row(g + 1);
        left = ppg;
      }
      process_pair(slot, bstage[warp][slot]);
      left--;
      if (++slot == kStagesT) slot = 0;
    }

    // Segment flush into the launcher-zeroed C.
#pragma unroll
    for (int t = 0; t < M_TILES; t++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          const int cc = col_base + 8 * (2 * j + b) + 2 * tig;
          const float4 v = *reinterpret_cast<float4*>(&acc[t][j][b]);
          if (r0ok[t])
            red_add2(
                reinterpret_cast<scalar_t2*>(C + int64_t(row0[t]) * N + cc),
                Dtype::nums2num2(Dtype::float2num(v.x), Dtype::float2num(v.y)));
          if (r1ok[t])
            red_add2(
                reinterpret_cast<scalar_t2*>(C + int64_t(row0[t] + 8) * N + cc),
                Dtype::nums2num2(Dtype::float2num(v.z), Dtype::float2num(v.w)));
        }
      }
    }
  }
}

// M-shared decode for M 33 to 96 at narrow N. W warps each own one m16 tile
// for the WHOLE K range, so the CTA streams each weight chunk exactly once
// for up to 96 rows (the Marlin CTA shape). Weights stage once per CTA in
// two-pair chunks that all warps consume in lockstep; activations stage per
// warp. The pipeline step is [issue chunk into the freed slot] [wait]
// [barrier] [process two pairs] [barrier]. The trailing barrier gates slot
// reuse, since the issue at step c writes the slot processed at step c-1.
// Work distribution is per-CTA Stream-K over (m-group, column, chunk) with
// the m-group index fastest, flushed through red.global at boundaries.
template <aphrodite::ScalarTypeId type_id, int W>
__global__ void swordfish_decode_mshare_kernel(
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ A,
    const int32_t* __restrict__ B,
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ S,
    typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ C,
    int M, int K, int N, int group_size, int m_groups) {
  static_assert(W == 4 || W == 6, "supported m-share widths");
  // Chunk size trades barrier rate against pipeline depth inside the 48 KB
  // smem budget. W=4 affords 4-pair chunks at 2 stages (one barrier pair per
  // 64 mmas per warp); W=6 fits 2-pair chunks at 3 stages.
  constexpr int kCP = 2;  // pairs per chunk (4-pair chunks at 2 stages
                          // measured worse; shallow pipelines lose more
                          // than the halved barrier rate gains)
  constexpr int kSt = 3;
  using Dtype = marlin::MarlinScalarType<type_id>;
  using scalar_t = typename Dtype::scalar_t;
  using scalar_t2 = typename Dtype::scalar_t2;
  using FragA = typename Dtype::FragA;
  using FragB = typename Dtype::FragB;
  using FragC = typename Dtype::FragC;

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int group_id = lane / 4;
  const int tig = lane % 4;
  const int ldsm_row = lane & 15;
  const int ldsm_col = (lane >> 4) << 3;
  const int ia_row = lane >> 1;
  const int ia_c0 = 16 * (lane & 1);

  const int num_pairs = K / (2 * kMarlinTileK);
  const int num_chunks = num_pairs / kCP;
  const int nb_cnt = N / kBlockN;
  const int num_kb = K / kBlockK;
  const int ppg = group_size > 0 ? group_size / 32 : INT_MAX;
  const scalar_t2 zero2 = Dtype::num2num2(Dtype::float2num(0.0f));
  const uint64_t bpol = l2_evict_first_policy();

  const int64_t total = int64_t(m_groups) * nb_cnt * num_chunks;
  const int64_t per = (total + gridDim.x - 1) / gridDim.x;
  int64_t w = int64_t(blockIdx.x) * per;
  const int64_t w_end = w + per < total ? w + per : total;

  // Weights once per CTA, activations per warp.
  __shared__ int4 bstage[kSt][kCP][64];
  __shared__ scalar_t astage[W][kSt][kCP][16][32];

  FragC acc[4][2];
  scalar_t2 s_reg[4][2];

  // B copy assignment. The chunk is kCP 1 KB pairs = kCP*64 16-byte units,
  // spread over the CTA's threads; each thread copies kBU units.
  constexpr int kThreads = W * 32;
  constexpr int kBU = (kCP * 64 + kThreads - 1) / kThreads;  // units/thread
  const int b_unit0 = tid;  // unit u covers pair u/64, slot (u%64)
  auto b_src_off = [&](int u) {
    const int uu = u % 64;
    return (u / 64) * kPairInt32 +
           (uu < 32 ? 4 * uu : kPairInt32 / 2 + 4 * (uu - 32));
  };

  while (w < w_end) {
    const int64_t cg = w / num_chunks;
    const int c_beg = int(w - cg * num_chunks);
    const int col = int(cg / m_groups);
    const int g_idx = int(cg - int64_t(col) * m_groups);
    const int c_end =
        int(int64_t(num_chunks) < c_beg + (w_end - w) ? int64_t(num_chunks)
                                                      : c_beg + (w_end - w));
    w += c_end - c_beg;

    const int m_base = g_idx * (16 * W) + 16 * warp;  // this warp's tile
    const int col_base = col * kBlockN;
    const int row0 = m_base + group_id;
    const bool r0ok = row0 < M;
    const bool r1ok = row0 + 8 < M;
    const bool a_ok = (m_base + ia_row) < M;

#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int b = 0; b < 2; b++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[j][b][i] = 0.0f;

    auto fetch_scale_row = [&](int g) -> scalar_t2 {
      return reinterpret_cast<const scalar_t2*>(S + int64_t(g) * N +
                                                col_base)[lane];
    };
    auto expand_scales = [&](scalar_t2 mine) {
      const uint32_t sel = group_id & 1;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          const scalar_t2 v = __shfl_sync(
              0xffffffffu, mine, 8 * j + 4 * b + (group_id >> 1));
          const uint32_t bits = reinterpret_cast<const uint32_t&>(v);
          const uint16_t h16 = sel ? uint16_t(bits >> 16) : uint16_t(bits);
          s_reg[j][b] =
              Dtype::num2num2(reinterpret_cast<const scalar_t&>(h16));
        }
      }
    };
    auto process_slice = [&](const FragA& fa, const marlin::I4& bq) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int b_quant_0 = bq.elems[j];
        const int b_quant_1 = b_quant_0 >> 8;
        FragB frag_b0, frag_b1;
        marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
            b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
        marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
            b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));
        frag_b0[0] = __hmul2(frag_b0[0], s_reg[j][0]);
        frag_b0[1] = __hmul2(frag_b0[1], s_reg[j][0]);
        frag_b1[0] = __hmul2(frag_b1[0], s_reg[j][1]);
        frag_b1[1] = __hmul2(frag_b1[1], s_reg[j][1]);
        marlin::mma<type_id, false>(fa, frag_b0, acc[j][0]);
        marlin::mma<type_id, false>(fa, frag_b1, acc[j][1]);
      }
    };
    auto process_pair = [&](int slot, int pi) {
      FragA fa0, fa1;
      ldsm4<type_id>(fa0, &astage[warp][slot][pi][ldsm_row][ldsm_col]);
      ldsm4<type_id>(fa1, &astage[warp][slot][pi][ldsm_row][ldsm_col + 16]);
      const int4* buf = bstage[slot][pi];
      const marlin::I4 bq0 = *reinterpret_cast<const marlin::I4*>(&buf[lane]);
      const marlin::I4 bq1 =
          *reinterpret_cast<const marlin::I4*>(&buf[32 + lane]);
      process_slice(fa0, bq0);
      process_slice(fa1, bq1);
    };

    const int32_t* b_col = B + int64_t(col) * num_kb * kBlockInt32;
    const int32_t* ichunk_base = b_col + kCP * c_beg * kPairInt32;
    const scalar_t* ia_ptr =
        A + (a_ok ? m_base + ia_row : 0) * K + 32 * kCP * c_beg + ia_c0;
    int icend = c_end - c_beg;

    auto issue_chunk = [&](int slot) {
      if (icend > 0) {
        icend--;
#pragma unroll
        for (int bu = 0; bu < kBU; bu++) {
          const int u = b_unit0 + bu * kThreads;
          if (u < kCP * 64) {
            cp_async4_evict_first(
                &bstage[slot][u / 64][u % 64], ichunk_base + b_src_off(u),
                bpol);
          }
        }
        ichunk_base += kCP * kPairInt32;
        if (a_ok) {
#pragma unroll
          for (int pi = 0; pi < kCP; pi++) {
            marlin::cp_async4(&astage[warp][slot][pi][ia_row][ia_c0],
                              ia_ptr + 32 * pi);
            marlin::cp_async4(&astage[warp][slot][pi][ia_row][ia_c0 + 8],
                              ia_ptr + 32 * pi + 8);
          }
        }
        ia_ptr += 32 * kCP;
      }
      marlin::cp_async_fence();
    };

    int g = 0;
    int left = INT_MAX;
    const int p_last = kCP * c_end - 1;
    const int g_last = group_size > 0 ? (32 * p_last) / group_size : 0;
    scalar_t2 s_next = zero2;
    if (c_beg < c_end) {
      const int p0 = kCP * c_beg;
      if (group_size > 0) {
        g = p0 / ppg;
        left = ppg - p0 % ppg;
      }
      expand_scales(fetch_scale_row(g));
      if (g + 1 <= g_last) s_next = fetch_scale_row(g + 1);
    }

#pragma unroll
    for (int st = 0; st < kSt - 1; st++) issue_chunk(st);

    int slot = 0;
    int islot = kSt - 1;
    for (int c = c_beg; c < c_end; c++) {
      issue_chunk(islot);
      if (++islot == kSt) islot = 0;
      marlin::cp_async_wait<kSt - 2>();
      __syncthreads();
#pragma unroll
      for (int pi = 0; pi < kCP; pi++) {
        if (left == 0) {
          ++g;
          expand_scales(s_next);
          if (g + 1 <= g_last) s_next = fetch_scale_row(g + 1);
          left = ppg;
        }
        process_pair(slot, pi);
        left--;
      }
      __syncthreads();
      if (++slot == kSt) slot = 0;
    }

    // Per-warp segment flush into the launcher-zeroed C.
    if (r0ok || r1ok) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int b = 0; b < 2; b++) {
          const int cc = col_base + 8 * (2 * j + b) + 2 * tig;
          const float4 v = *reinterpret_cast<float4*>(&acc[j][b]);
          if (r0ok)
            red_add2(reinterpret_cast<scalar_t2*>(C + int64_t(row0) * N + cc),
                     Dtype::nums2num2(Dtype::float2num(v.x),
                                      Dtype::float2num(v.y)));
          if (r1ok)
            red_add2(
                reinterpret_cast<scalar_t2*>(C + int64_t(row0 + 8) * N + cc),
                Dtype::nums2num2(Dtype::float2num(v.z),
                                 Dtype::float2num(v.w)));
        }
      }
    }
  }
}

}  // namespace swordfish
