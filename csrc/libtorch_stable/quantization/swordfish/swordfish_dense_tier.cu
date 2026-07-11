// Dense-GEMM tier for very large M. Past a few thousand rows the problem is
// compute-bound and Blackwell's dense fp16/bf16 rate through cuBLAS beats
// any fused mixed-input mainloop, so the weight dequantizes once into a
// transient dense buffer and cuBLAS takes the GEMM. The dequant cost is a
// few weight-reads and amortizes to nothing at this scale.

#include <cublas_v2.h>

#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"
#include "swordfish_decode.cuh"

#define SWORDFISH_CHECK_CUBLAS(cmd)                             \
  do {                                                          \
    cublasStatus_t e = cmd;                                     \
    STD_TORCH_CHECK(e == CUBLAS_STATUS_SUCCESS,                 \
                    "swordfish dense tier cuBLAS error ", int(e)); \
  } while (0)

namespace swordfish {

namespace {

// One warp per marlin 16x64 sub-tile, staged through smem so the dense
// stores coalesce. The fragment contract matches the decode kernels: word j
// of lane T covers columns {16j + T/4, 16j + 8 + T/4} at sub-tile rows
// 2(T%4) + {0, 1, 8, 9}.
template <aphrodite::ScalarTypeId type_id, bool W8, bool HAS_ZP>
__global__ void swordfish_dequant_dense_kernel(
    const int32_t* __restrict__ B,
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ S,
    const typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ Z,
    const int32_t* __restrict__ perm,
    typename marlin::MarlinScalarType<type_id>::scalar_t* __restrict__ W,
    int K, int N, int group_size) {
  using Dtype = marlin::MarlinScalarType<type_id>;
  using scalar_t = typename Dtype::scalar_t;
  using scalar_t2 = typename Dtype::scalar_t2;
  using FragB = typename Dtype::FragB;

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;  // sub-tile within the block
  const int c = lane >> 2;
  const int t = lane & 3;
  const int nb = blockIdx.x;
  const int kb = blockIdx.y;
  const int num_kb = K / kBlockK;
  constexpr int64_t kBlockW = W8 ? kBlockInt32_8 : kBlockInt32;

  __shared__ scalar_t tile[4][16][64];

  const int4* buf = reinterpret_cast<const int4*>(
      B + (int64_t(nb) * num_kb + kb) * kBlockW +
      warp * (W8 ? 2 * kPairInt32 / 2 : kPairInt32 / 2));

  // One scale group covers the whole 16-row sub-tile for every supported
  // group size.
  const int g = group_size > 0 ? (kb * kBlockK + 16 * warp) / group_size : 0;
  const scalar_t* srow = S + int64_t(g) * N + nb * kBlockN;
  const scalar_t* zrow = HAS_ZP ? Z + int64_t(g) * N + nb * kBlockN : nullptr;

  const marlin::I4 bq0 =
      *reinterpret_cast<const marlin::I4*>(&buf[W8 ? 2 * lane : lane]);
  marlin::I4 bq1;
  if constexpr (W8) {
    bq1 = *reinterpret_cast<const marlin::I4*>(&buf[2 * lane + 1]);
  }

  auto put = [&](int row, int col, scalar_t2 v2, bool hi) {
    tile[warp][row][col] = reinterpret_cast<const scalar_t*>(&v2)[hi ? 1 : 0];
  };

#pragma unroll
  for (int j = 0; j < 4; j++) {
    FragB frag_b0, frag_b1;
    if constexpr (W8) {
      const marlin::I4& src = j < 2 ? bq0 : bq1;
      marlin::dequant<scalar_t2, aphrodite::kU8B128.id(), false>(
          src.elems[(2 * j) & 3], reinterpret_cast<scalar_t2*>(&frag_b0));
      marlin::dequant<scalar_t2, aphrodite::kU8B128.id(), false>(
          src.elems[(2 * j + 1) & 3], reinterpret_cast<scalar_t2*>(&frag_b1));
    } else {
      const int q = bq0.elems[j];
      marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
          q, reinterpret_cast<scalar_t2*>(&frag_b0));
      marlin::dequant<scalar_t2, aphrodite::kU4B8.id(), false>(
          q >> 8, reinterpret_cast<scalar_t2*>(&frag_b1));
    }
    const int n0 = 16 * j + c;
    const int n1 = n0 + 8;
    const scalar_t2 s0 = Dtype::num2num2(srow[n0]);
    const scalar_t2 s1 = Dtype::num2num2(srow[n1]);
    if constexpr (HAS_ZP) {
      const scalar_t2 z0 = Dtype::num2num2(zrow[n0]);
      const scalar_t2 z1 = Dtype::num2num2(zrow[n1]);
      frag_b0[0] = __hfma2(frag_b0[0], s0, z0);
      frag_b0[1] = __hfma2(frag_b0[1], s0, z0);
      frag_b1[0] = __hfma2(frag_b1[0], s1, z1);
      frag_b1[1] = __hfma2(frag_b1[1], s1, z1);
    } else {
      frag_b0[0] = __hmul2(frag_b0[0], s0);
      frag_b0[1] = __hmul2(frag_b0[1], s0);
      frag_b1[0] = __hmul2(frag_b1[0], s1);
      frag_b1[1] = __hmul2(frag_b1[1], s1);
    }
    // frag[0] holds rows {2t, 2t+1}, frag[1] rows {2t+8, 2t+9}.
    put(2 * t, n0, frag_b0[0], false);
    put(2 * t + 1, n0, frag_b0[0], true);
    put(2 * t + 8, n0, frag_b0[1], false);
    put(2 * t + 9, n0, frag_b0[1], true);
    put(2 * t, n1, frag_b1[0], false);
    put(2 * t + 1, n1, frag_b1[0], true);
    put(2 * t + 8, n1, frag_b1[1], false);
    put(2 * t + 9, n1, frag_b1[1], true);
  }
  __syncwarp();

  // Coalesced stores: 16-byte runs per lane over the sub-tile's rows.
  // Under act_order the packed rows are group-sorted, so row r scatters to
  // its original position and the activations stay unpermuted.
  const int k_base = kb * kBlockK + 16 * warp;
  const int c0 = 8 * (lane % 8);
#pragma unroll
  for (int r = lane / 8; r < 16; r += 4) {
    const int64_t k_dst = perm != nullptr ? perm[k_base + r] : k_base + r;
    *reinterpret_cast<int4*>(&W[k_dst * N + nb * kBlockN + c0]) =
        *reinterpret_cast<const int4*>(&tile[warp][r][c0]);
  }
}

}  // namespace

void swordfish_dense_tier_mm(const void* a, const int32_t* b, const void* s,
                             const void* z, const int32_t* perm, void* c,
                             void* w_dense, bool is_half, bool w8,
                             bool has_zp, int m, int k, int n, int group_size,
                             cudaStream_t stream) {
  dim3 grid(n / kBlockN, k / kBlockK);
  const auto run = [&](auto tid, auto w8c, auto zpc) {
    constexpr aphrodite::ScalarTypeId kTid = decltype(tid)::value;
    using scalar_t = typename marlin::MarlinScalarType<kTid>::scalar_t;
    swordfish_dequant_dense_kernel<kTid, decltype(w8c)::value,
                                   decltype(zpc)::value>
        <<<grid, 128, 0, stream>>>(
            b, reinterpret_cast<const scalar_t*>(s),
            reinterpret_cast<const scalar_t*>(z), perm,
            reinterpret_cast<scalar_t*>(w_dense), k, n, group_size);
  };
  using kF16 = std::integral_constant<aphrodite::ScalarTypeId,
                                      aphrodite::kFloat16.id()>;
  using kBF16 = std::integral_constant<aphrodite::ScalarTypeId,
                                       aphrodite::kBFloat16.id()>;
  using kT = std::true_type;
  using kF = std::false_type;
  if (is_half) {
    if (has_zp) {
      run(kF16{}, kF{}, kT{});
    } else if (w8) {
      run(kF16{}, kT{}, kF{});
    } else {
      run(kF16{}, kF{}, kF{});
    }
  } else {
    if (has_zp) {
      run(kBF16{}, kF{}, kT{});
    } else if (w8) {
      run(kBF16{}, kT{}, kF{});
    } else {
      run(kBF16{}, kF{}, kF{});
    }
  }

  // C = A W with row-major tensors through column-major cuBLAS as
  // C^T = W^T A^T.
  cublasHandle_t handle = get_current_cuda_blas_handle();
  SWORDFISH_CHECK_CUBLAS(cublasSetStream(handle, stream));
  const float alpha = 1.0f, beta = 0.0f;
  const cudaDataType_t ct = is_half ? CUDA_R_16F : CUDA_R_16BF;
  SWORDFISH_CHECK_CUBLAS(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, w_dense, ct, n, a, ct,
      k, &beta, c, ct, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace swordfish
