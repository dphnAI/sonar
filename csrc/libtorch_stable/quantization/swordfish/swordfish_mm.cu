// w4a16 decode GEMM over the Swordfish ABI v1 packed weight
//. The kernel lives in swordfish_decode.cuh.

#include <algorithm>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"
#include "swordfish_decode.cuh"

namespace swordfish {

// Defined in swordfish_prefill.cu (same extension).
torch::stable::Tensor swordfish_prefill_mm(torch::stable::Tensor& a,
                                           torch::stable::Tensor& b_packed,
                                           torch::stable::Tensor& group_scales,
                                           int64_t group_size, int64_t size_k,
                                           int64_t size_n);

namespace {

// Decode/prefill crossover. It must live in C++ because a Python-side
// branch is traced by torch.compile at one representative M and baked into
// the compiled graph. Here the true runtime M decides on every call and on
// every captured CUDA graph.
inline constexpr int64_t kPrefillMinM = 48;

inline bool use_prefill(int64_t m, torch::headeronly::ScalarType a_st,
                        int64_t group_size, int64_t k, int64_t n) {
  return m >= kPrefillMinM &&
         a_st == torch::headeronly::ScalarType::BFloat16 &&
         group_size == 128 && k % 128 == 0 && n % 128 == 0;
}

template <aphrodite::ScalarTypeId type_id, int T>
void launch_decode_atomic_t(const void* a, const int32_t* b, const void* s,
                            void* c, int m, int k, int n, int group_size,
                            cudaStream_t stream) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  constexpr int kStagesT = T == 1 ? kStages : (T == 2 ? 4 : 3);
  static int ctas_per_sm = 0;  // per (type, T) instantiation
  if (ctas_per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, swordfish_decode_kernel<type_id, true, T>,
        kDecodeThreads, 0);
    if (ctas_per_sm <= 0) ctas_per_sm = 4;
  }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int m_ctas = (m + 16 * T - 1) / (16 * T);
  const int nb = n / kBlockN;
  const int num_pairs = k / 32;
  const int target = ctas_per_sm * sms;
  // Floor division. Splitting K pays only when columns fall well short of
  // the machine, since slicing shortens the per-warp pipeline and raises
  // atomic contention.
  int split = target / (nb * m_ctas);
  // Keep at least 2*kStagesT pairs per warp so the pipeline reaches steady
  // state within a slice.
  const int max_split =
      std::max(1, num_pairs / (kDecodeWarps * 2 * kStagesT));
  split = std::min(std::max(split, 1), max_split);
  dim3 sgrid(m_ctas, nb, split);
  // At split 1 each CTA zeroes its exclusive C tile in-kernel. At split > 1
  // tiles are shared and the memset is required.
  if (split > 1) {
    cudaMemsetAsync(c, 0, size_t(m) * n * sizeof(scalar_t), stream);
  }
  swordfish_decode_kernel<type_id, true, T><<<sgrid, kDecodeThreads, 0,
                                              stream>>>(
      reinterpret_cast<const scalar_t*>(a), b,
      reinterpret_cast<const scalar_t*>(s), reinterpret_cast<scalar_t*>(c),
      m, k, n, group_size);
}

template <aphrodite::ScalarTypeId type_id>
void launch_decode_atomic(int T, const void* a, const int32_t* b,
                          const void* s, void* c, int m, int k, int n,
                          int group_size, cudaStream_t stream) {
  if (T == 1) {
    launch_decode_atomic_t<type_id, 1>(a, b, s, c, m, k, n, group_size,
                                       stream);
  } else if (T == 2) {
    launch_decode_atomic_t<type_id, 2>(a, b, s, c, m, k, n, group_size,
                                       stream);
  } else {
    launch_decode_atomic_t<type_id, 3>(a, b, s, c, m, k, n, group_size,
                                       stream);
  }
}

template <aphrodite::ScalarTypeId type_id>
void launch_decode(const void* a, const int32_t* b, const void* s, void* c,
                   int m, int k, int n, int group_size, cudaStream_t stream) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  // Sub-prefill window. One CTA fuses T m16 tiles against one weight stream
  // (each packed word dequants once and fans out to T mmas), so the window
  // streams the weights exactly once per column, and split-K fills the
  // machine at narrow N.
  if (m <= 47) {
    const int T = m <= 16 ? 1 : (m <= 32 ? 2 : 3);
    launch_decode_atomic<type_id>(T, a, b, s, c, m, k, n, group_size, stream);
  } else {
    dim3 grid((m + 15) / 16, n / kBlockN);
    swordfish_decode_kernel<type_id, false><<<grid, kDecodeThreads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(a), b,
        reinterpret_cast<const scalar_t*>(s), reinterpret_cast<scalar_t*>(c),
        m, k, n, group_size);
  }
}

}  // namespace

torch::stable::Tensor swordfish_mm(torch::stable::Tensor& a,
                                   torch::stable::Tensor& b_packed,
                                   torch::stable::Tensor& group_scales,
                                   int64_t group_size, int64_t size_k,
                                   int64_t size_n) {
  STD_TORCH_CHECK(shape_ok(size_k, size_n), "swordfish ABI v1 requires K % ",
                  kBlockK, " == 0 and N % ", kBlockN, " == 0; got K=", size_k,
                  " N=", size_n);
  STD_TORCH_CHECK(a.dim() == 2 && a.size(1) == size_k,
                  "a must be [M, K] with K=", size_k);
  STD_TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == size_k,
                  "a must be contiguous");
  const auto a_st = a.scalar_type();
  STD_TORCH_CHECK(a_st == torch::headeronly::ScalarType::Half ||
                      a_st == torch::headeronly::ScalarType::BFloat16,
                  "a must be fp16 or bf16");
  STD_TORCH_CHECK(group_scales.scalar_type() == a_st,
                  "group_scales dtype must match a");

  const int64_t nb = num_blocks_n(size_n);
  const int64_t kb = num_blocks_k(size_k);
  STD_TORCH_CHECK(
      b_packed.scalar_type() == torch::headeronly::ScalarType::Int &&
          b_packed.dim() == 3 && b_packed.size(0) == nb &&
          b_packed.size(1) == kb && b_packed.size(2) == kBlockInt32,
      "b_packed must be int32 [", nb, ", ", kb, ", ", kBlockInt32, "]");

  int64_t num_groups = 1;
  if (group_size != -1) {
    // The decode mainloop consumes k16-slice PAIRS (k32) with scales hoisted
    // per pair, so a scale group must cover whole pairs.
    STD_TORCH_CHECK(group_size > 0 && size_k % group_size == 0 &&
                        group_size % (2 * kMarlinTileK) == 0,
                    "group_size must be -1 or a multiple of ",
                    2 * kMarlinTileK, " dividing K; got ", group_size);
    num_groups = size_k / group_size;
  }
  STD_TORCH_CHECK(group_scales.dim() == 2 &&
                      group_scales.size(0) == num_groups &&
                      group_scales.size(1) == size_n,
                  "group_scales must be [", num_groups, ", ", size_n, "]");

  const int64_t size_m = a.size(0);

  if (use_prefill(size_m, a_st, group_size, size_k, size_n)) {
    return swordfish_prefill_mm(a, b_packed, group_scales, group_size, size_k,
                                size_n);
  }

  const int32_t device_index = a.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  torch::stable::Tensor c =
      torch::stable::empty({size_m, size_n}, a_st, std::nullopt, a.device());
  if (size_m == 0) return c;

  const auto* b_ptr = reinterpret_cast<const int32_t*>(b_packed.const_data_ptr());
  const void* a_ptr = a.const_data_ptr();
  const void* s_ptr = group_scales.const_data_ptr();
  void* c_ptr = c.mutable_data_ptr();

  if (a_st == torch::headeronly::ScalarType::Half) {
    launch_decode<aphrodite::kFloat16.id()>(a_ptr, b_ptr, s_ptr, c_ptr, size_m,
                                            size_k, size_n, group_size,
                                            stream);
  } else {
    launch_decode<aphrodite::kBFloat16.id()>(a_ptr, b_ptr, s_ptr, c_ptr,
                                             size_m, size_k, size_n,
                                             group_size, stream);
  }

  return c;
}

}  // namespace swordfish

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("swordfish_mm", TORCH_BOX(&swordfish::swordfish_mm));
}
