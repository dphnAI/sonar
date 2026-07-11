// w4a16 decode GEMM over the Swordfish ABI v1 packed weight
//. The kernel lives in swordfish_decode.cuh.

#include <algorithm>
#include <optional>

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
torch::stable::Tensor swordfish_prefill_mm(
    torch::stable::Tensor& a, torch::stable::Tensor& b_packed,
    torch::stable::Tensor& group_scales,
    std::optional<torch::stable::Tensor> const& group_zps, int64_t group_size,
    int64_t size_k, int64_t size_n);

namespace {

// Decode/prefill crossover. It must live in C++ because a Python-side
// branch is traced by torch.compile at one representative M and baked into
// the compiled graph. Here the true runtime M decides on every call and on
// every captured CUDA graph.
inline bool use_prefill(int64_t m, torch::headeronly::ScalarType a_st,
                        int64_t group_size, int64_t k, int64_t n) {
  if (a_st != torch::headeronly::ScalarType::BFloat16 || group_size != 128 ||
      k % 128 != 0 || n % 128 != 0) {
    return false;
  }
  // The prefill grid launches about n/128 CTAs per M tile. When that fills
  // the machine the tcgen05 path wins from M 48 up; when it underfills
  // (many SMs, narrow N), the Stream-K decode window carries [17, 96).
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    if (sms <= 0) sms = 1;
  }
  const bool prefill_fills = n / 128 >= sms;
  return m >= 96 || (m >= 48 && prefill_fills);
}

inline int cached_sm_count() {
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    if (sms <= 0) sms = 1;
  }
  return sms;
}

template <aphrodite::ScalarTypeId type_id, int T, bool HAS_ZP>
void launch_decode_streamk_t(const void* a, const int32_t* b, const void* s,
                             const void* z, void* c, int m, int k, int n,
                             int group_size, cudaStream_t stream) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  constexpr int kStagesT = T == 1 ? kStages : (T == 2 ? 4 : 3);
  static int ctas_per_sm = 0;
  if (ctas_per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, swordfish_decode_streamk_kernel<type_id, T, HAS_ZP>,
        kDecodeThreads, 0);
    if (ctas_per_sm <= 0) ctas_per_sm = 2;
  }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int m_tiles = (m + 15) / 16;
  const int m_groups = (m_tiles + T - 1) / T;
  const int nb = n / kBlockN;
  const int num_pairs = k / 32;
  const int64_t total = int64_t(m_groups) * nb * num_pairs;
  // Cap the grid so every warp gets at least a pipeline's worth of pairs.
  const int64_t max_warps = total / (2 * kStagesT) > 0 ? total / (2 * kStagesT) : 1;
  int ctas = ctas_per_sm * sms;
  if (int64_t(ctas) * kDecodeWarps > max_warps) {
    ctas = int((max_warps + kDecodeWarps - 1) / kDecodeWarps);
  }
  if (ctas < 1) ctas = 1;
  cudaMemsetAsync(c, 0, size_t(m) * n * sizeof(scalar_t), stream);
  swordfish_decode_streamk_kernel<type_id, T, HAS_ZP>
      <<<ctas, kDecodeThreads, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(a), b,
          reinterpret_cast<const scalar_t*>(s),
          reinterpret_cast<const scalar_t*>(z),
          reinterpret_cast<scalar_t*>(c), m, k, n, group_size, m_groups);
}

template <aphrodite::ScalarTypeId type_id, int W, bool HAS_ZP>
void launch_decode_mshare_t(const void* a, const int32_t* b, const void* s,
                            const void* z, void* c, int m, int k, int n,
                            int group_size, cudaStream_t stream) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  static int ctas_per_sm = 0;
  if (ctas_per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, swordfish_decode_mshare_kernel<type_id, W, HAS_ZP>,
        W * 32, 0);
    if (ctas_per_sm <= 0) ctas_per_sm = 1;
  }
  int sms = cached_sm_count();
  const int m_tiles = (m + 15) / 16;
  const int m_groups = (m_tiles + W - 1) / W;
  const int nb = n / kBlockN;
  const int num_chunks = k / 64;
  const int64_t total = int64_t(m_groups) * nb * num_chunks;
  int ctas = ctas_per_sm * sms;
  // Fill is load-bearing; a grid capped toward long unbroken segments
  // measured far worse. Only bound by available work.
  const int64_t max_ctas = total / 6 > 0 ? total / 6 : 1;
  if (ctas > max_ctas) ctas = int(max_ctas);
  if (ctas < 1) ctas = 1;
  cudaMemsetAsync(c, 0, size_t(m) * n * sizeof(scalar_t), stream);
  swordfish_decode_mshare_kernel<type_id, W, HAS_ZP>
      <<<ctas, W * 32, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(a), b,
          reinterpret_cast<const scalar_t*>(s),
          reinterpret_cast<const scalar_t*>(z),
          reinterpret_cast<scalar_t*>(c), m, k, n, group_size, m_groups);
}

template <aphrodite::ScalarTypeId type_id, int T, bool HAS_ZP>
void launch_decode_atomic_t(const void* a, const int32_t* b, const void* s,
                            const void* z, void* c, int m, int k, int n,
                            int group_size, cudaStream_t stream) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  constexpr int kStagesT = T == 1 ? kStages : (T == 2 ? 4 : 3);
  static int ctas_per_sm = 0;  // per (type, T) instantiation
  if (ctas_per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, swordfish_decode_kernel<type_id, true, T, HAS_ZP>,
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
  swordfish_decode_kernel<type_id, true, T, HAS_ZP>
      <<<sgrid, kDecodeThreads, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(a), b,
          reinterpret_cast<const scalar_t*>(s),
          reinterpret_cast<const scalar_t*>(z),
          reinterpret_cast<scalar_t*>(c), m, k, n, group_size);
}

template <aphrodite::ScalarTypeId type_id, bool HAS_ZP>
void launch_decode_atomic(int T, const void* a, const int32_t* b,
                          const void* s, const void* z, void* c, int m, int k,
                          int n, int group_size, cudaStream_t stream) {
  if (T == 1) {
    launch_decode_atomic_t<type_id, 1, HAS_ZP>(a, b, s, z, c, m, k, n,
                                               group_size, stream);
  } else if (T == 2) {
    launch_decode_atomic_t<type_id, 2, HAS_ZP>(a, b, s, z, c, m, k, n,
                                               group_size, stream);
  } else {
    launch_decode_atomic_t<type_id, 3, HAS_ZP>(a, b, s, z, c, m, k, n,
                                               group_size, stream);
  }
}

template <aphrodite::ScalarTypeId type_id, bool HAS_ZP>
void launch_decode(const void* a, const int32_t* b, const void* s,
                   const void* z, void* c, int m, int k, int n, int group_size,
                   cudaStream_t stream) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  if (m <= 16) {
    // Tuned single-tile path with in-kernel C zeroing and heuristic split-K.
    launch_decode_atomic<type_id, HAS_ZP>(1, a, b, s, z, c, m, k, n,
                                          group_size, stream);
  } else if (m <= 96) {
    // Window dispatch. When columns alone fill the machine the fused atomic
    // grid (in-kernel zeroing, no memset) is already balanced; otherwise
    // Stream-K hands each warp a contiguous flat range of (fused tile,
    // column, pair) work and the atomic epilogue merges segments, removing
    // split-K heuristics and wave quantization.
    const bool wide_n = n / kBlockN >= 4 * cached_sm_count();
    if (wide_n && m <= 47) {
      launch_decode_atomic<type_id, HAS_ZP>(m <= 32 ? 2 : 3, a, b, s, z, c, m,
                                            k, n, group_size, stream);
    } else if (m <= 32) {
      launch_decode_streamk_t<type_id, 2, HAS_ZP>(a, b, s, z, c, m, k, n,
                                                  group_size, stream);
    } else if (m <= 47) {
      launch_decode_streamk_t<type_id, 3, HAS_ZP>(a, b, s, z, c, m, k, n,
                                                  group_size, stream);
    } else if (m <= 64) {
      // M-shared CTA for the band the prefill crossover leaves to decode on
      // many-SM parts. W warps, one m16 tile each for the whole K range, so
      // the weight stream covers all rows once.
      launch_decode_mshare_t<type_id, 4, HAS_ZP>(a, b, s, z, c, m, k, n,
                                                 group_size, stream);
    } else {
      launch_decode_mshare_t<type_id, 6, HAS_ZP>(a, b, s, z, c, m, k, n,
                                                 group_size, stream);
    }
  } else {
    dim3 grid((m + 15) / 16, n / kBlockN);
    swordfish_decode_kernel<type_id, false, 1, HAS_ZP>
        <<<grid, kDecodeThreads, 0, stream>>>(
            reinterpret_cast<const scalar_t*>(a), b,
            reinterpret_cast<const scalar_t*>(s),
            reinterpret_cast<const scalar_t*>(z),
            reinterpret_cast<scalar_t*>(c), m, k, n, group_size);
  }
}

}  // namespace

torch::stable::Tensor swordfish_mm(
    torch::stable::Tensor& a, torch::stable::Tensor& b_packed,
    torch::stable::Tensor& group_scales,
    std::optional<torch::stable::Tensor> const& group_zps, int64_t group_size,
    int64_t size_k, int64_t size_n) {
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
  const bool has_zp = group_zps.has_value();
  if (has_zp) {
    STD_TORCH_CHECK(group_zps->scalar_type() == a_st &&
                        group_zps->dim() == 2 &&
                        group_zps->size(0) == num_groups &&
                        group_zps->size(1) == size_n,
                    "group_zps must be [", num_groups, ", ", size_n,
                    "] with a's dtype");
  }

  const int64_t size_m = a.size(0);

  if (use_prefill(size_m, a_st, group_size, size_k, size_n)) {
    return swordfish_prefill_mm(a, b_packed, group_scales, group_zps,
                                group_size, size_k, size_n);
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
  const void* z_ptr = has_zp ? group_zps->const_data_ptr() : nullptr;
  void* c_ptr = c.mutable_data_ptr();

  if (a_st == torch::headeronly::ScalarType::Half) {
    if (has_zp) {
      launch_decode<aphrodite::kFloat16.id(), true>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream);
    } else {
      launch_decode<aphrodite::kFloat16.id(), false>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream);
    }
  } else {
    if (has_zp) {
      launch_decode<aphrodite::kBFloat16.id(), true>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream);
    } else {
      launch_decode<aphrodite::kBFloat16.id(), false>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream);
    }
  }

  return c;
}

}  // namespace swordfish

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("swordfish_mm", TORCH_BOX(&swordfish::swordfish_mm));
}
