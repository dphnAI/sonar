// w4a16 decode GEMM over the Swordfish ABI v1 packed weight
//. The kernel lives in swordfish_decode.cuh.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/ops.h"
#include "libtorch_stable/torch_utils.h"
#include "swordfish_decode.cuh"

namespace swordfish {

// Defined in swordfish_dense_tier.cu (same extension).
void swordfish_dense_tier_mm(const void* a, const int32_t* b, const void* s,
                             const void* z, const int32_t* perm, void* c,
                             void* w_dense, bool is_half, bool w8,
                             bool has_zp, int m, int k, int n, int group_size,
                             cudaStream_t stream);

// Defined in swordfish_prefill.cu (same extension).
torch::stable::Tensor swordfish_prefill_mm(
    torch::stable::Tensor& a, torch::stable::Tensor& b_packed,
    torch::stable::Tensor& group_scales,
    std::optional<torch::stable::Tensor> const& group_zps, int64_t num_bits,
    int64_t group_size, int64_t size_k, int64_t size_n);

namespace {

// Decode/prefill crossover. It must live in C++ because a Python-side
// branch is traced by torch.compile at one representative M and baked into
// the compiled graph. Here the true runtime M decides on every call and on
// every captured CUDA graph.
inline bool use_prefill(int64_t m, torch::headeronly::ScalarType a_st,
                        bool w8, int64_t group_size, int64_t k, int64_t n) {
  if (a_st != torch::headeronly::ScalarType::BFloat16 &&
      a_st != torch::headeronly::ScalarType::Half) {
    return false;
  }
  if (w8 ? group_size != 128
         : (group_size != 32 && group_size != 64 && group_size != 128)) {
    return false;
  }
  if (k % 128 != 0 || n % 128 != 0) return false;
  // The prefill grid launches about n/128 CTAs per M tile. When that fills
  // the machine the tcgen05 path wins from M 48 up; when it underfills
  // (many SMs, narrow N), the Stream-K decode window carries [17, 96).
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    if (sms <= 0) sms = 1;
  }
  const bool prefill_fills = n / 128 >= sms;
  // The four-tile decode window carries [48, 56) even at wide N; the
  // tcgen05 wave only pulls ahead of it from 56 rows up. K-heavy narrow-N
  // shapes keep Stream-K through [96, 128) on many-SM parts, where a single
  // underfilled tcgen05 wave loses to it.
  if (m >= 96 && m < 128 && k >= 2 * n && n / 128 < sms / 4) return false;
  return m >= 96 || (m >= 56 && prefill_fills);
}

// Above this M the problem is compute-bound and dequant-once + dense
// cuBLAS outruns the fused mixed-input mainloops, so the tier takes over.
// 8-bit crosses earlier than 4-bit because the mixed-input pipeline moves
// twice the weight bytes through the transform. On few-SM parts the dense
// rate only clears the 4-bit weight stream's bandwidth win at 8 bits, and
// K-heavy shapes pay proportionally more dequant per flop, so they cross
// later still.
inline int dense_tier_min_m(int sms, bool w8, int64_t size_k, int64_t size_n) {
  static const int v = [] {
    const char* e = std::getenv("APHRODITE_SWORDFISH_DENSE_M");
    return e != nullptr && e[0] != '\0' ? std::atoi(e) : 0;
  }();
  if (v != 0) return v;
  if (sms < 100) {
    if (!w8) return INT_MAX;
    return size_k >= 2 * size_n ? 8192 : 2048;
  }
  return w8 ? 1024 : 4096;
}

// APHRODITE_SWORDFISH_DETERMINISTIC forces the run-stable decode paths:
// the smem-reduction epilogue replaces every atomic-window kernel, at the
// decode window's cost. The tcgen05 prefill is deterministic either way.
// The fused-MoE kernels keep their atomic merge regardless.
inline bool force_deterministic() {
  static const bool v = [] {
    const char* e = std::getenv("APHRODITE_SWORDFISH_DETERMINISTIC");
    return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
  }();
  return v;
}

inline int cached_sm_count() {
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    if (sms <= 0) sms = 1;
  }
  return sms;
}

template <aphrodite::ScalarTypeId type_id, int T, bool HAS_ZP,
          bool W8 = false>
void launch_decode_streamk_t(const void* a, const int32_t* b, const void* s,
                             const void* z, void* c, int m, int k, int n,
                             int group_size, cudaStream_t stream,
                             bool c_zeroed = false) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  constexpr int kStagesT =
      T == 1 ? kStages
             : (T == 2 ? (W8 ? 5 : 4) : (T == 3 ? 3 : (W8 ? 4 : 2)));
  constexpr int kUnitK = W8 ? 16 : 32;
  static int ctas_per_sm = 0;
  if (ctas_per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm,
        swordfish_decode_streamk_kernel<type_id, T, HAS_ZP, W8>,
        kDecodeThreads, 0);
    if (ctas_per_sm <= 0) ctas_per_sm = 2;
  }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int m_tiles = (m + 15) / 16;
  const int m_groups = (m_tiles + T - 1) / T;
  const int nb = n / kBlockN;
  const int num_pairs = k / kUnitK;
  const int64_t total = int64_t(m_groups) * nb * num_pairs;
  // Cap the grid so every warp gets at least a pipeline's worth of pairs.
  const int64_t max_warps = total / (2 * kStagesT) > 0 ? total / (2 * kStagesT) : 1;
  int ctas = ctas_per_sm * sms;
  if (int64_t(ctas) * kDecodeWarps > max_warps) {
    ctas = int((max_warps + kDecodeWarps - 1) / kDecodeWarps);
  }
  if (ctas < 1) ctas = 1;
  if (!c_zeroed) launch_zero_c<scalar_t>(c, m, n, stream);
  swordfish_decode_streamk_kernel<type_id, T, HAS_ZP, W8>
      <<<ctas, kDecodeThreads, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(a), b,
          reinterpret_cast<const scalar_t*>(s),
          reinterpret_cast<const scalar_t*>(z),
          reinterpret_cast<scalar_t*>(c), m, k, n, group_size, m_groups);
}

template <aphrodite::ScalarTypeId type_id, int T, bool HAS_ZP,
          bool W8 = false>
void launch_decode_atomic_t(const void* a, const int32_t* b, const void* s,
                            const void* z, void* c, int m, int k, int n,
                            int group_size, cudaStream_t stream,
                            bool c_zeroed = false) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  constexpr int kStagesT = T == 1 ? kStages : (T == 2 ? 4 : 3);
  static int ctas_per_sm = 0;  // per (type, T) instantiation
  if (ctas_per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm,
        swordfish_decode_kernel<type_id, true, T, HAS_ZP, W8>,
        kDecodeThreads, 0);
    if (ctas_per_sm <= 0) ctas_per_sm = 4;
  }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int m_ctas = (m + 16 * T - 1) / (16 * T);
  const int nb = n / kBlockN;
  const int num_pairs = k / (W8 ? 16 : 32);
  const int target = ctas_per_sm * sms;
  // Floor division. Splitting K pays only when columns fall well short of
  // the machine, since slicing shortens the per-warp pipeline and raises
  // atomic contention.
  int split = target / (nb * m_ctas);
  // On few-SM parts, columns alone reaching the machine is enough. K-slices
  // beyond that shorten the latency-bound per-warp chains and buy the
  // launcher memset, measured 8-15 percent slower at M=1 narrow N on 20
  // SMs. Many-SM parts want the occupancy target: the same sweep run the
  // other way is 2-3x slower at split 1 on 148 SMs.
  if (sms < 100 && nb * m_ctas >= sms) split = 1;
  // Keep at least 2*kStagesT pairs per warp so the pipeline reaches steady
  // state within a slice.
  const int max_split =
      std::max(1, num_pairs / (kDecodeWarps * 2 * kStagesT));
  split = std::min(std::max(split, 1), max_split);
  dim3 sgrid(m_ctas, nb, split);
  // At split 1 each CTA zeroes its exclusive C tile in-kernel. At split > 1
  // tiles are shared and the memset is required.
  if (split > 1 && !c_zeroed) {
    launch_zero_c<scalar_t>(c, m, n, stream);
  }
  swordfish_decode_kernel<type_id, true, T, HAS_ZP, W8>
      <<<sgrid, kDecodeThreads, 0, stream>>>(
          reinterpret_cast<const scalar_t*>(a), b,
          reinterpret_cast<const scalar_t*>(s),
          reinterpret_cast<const scalar_t*>(z),
          reinterpret_cast<scalar_t*>(c), m, k, n, group_size);
}

template <aphrodite::ScalarTypeId type_id, bool HAS_ZP, bool W8 = false>
void launch_decode_atomic(int T, const void* a, const int32_t* b,
                          const void* s, const void* z, void* c, int m, int k,
                          int n, int group_size, cudaStream_t stream,
                          bool c_zeroed = false) {
  if (T == 1) {
    launch_decode_atomic_t<type_id, 1, HAS_ZP, W8>(a, b, s, z, c, m, k, n,
                                                   group_size, stream,
                                                   c_zeroed);
  } else if (T == 2) {
    launch_decode_atomic_t<type_id, 2, HAS_ZP, W8>(a, b, s, z, c, m, k, n,
                                                   group_size, stream,
                                                   c_zeroed);
  } else {
    launch_decode_atomic_t<type_id, 3, HAS_ZP, W8>(a, b, s, z, c, m, k, n,
                                                   group_size, stream,
                                                   c_zeroed);
  }
}

template <aphrodite::ScalarTypeId type_id, bool HAS_ZP, bool W8 = false>
void launch_decode(const void* a, const int32_t* b, const void* s,
                   const void* z, void* c, int m, int k, int n, int group_size,
                   cudaStream_t stream, bool c_zeroed = false) {
  using scalar_t = typename marlin::MarlinScalarType<type_id>::scalar_t;
  if (force_deterministic()) {
    dim3 grid((m + 15) / 16, n / kBlockN);
    swordfish_decode_kernel<type_id, false, 1, HAS_ZP, W8>
        <<<grid, kDecodeThreads, 0, stream>>>(
            reinterpret_cast<const scalar_t*>(a), b,
            reinterpret_cast<const scalar_t*>(s),
            reinterpret_cast<const scalar_t*>(z),
            reinterpret_cast<scalar_t*>(c), m, k, n, group_size);
  } else if (m <= 16) {
    // Tuned single-tile path with in-kernel C zeroing and heuristic split-K.
    launch_decode_atomic<type_id, HAS_ZP, W8>(1, a, b, s, z, c, m, k, n,
                                              group_size, stream, c_zeroed);
  } else if (m <= 127) {
    // Window dispatch. When columns alone fill the machine the fused atomic
    // grid (in-kernel zeroing, no memset) is already balanced; otherwise
    // Stream-K hands each warp a contiguous flat range of (fused tile,
    // column, pair) work and the atomic epilogue merges segments, removing
    // split-K heuristics and wave quantization.
    const bool wide_n = n / kBlockN >= 4 * cached_sm_count();
    // On many-SM parts the whole [17, 48] band belongs to the fused atomic
    // grid at any width: its CTA shares one weight stream across four
    // warps, where Stream-K's warp-private B and A staging pays for itself
    // only when the machine is small enough for warps to get long claims.
    // Few-SM parts keep Stream-K outside wide N (measured 0.73-0.96 of
    // marlin the other way around on 20 SMs).
    const bool band_atomic = cached_sm_count() >= 100 && m <= 48;
    if (band_atomic || (wide_n && m <= 47)) {
      launch_decode_atomic<type_id, HAS_ZP, W8>(m <= 32 ? 2 : 3, a, b, s, z,
                                                c, m, k, n, group_size,
                                                stream, c_zeroed);
    } else if (m <= 32) {
      launch_decode_streamk_t<type_id, 2, HAS_ZP, W8>(a, b, s, z, c, m, k, n,
                                                      group_size, stream,
                                                      c_zeroed);
    } else if (m <= 48) {
      launch_decode_streamk_t<type_id, 3, HAS_ZP, W8>(a, b, s, z, c, m, k, n,
                                                      group_size, stream,
                                                      c_zeroed);
    } else if (m <= 64) {
      // Four-tile fusion amortizes the dequant across the whole band; the
      // m-shared CTA it replaces dequantized the same weights once per warp
      // and issued 2.7x the instructions for it.
      launch_decode_streamk_t<type_id, 4, HAS_ZP, W8>(a, b, s, z, c, m, k, n,
                                                      group_size, stream,
                                                      c_zeroed);
    } else {
      launch_decode_streamk_t<type_id, 3, HAS_ZP, W8>(a, b, s, z, c, m, k, n,
                                                      group_size, stream,
                                                      c_zeroed);
    }
  } else {
    dim3 grid((m + 15) / 16, n / kBlockN);
    swordfish_decode_kernel<type_id, false, 1, HAS_ZP, W8>
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
    std::optional<torch::stable::Tensor> const& group_zps,
    std::optional<torch::stable::Tensor> const& perm, int64_t num_bits,
    int64_t group_size, int64_t size_k, int64_t size_n) {
  STD_TORCH_CHECK(num_bits == 4 || num_bits == 8,
                  "swordfish supports 4-bit and 8-bit weights");
  const bool w8 = num_bits == 8;
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
  const int64_t words = w8 ? kBlockInt32_8 : kBlockInt32;
  STD_TORCH_CHECK(
      b_packed.scalar_type() == torch::headeronly::ScalarType::Int &&
          b_packed.dim() == 3 && b_packed.size(0) == nb &&
          b_packed.size(1) == kb && b_packed.size(2) == words,
      "b_packed must be int32 [", nb, ", ", kb, ", ", words, "]");

  int64_t num_groups = 1;
  // Channelwise checkpoints arrive as group -1 with the single scale row
  // replicated to group 128 by the weight loader. The grouped tiers
  // (tcgen05 prefill, dense dequant) consume the replicated rows as g128;
  // the decode kernels take the native -1 path, which reads row 0 once and
  // skips the per-group fetch/expand bookkeeping the duplicate rows would
  // otherwise re-run at every 128-row boundary.
  int64_t tier_group = group_size;
  if (group_size == -1) {
    if (group_scales.size(0) == size_k / 128) {
      num_groups = size_k / 128;
      tier_group = 128;
    }
  } else {
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
    STD_TORCH_CHECK(!w8, "zero points are a 4-bit (AWQ/HQQ) feature");
    STD_TORCH_CHECK(group_zps->scalar_type() == a_st &&
                        group_zps->dim() == 2 &&
                        group_zps->size(0) == num_groups &&
                        group_zps->size(1) == size_n,
                    "group_zps must be [", num_groups, ", ", size_n,
                    "] with a's dtype");
  }

  const int64_t size_m = a.size(0);

  const bool has_perm = perm.has_value() && perm->numel() > 0;
  if (has_perm) {
    STD_TORCH_CHECK(perm->scalar_type() == torch::headeronly::ScalarType::Int &&
                        perm->numel() == size_k,
                    "perm must be int32 [", size_k, "]");
  }

  if (size_m >= dense_tier_min_m(cached_sm_count(), w8, size_k, size_n) &&
      !force_deterministic()) {
    const int32_t device_index = a.get_device_index();
    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    const cudaStream_t stream = get_current_cuda_stream(device_index);
    torch::stable::Tensor c = torch::stable::empty({size_m, size_n}, a_st,
                                                   std::nullopt, a.device());
    torch::stable::Tensor w_dense = torch::stable::empty(
        {size_k, size_n}, a_st, std::nullopt, a.device());
    // Act_order folds into the weight scatter here, so the activations are
    // consumed unpermuted.
    swordfish_dense_tier_mm(
        a.const_data_ptr(),
        reinterpret_cast<const int32_t*>(b_packed.const_data_ptr()),
        group_scales.const_data_ptr(),
        has_zp ? group_zps->const_data_ptr() : nullptr,
        has_perm ? reinterpret_cast<const int32_t*>(perm->const_data_ptr())
                 : nullptr,
        c.mutable_data_ptr(), w_dense.mutable_data_ptr(),
        a_st == torch::headeronly::ScalarType::Half, w8, has_zp, int(size_m),
        int(size_k), int(size_n), int(tier_group), stream);
    return c;
  }

  // The fused paths consume group-sorted K. In the decode window the sort
  // and the output zeroing fuse into one prep launch (a separate
  // permute_cols node plus a zero node cost several microseconds of launch
  // and engine gaps per GEMM at bs=1); prefill and the tall tail keep the
  // plain sorted copy.
  const bool will_prefill =
      use_prefill(size_m, a_st, w8, tier_group, size_k, size_n);
  const bool prep_perm = has_perm && !will_prefill && size_m <= 127 &&
                         !force_deterministic();
  torch::stable::Tensor a_used =
      has_perm && !prep_perm ? permute_cols(a, *perm) : a;

  if (will_prefill) {
    return swordfish_prefill_mm(a_used, b_packed, group_scales, group_zps,
                                num_bits, tier_group, size_k, size_n);
  }

  const int32_t device_index = a.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  torch::stable::Tensor c =
      torch::stable::empty({size_m, size_n}, a_st, std::nullopt, a.device());
  if (size_m == 0) return c;

  torch::stable::Tensor a_perm = a_used;
  if (prep_perm) {
    a_perm = torch::stable::empty({size_m, size_k}, a_st, std::nullopt,
                                  a.device());
    if (a_st == torch::headeronly::ScalarType::Half) {
      launch_prep<half>(a_used.const_data_ptr(),
                        reinterpret_cast<const int32_t*>(
                            perm->const_data_ptr()),
                        a_perm.mutable_data_ptr(), c.mutable_data_ptr(),
                        size_m, size_k, size_n, stream);
    } else {
      launch_prep<nv_bfloat16>(a_used.const_data_ptr(),
                               reinterpret_cast<const int32_t*>(
                                   perm->const_data_ptr()),
                               a_perm.mutable_data_ptr(),
                               c.mutable_data_ptr(), size_m, size_k, size_n,
                               stream);
    }
  }

  const auto* b_ptr = reinterpret_cast<const int32_t*>(b_packed.const_data_ptr());
  const void* a_ptr = a_perm.const_data_ptr();
  const void* s_ptr = group_scales.const_data_ptr();
  const void* z_ptr = has_zp ? group_zps->const_data_ptr() : nullptr;
  void* c_ptr = c.mutable_data_ptr();

  if (a_st == torch::headeronly::ScalarType::Half) {
    if (has_zp) {
      launch_decode<aphrodite::kFloat16.id(), true>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream, prep_perm);
    } else if (w8) {
      launch_decode<aphrodite::kFloat16.id(), false, true>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream, prep_perm);
    } else {
      launch_decode<aphrodite::kFloat16.id(), false>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream, prep_perm);
    }
  } else {
    if (has_zp) {
      launch_decode<aphrodite::kBFloat16.id(), true>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream, prep_perm);
    } else if (w8) {
      launch_decode<aphrodite::kBFloat16.id(), false, true>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream, prep_perm);
    } else {
      launch_decode<aphrodite::kBFloat16.id(), false>(
          a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, size_m, size_k, size_n,
          group_size, stream, prep_perm);
    }
  }

  return c;
}

}  // namespace swordfish

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("swordfish_mm", TORCH_BOX(&swordfish::swordfish_mm));
}
