// w4a16/w8a16 prefill GEMM over the Swordfish ABI v1 packed weight, using
// the forked sm100 tcgen05 mixed-input collective. This TU instantiates the
// bf16-activation configurations and hosts the op entry; the fp16 set
// compiles in swordfish_prefill_f16.cu.

#include "swordfish_prefill_impl.cuh"

namespace swordfish {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Defined in swordfish_prefill_f16.cu.
namespace prefill {
extern template void run_prefill_all<cutlass::half_t>(
    torch::stable::Tensor&, torch::stable::Tensor&, torch::stable::Tensor&,
    const void*, bool, bool, int, torch::stable::Tensor&, int, int, int,
    cudaStream_t);
}
#endif

torch::stable::Tensor swordfish_prefill_mm(
    torch::stable::Tensor& a, torch::stable::Tensor& b_packed,
    torch::stable::Tensor& group_scales,
    std::optional<torch::stable::Tensor> const& group_zps, int64_t num_bits,
    int64_t group_size, int64_t size_k, int64_t size_n) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  STD_TORCH_CHECK(shape_ok(size_k, size_n) && size_k % 128 == 0 &&
                      size_n % 128 == 0,
                  "swordfish prefill v1 requires K % 128 == 0 and "
                  "N % 128 == 0; got K=", size_k, " N=", size_n);
  STD_TORCH_CHECK(a.dim() == 2 && a.size(1) == size_k,
                  "a must be [M, K] with K=", size_k);
  STD_TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == size_k,
                  "a must be contiguous");
  const auto a_st = a.scalar_type();
  STD_TORCH_CHECK(a_st == torch::headeronly::ScalarType::BFloat16 ||
                      a_st == torch::headeronly::ScalarType::Half,
                  "swordfish prefill requires fp16 or bf16 activations");
  STD_TORCH_CHECK(group_scales.scalar_type() == a_st,
                  "group_scales dtype must match a");
  STD_TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
                  "swordfish prefill supports group sizes 32, 64 and 128; "
                  "got ", group_size);

  STD_TORCH_CHECK(num_bits == 4 || num_bits == 8,
                  "swordfish supports 4-bit and 8-bit weights");
  const bool w8 = num_bits == 8;
  const int64_t nb = num_blocks_n(size_n);
  const int64_t kb = num_blocks_k(size_k);
  const int64_t words = w8 ? kBlockInt32_8 : kBlockInt32;
  STD_TORCH_CHECK(
      b_packed.scalar_type() == torch::headeronly::ScalarType::Int &&
          b_packed.dim() == 3 && b_packed.size(0) == nb &&
          b_packed.size(1) == kb && b_packed.size(2) == words,
      "b_packed must be int32 [", nb, ", ", kb, ", ", words, "]");

  const int64_t num_groups = size_k / group_size;
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

  const int32_t device_index = a.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  torch::stable::Tensor c = torch::stable::empty(
      {size_m, size_n}, a.scalar_type(), std::nullopt, a.device());
  if (size_m == 0) return c;

  const int M = int(size_m), N = int(size_n), K = int(size_k);

  STD_TORCH_CHECK(!(has_zp && w8), "zero points are a 4-bit feature");
  STD_TORCH_CHECK(!(w8 && group_size != 128),
                  "8-bit prefill supports group_size 128 only");
  const void* zp_ptr = has_zp ? group_zps->const_data_ptr() : nullptr;
  if (a_st == torch::headeronly::ScalarType::Half) {
    prefill::run_prefill_all<cutlass::half_t>(a, b_packed, group_scales,
                                              zp_ptr, has_zp, w8,
                                              int(group_size), c, M, N, K,
                                              stream);
  } else {
    prefill::run_prefill_all<cutlass::bfloat16_t>(a, b_packed, group_scales,
                                                  zp_ptr, has_zp, w8,
                                                  int(group_size), c, M, N, K,
                                                  stream);
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
