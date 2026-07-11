// Packs a GPTQ int4 weight into the Swordfish ABI v1 block-linear layout
//. Reuses the Marlin repack kernel for the in-tile
// permutation, which ABI v1 adopts verbatim, then re-tiles its flat
// [K/16, N*2] int32 layout into (NB, KB, 512) int32 blocks. Runs once at
// weight load.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"
#include "swordfish_abi.cuh"

// Defined in quantization/marlin/gptq_marlin_repack.cu (same extension).
torch::stable::Tensor gptq_marlin_repack(torch::stable::Tensor& b_q_weight,
                                         torch::stable::Tensor& perm,
                                         int64_t size_k, int64_t size_n,
                                         int64_t num_bits, bool is_a_8bit);

namespace swordfish {

namespace {

// One CTA per output block: 512 threads gather the block's int32 words from
// the flat Marlin layout (two words each at 8-bit). Writes are perfectly
// coalesced; reads are tile-sized runs from 4 Marlin rows.
template <int kTileInt32>
__global__ void retile_marlin_to_blocks(const int32_t* __restrict__ marlin,
                                        int32_t* __restrict__ out,
                                        int64_t num_kb, int64_t size_n) {
  constexpr int kWords = 4 * kTileInt32;
  const int64_t nb = blockIdx.x;
  const int64_t kb = blockIdx.y;
  for (int w = threadIdx.x; w < kWords; w += kBlockInt32) {
    const int64_t dst = (nb * num_kb + kb) * kWords + w;
    out[dst] = marlin[marlin_word_index<kTileInt32>(nb, kb, w, size_n)];
  }
}

}  // namespace

torch::stable::Tensor swordfish_prepack_B(torch::stable::Tensor& b_q_weight,
                                          int64_t size_k, int64_t size_n,
                                          int64_t num_bits) {
  STD_TORCH_CHECK(shape_ok(size_k, size_n), "swordfish ABI v1 requires K % ",
                  kBlockK, " == 0 and N % ", kBlockN, " == 0; got K=", size_k,
                  " N=", size_n, " (v1 tail policy: reject)");
  STD_TORCH_CHECK(num_bits == 4 || num_bits == 8,
                  "swordfish supports 4-bit and 8-bit weights");

  const int32_t device_index = b_q_weight.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);

  // Stage 1: Marlin in-tile permutation (no act_order in v1 -> empty perm).
  torch::stable::Tensor empty_perm = torch::stable::empty(
      {0}, torch::headeronly::ScalarType::Int, std::nullopt,
      b_q_weight.device());
  torch::stable::Tensor marlin_flat = gptq_marlin_repack(
      b_q_weight, empty_perm, size_k, size_n, num_bits,
      /*is_a_8bit=*/false);

  // Stage 2: re-tile to (NB, KB, 512|1024) int32 blocks.
  const int64_t nb = num_blocks_n(size_n);
  const int64_t kb = num_blocks_k(size_k);
  const int64_t words = num_bits == 8 ? kBlockInt32_8 : kBlockInt32;
  torch::stable::Tensor out = torch::stable::empty(
      {nb, kb, words}, torch::headeronly::ScalarType::Int, std::nullopt,
      b_q_weight.device());

  dim3 grid(nb, kb);
  if (num_bits == 8) {
    retile_marlin_to_blocks<256><<<grid, kBlockInt32, 0, stream>>>(
        reinterpret_cast<const int32_t*>(marlin_flat.const_data_ptr()),
        reinterpret_cast<int32_t*>(out.mutable_data_ptr()), kb, size_n);
  } else {
    retile_marlin_to_blocks<128><<<grid, kBlockInt32, 0, stream>>>(
        reinterpret_cast<const int32_t*>(marlin_flat.const_data_ptr()),
        reinterpret_cast<int32_t*>(out.mutable_data_ptr()), kb, size_n);
  }

  return out;
}

}  // namespace swordfish

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("swordfish_prepack_B", TORCH_BOX(&swordfish::swordfish_prepack_B));
}
