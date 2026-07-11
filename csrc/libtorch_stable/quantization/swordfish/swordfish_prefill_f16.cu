// fp16-activation instantiations of the Swordfish prefill configurations.

#include "swordfish_prefill_impl.cuh"

namespace swordfish {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
namespace prefill {
template void run_prefill_all<cutlass::half_t>(
    torch::stable::Tensor&, torch::stable::Tensor&, torch::stable::Tensor&,
    const void*, bool, bool, int, torch::stable::Tensor&, int, int, int,
    cudaStream_t);
}  // namespace prefill
#endif
}  // namespace swordfish
