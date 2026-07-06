// Fork-specific op registrations (extension: _C_fork).
//
// Upstream vLLM registers core/quant ops in the _C_stable_libtorch extension
// via STABLE_TORCH_LIBRARY. Aphrodite keeps a handful of fork-only kernels that
// have no upstream equivalent; this file registers them into the SAME
// torch.ops._C namespace using TORCH_LIBRARY_FRAGMENT so Python can keep
// calling torch.ops._C.<op>. The heavy kernels live in their own translation
// units; here we only declare the entry points, adapt argument types, and
// register.
//
// Currently built: DRY sampler (CPU) and EXL3 quantization (CUDA). The other
// fork kernels (exl2/aqlm/vptq/gguf/quip/...) remain in csrc but are not built.

#include <torch/library.h>
#include <torch/torch.h>

#include <optional>
#include <tuple>

#include "core/registration.h"

// ---------------------------------------------------------------------------
// DRY sampler (CPU). Implementation in csrc/cpu/dry.cpp.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dry_scan_penalties_cpu(
    const torch::Tensor& token_history_ids,
    const torch::Tensor& token_history_lens,
    const torch::Tensor& dry_multiplier, const torch::Tensor& allowed_lengths,
    const torch::Tensor& sequence_breakers_ids, const torch::Tensor& ranges,
    const torch::Tensor& max_ngram, const torch::Tensor& max_occurrences,
    const torch::Tensor& early_exit_match_len, int64_t vocab_size);

#ifndef USE_ROCM

  // -------------------------------------------------------------------------
  // EXL3 quantization (CUDA). Kernel entry points live in the exllamav3_ext
  // translation units; the .cuh headers only declare them.
  // -------------------------------------------------------------------------
  #include "quantization/exl3/exllamav3_ext/hgemm.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/exl3_gemm.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/exl3_moe.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/hadamard.cuh"
  #include "quantization/exl3/exllamav3_ext/quant/reconstruct.cuh"

// Thin adapters between the torch schema (int64/double/bool) and the native
// kernel entry points (int/float). Defined here rather than in a shared header
// so exl3 stays self-contained and conventional.
static void aphrodite_exl3_gemm(const at::Tensor& A, const at::Tensor& B,
                                at::Tensor& C,
                                const std::optional<at::Tensor>& suh,
                                const std::optional<at::Tensor>& A_had,
                                const std::optional<at::Tensor>& svh,
                                int64_t force_shape_idx, bool mcg, bool mul1,
                                int64_t force_num_sms) {
  exl3_gemm(A, B, C, suh, A_had, svh, static_cast<int>(force_shape_idx), mcg,
            mul1, static_cast<int>(force_num_sms));
}

static void aphrodite_exl3_mgemm(const at::Tensor& A, const at::Tensor& B,
                                 at::Tensor& C, const at::Tensor& suh,
                                 const at::Tensor& A_had, const at::Tensor& svh,
                                 const std::optional<at::Tensor>& indices,
                                 const std::optional<at::Tensor>& weights,
                                 int64_t k, int64_t force_shape_idx, bool mcg,
                                 bool mul1, int64_t min_index,
                                 int64_t max_index, int64_t force_num_sms) {
  exl3_mgemm(A, B, C, suh, A_had, svh, indices, weights, static_cast<int>(k),
             static_cast<int>(force_shape_idx), static_cast<uint32_t>(mcg),
             static_cast<uint32_t>(mul1), static_cast<int>(min_index),
             static_cast<int>(max_index), static_cast<int>(force_num_sms));
}

static void aphrodite_exl3_reconstruct(at::Tensor unpacked, at::Tensor packed,
                                       int64_t k, bool mcg, bool mul1) {
  reconstruct(unpacked, packed, static_cast<int>(k), mcg, mul1);
}

static void aphrodite_exl3_had_r_128(
    const at::Tensor& input, const at::Tensor& output,
    const std::optional<at::Tensor>& pre_scale,
    const std::optional<at::Tensor>& post_scale, double scale) {
  had_r_128(input, output, pre_scale, post_scale, static_cast<float>(scale));
}

static void aphrodite_exl3_hgemm(at::Tensor a, at::Tensor b, at::Tensor c) {
  hgemm(a, b, c);
}

static void aphrodite_exl3_moe(
    const at::Tensor& hidden_state, const at::Tensor& output_state,
    const at::Tensor& expert_count, const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted, const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u, const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u, int64_t act_function, int64_t K_gate,
    int64_t K_up, int64_t K_down, const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis, const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh, const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,
    bool gate_mcg, bool gate_mul1, bool up_mcg, bool up_mul1, bool down_mcg,
    bool down_mul1, double act_limit) {
  exl3_moe(hidden_state, output_state, expert_count, token_sorted,
           weight_sorted, temp_state_g, temp_state_u, temp_intermediate_g,
           temp_intermediate_u, static_cast<int>(act_function),
           static_cast<int>(K_gate), static_cast<int>(K_up),
           static_cast<int>(K_down), gate_ptrs_trellis, gate_ptrs_suh,
           gate_ptrs_svh, up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
           down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh, gate_mcg, gate_mul1,
           up_mcg, up_mul1, down_mcg, down_mul1, static_cast<float>(act_limit));
}

#endif  // not USE_ROCM

// ---------------------------------------------------------------------------
// Schemas — registered as a fragment of the existing torch.ops._C library.
// ---------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "dry_scan_penalties("
      "    Tensor token_history_ids,"
      "    Tensor token_history_lens,"
      "    Tensor dry_multiplier,"
      "    Tensor allowed_lengths,"
      "    Tensor sequence_breakers_ids,"
      "    Tensor ranges,"
      "    Tensor max_ngram,"
      "    Tensor max_occurrences,"
      "    Tensor early_exit_match_len,"
      "    int vocab_size) -> (Tensor, Tensor, Tensor)");

#ifndef USE_ROCM
  ops.def(
      "exl3_gemm(Tensor a, Tensor b, Tensor! c, Tensor? suh, Tensor? a_had, "
      "Tensor? svh, int force_shape_idx, bool mcg, bool mul1, "
      "int force_num_sms) -> ()");
  ops.def(
      "exl3_mgemm(Tensor a, Tensor b, Tensor! c, Tensor suh, Tensor! a_had, "
      "Tensor svh, Tensor? indices, Tensor? weights, int k, "
      "int force_shape_idx, bool mcg, bool mul1, int min_index, "
      "int max_index, int force_num_sms) -> ()");
  ops.def(
      "exl3_reconstruct(Tensor! unpacked, Tensor packed, int k, bool mcg, "
      "bool mul1) -> ()");
  ops.def(
      "exl3_had_r_128(Tensor input, Tensor! output, Tensor? pre_scale, "
      "Tensor? post_scale, float scale) -> ()");
  ops.def("exl3_hgemm(Tensor a, Tensor b, Tensor! c) -> ()");
  ops.def(
      "exl3_moe(Tensor hidden_state, Tensor! output_state, Tensor "
      "expert_count, Tensor token_sorted, Tensor weight_sorted, Tensor "
      "temp_state_g, Tensor temp_state_u, Tensor temp_intermediate_g, Tensor "
      "temp_intermediate_u, int act_function, int K_gate, int K_up, int "
      "K_down, Tensor gate_ptrs_trellis, Tensor gate_ptrs_suh, Tensor "
      "gate_ptrs_svh, Tensor up_ptrs_trellis, Tensor up_ptrs_suh, Tensor "
      "up_ptrs_svh, Tensor down_ptrs_trellis, Tensor down_ptrs_suh, Tensor "
      "down_ptrs_svh, bool gate_mcg, bool gate_mul1, bool up_mcg, bool "
      "up_mul1, bool down_mcg, bool down_mul1, float act_limit) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(_C, CPU, ops) {
  ops.impl("dry_scan_penalties", &dry_scan_penalties_cpu);
}

#ifndef USE_ROCM
TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("exl3_gemm", &aphrodite_exl3_gemm);
  ops.impl("exl3_mgemm", &aphrodite_exl3_mgemm);
  ops.impl("exl3_reconstruct", &aphrodite_exl3_reconstruct);
  ops.impl("exl3_had_r_128", &aphrodite_exl3_had_r_128);
  ops.impl("exl3_hgemm", &aphrodite_exl3_hgemm);
  ops.impl("exl3_moe", &aphrodite_exl3_moe);
}
#endif

REGISTER_EXTENSION(_C_fork)
