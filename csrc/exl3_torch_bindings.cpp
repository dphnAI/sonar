// Sonar EXL3 op registrations in the libtorch stable ABI extension.
//
// Upstream vLLM registers core/quant ops in the _C_stable_libtorch extension
// Sonar keeps these fork-only kernels in the torch.ops._C namespace so
// Python callers remain unchanged. The heavy kernels live in their own
// translation units; here we only adapt scalar argument types and register.
//
// Currently built: EXL3 quantization (CUDA). The other
// fork kernels (exl2/aqlm/vptq/gguf/quip/...) remain in csrc but are not built.

#include <torch/csrc/stable/library.h>

#include <optional>

#include "quantization/exl3/exllamav3_ext/hgemm.cuh"
#include "quantization/exl3/exllamav3_ext/quant/exl3_gemm.cuh"
#include "quantization/exl3/exllamav3_ext/quant/exl3_moe.cuh"
#include "quantization/exl3/exllamav3_ext/quant/hadamard.cuh"
#include "quantization/exl3/exllamav3_ext/quant/reconstruct.cuh"
#include "quantization/exl3/exllamav3_ext/quant/util.cuh"

// Thin adapters between the torch schema (int64/double/bool) and the native
// kernel entry points (int/float). Defined here rather than in a shared header
// so exl3 stays self-contained and conventional.
static void sonar_exl3_gemm(const torch::stable::Tensor& A,
                            const torch::stable::Tensor& B,
                            torch::stable::Tensor& C,
                            const std::optional<torch::stable::Tensor>& suh,
                            const std::optional<torch::stable::Tensor>& A_had,
                            const std::optional<torch::stable::Tensor>& svh,
                            int64_t force_shape_idx, bool mcg, bool mul1,
                            int64_t force_num_sms) {
  exl3_gemm(A, B, C, suh, A_had, svh, static_cast<int>(force_shape_idx), mcg,
            mul1, static_cast<int>(force_num_sms));
}

static void sonar_exl3_mgemm(
    const torch::stable::Tensor& A, const torch::stable::Tensor& B,
    torch::stable::Tensor& C, const torch::stable::Tensor& suh,
    const torch::stable::Tensor& A_had, const torch::stable::Tensor& svh,
    const std::optional<torch::stable::Tensor>& indices,
    const std::optional<torch::stable::Tensor>& weights, int64_t k,
    int64_t force_shape_idx, bool mcg, bool mul1, int64_t min_index,
    int64_t max_index, int64_t force_num_sms) {
  exl3_mgemm(A, B, C, suh, A_had, svh, indices, weights, static_cast<int>(k),
             static_cast<int>(force_shape_idx), static_cast<uint32_t>(mcg),
             static_cast<uint32_t>(mul1), static_cast<int>(min_index),
             static_cast<int>(max_index), static_cast<int>(force_num_sms));
}

static void sonar_exl3_reconstruct(torch::stable::Tensor unpacked,
                                   torch::stable::Tensor packed, int64_t k,
                                   bool mcg, bool mul1) {
  reconstruct(unpacked, packed, static_cast<int>(k), mcg, mul1);
}

static void sonar_exl3_reconstruct_slice(torch::stable::Tensor unpacked,
                                         torch::stable::Tensor packed,
                                         int64_t k, bool mcg, bool mul1,
                                         int64_t n_offset) {
  reconstruct_slice(unpacked, packed, static_cast<int>(k), mcg, mul1, n_offset);
}

static void sonar_exl3_had_r_128(
    const torch::stable::Tensor& input, const torch::stable::Tensor& output,
    const std::optional<torch::stable::Tensor>& pre_scale,
    const std::optional<torch::stable::Tensor>& post_scale, double scale) {
  had_r_128(input, output, pre_scale, post_scale, static_cast<float>(scale));
}

static void sonar_exl3_had_r_128_dual(
    const torch::stable::Tensor& input1, const torch::stable::Tensor& output1,
    const std::optional<torch::stable::Tensor>& pre_scale1,
    const std::optional<torch::stable::Tensor>& post_scale1,
    const torch::stable::Tensor& input2, const torch::stable::Tensor& output2,
    const std::optional<torch::stable::Tensor>& pre_scale2,
    const std::optional<torch::stable::Tensor>& post_scale2, double scale) {
  had_r_128_dual(input1, output1, pre_scale1, post_scale1, input2, output2,
                 pre_scale2, post_scale2, static_cast<float>(scale));
}

static void sonar_exl3_hgemm(torch::stable::Tensor a, torch::stable::Tensor b,
                             torch::stable::Tensor c) {
  hgemm(a, b, c);
}

static void sonar_exl3_moe(const torch::stable::Tensor& hidden_state,
                           const torch::stable::Tensor& output_state,
                           const torch::stable::Tensor& expert_count,
                           const torch::stable::Tensor& token_sorted,
                           const torch::stable::Tensor& weight_sorted,
                           const torch::stable::Tensor& temp_state_g,
                           const torch::stable::Tensor& temp_state_u,
                           const torch::stable::Tensor& temp_intermediate_g,
                           const torch::stable::Tensor& temp_intermediate_u,
                           int64_t act_function, int64_t K_gate, int64_t K_up,
                           int64_t K_down,
                           const torch::stable::Tensor& gate_ptrs_trellis,
                           const torch::stable::Tensor& gate_ptrs_suh,
                           const torch::stable::Tensor& gate_ptrs_svh,
                           const torch::stable::Tensor& up_ptrs_trellis,
                           const torch::stable::Tensor& up_ptrs_suh,
                           const torch::stable::Tensor& up_ptrs_svh,
                           const torch::stable::Tensor& down_ptrs_trellis,
                           const torch::stable::Tensor& down_ptrs_suh,
                           const torch::stable::Tensor& down_ptrs_svh,
                           bool gate_mcg, bool gate_mul1, bool up_mcg,
                           bool up_mul1, bool down_mcg, bool down_mul1,
                           double act_limit) {
  exl3_moe(
      hidden_state, output_state, expert_count, token_sorted, weight_sorted,
      temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
      static_cast<int>(act_function), static_cast<int>(K_gate),
      static_cast<int>(K_up), static_cast<int>(K_down), gate_ptrs_trellis,
      gate_ptrs_suh, gate_ptrs_svh, up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
      down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh, gate_mcg, gate_mul1,
      up_mcg, up_mul1, down_mcg, down_mul1, static_cast<float>(act_limit), -1);
}

static void sonar_exl3_moe_active(
    const torch::stable::Tensor& hidden_state,
    const torch::stable::Tensor& output_state,
    const torch::stable::Tensor& expert_count,
    const torch::stable::Tensor& token_sorted,
    const torch::stable::Tensor& weight_sorted,
    const torch::stable::Tensor& temp_state_g,
    const torch::stable::Tensor& temp_state_u,
    const torch::stable::Tensor& temp_intermediate_g,
    const torch::stable::Tensor& temp_intermediate_u, int64_t act_function,
    int64_t K_gate, int64_t K_up, int64_t K_down,
    const torch::stable::Tensor& gate_ptrs_trellis,
    const torch::stable::Tensor& gate_ptrs_suh,
    const torch::stable::Tensor& gate_ptrs_svh,
    const torch::stable::Tensor& up_ptrs_trellis,
    const torch::stable::Tensor& up_ptrs_suh,
    const torch::stable::Tensor& up_ptrs_svh,
    const torch::stable::Tensor& down_ptrs_trellis,
    const torch::stable::Tensor& down_ptrs_suh,
    const torch::stable::Tensor& down_ptrs_svh, bool gate_mcg, bool gate_mul1,
    bool up_mcg, bool up_mul1, bool down_mcg, bool down_mul1, double act_limit,
    int64_t num_active) {
  exl3_moe(hidden_state, output_state, expert_count, token_sorted,
           weight_sorted, temp_state_g, temp_state_u, temp_intermediate_g,
           temp_intermediate_u, static_cast<int>(act_function),
           static_cast<int>(K_gate), static_cast<int>(K_up),
           static_cast<int>(K_down), gate_ptrs_trellis, gate_ptrs_suh,
           gate_ptrs_svh, up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
           down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh, gate_mcg, gate_mul1,
           up_mcg, up_mul1, down_mcg, down_mul1, static_cast<float>(act_limit),
           static_cast<int>(num_active));
}

// ---------------------------------------------------------------------------
// Schemas and implementations in the existing torch.ops._C library.
// ---------------------------------------------------------------------------
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
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
      "exl3_reconstruct_slice(Tensor! unpacked, Tensor packed, int k, bool "
      "mcg, bool mul1, int n_offset) -> ()");
  ops.def(
      "exl3_had_r_128(Tensor input, Tensor! output, Tensor? pre_scale, "
      "Tensor? post_scale, float scale) -> ()");
  ops.def(
      "exl3_had_r_128_dual(Tensor input1, Tensor! output1, Tensor? "
      "pre_scale1, Tensor? post_scale1, Tensor input2, Tensor! output2, "
      "Tensor? pre_scale2, Tensor? post_scale2, float scale) -> ()");
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
  ops.def(
      "exl3_moe_active(Tensor hidden_state, Tensor! output_state, Tensor "
      "expert_count, Tensor token_sorted, Tensor weight_sorted, Tensor "
      "temp_state_g, Tensor temp_state_u, Tensor temp_intermediate_g, Tensor "
      "temp_intermediate_u, int act_function, int K_gate, int K_up, int "
      "K_down, Tensor gate_ptrs_trellis, Tensor gate_ptrs_suh, Tensor "
      "gate_ptrs_svh, Tensor up_ptrs_trellis, Tensor up_ptrs_suh, Tensor "
      "up_ptrs_svh, Tensor down_ptrs_trellis, Tensor down_ptrs_suh, Tensor "
      "down_ptrs_svh, bool gate_mcg, bool gate_mul1, bool up_mcg, bool "
      "up_mul1, bool down_mcg, bool down_mul1, float act_limit, int "
      "num_active) -> ()");
  ops.def(
      "make_gate_up_indices(Tensor! out, Tensor indices, int offset) -> ()");
  ops.def("silu_mul(Tensor! out, Tensor gate, Tensor up) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("exl3_gemm", TORCH_BOX(&sonar_exl3_gemm));
  ops.impl("exl3_mgemm", TORCH_BOX(&sonar_exl3_mgemm));
  ops.impl("exl3_reconstruct", TORCH_BOX(&sonar_exl3_reconstruct));
  ops.impl("exl3_reconstruct_slice", TORCH_BOX(&sonar_exl3_reconstruct_slice));
  ops.impl("exl3_had_r_128", TORCH_BOX(&sonar_exl3_had_r_128));
  ops.impl("exl3_had_r_128_dual", TORCH_BOX(&sonar_exl3_had_r_128_dual));
  ops.impl("exl3_hgemm", TORCH_BOX(&sonar_exl3_hgemm));
  ops.impl("exl3_moe", TORCH_BOX(&sonar_exl3_moe));
  ops.impl("exl3_moe_active", TORCH_BOX(&sonar_exl3_moe_active));
  ops.impl("make_gate_up_indices", TORCH_BOX(&exl3_make_gate_up_indices));
  ops.impl("silu_mul", TORCH_BOX(&exl3_silu_mul));
}
